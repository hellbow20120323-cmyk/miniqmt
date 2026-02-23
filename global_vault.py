"""
Windows 端：全局资金管理器 (GlobalVault)，双标的共享物理资金池。
单例模式，跨进程通过 shared_pool.json 持久化，写入时使用文件锁防止 159201/512890 两进程同时写冲突。
参见 开发文档_双标的共享资金池.md 第三节。
"""
import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

# --- 配置（与 order_executor 共用 SHARED_DIR）---
SHARED_DIR = os.environ.get("MINIQMT_SHARED_DIR", r"C:\Mac\Home\Documents\miniqmt")
SHARED_POOL_PATH = os.path.join(SHARED_DIR, "shared_pool.json")
LOCK_FILE_PATH = os.path.join(SHARED_DIR, "shared_pool.lock")

PHYSICAL_POOL = 300_000
MIN_RESERVE_RATIO = 0.15  # 剩余 15% 禁止新开 Layer 0

# 标的键：内部统一用 159201 / 512890
SYMBOLS = ("159201", "512890")

logger = logging.getLogger(__name__)

# Windows 文件锁
if sys.platform == "win32":
    import msvcrt

    def _lock_file(fd):
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    def _unlock_file(fd):
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    # 非 Windows（如 Mac 开发/测试）可用 fcntl 或仅单进程
    try:
        import fcntl
        FCNTL_AVAILABLE = True
    except ImportError:
        FCNTL_AVAILABLE = False

    def _lock_file(fd):
        if FCNTL_AVAILABLE:
            fcntl.flock(fd, fcntl.LOCK_EX)
        # else no-op for single process

    def _unlock_file(fd):
        if FCNTL_AVAILABLE:
            fcntl.flock(fd, fcntl.LOCK_UNLCK)


def _normalize_symbol(symbol: str) -> str:
    """159201.SZ / 512890.SH -> 159201 / 512890"""
    s = (symbol or "").strip().upper()
    for sym in SYMBOLS:
        if s.startswith(sym) or s == sym:
            return sym
    if "159201" in s:
        return "159201"
    if "512890" in s:
        return "512890"
    return s


def _default_state() -> dict:
    return {
        "used_159201": 0.0,
        "used_512890": 0.0,
        "frozen_159201": 0.0,
        "frozen_512890": 0.0,
        "allocations_159201": [],  # list of {"client_order_id": str, "amount": float}
        "allocations_512890": [],
        "frozen_159201_entries": [],
        "frozen_512890_entries": [],
        "acc_alpha_159201": 0.0,
        "acc_alpha_512890": 0.0,
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }


class GlobalVault:
    """
    全局资金管理器单例。跨进程通过 shared_pool.json + 文件锁 保证读写互斥。
    """

    _instance: Optional["GlobalVault"] = None
    _lock_holder: Optional[int] = None  # fd of lock file when held

    def __new__(cls, shared_dir: Optional[str] = None) -> "GlobalVault":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, shared_dir: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
        self._shared_dir = shared_dir or SHARED_DIR
        self._pool_path = os.path.join(self._shared_dir, "shared_pool.json")
        self._lock_path = os.path.join(self._shared_dir, "shared_pool.lock")
        self._state = _default_state()
        self._lock_fd: Optional[int] = None
        self._initialized = True

    def _acquire_lock(self, timeout_sec: float = 30.0) -> bool:
        """打开锁文件并加锁，返回是否成功。"""
        if self._lock_fd is not None:
            return True
        start = time.time()
        while time.time() - start < timeout_sec:
            try:
                os.makedirs(os.path.dirname(self._lock_path) or ".", exist_ok=True)
                fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
                _lock_file(fd)
                self._lock_fd = fd
                return True
            except (OSError, IOError) as e:
                logger.debug("GlobalVault lock wait: %s", e)
                time.sleep(0.05)
        logger.error("GlobalVault: failed to acquire lock within %.1fs", timeout_sec)
        return False

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            _unlock_file(self._lock_fd)
            os.close(self._lock_fd)
        except Exception as e:
            logger.warning("GlobalVault unlock: %s", e)
        self._lock_fd = None

    def _load(self) -> None:
        """从 shared_pool.json 加载到 self._state（调用前需已持锁）。先重置为默认再合并文件。"""
        self._state = _default_state()
        if not os.path.exists(self._pool_path):
            return
        try:
            with open(self._pool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in _default_state():
                if k in data:
                    self._state[k] = data[k]
            if "frozen_159201_entries" not in self._state or not isinstance(self._state["frozen_159201_entries"], list):
                self._state["frozen_159201_entries"] = []
            if "frozen_512890_entries" not in self._state or not isinstance(self._state["frozen_512890_entries"], list):
                self._state["frozen_512890_entries"] = []
            if "allocations_159201" not in self._state or not isinstance(self._state["allocations_159201"], list):
                self._state["allocations_159201"] = []
            if "allocations_512890" not in self._state or not isinstance(self._state["allocations_512890"], list):
                self._state["allocations_512890"] = []
        except Exception as e:
            logger.warning("GlobalVault load %s: %s", self._pool_path, e)
            self._state = _default_state()

    def _save(self) -> None:
        """将 self._state 写入 shared_pool.json（调用前需已持锁）。"""
        self._state["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        out = {
            "used_159201": round(self._state["used_159201"], 2),
            "used_512890": round(self._state["used_512890"], 2),
            "frozen_159201": round(self._state["frozen_159201"], 2),
            "frozen_512890": round(self._state["frozen_512890"], 2),
            "acc_alpha_159201": round(self._state["acc_alpha_159201"], 2),
            "acc_alpha_512890": round(self._state["acc_alpha_512890"], 2),
            "updated_at": self._state["updated_at"],
            "allocations_159201": self._state.get("allocations_159201", []),
            "allocations_512890": self._state.get("allocations_512890", []),
            "frozen_159201_entries": self._state.get("frozen_159201_entries", []),
            "frozen_512890_entries": self._state.get("frozen_512890_entries", []),
        }
        tmp = self._pool_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            if os.path.exists(self._pool_path):
                os.remove(self._pool_path)
            os.rename(tmp, self._pool_path)
        except Exception as e:
            logger.error("GlobalVault save %s: %s", self._pool_path, e)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _with_lock(self, fn):
        """持锁执行 fn()，先 load 再执行 fn（fn 内可改 self._state）再 save。"""
        if not self._acquire_lock():
            raise RuntimeError("GlobalVault: could not acquire lock")
        try:
            self._load()
            result = fn()
            self._save()
            return result
        finally:
            self._release_lock()

    # ---------- 对外接口 ----------

    def request_allocation(
        self,
        symbol: str,
        amount: float,
        layer_index: int,
        client_order_id: str,
    ) -> bool:
        """
        申请占用：将 amount 划入该标的 frozen。
        layer_index==0 时需满足 (used+frozen) 合计后物理池剩余 >= 15%。
        返回 True 表示成功，False 表示拒绝。
        """
        amount = float(amount)
        if amount <= 0:
            return False
        key = _normalize_symbol(symbol)
        if key not in SYMBOLS:
            return False

        def _do():
            used = self._state["used_159201"] + self._state["used_512890"]
            frozen = self._state["frozen_159201"] + self._state["frozen_512890"]
            committed = used + frozen
            if layer_index == 0:
                if committed + amount > PHYSICAL_POOL * (1 - MIN_RESERVE_RATIO):
                    return False
            if committed + amount > PHYSICAL_POOL:
                return False
            self._state[f"frozen_{key}"] = self._state.get(f"frozen_{key}", 0) + amount
            entries = self._state.get(f"frozen_{key}_entries", [])
            if not isinstance(entries, list):
                entries = []
            entries.append({"client_order_id": client_order_id, "amount": amount})
            self._state[f"frozen_{key}_entries"] = entries
            return True

        try:
            return self._with_lock(lambda: _do())
        except Exception as e:
            logger.exception("request_allocation: %s", e)
            return False

    def on_fill(self, symbol: str, client_order_id: str) -> None:
        """成交回报：将 client_order_id 对应金额从 frozen 转入 used，并加入 allocations。"""
        key = _normalize_symbol(symbol)
        if key not in SYMBOLS:
            return

        def _do():
            entries = self._state.get(f"frozen_{key}_entries", []) or []
            for i, e in enumerate(entries):
                if e.get("client_order_id") == client_order_id:
                    amt = float(e.get("amount", 0))
                    self._state[f"frozen_{key}"] = max(0, self._state.get(f"frozen_{key}", 0) - amt)
                    self._state[f"used_{key}"] = self._state.get(f"used_{key}", 0) + amt
                    alloc = self._state.get(f"allocations_{key}", []) or []
                    alloc.append({"client_order_id": client_order_id, "amount": amt})
                    self._state[f"allocations_{key}"] = alloc
                    del entries[i]
                    self._state[f"frozen_{key}_entries"] = entries
                    return
            logger.warning("on_fill: client_order_id %s not found in frozen_%s", client_order_id, key)

        self._with_lock(_do)

    def on_cancel(self, symbol: str, client_order_id: str) -> None:
        """撤单/废单：释放该 client_order_id 对应的 frozen。"""
        key = _normalize_symbol(symbol)
        if key not in SYMBOLS:
            return

        def _do():
            entries = self._state.get(f"frozen_{key}_entries", []) or []
            for i, e in enumerate(entries):
                if e.get("client_order_id") == client_order_id:
                    amt = float(e.get("amount", 0))
                    self._state[f"frozen_{key}"] = max(0, self._state.get(f"frozen_{key}", 0) - amt)
                    del entries[i]
                    self._state[f"frozen_{key}_entries"] = entries
                    return
            logger.warning("on_cancel: client_order_id %s not found in frozen_%s", client_order_id, key)

        self._with_lock(_do)

    def release(self, symbol: str, client_order_id: str) -> Optional[float]:
        """
        平仓释放：按 client_order_id 从 used 及 allocations 中移除该笔金额。
        返回释放的金额，若未找到返回 None。
        """
        key = _normalize_symbol(symbol)
        if key not in SYMBOLS:
            return None

        def _do():
            alloc = self._state.get(f"allocations_{key}", []) or []
            for i, e in enumerate(alloc):
                if e.get("client_order_id") == client_order_id:
                    amt = float(e.get("amount", 0))
                    self._state[f"used_{key}"] = max(0, self._state.get(f"used_{key}", 0) - amt)
                    del alloc[i]
                    self._state[f"allocations_{key}"] = alloc
                    return amt
            logger.warning("release: client_order_id %s not found in allocations_%s", client_order_id, key)
            return None

        return self._with_lock(_do)

    def add_alpha(self, symbol: str, pnl: float) -> None:
        """累计 Alpha：平仓实现盈亏累加到对应标的 acc_alpha。"""
        key = _normalize_symbol(symbol)
        if key not in SYMBOLS:
            return
        pnl = float(pnl)

        def _do():
            self._state[f"acc_alpha_{key}"] = self._state.get(f"acc_alpha_{key}", 0) + pnl

        self._with_lock(_do)

    def get_used_map(self) -> Dict[str, float]:
        """返回各标的已占用金额（used）。"""
        def _do():
            return {
                "159201": self._state.get("used_159201", 0),
                "512890": self._state.get("used_512890", 0),
            }
        return self._with_lock(_do)

    def get_frozen_map(self) -> Dict[str, float]:
        """返回各标的冻结中金额（frozen）。"""
        def _do():
            return {
                "159201": self._state.get("frozen_159201", 0),
                "512890": self._state.get("frozen_512890", 0),
            }
        return self._with_lock(_do)

    def get_committed_total(self) -> Tuple[float, float]:
        """返回 (used_total, frozen_total)。"""
        def _do():
            u = self._state.get("used_159201", 0) + self._state.get("used_512890", 0)
            f = self._state.get("frozen_159201", 0) + self._state.get("frozen_512890", 0)
            return (u, f)
        return self._with_lock(_do)

    def can_open_new_layer(self, symbol: str, amount: float) -> bool:
        """物理池剩余 >= 15% 且 committed+amount <= PHYSICAL_POOL 时允许新开第一层。"""
        amount = float(amount)
        if amount <= 0:
            return False

        def _do():
            used = self._state.get("used_159201", 0) + self._state.get("used_512890", 0)
            frozen = self._state.get("frozen_159201", 0) + self._state.get("frozen_512890", 0)
            committed = used + frozen
            if committed + amount > PHYSICAL_POOL * (1 - MIN_RESERVE_RATIO):
                return False
            if committed + amount > PHYSICAL_POOL:
                return False
            return True

        return self._with_lock(_do)


def get_vault(shared_dir: Optional[str] = None) -> GlobalVault:
    """获取 GlobalVault 单例。"""
    return GlobalVault(shared_dir=shared_dir)
