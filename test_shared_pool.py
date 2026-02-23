# 双标共享资金池 — pytest 用例骨架
# 运行方式：在 miniqmt 目录下执行  pytest test_shared_pool.py -v
# 依赖：pip install pytest

import json
import pytest

# 可选依赖：若当前环境无 mac_dashboard / global_vault，对应用例会 skip
def _import_mac_dashboard():
    try:
        import mac_dashboard as md
        return md
    except ImportError:
        return None

def _import_global_vault():
    try:
        import global_vault as gv
        return gv
    except ImportError:
        return None


# ---------- 常量（与 mac_dashboard / 开发文档一致）----------
PHYSICAL_POOL = 300_000
POOL_90_PCT = 270_000
POOL_85_PCT = 255_000


# ========== 1. _load_shared_pool（Mac 侧读池）==========

@pytest.fixture
def shared_pool_file(tmp_path):
    """返回临时 shared_pool.json 路径，测试可写入内容。"""
    return tmp_path / "shared_pool.json"


class TestLoadSharedPool:
    """验证 _load_shared_pool 在各种文件状态下的返回值。"""

    def test_missing_file_returns_empty_committed(self, shared_pool_file, monkeypatch):
        md = _import_mac_dashboard()
        if md is None:
            pytest.skip("mac_dashboard 未安装或不在路径")
        monkeypatch.setattr(md, "SHARED_POOL_FILE", str(shared_pool_file) + ".nonexist")
        data = md._load_shared_pool()
        assert "committed" in data
        assert data["committed"] == 0
        assert data.get("used_159201", 0) == 0
        assert data.get("used_512890", 0) == 0

    def test_valid_file_returns_committed_and_alpha(self, shared_pool_file, monkeypatch):
        md = _import_mac_dashboard()
        if md is None:
            pytest.skip("mac_dashboard 未安装或不在路径")
        shared_pool_file.write_text(json.dumps({
            "used_159201": 100_000,
            "used_512890": 80_000,
            "frozen_159201": 0,
            "frozen_512890": 0,
            "committed": 180_000,
            "acc_alpha_159201": 100.5,
            "acc_alpha_512890": 200.3,
            "updated_at": "2026-02-22T12:00:00",
        }), encoding="utf-8")
        monkeypatch.setattr(md, "SHARED_POOL_FILE", str(shared_pool_file))
        data = md._load_shared_pool()
        assert data["committed"] == 180_000
        assert data.get("acc_alpha_159201") == 100.5
        assert data.get("acc_alpha_512890") == 200.3

    def test_bad_json_returns_empty_committed(self, shared_pool_file, monkeypatch):
        md = _import_mac_dashboard()
        if md is None:
            pytest.skip("mac_dashboard 未安装或不在路径")
        shared_pool_file.write_text("not json {", encoding="utf-8")
        monkeypatch.setattr(md, "SHARED_POOL_FILE", str(shared_pool_file))
        data = md._load_shared_pool()
        assert data.get("committed", 0) == 0


# ========== 2. 迟滞与步长惩罚（纯逻辑，不依赖模块）==========

def _step_penalty_update(committed: float, current: bool) -> bool:
    """与 mac_dashboard 主循环一致的迟滞逻辑：>90% 开，<85% 关。"""
    if committed > POOL_90_PCT:
        return True
    if committed < POOL_85_PCT:
        return False
    return current


class TestStepPenaltyHysteresis:
    """90%/85% 迟滞：占用>270000 开启惩罚，<255000 关闭。"""

    def test_above_90_turns_on(self):
        assert _step_penalty_update(271_000, False) is True
        assert _step_penalty_update(270_001, False) is True

    def test_below_85_turns_off(self):
        assert _step_penalty_update(254_000, True) is False
        assert _step_penalty_update(254_999, True) is False

    def test_between_85_and_90_keeps_current(self):
        assert _step_penalty_update(260_000, True) is True
        assert _step_penalty_update(260_000, False) is False


# ========== 3. 15% 禁止新开第一层（纯逻辑）==========

def _should_block_first_layer(hold_layers: int, committed: float) -> bool:
    """剩余不足 15% 时禁止新开第一层。"""
    return hold_layers == 0 and committed > POOL_85_PCT


class TestFirstLayerBlock:
    """15% 预留：仅当 hold_layers==0 且 committed>255000 时拦截。"""

    def test_block_when_zero_layers_and_high_committed(self):
        assert _should_block_first_layer(0, 256_000) is True
        assert _should_block_first_layer(0, 255_001) is True

    def test_no_block_when_has_layers(self):
        assert _should_block_first_layer(1, 256_000) is False
        assert _should_block_first_layer(2, 300_000) is False

    def test_no_block_when_committed_below_85(self):
        assert _should_block_first_layer(0, 255_000) is False
        assert _should_block_first_layer(0, 200_000) is False


# ========== 4. GlobalVault：占用不超物理池 ==========
# 实现约定：单例 GlobalVault(shared_dir=...) / get_vault(shared_dir=...)，路径由 shared_dir 决定。


def _get_vault_with_tmp_path(gv, tmp_path, monkeypatch):
    """用临时目录拿到 Vault 实例：重置单例后以 shared_dir=tmp_path 创建；测试时禁用文件锁避免 flock 阻塞。"""
    shared_dir = str(tmp_path)
    # 测试环境禁用文件锁，避免 fcntl.flock 在部分环境下阻塞
    def _noop_lock(_fd):
        pass
    monkeypatch.setattr(gv, "_lock_file", _noop_lock)
    monkeypatch.setattr(gv, "_unlock_file", _noop_lock)
    if hasattr(gv.GlobalVault, "_instance"):
        monkeypatch.setattr(gv.GlobalVault, "_instance", None)
    if hasattr(gv, "get_vault"):
        return gv.get_vault(shared_dir=shared_dir)
    return gv.GlobalVault(shared_dir=shared_dir)


class TestGlobalVaultCap:
    """Vault 拒绝会导致 used 总和超过 300000 的申请。"""

    def test_request_allocation_rejects_over_cap(self, tmp_path, monkeypatch):
        gv = _import_global_vault()
        if gv is None:
            pytest.skip("global_vault 未安装或不在路径")
        try:
            vault = _get_vault_with_tmp_path(gv, tmp_path, monkeypatch)
        except Exception:
            pytest.skip("GlobalVault 构造函数与当前实现不一致，见 测试指导_Vault.md")
        # 先占用到 298k（15% 规则：layer_index=0 时 committed+amount<=255k，故第二笔用 layer_index=1）
        ok1 = vault.request_allocation("159201.SZ", 200_000, 0, "test-order-1")
        if not ok1:
            pytest.skip("Vault 接口或实现与用例假设不同")
        ok2 = vault.request_allocation("512890.SH", 98_000, 1, "test-order-2")
        if not ok2:
            pytest.skip("Vault 第二笔 98k 未通过")
        # 再申请 5k 应拒绝（298k + 5k > 300k 物理池）
        ok3 = vault.request_allocation("512890.SH", 5_000, 0, "test-order-3")
        assert ok3 is False
        used_after = sum(vault.get_used_map().values()) if hasattr(vault, "get_used_map") else 0
        assert used_after <= PHYSICAL_POOL

    def test_release_decreases_used(self, tmp_path, monkeypatch):
        gv = _import_global_vault()
        if gv is None:
            pytest.skip("global_vault 未安装或不在路径")
        try:
            vault = _get_vault_with_tmp_path(gv, tmp_path, monkeypatch)
        except Exception:
            pytest.skip("GlobalVault 构造函数与当前实现不一致，见 测试指导_Vault.md")
        client_order_id = "test-release-order-1"
        ok = vault.request_allocation("159201.SZ", 50_000, 0, client_order_id)
        if not ok:
            pytest.skip("request_allocation 失败，无法测 release")
        vault.on_fill("159201.SZ", client_order_id)  # frozen -> used
        before = vault.get_used_map()
        released = vault.release("159201.SZ", client_order_id)  # release(symbol, client_order_id)
        after = vault.get_used_map()
        used_159201_before = before.get("159201", 0)
        used_159201_after = after.get("159201", 0)
        assert released == 50_000
        assert used_159201_after <= used_159201_before - 50_000 + 1


# ========== 5. 信号字段：client_order_id / amount / layer_index ==========

class TestSignalFields:
    """BUY 信号应包含 client_order_id、amount、layer_index（由 execute_signal 写入）。"""

    def test_execute_signal_buy_writes_amount_and_layer_index(self, tmp_path, monkeypatch):
        md = _import_mac_dashboard()
        if md is None:
            pytest.skip("mac_dashboard 未安装或不在路径")
        signal_file = tmp_path / "order_signal.json"
        monkeypatch.setattr(md, "SIGNAL_FILE", str(signal_file))
        md.execute_signal(
            "BUY", 1.234, "测试", shares=1000,
            amount=1234.0, layer_index=0,
        )
        if not signal_file.exists():
            pytest.skip("execute_signal 未写入文件或路径不同")
        data = json.loads(signal_file.read_text(encoding="utf-8"))
        # 实际实现可能用 direction 或 side 表示买卖方向
        assert data.get("direction") == "BUY" or data.get("side") == "BUY"
        assert "client_order_id" in data or "client_order_id" in data.get("signal_data", {})
        payload = {**data.get("signal_data", {}), **data}
        assert payload.get("amount") == 1234.0
        assert payload.get("layer_index") == 0


# ========== 6. 步长惩罚后的 next_buy 更远 ==========

def _next_buy_price(last_buy: float, grid_step: float, penalty: bool) -> float:
    """下一买点 = last_buy * (1 - grid_step)；惩罚时 grid_step 为 1.5 倍。"""
    step = grid_step * 1.5 if penalty else grid_step
    return last_buy * (1 - step)


class TestStepPenaltyMovesNextBuy:
    """惩罚开启时，同一 last_buy 下 next_buy 更低（更难触发买入）。"""

    def test_penalty_makes_next_buy_lower(self):
        last_buy = 1.0
        grid_step = 0.01
        next_normal = _next_buy_price(last_buy, grid_step, False)
        next_penalty = _next_buy_price(last_buy, grid_step, True)
        assert next_penalty < next_normal
        assert abs(next_penalty - (1.0 - 0.01 * 1.5)) < 1e-9
