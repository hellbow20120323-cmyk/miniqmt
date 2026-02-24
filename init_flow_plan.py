"""
初始化新建仓计划：沿用现有持仓、重置成本基准（单标 159201）。

用途：
- 收盘后读取 shared_quote_159201.json
- 以当日收盘价（或当前价）作为新的成本基准，将已有持仓合并为一笔仓位
- 写入 dashboard_state.json，供 mac_dashboard.py 使用

运行：
    python init_flow_plan.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple


_SCRIPT_DIR = os.environ.get("DASHBOARD_WORK_DIR") or os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_SCRIPT_DIR, "dashboard_state.json")
SHARED_FILE_159201 = os.path.join(_SCRIPT_DIR, "shared_quote_159201.json")


def _fresh_state() -> Dict[str, Any]:
    """基础状态模板，字段与 mac_dashboard 的使用保持一致。"""
    return {
        "last_buy_price": None,
        "hold_layers": 0,
        "total_cost": 0.0,
        "hold_t0_volume": 0,
        "last_sell_timestamp": None,
        "positions": [],
        "last_sent_signal_id": None,
        "last_sent_signal_direction": None,
        "last_sent_signal_shares": None,
        "last_sent_signal_price": None,
        "last_sent_buy_prev_anchor": None,
        "last_sent_sell_removed_lots": [],
        "last_sent_was_topup": False,
        "last_applied_result_signal_id": None,
        "pending_buy_shares": 0,
        "pending_buy_price": None,
        "pending_buy_since": None,
        # 卖出相关 pending 字段，mac_dashboard 里会用到
        "pending_sell_since": None,
        "pending_sell_volume": 0,
    }


def _load_state(path: str) -> Dict[str, Any]:
    """加载已有 state 文件，如不存在则使用 fresh 模板。"""
    base = _fresh_state()
    if not os.path.exists(path):
        return base
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # 用已有字段覆盖默认模板，保证缺失字段也有默认值
        base.update(data)
        return base
    except Exception:
        return base


def _read_quote(shared_file: str) -> Tuple[int, Optional[float]]:
    """
    从 shared_quote 文件中读取：
    - 当前真实持仓股数：position.volume
    - 今日收盘价 today_close：优先 history[-1]，否则 price
    """
    if not os.path.exists(shared_file):
        return 0, None

    try:
        with open(shared_file, "r") as f:
            data = json.load(f)
    except Exception:
        return 0, None

    # 持仓股数
    vol = 0
    try:
        pos = data.get("position") or {}
        vol = int(pos.get("volume") or 0)
    except Exception:
        vol = 0

    # 今日收盘价：优先 K 线 history[-1]，否则用当前价 price
    today_close: Optional[float] = None
    try:
        history = data.get("history") or []
        if isinstance(history, list) and history:
            today_close = float(history[-1])
    except Exception:
        today_close = None

    if (today_close is None or today_close <= 0) and data.get("price"):
        try:
            today_close = float(data.get("price") or 0)
        except Exception:
            today_close = None

    if today_close is not None and today_close <= 0:
        today_close = None

    return vol, today_close


def _apply_flow_plan(state_path: str, shared_file: str, label: str) -> None:
    """
    将 shared_quote 中的持仓与价格，写入 state 文件：
    - vol > 0：合并为一笔仓位，成本=vol * today_close，buy_price=today_close
    - vol == 0：清空 positions，但仍可设置 last_buy_price 为今日收盘价作为新锚点
    - 清空 pending_*、last_sent_* 等短期状态
    """
    vol, today_close = _read_quote(shared_file)
    if today_close is None:
        print(f"{label}: 未能从 {os.path.basename(shared_file)} 读取到有效价格，跳过。")
        return

    state = _load_state(state_path)

    if vol > 0:
        cost = vol * today_close
        positions = [
            {
                "shares": vol,
                "cost": cost,
                "buy_price": today_close,
                # 历史仓位没有对应的 client_order_id，留空即可（不参与 GlobalVault 回收）
                # 下游代码会安全处理缺失/None 的 client_order_id
            }
        ]
        state["positions"] = positions
        state["hold_layers"] = 1
        state["hold_t0_volume"] = vol
        state["total_cost"] = cost
        print(f"{label}: 检测到持仓 {vol} 股，已按 {today_close:.3f} 合并为一笔仓位。")
    else:
        state["positions"] = []
        state["hold_layers"] = 0
        state["hold_t0_volume"] = 0
        state["total_cost"] = 0.0
        print(f"{label}: 无持仓，已清空策略层 positions。")

    # 新建仓计划的基准价：以今日收盘价为锚点
    state["last_buy_price"] = today_close

    # 清空短期状态，避免旧信号/挂单干扰新计划
    state["last_sent_signal_id"] = None
    state["last_sent_signal_direction"] = None
    state["last_sent_signal_shares"] = None
    state["last_sent_signal_price"] = None
    state["last_sent_buy_prev_anchor"] = None
    state["last_sent_sell_removed_lots"] = []
    state["last_sent_was_topup"] = False
    state["last_applied_result_signal_id"] = None
    state["pending_buy_shares"] = 0
    state["pending_buy_price"] = None
    state["pending_buy_since"] = None
    state["pending_sell_since"] = None
    state["pending_sell_volume"] = 0

    # 冷却时间重置为“未卖出”状态，由新计划重新开始
    state["last_sell_timestamp"] = None

    tmp = state_path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state_path)
        print(f"{label}: 已写入状态文件 -> {state_path}")
    except Exception as e:
        print(f"{label}: 写入状态文件失败 {state_path}: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def main() -> None:
    # 收盘后运行：避免还在交易中间就重置锚点
    now = time.localtime()
    if now.tm_hour < 15:
        print("当前时间尚未到 15:00，请收盘后（建议 15:05 之后）再运行本脚本。")
        sys.exit(1)

    print("=== 初始化新建仓计划：沿用持仓、重置成本基准（159201 单标） ===")
    print(f"工作目录: {_SCRIPT_DIR}")

    _apply_flow_plan(STATE_FILE, SHARED_FILE_159201, "159201 自由现金流")

    print("\n初始化完成。请重启 mac_dashboard.py，使新建仓计划生效。")


if __name__ == "__main__":
    main()

