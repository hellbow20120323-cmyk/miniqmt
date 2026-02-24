"""
初始化固定 80% + 流动 20% 网格计划（单标 159201）。

用途：
- 收盘后读取 shared_quote_159201.json
- 将当前真实持仓按 80% 设为固定仓位、不参与网格卖出
- 剩余 20% 作为流动仓位，按当日收盘价合并为一笔网格仓，并写入 dashboard_state.json

运行：
    python init_fixed_flow_plan.py
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
    """基础状态模板，应与 mac_dashboard.py 中 _load_state 的默认结构保持兼容。"""
    return {
        "last_buy_price": None,
        "hold_layers": 0,
        "total_cost": 0.0,
        "hold_t0_volume": 0,
        "fixed_volume": 0,
        "fixed_base_price": None,
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
        "pending_sell_since": None,
        "pending_sell_volume": 0,
    }


def _load_state(path: str) -> Dict[str, Any]:
    base = _fresh_state()
    if not os.path.exists(path):
        return base
    try:
        with open(path, "r") as f:
            data = json.load(f)
        base.update(data)
        return base
    except Exception:
        return base


def _read_quote(shared_file: str) -> Tuple[int, Optional[float]]:
    """
    从 shared_quote_159201.json 读取：
    - 当前真实持仓股数：position.volume
    - 今日收盘价 today_close：优先 history[-1]，否则 price
    """
    if not os.path.exists(shared_file):
        print(f"未找到行情文件: {shared_file}")
        return 0, None

    try:
        with open(shared_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取行情文件失败 {shared_file}: {e}")
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


def _split_fixed_flow(total_vol: int) -> Tuple[int, int]:
    """
    按 80/20 拆分固定仓与流动仓股数，按 100 股整数倍对齐。
    - total_vol < 100: 全部视为固定仓，流动仓为 0
    - flow_vol < 100: 提示后也视为全部固定仓，避免不足 1 手的流动仓
    """
    if total_vol <= 0:
        return 0, 0
    if total_vol < 100:
        print(f"当前总持仓 {total_vol} 股 < 100 股，视为无流动仓，全部作为固定仓。")
        return total_vol, 0

    raw_fixed = int(total_vol * 0.8)
    fixed_vol = (raw_fixed // 100) * 100
    if fixed_vol < 0:
        fixed_vol = 0
    if fixed_vol > total_vol:
        fixed_vol = total_vol

    flow_vol = total_vol - fixed_vol
    if flow_vol < 100:
        print(f"按 80% 计算得到流动仓 {flow_vol} 股 < 100 股，全部视为固定仓。")
        return total_vol, 0

    return fixed_vol, flow_vol


def main() -> None:
    # 收盘后运行：避免还在交易中间就重置锚点
    now = time.localtime()
    if now.tm_hour < 15:
        print("当前时间尚未到 15:00，请收盘后（建议 15:05 之后）再运行本脚本。")
        sys.exit(1)

    print("=== 初始化固定 80% + 流动 20% 网格计划（159201 单标） ===")
    print(f"工作目录: {_SCRIPT_DIR}")

    total_vol, today_close = _read_quote(SHARED_FILE_159201)
    if today_close is None:
        print("未能从 shared_quote_159201.json 读取到有效价格，放弃初始化。")
        sys.exit(1)

    fixed_vol, flow_vol = _split_fixed_flow(total_vol)
    print(f"当前真实持仓: {total_vol} 股 -> 固定仓: {fixed_vol} 股, 流动仓: {flow_vol} 股")
    print(f"流动仓基准价(今日收盘/当前价): {today_close:.3f}")

    state = _load_state(STATE_FILE)

    # 固定仓设置
    state["fixed_volume"] = fixed_vol
    state["fixed_base_price"] = today_close

    # 流动仓初始仓位：仅当 flow_vol >= 100 时才作为一笔网格仓位
    if flow_vol >= 100:
        cost = flow_vol * today_close
        state["positions"] = [
            {
                "shares": flow_vol,
                "cost": cost,
                "buy_price": today_close,
                # 历史固定仓部分不参与网格，也不挂 client_order_id
            }
        ]
        state["hold_layers"] = 1
        state["hold_t0_volume"] = flow_vol
        state["total_cost"] = cost
        print(f"已将流动仓 {flow_vol} 股合并为一笔网格仓位。")
    else:
        state["positions"] = []
        state["hold_layers"] = 0
        state["hold_t0_volume"] = 0
        state["total_cost"] = 0.0
        if flow_vol > 0:
            print(f"流动仓 {flow_vol} 股不足 100 股，不纳入网格，仅保留为固定仓。")

    # 网格锚点：以今日收盘价为 last_buy_price
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
    state["last_sell_timestamp"] = None

    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
        print(f"已写入状态文件: {STATE_FILE}")
    except Exception as e:
        print(f"写入状态文件失败 {STATE_FILE}: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        sys.exit(1)

    print("\n初始化完成。请重启 mac_dashboard.py，使 80% 固定 + 20% 网格策略生效。")


if __name__ == "__main__":
    main()

