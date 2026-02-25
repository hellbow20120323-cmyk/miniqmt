"""
重置 159201 自由现金流 的建仓计划。
清空持仓、成本价等状态，以当前价为新锚点开启新计划。

⚠️ 使用前请确认：
1. 若 QMT 仍有持仓，需先手动平仓或接受「策略状态与真实持仓不一致」
2. 若使用 GlobalVault 共享池，used 需与真实持仓一致（平仓后 release 会释放）

运行: python reset_build_plan.py
"""
import json
import os
from typing import Optional

_SCRIPT_DIR = os.environ.get("DASHBOARD_WORK_DIR") or os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_SCRIPT_DIR, 'dashboard_state.json')
SHARED_FILE_159201 = os.path.join(_SCRIPT_DIR, 'shared_quote_159201.json')


def _fresh_state():
    """全新建仓状态：无持仓，last_buy_price 由 dashboard 首次运行时从当前价初始化"""
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
    }


def _get_current_price(shared_file: str) -> Optional[float]:
    """从 shared_quote 读取当前价，用于可选预填 last_buy_price"""
    if not os.path.exists(shared_file):
        return None
    try:
        with open(shared_file, 'r') as f:
            data = json.load(f)
        return float(data.get('price', 0) or 0)
    except Exception:
        return None


def main():
    state = _fresh_state()

    # 可选：若存在行情，用当前价预填 last_buy_price，否则保持 None（dashboard 首次运行时会从 curr_p 初始化）
    p159201 = _get_current_price(SHARED_FILE_159201)
    if p159201 and p159201 > 0:
        state["last_buy_price"] = p159201
        print(f"159201 当前价: {p159201:.3f}，已设为新锚点")
    else:
        print("159201: 无行情文件，last_buy_price 将待 dashboard 运行时从当前价初始化")
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
        print(f"已重置: 159201 自由现金流 -> {STATE_FILE}")
    except Exception as e:
        print(f"写入失败 {STATE_FILE}: {e}")

    print("\n新建仓计划已就绪。请重启 mac_dashboard.py。")


if __name__ == "__main__":
    main()
