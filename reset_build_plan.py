"""
重置 159201 自由现金流 的建仓计划。
清空持仓、成本价等状态，以当前价为新锚点开启新计划。

⚠️ 使用前请确认：
1. 若 QMT 仍有持仓，需先手动平仓或接受「策略状态与真实持仓不一致」
2. 若使用 GlobalVault 共享池，used 需与真实持仓一致（平仓后 release 会释放）

运行: python reset_build_plan.py
"""
import argparse
import json
import os
from typing import Optional

_SCRIPT_DIR = os.environ.get("DASHBOARD_WORK_DIR") or os.path.dirname(os.path.abspath(__file__))


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
    parser = argparse.ArgumentParser(description="重置建仓计划：清空策略状态，以当前价为新锚点开启新计划（支持多标的）。")
    parser.add_argument(
        "--symbol",
        type=str,
        default="159201",
        help="标的代码（不含交易所后缀），如 159201 或 512890；默认 159201。",
    )
    args = parser.parse_args()
    symbol = args.symbol.strip()

    if symbol == "159201":
        state_file = os.path.join(_SCRIPT_DIR, "dashboard_state.json")
        shared_file = os.path.join(_SCRIPT_DIR, "shared_quote_159201.json")
        label = "159201 自由现金流"
    else:
        state_file = os.path.join(_SCRIPT_DIR, f"dashboard_state_{symbol}.json")
        shared_file = os.path.join(_SCRIPT_DIR, f"shared_quote_{symbol}.json")
        label = f"{symbol} 网格策略"

    state = _fresh_state()

    # 可选：若存在行情，用当前价预填 last_buy_price，否则保持 None（dashboard 首次运行时会从 curr_p 初始化）
    p = _get_current_price(shared_file)
    if p and p > 0:
        state["last_buy_price"] = p
        print(f"{symbol} 当前价: {p:.3f}，已设为新锚点")
    else:
        print(f"{symbol}: 无行情文件，last_buy_price 将待 dashboard 运行时从当前价初始化")
    try:
        tmp = state_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state_file)
        print(f"已重置: {label} -> {state_file}")
    except Exception as e:
        print(f"写入失败 {state_file}: {e}")

    print("\n新建仓计划已就绪。请重启 mac_dashboard.py。")


if __name__ == "__main__":
    main()
