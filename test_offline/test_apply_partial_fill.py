#!/usr/bin/env python3
"""
离线测试：部分成交状态对齐（_apply_order_result 逻辑）
在 test_offline 目录下准备 dashboard_state.json、order_result.json，运行后检查 state 是否按预期更新。
用法：cd miniqmt/test_offline && python3 test_apply_partial_fill.py [--buy|--sell]
"""
import json
import os
import sys
import time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(TEST_DIR, "dashboard_state.json")
ORDER_RESULT_FILE = os.path.join(TEST_DIR, "order_result.json")


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


def apply_order_result(state, result):
    """与 mac_dashboard._apply_order_result 一致的核心逻辑（不依赖 _save_state 抽象）"""
    if not isinstance(result, dict):
        return
    sid = result.get("signal_id")
    if not sid or sid != state.get("last_sent_signal_id") or sid == state.get("last_applied_result_signal_id"):
        return
    requested = int(result.get("requested_shares") or 0)
    filled = int(result.get("filled_shares", requested) or requested)
    direction = (result.get("direction") or "").upper()
    price = float(result.get("price") or state.get("last_sent_signal_price") or 0)
    if price <= 0:
        price = state.get("last_sent_signal_price") or 0

    state["last_applied_result_signal_id"] = sid
    if filled >= requested:
        return

    if direction == "BUY":
        is_topup = state.get("last_sent_was_topup", False)
        positions = list(state.get("positions", []))
        if is_topup:
            if filled > 0:
                positions.append({"shares": filled, "cost": filled * price, "buy_price": price})
            remaining = requested - filled
            if remaining > 0:
                state["pending_buy_shares"] = remaining
                state["pending_buy_price"] = price
                state["pending_buy_since"] = time.time()
            else:
                state["pending_buy_shares"] = 0
                state["pending_buy_price"] = None
                state["pending_buy_since"] = None
        else:
            if not positions:
                state["last_buy_price"] = state.get("last_sent_buy_prev_anchor") or state.get("last_buy_price")
            else:
                if filled > 0:
                    positions[-1] = {"shares": filled, "cost": filled * price, "buy_price": price}
                else:
                    positions.pop()
                    state["last_buy_price"] = state.get("last_sent_buy_prev_anchor") or state.get("last_buy_price")
                remaining = requested - filled
                if remaining > 0:
                    state["pending_buy_shares"] = remaining
                    state["pending_buy_price"] = price
                    state["pending_buy_since"] = time.time()
                else:
                    state["pending_buy_shares"] = 0
                    state["pending_buy_price"] = None
                    state["pending_buy_since"] = None
        state["positions"] = positions
        state["hold_layers"] = len(positions)
        state["hold_t0_volume"] = sum(p["shares"] for p in positions)
        state["total_cost"] = sum(p["cost"] for p in positions)
    elif direction == "SELL":
        removed = state.get("last_sent_sell_removed_lots", [])
        if not removed:
            return
        total_removed = sum(lot["shares"] for lot in removed)
        total_cost_removed = sum(lot["cost"] for lot in removed)
        remaining = requested - filled
        if remaining <= 0:
            return
        avg_price = total_cost_removed / total_removed if total_removed else price
        back_cost = total_cost_removed * (remaining / total_removed) if total_removed else remaining * price
        positions = list(state.get("positions", []))
        positions.append({"shares": remaining, "cost": back_cost, "buy_price": avg_price})
        state["positions"] = positions
        state["hold_layers"] = len(positions)
        state["hold_t0_volume"] = sum(p["shares"] for p in positions)
        state["total_cost"] = sum(p["cost"] for p in positions)
        state["pending_sell_since"] = None
        state["pending_sell_volume"] = 0


def main():
    os.chdir(TEST_DIR)
    if not os.path.exists(ORDER_RESULT_FILE):
        print("请先在此目录放置 order_result.json（含 signal_id, requested_shares, filled_shares, direction）")
        sys.exit(1)
    state = load_state()
    with open(ORDER_RESULT_FILE, "r") as f:
        result = json.load(f)
    apply_order_result(state, result)
    save_state(state)
    print("已应用 order_result，当前 state 关键字段：")
    print("  positions:", state.get("positions"))
    print("  hold_layers:", state.get("hold_layers"))
    print("  hold_t0_volume:", state.get("hold_t0_volume"))
    print("  pending_buy_shares:", state.get("pending_buy_shares"))
    print("  pending_buy_price:", state.get("pending_buy_price"))
    print("  last_applied_result_signal_id:", state.get("last_applied_result_signal_id"))


if __name__ == "__main__":
    main()
