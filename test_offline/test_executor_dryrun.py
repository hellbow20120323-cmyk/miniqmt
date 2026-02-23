#!/usr/bin/env python3
"""
离线测试：执行器 dry-run（不连 QMT，只校验 + 写 order_result + 消费信号）
在 test_offline 目录下放置 order_signal.json，运行后检查 order_result.json、executed_signals.json 及信号是否被移走。
用法：cd miniqmt/test_offline && python3 test_executor_dryrun.py
"""
import json
import os
import sys

# 使用当前脚本所在目录为“共享目录”
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ORDER_SIGNAL_PATH = os.path.join(TEST_DIR, "order_signal.json")
ORDER_RESULT_PATH = os.path.join(TEST_DIR, "order_result.json")
EXECUTED_SIGNALS_FILE = os.path.join(TEST_DIR, "executed_signals.json")
DONE_DIR = os.path.join(TEST_DIR, "order_signal_done")
ALLOWED_CODES = ["159201.SZ"]
MAX_SHARES_PER_ORDER = 100000
MAX_EXECUTED_IDS = 5000


def load_executed_ids():
    if not os.path.exists(EXECUTED_SIGNALS_FILE):
        return []
    try:
        with open(EXECUTED_SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = data.get("signal_ids", [])
        return ids if isinstance(ids, list) else []
    except Exception:
        return []


def save_executed_id(signal_id):
    ids = load_executed_ids()
    if signal_id in ids:
        return
    ids.append(signal_id)
    if len(ids) > MAX_EXECUTED_IDS:
        ids = ids[-MAX_EXECUTED_IDS:]
    with open(EXECUTED_SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump({"signal_ids": ids}, f, indent=0)


def consume_signal_file(signal_id):
    if not os.path.exists(ORDER_SIGNAL_PATH):
        return
    os.makedirs(DONE_DIR, exist_ok=True)
    dest = os.path.join(DONE_DIR, f"order_signal_{signal_id}.done")
    if os.path.exists(dest):
        os.remove(dest)
    os.rename(ORDER_SIGNAL_PATH, dest)


def write_order_result(signal_id, status, requested_shares=None, filled_shares=None, direction=None, message=None):
    result = {
        "signal_id": signal_id,
        "status": status,
        "order_id": 0 if status == "success" else None,
        "message": message,
        "timestamp": __import__("time").time(),
        "direction": direction,
        "requested_shares": requested_shares,
        "filled_shares": filled_shares if filled_shares is not None else requested_shares,
    }
    with open(ORDER_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def validate_signal(sig):
    if not isinstance(sig, dict):
        return "signal not dict"
    for key in ("signal_id", "code", "direction", "shares", "timestamp"):
        if key not in sig:
            return f"missing field: {key}"
    try:
        sid = str(sig["signal_id"]).strip()
        code = str(sig["code"]).strip()
        direction = str(sig["direction"]).upper()
        shares = int(sig["shares"])
        price = float(sig.get("price", 0))
    except (TypeError, ValueError) as e:
        return f"invalid types: {e}"
    if not sid:
        return "signal_id empty"
    if direction not in ("BUY", "SELL"):
        return f"invalid direction: {direction}"
    if shares <= 0 or shares % 100 != 0:
        return f"shares must be positive and multiple of 100: {shares}"
    if shares > MAX_SHARES_PER_ORDER:
        return f"shares exceed limit: {shares}"
    if code not in ALLOWED_CODES:
        return f"code not allowed: {code}"
    if price <= 0:
        return "price must be positive"
    return None


def main():
    os.chdir(TEST_DIR)
    if not os.path.exists(ORDER_SIGNAL_PATH):
        print("未找到 order_signal.json，请先放入测试信号")
        sys.exit(0)
    with open(ORDER_SIGNAL_PATH, "r", encoding="utf-8") as f:
        sig = json.load(f)
    signal_id = sig.get("signal_id", "")
    if not signal_id:
        write_order_result("no_id", "failed", message="missing signal_id")
        consume_signal_file("no_id")
        print("已消费无效信号（无 signal_id）")
        return

    executed = load_executed_ids()
    if signal_id in executed:
        consume_signal_file(signal_id)
        print("幂等：signal_id 已执行过，仅消费")
        return

    err = validate_signal(sig)
    if err:
        requested = int(sig.get("shares", 0) or 0)
        write_order_result(signal_id, "failed", requested_shares=requested, filled_shares=0, direction=sig.get("direction"), message=err)
        save_executed_id(signal_id)
        consume_signal_file(signal_id)
        print("校验失败:", err, "已写 result 并消费")
        return

    # dry-run：不下单，直接视为全成
    requested = int(sig["shares"])
    write_order_result(signal_id, "success", requested_shares=requested, filled_shares=requested, direction=sig.get("direction"))
    save_executed_id(signal_id)
    consume_signal_file(signal_id)
    print("dry-run 成功：已写 order_result、记录 executed、消费信号")


if __name__ == "__main__":
    main()
