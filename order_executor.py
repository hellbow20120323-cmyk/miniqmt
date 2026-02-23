"""
Windows 端：读取 Mac 写入的 order_signal.json，校验、资金预检后调用 QMT 下单，并消费信号防重。
与 mac_dashboard.py 配套：signal_id 幂等、可用资金预检、校验与限幅、消费协议。
"""
import json
import os
import time
import random
import logging
from pathlib import Path

from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtconstant

try:
    from global_vault import get_vault
    GLOBAL_VAULT_AVAILABLE = True
except ImportError:
    get_vault = None
    GLOBAL_VAULT_AVAILABLE = False

# --- 配置（与 bridge_producer 共用 SHARED_DIR）---
SHARED_DIR = r'C:\Mac\Home\Documents\miniqmt'
ORDER_SIGNAL_PATH = os.path.join(SHARED_DIR, 'order_signal.json')
ORDER_RESULT_PATH = os.path.join(SHARED_DIR, 'order_result.json')
EXECUTED_SIGNALS_FILE = os.path.join(SHARED_DIR, 'executed_signals.json')
DONE_DIR = os.path.join(SHARED_DIR, 'order_signal_done')
QMT_USERDATA_PATH = r'C:\国金证券QMT交易端\userdata_mini'
ACCOUNT_ID = '8883921646'
ACCOUNT_TYPE = 'STOCK'
# 双标的：159201 + 512890，参见 开发文档_双标的共享资金池.md
ALLOWED_CODES = ['159201.SZ', '512890.SH']
MAX_SHARES_PER_ORDER = 100000
# 是否启用 GlobalVault 共享资金池审批（需 global_vault 模块）
USE_GLOBAL_VAULT = True
CASH_BUFFER_RATIO = 1.01
POLL_INTERVAL_SEC = 3
SIGNAL_EXPIRE_SEC = 300
MAX_EXECUTED_IDS = 5000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SHARED_DIR, 'order_executor.log'), encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_executed_ids():
    if not os.path.exists(EXECUTED_SIGNALS_FILE):
        return []
    try:
        with open(EXECUTED_SIGNALS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ids = data.get("signal_ids", [])
        return ids if isinstance(ids, list) else []
    except Exception as e:
        logger.warning("load_executed_ids failed: %s", e)
        return []


def save_executed_id(signal_id):
    ids = load_executed_ids()
    if signal_id in ids:
        return
    ids.append(signal_id)
    if len(ids) > MAX_EXECUTED_IDS:
        ids = ids[-MAX_EXECUTED_IDS:]
    try:
        with open(EXECUTED_SIGNALS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"signal_ids": ids}, f, indent=0)
    except Exception as e:
        logger.error("save_executed_id failed: %s", e)


def consume_signal_file(signal_id, success):
    try:
        if not os.path.exists(ORDER_SIGNAL_PATH):
            return
        os.makedirs(DONE_DIR, exist_ok=True)
        dest = os.path.join(DONE_DIR, f"order_signal_{signal_id}.done")
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(ORDER_SIGNAL_PATH, dest)
    except Exception as e:
        logger.error("consume_signal_file failed: %s", e)
        try:
            os.remove(ORDER_SIGNAL_PATH)
        except Exception:
            pass


def write_order_result(signal_id, status, order_id=None, message=None, direction=None, requested_shares=None, filled_shares=None):
    """filled_shares 可由后续 QMT 回报更新；当前无回报时默认与 requested_shares 一致（视为全成）"""
    try:
        result = {
            "signal_id": signal_id,
            "status": status,
            "order_id": order_id,
            "message": message,
            "timestamp": time.time(),
            "direction": direction,
            "requested_shares": requested_shares,
            "filled_shares": filled_shares if filled_shares is not None else requested_shares,
        }
        with open(ORDER_RESULT_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("write_order_result failed: %s", e)


def validate_signal(sig):
    if not isinstance(sig, dict):
        return "signal not dict"
    for key in ('signal_id', 'code', 'direction', 'shares', 'timestamp'):
        if key not in sig:
            return f"missing field: {key}"
    try:
        sid = str(sig['signal_id']).strip()
        code = str(sig['code']).strip()
        direction = str(sig['direction']).upper()
        shares = int(sig['shares'])
        ts = float(sig['timestamp'])
        price = float(sig.get('price', 0))
    except (TypeError, ValueError) as e:
        return f"invalid types: {e}"
    if not sid:
        return "signal_id empty"
    if direction not in ('BUY', 'SELL'):
        return f"invalid direction: {direction}"
    if shares <= 0 or shares % 100 != 0:
        return f"shares must be positive and multiple of 100: {shares}"
    if shares > MAX_SHARES_PER_ORDER:
        return f"shares exceed limit {MAX_SHARES_PER_ORDER}: {shares}"
    if code not in ALLOWED_CODES:
        return f"code not allowed: {code}"
    if price <= 0:
        return "price must be positive"
    if time.time() - ts > SIGNAL_EXPIRE_SEC:
        return "signal expired"
    return None


def run_once(trader, acc):
    if not os.path.exists(ORDER_SIGNAL_PATH):
        return
    try:
        with open(ORDER_SIGNAL_PATH, 'r', encoding='utf-8') as f:
            sig = json.load(f)
    except Exception as e:
        logger.warning("read signal failed: %s", e)
        return
    signal_id = sig.get('signal_id', '')
    if not signal_id:
        err = "missing signal_id"
        logger.warning("%s", err)
        consume_signal_file(signal_id or 'no_id', False)
        write_order_result(signal_id or 'no_id', 'failed', message=err, direction=sig.get('direction'), requested_shares=sig.get('shares'))
        return

    executed = load_executed_ids()
    if signal_id in executed:
        logger.info("signal_id already executed: %s", signal_id)
        consume_signal_file(signal_id, True)
        return

    err = validate_signal(sig)
    if err:
        logger.warning("validate failed: %s", err)
        consume_signal_file(signal_id, False)
        write_order_result(signal_id, 'failed', message=err, direction=sig.get('direction'), requested_shares=int(sig.get('shares', 0) or 0), filled_shares=0)
        save_executed_id(signal_id)
        return

    code = str(sig['code']).strip()
    direction = str(sig['direction']).upper()
    shares = int(sig['shares'])
    price = float(sig['price'])
    client_order_id = str(sig.get('client_order_id') or signal_id).strip()
    amount = float(sig.get('amount') or 0) or round(price * shares, 2)
    layer_index = int(sig.get('layer_index', 0))

    vault = None
    if USE_GLOBAL_VAULT and GLOBAL_VAULT_AVAILABLE:
        try:
            vault = get_vault(SHARED_DIR)
        except Exception as e:
            logger.warning("get_vault failed: %s", e)

    if direction == 'BUY' and vault:
        ok = vault.request_allocation(code, amount, layer_index, client_order_id)
        if not ok:
            msg = "GlobalVault 拒绝: 物理池剩余不足或超限"
            logger.warning("%s", msg)
            consume_signal_file(signal_id, False)
            write_order_result(signal_id, 'failed', message=msg, direction=direction, requested_shares=shares, filled_shares=0)
            save_executed_id(signal_id)
            return

    asset = trader.query_stock_asset(acc)
    if asset is None:
        logger.warning("query_stock_asset returned None")
        consume_signal_file(signal_id, False)
        write_order_result(signal_id, 'failed', message='资金查询失败', direction=direction, requested_shares=shares, filled_shares=0)
        save_executed_id(signal_id)
        return
    required = price * shares * CASH_BUFFER_RATIO
    if direction == 'BUY' and asset.cash < required:
        msg = f"资金不足: 可用={asset.cash:.2f} 需={required:.2f}"
        logger.warning("%s", msg)
        if vault:
            try:
                vault.on_cancel(code, client_order_id)
            except Exception as e:
                logger.warning("vault.on_cancel after cash fail: %s", e)
        consume_signal_file(signal_id, False)
        write_order_result(signal_id, 'failed', message=msg, direction=direction, requested_shares=shares, filled_shares=0)
        save_executed_id(signal_id)
        return

    order_type = xtconstant.STOCK_BUY if direction == 'BUY' else xtconstant.STOCK_SELL
    strategy_name = "mac_dashboard_159201"
    order_remark = sig.get('reason', '') or signal_id[:8]

    order_id = trader.order_stock(
        acc, code, order_type, shares,
        xtconstant.FIX_PRICE, price,
        strategy_name, order_remark[:50],
    )
    save_executed_id(signal_id)
    consume_signal_file(signal_id, order_id > 0)

    if order_id and order_id > 0:
        if vault:
            try:
                if direction == 'BUY':
                    vault.on_fill(code, client_order_id)
                else:
                    vault.release(code, client_order_id)
            except Exception as e:
                logger.warning("vault on_fill/release: %s", e)
        logger.info("order placed: signal_id=%s code=%s %s %s @ %s order_id=%s",
                    signal_id, code, direction, shares, price, order_id)
        write_order_result(signal_id, 'success', order_id=order_id, direction=direction, requested_shares=shares, filled_shares=shares)
    else:
        if direction == 'BUY' and vault:
            try:
                vault.on_cancel(code, client_order_id)
            except Exception as e:
                logger.warning("vault.on_cancel after order fail: %s", e)
        logger.warning("order failed: signal_id=%s order_id=%s", signal_id, order_id)
        write_order_result(signal_id, 'failed', order_id=order_id, message='委托返回失败', direction=direction, requested_shares=shares, filled_shares=0)


def main():
    logger.info("order_executor starting, SHARED_DIR=%s", SHARED_DIR)
    session_id = random.randint(100000, 999999)
    trader = XtQuantTrader(QMT_USERDATA_PATH, session_id)
    trader.start()
    res = trader.connect()
    if res != 0:
        logger.error("trader.connect failed: %s", res)
        return
    acc = StockAccount(ACCOUNT_ID, ACCOUNT_TYPE)
    logger.info("connected, polling every %s sec", POLL_INTERVAL_SEC)
    while True:
        try:
            run_once(trader, acc)
        except Exception as e:
            logger.exception("run_once error: %s", e)
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
