"""
Windows 端：从 QMT 查询 159201 自由现金流 + 512890 红利低波 上一笔历史成交。
需在 Windows 上运行，QMT 已登录。

运行: cd C:\\Mac\\Home\\Documents\\miniqmt && python fetch_last_trade.py
"""
import random
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtconstant

# --- 配置（与 bridge_producer / order_executor 一致）---
QMT_USERDATA_PATH = r'C:\国金证券QMT交易端\userdata_mini'
ACCOUNT_ID = '8883921646'
ACCOUNT_TYPE = 'STOCK'
CODES = ['159201.SZ', '512890.SH']  # 自由现金流、红利低波


def _format_time(traded_time: int) -> str:
    """traded_time 为整数，如 93105 表示 09:31:05，93105000 表示 09:31:05.000"""
    if traded_time is None or traded_time <= 0:
        return "--"
    s = str(int(traded_time)).zfill(9)
    h = int(s[:2]) if len(s) >= 2 else 0
    m = int(s[2:4]) if len(s) >= 4 else 0
    sec = int(s[4:6]) if len(s) >= 6 else 0
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _direction_str(t) -> str:
    """根据 order_type 或 offset_flag 判断买卖"""
    order_type = getattr(t, "order_type", None)
    if order_type == xtconstant.STOCK_BUY:
        return "买入"
    if order_type == xtconstant.STOCK_SELL:
        return "卖出"
    offset_flag = getattr(t, "offset_flag", 0)
    if offset_flag in (48, getattr(xtconstant, "OFFSET_FLAG_OPEN", 48)):
        return "买入"
    return "卖出"


def main():
    session_id = random.randint(100000, 999999)
    trader = XtQuantTrader(QMT_USERDATA_PATH, session_id)
    trader.start()
    res = trader.connect()
    if res != 0:
        print("连接 QMT 失败，请检查路径与登录状态")
        return
    acc = StockAccount(ACCOUNT_ID, ACCOUNT_TYPE)

    trades = trader.query_stock_trades(acc)
    if not trades:
        print("当日无成交记录")
        return

    # 按标的筛选，取每标的最后一笔（按 traded_time 排序）
    by_code = {}
    for t in trades:
        code = getattr(t, "stock_code", None) or getattr(t, "stock_code1", "")
        if not code:
            continue
        if code not in CODES:
            continue
        if code not in by_code or (getattr(t, "traded_time", 0) or 0) > (getattr(by_code[code], "traded_time", 0) or 0):
            by_code[code] = t

    name_map = {"159201.SZ": "159201 自由现金流", "512890.SH": "512890 红利低波"}
    print("=" * 60)
    print("159201 自由现金流 + 512890 红利低波 — 上一笔成交（当日）")
    print("=" * 60)
    for code in CODES:
        name = name_map.get(code, code)
        if code not in by_code:
            print(f"\n{name}: 当日无成交")
            continue
        t = by_code[code]
        direction = _direction_str(t)
        print(f"\n{name}:")
        print(f"  方向: {direction}")
        print(f"  成交价: {getattr(t, 'traded_price', 0):.3f}")
        print(f"  成交数量: {getattr(t, 'traded_volume', 0)} 股")
        print(f"  成交金额: {getattr(t, 'traded_amount', 0):.2f} 元")
        print(f"  成交时间: {_format_time(getattr(t, 'traded_time', 0))}")
        print(f"  成交编号: {getattr(t, 'traded_id', '')}")
        print(f"  订单编号: {getattr(t, 'order_id', '')}")
        print(f"  手续费: {getattr(t, 'commission', 0):.2f} 元")
    print("\n" + "=" * 60)
    print("说明: query_stock_trades 仅返回当日成交，非当日需在 QMT 客户端查看")
    print("=" * 60)


if __name__ == "__main__":
    main()
