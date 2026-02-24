import json
import time
import os
import random
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount

# --- 配置区（与 mac_dashboard / order_executor 标的一致）---
# 单标的：159201 自由现金流
CODES = [
    ('159201.SZ', 'shared_quote_159201.json'),  # 自由现金流
]
ACCOUNT_ID = '8883921646'  # ⚠️ 请修改为你的实际账号
ACCOUNT_TYPE = 'STOCK'      # 股票账户通常是 STOCK
# Windows 侧对应的共享目录，与 dashboard/executor 共用
SHARED_DIR = r'C:\Mac\Home\Documents\miniqmt'

# QMT 这里的路径通常是指向 userdata_mini 文件夹
# 假设你的 QMT 安装在 C 盘，这里需要根据你的实际安装路径微调
QMT_USERDATA_PATH = r'C:\国金证券QMT交易端\userdata_mini'

def start():
    # 1. 初始化行情（双标的）
    codes = [c[0] for c in CODES]
    for code in codes:
        xtdata.subscribe_quote(code, period='1m', count=-1)

    # 2. 初始化交易接口 (用于获取持仓)
    session_id = random.randint(100000, 999999)
    trader = XtQuantTrader(QMT_USERDATA_PATH, session_id)
    trader.start()
    res = trader.connect()
    acc = StockAccount(ACCOUNT_ID, ACCOUNT_TYPE)
    if res != 0:
        print("交易接口连接失败，请检查 QMT 路径和登录状态")

    print(f">>> Windows 采集端已启动，监控: {', '.join(codes)}")

    while True:
        tick_res = xtdata.get_full_tick(codes)
        positions = trader.query_stock_positions(acc)

        for STOCK_CODE, filename in CODES:
            current_pos = {}
            if positions:
                for pos in positions:
                    if pos.stock_code == STOCK_CODE:
                        current_pos = {
                            "volume": pos.volume,
                            "can_use_volume": pos.can_use_volume,
                            "open_price": pos.open_price,
                            "market_value": pos.market_value
                        }
                        break

            if STOCK_CODE in tick_res:
                tick = tick_res[STOCK_CODE]
                k_data = xtdata.get_market_data_ex(['close', 'preClose'], [STOCK_CODE], period='1m', count=80)
                history = k_data[STOCK_CODE]['close'].tolist() if STOCK_CODE in k_data else []

                # 涨跌幅 = (现价 / 前一日收盘价 - 1) * 100；优先 tick.lastClose，否则用 K 线 preClose
                pre_close = tick.get('lastClose') or tick.get('preClose')
                if pre_close is None and STOCK_CODE in k_data and 'preClose' in k_data[STOCK_CODE]:
                    pc = k_data[STOCK_CODE]['preClose']
                    pre_close = float(pc.iloc[-1]) if hasattr(pc, 'iloc') and len(pc) else None
                payload = {
                    "code": STOCK_CODE,
                    "price": tick['lastPrice'],
                    "preClose": pre_close,
                    "history": history,
                    "position": current_pos,
                    "time": time.strftime("%H:%M:%S")
                }

                filepath = os.path.join(SHARED_DIR, filename)
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(payload, f)
                except Exception as e:
                    print(f"写入 {filename} 出错: {e}")

        time.sleep(4)

if __name__ == "__main__":
    start()