import json
import time
import os
import random
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount

# --- 配置区 ---
STOCK_CODE = '603060.SH'
ACCOUNT_ID = '8883921646'  # ⚠️ 请修改为你的实际账号
ACCOUNT_TYPE = 'STOCK'      # 股票账户通常是 STOCK
# Windows 侧对应的共享目录
SHARED_DIR = r'C:\Mac\Home\Documents\miniqmt'
SHARED_FILE_PATH = os.path.join(SHARED_DIR, 'shared_quote.json')

# QMT 这里的路径通常是指向 userdata_mini 文件夹
# 假设你的 QMT 安装在 C 盘，这里需要根据你的实际安装路径微调
QMT_USERDATA_PATH = r'C:\国金证券QMT交易端\userdata_mini'

def start():
    # 1. 初始化行情
    xtdata.subscribe_quote(STOCK_CODE, period='1m', count=-1)
    
    # 2. 初始化交易接口 (用于获取持仓)
    session_id = random.randint(100000, 999999)
    trader = XtQuantTrader(QMT_USERDATA_PATH, session_id)
    trader.start()
    res = trader.connect()
    # 创建账号对象，将 ID 和类型封装进去
    # 使用字符串 ID 和 账户类型进行显式初始化
    acc = StockAccount(ACCOUNT_ID, ACCOUNT_TYPE)
    if res != 0:
        print("交易接口连接失败，请检查 QMT 路径和登录状态")

    print(f">>> Windows 采集端已启动，监控: {STOCK_CODE}")

    while True:
        # 获取行情
        tick_res = xtdata.get_full_tick([STOCK_CODE])
        
        # 获取持仓 (重点新加部分)
        # 核心修复点：只传入封装好的 acc 对象
        positions = trader.query_stock_positions(acc)        
        
        current_pos = {}
        
        if positions:
            for pos in positions:
                # 在 Native API 中，pos.stock_code 已经是完整代码（如 '603060.SH'）
                if pos.stock_code == STOCK_CODE:
                    current_pos = {
                        "volume": pos.volume,
                        "can_use_volume": pos.can_use_volume,
                        "open_price": pos.open_price,
                        "market_value": pos.market_value
                    }
                    break # 找到目标后跳出

        if STOCK_CODE in tick_res:
            tick = tick_res[STOCK_CODE]
            # 获取 60 根 K 线用于 Mac 端计算 BOLL/RSI
            k_data = xtdata.get_market_data_ex(['close'], [STOCK_CODE], period='1m', count=60)
            history = k_data[STOCK_CODE]['close'].tolist() if STOCK_CODE in k_data else []

            payload = {
                "code": STOCK_CODE,
                "price": tick['lastPrice'],
                "history": history,
                "position": current_pos, # 把真实的持仓传给 Mac
                "time": time.strftime("%H:%M:%S")
            }
            
            try:
                with open(SHARED_FILE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(payload, f)
            except Exception as e:
                print(f"写入出错: {e}")
        
        time.sleep(4)

if __name__ == "__main__":
    start()