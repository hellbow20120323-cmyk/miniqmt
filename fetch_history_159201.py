#!/usr/bin/env python3
"""
Windows 端：从 QMT 行情接口一次性拉取 159201 自由现金流 ETF 历史 1 分钟 K 线数据。
使用 xtquant.xtdata：download_history_data + get_market_data_ex，保存为 history_159201_1m.csv。
运行前请确保 MiniQMT 客户端已启动并登录。
运行: 在 Windows 上 cd 到 miniqmt 目录后执行 python fetch_history_159201.py
"""
from xtquant import xtdata
import pandas as pd
import os

# --- 配置区 ---
STOCK_CODE = '159201.SZ'  # 159201 自由现金流 ETF，深交所
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(_SCRIPT_DIR, 'history_159201_1m.csv')


def download_fcf_data():
    print(f"正在准备下载 {STOCK_CODE} 的历史数据...")

    xtdata.download_history_data(STOCK_CODE, period='1m', start_time='20250101')

    data = xtdata.get_market_data_ex(
        field_list=[],
        stock_list=[STOCK_CODE],
        period='1m',
        start_time='20250101',
        dividend_type='front'
    )

    if STOCK_CODE in data:
        df = data[STOCK_CODE]
        df.index = pd.to_datetime(df.index)
        df.to_csv(OUTPUT_CSV)
        print(f"✅ 数据下载成功！已保存至: {OUTPUT_CSV}")
        print(f"统计：共获取 {len(df)} 条 1 分钟 K 线数据。")
    else:
        print("❌ 获取数据失败，请检查 QMT 是否开启了行情订阅。")


if __name__ == "__main__":
    download_fcf_data()
