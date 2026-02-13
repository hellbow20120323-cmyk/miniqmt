from xtquant import xtdata
import pandas as pd
import os

# --- 配置区 ---
# 159201 是深交所代码，后缀为 .SZ
STOCK_CODE = '159201.SZ' 
# 设定共享目录路径
SAVE_PATH = r'C:\Mac\Home\Documents\miniqmt\history_159201_1m.csv'

def download_fcf_data():
    # 1. 确保 xtdata 已经连接 QMT 客户端
    # 请确保你的 Windows QMT 已经登录并运行
    print(f"正在准备下载 {STOCK_CODE} 的历史数据...")

    # 2. 下载历史 1 分钟数据 (下载过去一年的数据)
    # 周期可选：'1m' (分钟), '1d' (日线)
    xtdata.download_history_data(STOCK_CODE, period='1m', start_time='20250101')

    # 3. 获取数据并进行前复权处理
    # dividend_type='front' 表示前复权，这对 ETF 尤其是除权息后的回测至关重要
    data = xtdata.get_market_data_ex(
        field_list=[], 
        stock_list=[STOCK_CODE], 
        period='1m', 
        start_time='20250101',
        dividend_type='front'
    )

    # 4. 数据清洗与保存
    if STOCK_CODE in data:
        df = data[STOCK_CODE]
        # 转换索引为时间格式，方便 Mac 端 Pandas 读取
        df.index = pd.to_datetime(df.index)
        
        # 保存到共享目录
        df.to_csv(SAVE_PATH)
        print(f"✅ 数据下载成功！已保存至: {SAVE_PATH}")
        print(f"统计：共获取 {len(df)} 条 1 分钟 K 线数据。")
    else:
        print("❌ 获取数据失败，请检查 QMT 是否开启了行情订阅。")

if __name__ == "__main__":
    download_fcf_data()