from xtquant import xtdata
import pandas as pd

# 下载 603060.SH 过去一年的 1 分钟数据
code = '600895.SH'
xtdata.download_history_data(code, period='1m', start_time='20250101')

# 获取并保存到共享目录
data = xtdata.get_market_data_ex([], [code], period='1m', start_time='20250101')
df = data[code]
df.to_csv(r'C:\Mac\Home\Documents\miniqmt\history_600895_1m.csv')
print("历史数据已导出到共享文件夹！")