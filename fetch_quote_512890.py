#!/usr/bin/env python3
"""
Windows 端：从 miniqmt（QMT）行情接口获取红利低波 ETF 512890 分钟级数据（默认近 5 年）。
使用 xtquant.xtdata：先 download_history_data 下载到本地，再用 get_market_data 读出并保存 CSV。
运行前请确保 MiniQMT 客户端已启动并登录。
依赖: 与 bridge_producer 相同（xtquant、MiniQMT 环境）
运行: 在 Windows 上 cd 到 miniqmt 目录后执行 python fetch_quote_512890.py
"""
import os
from datetime import datetime, timedelta

try:
    from xtquant import xtdata
    import pandas as pd
except ImportError as e:
    print("请确保已安装 xtquant（MiniQMT 自带）: %s" % (str(e)[:80]))
    raise

# 配置（标的与 bridge_producer 同源：QMT 行情）
STOCK_CODE = "512890.SH"   # 红利低波 ETF，上交所
PERIOD = "1m"              # 1 分钟 K 线（可选 5m/15m/30m/1h/1d）
YEARS = 1                  # 拉取最近 N 年（仅当 FIXED_YEAR 为 None 时生效）
FIXED_YEAR = None          # 指定年份则只拉该年整年：2024-01-01 ~ 2024-12-31；设为 None 则用 YEARS
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "quote_512890_1min_5y.csv")  # main() 里按 FIXED_YEAR 覆盖


def main():
    if FIXED_YEAR is not None:
        start_date = datetime(FIXED_YEAR, 1, 1)
        end_date = datetime(FIXED_YEAR, 12, 31)
        output_csv = os.path.join(OUTPUT_DIR, "quote_512890_1min_%d.csv" % FIXED_YEAR)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=YEARS * 365)
        output_csv = OUTPUT_CSV
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    print("标的: %s 红利低波 | 周期: %s | 范围: %s ~ %s" % (STOCK_CODE, PERIOD, start_str, end_str))
    print("正在通过 QMT 行情接口下载历史数据（需 MiniQMT 已启动）…")

    # 1. 下载历史行情到本地（xtdata 与 MiniQMT 通信）
    xtdata.download_history_data(STOCK_CODE, period=PERIOD, start_time=start_str, end_time=end_str)
    print("下载完成，正在读取…")

    # 2. 从缓存/本地获取数据（field_list 为空表示全部字段：time, open, high, low, close, volume 等）
    data = xtdata.get_market_data(
        field_list=[],
        stock_list=[STOCK_CODE],
        period=PERIOD,
        start_time=start_str,
        end_time=end_str,
        count=-1,
    )
    if not data:
        print("未获取到数据，请检查 MiniQMT 是否已登录、标的 %s 是否有权限。" % STOCK_CODE)
        return

    # 3. 组装为一张表：返回 dict{ field -> DataFrame }，DataFrame 的 index=stock_list、columns=time_list
    first_df = next((v for v in data.values() if v is not None and not v.empty), None)
    if first_df is None:
        print("返回数据为空（可能该时间区间无分钟数据）。可尝试缩短区间或换日线。")
        return
    # 兼容不同 index 格式：有的返回 "512890.SH"，有的为 "512890" 或 "512890.sh"
    effective_code = None
    for code in (STOCK_CODE, "512890", "512890.sh", "512890.SH"):
        if code in first_df.index:
            effective_code = code
            break
    if effective_code is None:
        print("返回数据中无标的 %s。实际 index: %s" % (STOCK_CODE, first_df.index.tolist()[:10]))
        return
    if effective_code != STOCK_CODE:
        print("使用标的: %s（接口返回格式与请求略有不同）" % effective_code)

    out = pd.DataFrame()
    for field_name, df in data.items():
        if df is None or df.empty or effective_code not in df.index:
            continue
        out[field_name] = df.loc[effective_code].values
    if out.empty:
        print("组装后无数据，请检查时间范围与标的。")
        return
    # 按时间排序列（若有 time 列）
    if "time" in out.columns:
        out = out.sort_values("time").reset_index(drop=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("已保存: %s | 行数: %d" % (output_csv, len(out)))


if __name__ == "__main__":
    main()
