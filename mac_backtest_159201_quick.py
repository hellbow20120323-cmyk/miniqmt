"""快速测试几组高灵敏度参数，目标 100-200 笔/年"""
import pandas as pd
import numpy as np

FILE_PATH = '/Users/yuhao/Documents/miniqmt/history_159201_1m.csv'
BATCH_MONEY = 100000
MAX_LAYERS = 5
ATR_PERIOD = 14
SELL_PROFIT_THRESHOLD = 0.006
TREND_MA_PERIOD = 60
UPTREND_SELL_FACTOR = 1.33
UPTREND_BATCH_FACTOR = 0.7
DOWNTREND_SELL_FACTOR = 0.83
DOWNTREND_BATCH_FACTOR = 1.2


def run(df, atr_f, floor, ug, dg, sell_pct=0.006):
    last_buy_price = df['close'].iloc[0]
    cash_alpha, hold_t0_volume, total_cost, hold_layers = 0, 0, 0, 0
    trades = []
    part_money = BATCH_MONEY / MAX_LAYERS

    for i in range(len(df)):
        curr_p = df['close'].iloc[i]
        atr = df['atr14'].iloc[i]
        slope = df['ma60_slope'].iloc[i]
        grid_step = max(floor, (atr / curr_p) * atr_f)
        sell_threshold = sell_pct
        batch_factor = 1.0
        if slope > 0:
            grid_step *= ug
            sell_threshold *= UPTREND_SELL_FACTOR  # 1.33
            batch_factor = UPTREND_BATCH_FACTOR
        elif slope < 0:
            grid_step *= dg
            sell_threshold *= DOWNTREND_SELL_FACTOR  # 0.83
            batch_factor = DOWNTREND_BATCH_FACTOR
        part_money_adj = part_money * batch_factor

        if hold_layers < MAX_LAYERS and curr_p <= last_buy_price * (1 - grid_step):
            shares = int(part_money_adj / curr_p // 100) * 100
            if shares > 0:
                cash_alpha -= shares * curr_p
                total_cost += shares * curr_p
                hold_t0_volume += shares
                hold_layers += 1
                last_buy_price = curr_p
                trades.append('B')
        elif hold_t0_volume > 0:
            avg_cost = total_cost / hold_t0_volume
            if (curr_p - avg_cost) / avg_cost >= sell_threshold:
                cash_alpha += hold_t0_volume * curr_p
                trades.append('S')
                hold_t0_volume = total_cost = hold_layers = 0
                last_buy_price = curr_p
    return len(trades), cash_alpha


def main():
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    prev_close = df['close'].shift(1)
    tr = np.maximum(df['high'] - df['low'],
        np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    df['atr14'] = tr.rolling(ATR_PERIOD).mean()
    df['ma60'] = df['close'].rolling(TREND_MA_PERIOD).mean()
    df['ma60_slope'] = df['ma60'].diff(5) / df['ma60'].shift(5)
    df = df.dropna()

    years = (df.index[-1] - df.index[0]).days / 365.0
    print(f"数据: {df.index[0].date()} ~ {df.index[-1].date()} ({years:.2f}年)\n")

    # 目标 100-200 笔/年：微调止盈与网格
    configs = [
        (0.35, 0.001, 1.0, 0.7, 0.0045),   # ~0.45% 止盈
        (0.35, 0.0012, 1.0, 0.7, 0.004),
        (0.4, 0.001, 1.0, 0.7, 0.0045),
        (0.4, 0.0012, 1.0, 0.7, 0.004),
        (0.35, 0.001, 1.05, 0.72, 0.004),   # 略放宽趋势
        (0.4, 0.001, 1.05, 0.75, 0.004),
        (0.35, 0.001, 1.0, 0.75, 0.004),
        (0.4, 0.001, 1.0, 0.75, 0.004),
    ]
    print("ATR  FLOOR   UP  DOWN  SELL% | 笔数  年化笔数  Alpha")
    print("-" * 60)
    for atr_f, floor, ug, dg, sell_pct in configs:
        n, alpha = run(df, atr_f, floor, ug, dg, sell_pct)
        tpy = n / years
        print(f"{atr_f}   {floor}   {ug}   {dg}   {sell_pct*100:.1f}% | {n:4}  {tpy:6.0f}   {alpha:,.0f}")


if __name__ == "__main__":
    main()
