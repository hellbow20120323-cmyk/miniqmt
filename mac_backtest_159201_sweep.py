"""
159201 ATR 参数扫描：目标年化 100-200 笔成交
通过微调 ATR_GRID_FACTOR、GRID_STEP_FLOOR 等，让网格更灵敏
"""
import pandas as pd
import numpy as np
from itertools import product

FILE_PATH = '/Users/yuhao/Documents/miniqmt/history_159201_1m.csv'
INITIAL_CAPITAL = 400000
BATCH_MONEY = 100000
MAX_LAYERS = 5
ATR_PERIOD = 14
SELL_PROFIT_THRESHOLD = 0.006
TREND_MA_PERIOD = 60
UPTREND_GRID_FACTOR = 1.2
UPTREND_SELL_FACTOR = 1.33
UPTREND_BATCH_FACTOR = 0.7
DOWNTREND_GRID_FACTOR = 0.85
DOWNTREND_SELL_FACTOR = 0.83
DOWNTREND_BATCH_FACTOR = 1.2


def run_backtest(df, atr_grid_factor, grid_step_floor, uptrend_grid_factor=1.2, downtrend_grid_factor=0.85):
    """单次回测，返回 (trades, cash_alpha)"""
    last_buy_price = df['close'].iloc[0]
    cash_alpha = 0
    hold_t0_volume = 0
    total_cost = 0
    hold_layers = 0
    trades = []
    part_money = BATCH_MONEY / MAX_LAYERS

    for i in range(len(df)):
        curr_p = df['close'].iloc[i]
        atr = df['atr14'].iloc[i]
        slope = df['ma60_slope'].iloc[i]

        grid_step = max(grid_step_floor, (atr / curr_p) * atr_grid_factor)
        sell_threshold = SELL_PROFIT_THRESHOLD
        batch_factor = 1.0

        if slope > 0:
            grid_step *= uptrend_grid_factor
            sell_threshold *= UPTREND_SELL_FACTOR
            batch_factor = UPTREND_BATCH_FACTOR
        elif slope < 0:
            grid_step *= downtrend_grid_factor
            sell_threshold *= DOWNTREND_SELL_FACTOR
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
                trades.append({'type': 'BUY', 'price': curr_p})

        elif hold_t0_volume > 0:
            avg_cost = total_cost / hold_t0_volume
            profit_pct = (curr_p - avg_cost) / avg_cost
            if profit_pct >= sell_threshold:
                cash_alpha += hold_t0_volume * curr_p
                trades.append({'type': 'SELL', 'price': curr_p})
                hold_t0_volume = 0
                total_cost = 0
                hold_layers = 0
                last_buy_price = curr_p

    return trades, cash_alpha


def main():
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    prev_close = df['close'].shift(1)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    df['atr14'] = tr.rolling(ATR_PERIOD).mean()
    df['ma60'] = df['close'].rolling(TREND_MA_PERIOD).mean()
    df['ma60_slope'] = df['ma60'].diff(5) / df['ma60'].shift(5)
    df = df.dropna()

    days = (df.index[-1] - df.index[0]).days
    years = days / 365.0

    # 参数扫描：更小的 ATR 系数和步长下限 = 更灵敏的网格
    atr_factors = [0.35, 0.4, 0.5, 0.6]
    grid_floors = [0.0015, 0.002, 0.0025]
    uptrend_grids = [1.0, 1.1]   # 1.0=不放宽，更灵敏
    downtrend_grids = [0.7, 0.75, 0.8]

    results = []
    total = len(atr_factors) * len(grid_floors) * len(uptrend_grids) * len(downtrend_grids)
    print(f"数据区间: {df.index[0].date()} ~ {df.index[-1].date()} ({years:.2f} 年)")
    print(f"扫描 {total} 组参数...")
    cnt = 0

    for atr_f, floor, ug, dg in product(atr_factors, grid_floors, uptrend_grids, downtrend_grids):
        cnt += 1
        if cnt % 20 == 0:
            print(f"  进度 {cnt}/{total}...")
        trades, alpha = run_backtest(df, atr_f, floor, ug, dg)
        n_trades = len(trades)
        trades_per_year = n_trades / years if years > 0 else 0

        results.append({
            'ATR_FACTOR': atr_f,
            'GRID_FLOOR': floor,
            'UPTREND_GRID': ug,
            'DOWNTREND_GRID': dg,
            'trades': n_trades,
            'trades_per_year': trades_per_year,
            'alpha': alpha,
            'alpha_per_trade': alpha / n_trades if n_trades > 0 else 0,
        })

    rdf = pd.DataFrame(results)

    # 筛选目标区间 100-200 笔/年
    target = rdf[(rdf['trades_per_year'] >= 100) & (rdf['trades_per_year'] <= 200)]
    if len(target) == 0:
        # 放宽：至少 80 笔/年
        target = rdf[rdf['trades_per_year'] >= 80].copy()
        target = target.sort_values('alpha', ascending=False).head(20)
        print("\n⚠️ 无参数达到 100-200 笔/年，展示 80+ 笔/年中 Alpha 最高的 20 组：")
    else:
        target = target.sort_values('alpha', ascending=False)
        print(f"\n✅ 达到 100-200 笔/年的参数共 {len(target)} 组，按 Alpha 排序：")

    print("\n" + "=" * 90)
    for _, row in target.head(15).iterrows():
        print(f"ATR={row['ATR_FACTOR']} FLOOR={row['GRID_FLOOR']:.4f} "
              f"UP_GRID={row['UPTREND_GRID']} DOWN_GRID={row['DOWNTREND_GRID']} | "
              f"笔数={row['trades']:.0f} 年化={row['trades_per_year']:.0f}笔/年 "
              f"Alpha={row['alpha']:,.0f}元 笔均={row['alpha_per_trade']:.1f}元")
    print("=" * 90)

    # 全参数中 Alpha 最高（不限笔数）
    best = rdf.loc[rdf['alpha'].idxmax()]
    print(f"\n📌 全参数最高 Alpha: ATR={best['ATR_FACTOR']} FLOOR={best['GRID_FLOOR']} "
          f"年化{best['trades_per_year']:.0f}笔 Alpha={best['alpha']:,.0f}元")

    # 保存完整结果供进一步分析
    rdf.to_csv('/Users/yuhao/Documents/miniqmt/sweep_results_159201.csv', index=False)
    print(f"\n完整结果已保存: sweep_results_159201.csv")


if __name__ == "__main__":
    main()
