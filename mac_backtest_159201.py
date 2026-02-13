import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- [1] 参数配置 ---
# 预设: "default"=原参数(22笔/年) | "high_freq"=高频率(100-200笔/年)
PRESET = "high_freq"

FILE_PATH = '/Users/yuhao/Documents/miniqmt/history_159201_1m.csv'
INITIAL_CAPITAL = 400000        # 40万总资金
BATCH_MONEY = 100000            # 单笔套利总金额（占40万资金25%）
MAX_LAYERS = 5                  # 最多 5 层补仓
ATR_PERIOD = 14
# 高频率预设：ATR 0.4 + 步长下限 0.1% → 年化 ~114 笔，Alpha ~10,665
# 默认预设：ATR 0.7 + 步长 0.3% → 年化 ~22 笔
ATR_GRID_FACTOR = 0.4 if PRESET == "high_freq" else 0.7
GRID_STEP_FLOOR = 0.001 if PRESET == "high_freq" else 0.003
SELL_PROFIT_THRESHOLD = 0.004 if PRESET == "high_freq" else 0.006  # 高频率用 0.4% 止盈
# 趋势自适应：涨多卖跌少买
TREND_MA_PERIOD = 60            # 趋势判断均线周期
UPTREND_GRID_FACTOR = 1.0 if PRESET == "high_freq" else 1.2   # 高频率不放宽，保持灵敏
UPTREND_SELL_FACTOR = 1.33      # 上升趋势：止盈提高（涨多卖）
UPTREND_BATCH_FACTOR = 0.7      # 上升趋势：每档买入金额缩小（跌少买）
DOWNTREND_GRID_FACTOR = 0.7 if PRESET == "high_freq" else 0.85  # 高频率更易触发补仓
DOWNTREND_SELL_FACTOR = 0.83    # 下降趋势：止盈降低（涨少卖）
DOWNTREND_BATCH_FACTOR = 1.2    # 下降趋势：每档买入金额放大（跌多买）

def run_backtest():
    # --- [2] 加载与处理数据 ---
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    
    # ATR(14) 用于动态网格
    prev_close = df['close'].shift(1)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    df['atr14'] = tr.rolling(ATR_PERIOD).mean()
    # 趋势判断：MA60 斜率
    df['ma60'] = df['close'].rolling(TREND_MA_PERIOD).mean()
    df['ma60_slope'] = df['ma60'].diff(5) / df['ma60'].shift(5)

    df = df.dropna()

    last_buy_price = df['close'].iloc[0]
    cash_alpha = 0
    hold_t0_volume = 0
    total_cost = 0
    hold_layers = 0
    trades = []
    equity_curve = []

    part_money = BATCH_MONEY / MAX_LAYERS

    for i in range(len(df)):
        curr_p = df['close'].iloc[i]
        atr = df['atr14'].iloc[i]
        slope = df['ma60_slope'].iloc[i]

        # 基准网格步长
        grid_step = max(GRID_STEP_FLOOR, (atr / curr_p) * ATR_GRID_FACTOR)
        sell_threshold = SELL_PROFIT_THRESHOLD

        # 趋势自适应：涨多卖跌少买 + 仓位分级
        batch_factor = 1.0
        if slope > 0:  # 上升趋势
            grid_step *= UPTREND_GRID_FACTOR   # 跌少买：需更大跌幅才触发
            sell_threshold *= UPTREND_SELL_FACTOR  # 涨多卖：持有到更高利润
            batch_factor = UPTREND_BATCH_FACTOR    # 跌少买：每档金额缩小
        elif slope < 0:  # 下降趋势
            grid_step *= DOWNTREND_GRID_FACTOR   # 跌多买：更易触发补仓
            sell_threshold *= DOWNTREND_SELL_FACTOR  # 涨少卖：见好就收
            batch_factor = DOWNTREND_BATCH_FACTOR   # 跌多买：每档金额放大

        part_money_adj = part_money * batch_factor

        # A. 买入：价格比上次买入价低一个动态步长即补仓，最多 5 层
        if hold_layers < MAX_LAYERS and curr_p <= last_buy_price * (1 - grid_step):
            shares = int(part_money_adj / curr_p // 100) * 100
            if shares > 0:
                cash_alpha -= shares * curr_p
                total_cost += shares * curr_p
                hold_t0_volume += shares
                hold_layers += 1
                last_buy_price = curr_p
                trades.append({'time': df.index[i], 'type': 'BUY', 'price': curr_p})

        # B. 卖出：T+0 仓位整体盈利达阈值即清仓（趋势自适应）
        elif hold_t0_volume > 0:
            avg_cost = total_cost / hold_t0_volume
            profit_pct = (curr_p - avg_cost) / avg_cost
            if profit_pct >= sell_threshold:
                cash_alpha += hold_t0_volume * curr_p
                trades.append({'time': df.index[i], 'type': 'SELL', 'price': curr_p})
                hold_t0_volume = 0
                total_cost = 0
                hold_layers = 0
                last_buy_price = curr_p

        equity_curve.append(cash_alpha + (hold_t0_volume * curr_p))

    # --- [4] 结果输出 ---
    beta_profit = (df['close'].iloc[-1] - df['close'].iloc[0]) * (INITIAL_CAPITAL / df['close'].iloc[0])
    
    print(f"\n📈 159201 (自由现金流 ETF) 回测报告")
    print("-" * 35)
    print(f"成交笔数: {len(trades)}")
    print(f"Alpha 套利净值: {cash_alpha:,.2f} 元")
    print(f"Beta 市值变动: {beta_profit:,.2f} 元")
    print(f"综合摊薄贡献: {cash_alpha + beta_profit:,.2f} 元")

    # 绘图可视化
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Cumulative T+0 Profit (Alpha)', color='green')
    plt.title('159201 Grid Strategy - Alpha Equity Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('/Users/yuhao/Documents/miniqmt/backtest_159201.png', dpi=120)
    plt.close()

if __name__ == "__main__":
    run_backtest()
