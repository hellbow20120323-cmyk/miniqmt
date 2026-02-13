import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- [1] 配置中心 ---
FILE_PATH = '/Users/yuhao/Documents/miniqmt/history_600895_1m.csv' # 你的历史数据路径
INITIAL_CAPITAL = 400000        # 初始资金 40 万
ATR_PERIOD = 14                 # ATR 周期
ATR_GRID_FACTOR = 1.2           # 动态网格系数：GRID_STEP = ATR/Price × 1.2
GRID_STEP_BOUNDS = (0.005, 0.05)  # 网格步长上下限 (0.5% ~ 5%)

BATCH_MONEY = 20000             # 单笔套利总金额 (分成 3 份分批买)
NUM_TRANCHES = 3                # 分 3 份，每跌 1% 买一份
BUY_GRID_STEP = 0.01            # 每下跌 1% 买入一份
SELL_PROFIT_THRESHOLD = 0.012   # 整体盈利超 1.2% 一次性全卖

def run_backtest():
    # --- [2] 加载数据与特征工程 ---
    print("正在加载并清洗 600895 历史数据...")
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    
    # 趋势指标：小时均线 (60min) 及斜率
    df['ma60'] = df['close'].rolling(60).mean()
    df['slope'] = df['ma60'].diff(5) 
    
    # 波动指标：14 日 ATR（用于动态网格）
    prev_close = df['close'].shift(1)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    df['atr14'] = tr.rolling(ATR_PERIOD).mean()

    # 波动指标：布林带 (20, 2)
    df['ma20'] = df['close'].rolling(20).mean()
    df['std'] = df['close'].rolling(20).std()
    df['upper'] = df['ma20'] + (df['std'] * 2)
    df['lower'] = df['ma20'] - (df['std'] * 2)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df = df.dropna()

    # --- [3] 核心模拟引擎：3 份分批买，盈利 1.2% 全卖 ---
    last_buy_price = df['close'].iloc[0]   # 上一档买入价，用于 1% 触发
    cash_alpha = 0
    hold_t0_volume = 0
    total_cost = 0           # 持仓总成本
    hold_tranches = 0        # 当前持有份数 (0~3)
    trades = []
    equity_curve = []

    part_money = BATCH_MONEY / NUM_TRANCHES

    print("开始执行 3 份分批网格策略回测...")
    for i in range(len(df)):
        curr_p = df['close'].iloc[i]

        # A. 买入：每跌 1% 买一份，最多 3 份
        if hold_tranches < NUM_TRANCHES and curr_p <= last_buy_price * (1 - BUY_GRID_STEP):
            shares = int(part_money / curr_p // 100) * 100
            if shares > 0:
                cash_alpha -= shares * curr_p
                total_cost += shares * curr_p
                hold_t0_volume += shares
                hold_tranches += 1
                last_buy_price = curr_p
                trades.append({'time': df.index[i], 'type': 'BUY', 'price': curr_p})

        # B. 卖出：整体盈利超 1.2% 一次性全卖
        elif hold_t0_volume > 0:
            avg_cost = total_cost / hold_t0_volume
            profit_pct = (curr_p - avg_cost) / avg_cost
            if profit_pct >= SELL_PROFIT_THRESHOLD:
                cash_alpha += hold_t0_volume * curr_p
                trades.append({'time': df.index[i], 'type': 'SELL', 'price': curr_p})
                hold_t0_volume = 0
                total_cost = 0
                hold_tranches = 0
                last_buy_price = curr_p  # 下一轮从当前价开始

        equity_curve.append(cash_alpha + (hold_t0_volume * curr_p))

    # --- [4] 绩效与可视化 ---
    final_price = df['close'].iloc[-1]
    start_price = df['close'].iloc[0]
    
    # Beta 收益：如果 40 万全买入持股不动
    beta_profit = (final_price - start_price) * (INITIAL_CAPITAL / start_price)
    
    print("\n" + "="*40)
    print(f"📊 600895 增强网格回测报告")
    print("-" * 40)
    print(f"成交总数: {len(trades)} (买/卖闭环)")
    print(f"Alpha 收益 (T+0套利纯利): {cash_alpha:,.2f} 元")
    print(f"Beta 收益 (底仓持股损益): {beta_profit:,.2f} 元")
    print(f"摊薄后最终综合表现: {cash_alpha + beta_profit:,.2f} 元")
    print("="*40)

    # 绘制可视化图表
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(df.index, df['close'], color='#2E86AB', linewidth=1, alpha=0.8)
    ax1.scatter([t['time'] for t in buy_trades], [t['price'] for t in buy_trades], marker='^', color='red', label='Grid Buy')
    ax1.scatter([t['time'] for t in sell_trades], [t['price'] for t in sell_trades], marker='v', color='green', label='Grid Sell')
    ax1.set_title("600895 Trading Points Analysis")
    ax1.legend()

    ax2.plot(df.index[:len(equity_curve)], equity_curve, label='Alpha Equity Curve', color='blue')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title("Cumulative Arbitrage Profit (Alpha)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()
