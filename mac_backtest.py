"""
网格策略回测模块
- 数据清洗：处理停牌、缺失数据
- 绩效评估：最大回撤、夏普比率、胜率
- 可视化：股价曲线 + 买卖点
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 配置 ---
CSV_PATH = '/Users/yuhao/Documents/miniqmt/history_600895_1m.csv'
OUTPUT_IMG = '/Users/yuhao/Documents/miniqmt/backtest_result.png'

# 策略参数（BASE_COST 需与标的初始价格匹配，如 600895 约 26，512480 约 1.6）
BASE_COST = 26.85
GRID_STEP = 0.008   # 0.8% 网格步长
SHARES_PER_TRADE = 3000
INITIAL_CAPITAL = 100000
BOLL_PERIOD = 20
RSI_PERIOD = 14


# --- 1. 数据清洗 ---
def load_and_clean_csv(path: str) -> pd.DataFrame:
    """读取 CSV 并处理停牌、缺失数据"""
    df = pd.read_csv(path, index_col=0)

    # 解析时间索引
    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H%M%S')
    df = df.sort_index()

    # 停牌处理：suspendFlag 1=停牌，0=正常，-1=当日起复牌
    if 'suspendFlag' in df.columns:
        df = df[df['suspendFlag'] != 1].copy()

    # 缺失值处理：OHLC 用前向填充，volume 用 0 填充
    ohlc_cols = ['open', 'high', 'low', 'close']
    for col in ohlc_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)

    # 删除仍含 NaN 的行（首尾可能因 ffill/bfill 失败）
    df = df.dropna(subset=['close'])

    return df


# --- 2. 指标计算 ---
def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算 BOLL、RSI（已去除斜率过滤）"""
    df = df.copy()
    df['ma20'] = df['close'].rolling(BOLL_PERIOD).mean()
    df['std'] = df['close'].rolling(BOLL_PERIOD).std()
    df['upper'] = df['ma20'] + df['std'] * 2
    df['lower'] = df['ma20'] - df['std'] * 2

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    return df.dropna()


# --- 3. 回测逻辑 ---
def run_backtest(df: pd.DataFrame, base_cost: float, grid_step: float):
    """运行回测，返回 trades 和 equity 曲线"""
    cash = INITIAL_CAPITAL
    position = 0
    last_trade_price = base_cost
    trades = []
    equity_curve = []

    for i in range(len(df)):
        curr_p = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        lower = df['lower'].iloc[i]
        upper = df['upper'].iloc[i]
        ts = df.index[i]

        # 买入逻辑（已去除斜率过滤）
        if curr_p <= last_trade_price * (1 - grid_step):
            if curr_p <= lower and rsi < 35:
                trades.append({'time': ts, 'type': 'BUY', 'price': curr_p})
                last_trade_price = curr_p
                cash -= curr_p * SHARES_PER_TRADE
                position += SHARES_PER_TRADE

        # 卖出逻辑
        elif curr_p >= last_trade_price * (1 + grid_step):
            if curr_p >= upper and rsi > 65 and position >= SHARES_PER_TRADE:
                trades.append({'time': ts, 'type': 'SELL', 'price': curr_p})
                last_trade_price = curr_p
                cash += curr_p * SHARES_PER_TRADE
                position -= SHARES_PER_TRADE

        # 权益曲线：现金 + 持仓市值
        equity = cash + position * curr_p
        equity_curve.append({'time': ts, 'equity': equity, 'price': curr_p})

    equity_df = pd.DataFrame(equity_curve).set_index('time')
    return trades, equity_df


# --- 4. 绩效评估 ---
def calc_max_drawdown(equity_series: pd.Series) -> tuple:
    """最大回撤 (比例, 绝对值)"""
    cummax = equity_series.cummax()
    # 避免除零
    denom = cummax.replace(0, np.nan)
    drawdown = (equity_series - cummax) / denom
    drawdown = drawdown.fillna(0)
    max_dd_pct = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    peak_idx = equity_series.loc[:max_dd_idx].idxmax()
    max_dd_abs = equity_series.loc[peak_idx] - equity_series.loc[max_dd_idx]
    return max_dd_pct, max_dd_abs


def calc_sharpe_ratio(equity_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """夏普比率（按日收益年化）"""
    # 重采样为日频，取每日末权益
    daily = equity_series.resample('D').last().dropna()
    returns = daily.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    sharpe = (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)
    return sharpe


def calc_win_rate(trades: list) -> float:
    """胜率：配对买卖，盈利次数/总次数"""
    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']
    n = min(len(buys), len(sells))
    if n == 0:
        return 0.0
    wins = sum(1 for i in range(n) if sells[i]['price'] > buys[i]['price'])
    return wins / n


def calc_t0_arbitrage_profit(trades: list, shares_per_trade: int) -> float:
    """T+0 累计套利收益：配对买卖的已实现盈亏（不含持仓浮盈）"""
    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']
    n = min(len(buys), len(sells))
    if n == 0:
        return 0.0
    return sum((sells[i]['price'] - buys[i]['price']) * shares_per_trade for i in range(n))


def calc_profit_loss_ratio(trades: list, shares_per_trade: int) -> float:
    """盈亏比：平均盈利/平均亏损（绝对值）"""
    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']
    n = min(len(buys), len(sells))
    if n == 0:
        return 0.0
    profits = [(sells[i]['price'] - buys[i]['price']) * shares_per_trade for i in range(n)]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0.0
    return avg_win / avg_loss


# --- 5. 可视化 ---
def plot_backtest(df: pd.DataFrame, trades: list, output_path: str):
    """绘制股价曲线 + 买卖点"""
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df['close'], color='#2E86AB', linewidth=1.2, label='Close', alpha=0.9)

    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']

    if buys:
        buy_times = [t['time'] for t in buys]
        buy_prices = [t['price'] for t in buys]
        ax.scatter(buy_times, buy_prices, color='#E94F37', marker='^', s=80, label='Buy', zorder=5)
    if sells:
        sell_times = [t['time'] for t in sells]
        sell_prices = [t['price'] for t in sells]
        ax.scatter(sell_times, sell_prices, color='#44AF69', marker='v', s=80, label='Sell', zorder=5)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Price', fontsize=11)
    ax.set_title('Grid Strategy Backtest - Price & Trades', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"图表已保存: {output_path}")


# --- 主流程 ---
def main():
    print("=" * 50)
    print("网格策略回测")
    print("=" * 50)

    # 1. 数据清洗
    print("\n[1] 加载并清洗数据...")
    df = load_and_clean_csv(CSV_PATH)
    print(f"    有效数据: {len(df)} 条")

    # 2. 指标
    print("\n[2] 计算指标...")
    df = prepare_indicators(df)
    print(f"    指标就绪: {len(df)} 条")

    # 3. 回测
    print("\n[3] 运行回测...")
    trades, equity_df = run_backtest(df, BASE_COST, GRID_STEP)
    print(f"    成交次数: {len(trades)} (买 {sum(1 for t in trades if t['type']=='BUY')} / 卖 {sum(1 for t in trades if t['type']=='SELL')})")

    # 4. 绩效
    print("\n[4] 绩效评估")
    print("-" * 30)
    eq = equity_df['equity']
    max_dd_pct, max_dd_abs = calc_max_drawdown(eq)
    sharpe = calc_sharpe_ratio(eq)
    win_rate = calc_win_rate(trades)
    pl_ratio = calc_profit_loss_ratio(trades, SHARES_PER_TRADE)
    t0_profit = calc_t0_arbitrage_profit(trades, SHARES_PER_TRADE)

    print(f"  最大回撤:     {max_dd_pct:.2%} (绝对: {max_dd_abs:,.2f})")
    print(f"  T+0 累计套利收益: {t0_profit:,.2f}")
    print(f"  夏普比率:     {sharpe:.2f}")
    print(f"  胜率:         {win_rate:.1%}")
    print(f"  盈亏比:       {pl_ratio:.2f}")

    final_cash = eq.iloc[-1] if len(eq) > 0 else 0
    print(f"  期末权益:     {final_cash:,.2f}")
    print("-" * 30)

    # 5. 可视化
    print("\n[5] 生成图表...")
    plot_backtest(df, trades, OUTPUT_IMG)

    print("\n回测完成。")


if __name__ == "__main__":
    main()
