"""
159201 自由现金流 ETF — ATR 动态网格 + 趋势自适应回测
优化平衡版：含手续费、滚动锚点、Alpha 最大回撤，参数偏稳健以降低过拟合与摩擦损耗。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- [1] 参数配置 (优化后的中庸均衡版) ---
# 数据与输出路径：相对本脚本所在目录，任意 cwd 下运行均可用
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(_SCRIPT_DIR, 'history_159201_1m.csv')
OUTPUT_IMAGE = os.path.join(_SCRIPT_DIR, 'optimized_backtest_159201.png')

INITIAL_CAPITAL = 500000        # 总资金 50 万 = 固定仓 30 万 + 流动仓 20 万（用于 Alpha 曲线基准）
BETA_CAPITAL = 300000           # 固定仓 30 万，用于 Beta 计算
BATCH_MONEY = 200000            # 单轮流动资金池 20 万（有限流动仓时用）
MAX_LAYERS = 9                  # 层数扫描综合最优（占用年化与回撤平衡）
# 无限流动仓：True=层数按 MAX_LAYERS 上限、权重 (1,1,2,2,3,3,4,4,...)，每层金额=BASE_UNIT*权重
UNLIMITED_FLOW = False          # 关闭无限流动仓，使用固定层数 + BATCH_MONEY 分配
BASE_UNIT = 15000               # 无限流动仓时每“1”权重对应金额（元），如 1.5 万
# 有限流动仓时的阶梯权重（UNLIMITED_FLOW=False 时用），长度须等于 MAX_LAYERS
LAYER_WEIGHTS = (1, 1, 1, 1, 1, 1, 1, 1, 1)  # 9 层均权
# 无限流动仓时的最大层数（UNLIMITED_FLOW=True 时生效）
MAX_LAYERS_UNLIMITED = 50
ATR_PERIOD = 14

# 甜点区参数：成本下调（万一佣金）+ 覆盖实盘 1 tick 滑点的步长下限
ATR_GRID_FACTOR = 0.38          # 网格弹性系数，加大间距保护子弹
GRID_STEP_FLOOR = 0.0012        # 步长下限 0.12%，覆盖实盘至少 1 tick 滑点损耗
LAYER_STEP_BONUS = 0.0001       # 层数加成：每多一层在基础步长上加 0.01%，第 9 层比第 1 层多约 0.08%
SELL_PROFIT_THRESHOLD = 0.005   # 止盈阈值 0.5%（单笔止盈为基准×涨多系数，无层数折扣）
SELL_BY_LOT = True              # True=单笔网格达到指定收益即卖；False=整体收益达阈值后一次性全平
# 跌少涨多：买入步长系数、止盈系数
BUY_STEP_FACTOR = 1.0           # 跌少：实际买入触发步长 = grid_step * 此值
SELL_THRESHOLD_FACTOR = 1.4     # 涨多：1.0=对称；1.4=温和(推荐)；1.8=激进
COMMISSION_RATE = 0.0001        # 交易手续费 万一 (0.01%)，ETF 无印花税
# 动态滑点压力测试：实盘价差约 0.05%，在回测中叠加 1–2 tick 做压力测试
STRESS_SLIPPAGE_ENABLED = False       # 是否在回测中叠加滑点（压力测试）
STRESS_SLIPPAGE_RATE = 0.0005         # 单边滑点 0.05%（约 1–2 tick）

# 时间窗口过滤：早盘/尾盘波动不理性，限制开仓
TIME_FILTER_ENABLED = False           # 是否启用时间过滤
NO_FIRST_LAYER_UNTIL = (9, 45)        # 9:30–9:45 只观察、不新开第一层
NO_BUY_AFTER = (14, 45)              # 14:45 之后只卖不买

# 趋势自适应参数
TREND_MA_PERIOD = 60
UPTREND_GRID_FACTOR = 1.2       # 上升趋势：放宽网格（跌少买）
UPTREND_SELL_FACTOR = 1.33      # 上升趋势：提高止盈（涨多卖）
UPTREND_BATCH_FACTOR = 0.7      # 上升趋势：减小仓位
DOWNTREND_GRID_FACTOR = 1.0    # 下降趋势：收紧网格（跌多买）
DOWNTREND_SELL_FACTOR = 0.83    # 下降趋势：降低止盈（涨少卖）
DOWNTREND_BATCH_FACTOR = 1.2    # 下降趋势：加大仓位

# 自动化风控：ATR 熔断 + 单周期最大浮亏止损
ATR_CIRCUIT_BREAKER_ENABLED = True   # 异常波动时暂停加仓
ATR_CIRCUIT_BREAKER_RATIO = 2.0      # 当前 ATR > 近期均值×此倍数则熔断
ATR_LOOKBACK = 60                    # 近期 ATR 用过去 60 根 K 线均值
ENABLE_FLOAT_LOSS_STOP = False       # 是否启用单周期最大浮亏止损（关闭则不再强制平仓）
MAX_CYCLE_FLOAT_LOSS = 15000         # 单周期最大浮亏（元），针对 20 万流动仓；超过则强制平仓（仅当 ENABLE 时生效）

# 动态冷静期：全平后禁止新开第一层；RSI<35 超跌缩短为 5 分钟，否则 15 分钟
COOLING_ENABLED = True               # 全量平仓后进入冷静期
COOLING_BARS = 15                    # 默认冷静期 15 根 K 线（分钟）
COOLING_BARS_SHORT = 5               # RSI < RSI_COOLING_THRESHOLD 时缩短为 5 分钟，便于超跌反弹二次进场
RSI_COOLING_THRESHOLD = 35           # RSI(14) < 35 视为超跌，使用缩短冷静期
COOLING_RSI_ENABLED = False          # 是否要求 RSI 回落至中轴以下才允许新开仓（与动态冷静期独立）
COOLING_RSI_BELOW = 50
RSI_PERIOD = 14

# 多周期趋势共振：大周期与 1 分钟同向时才用激进系数，否则用中性
MTF_ENABLED = False                   # 是否启用多周期趋势共振（False=仅 1 分钟斜率）
MTF_RESAMPLE = '15min'                # 大周期：15 分钟
MTF_MA_BARS = 20                      # 15 分钟 K 线 MA 周期（约 5 小时）
MTF_SLOPE_BARS = 2                    # 15 分钟斜率用 diff(2)


def run_backtest(return_metrics=False, max_layers_override=None, buy_step_factor=1.0, sell_threshold_factor=1.0, grid_step_floor_override=None, batch_money_override=None, beta_capital_override=None):
    """return_metrics=True 时只返回 dict。batch_money_override/beta_capital_override 用于固定仓｜流动仓对比。"""
    if not os.path.exists(FILE_PATH):
        print(f"错误: 未找到数据文件 {FILE_PATH}")
        print("请将 1 分钟 K 线数据放到该路径，格式需包含 open/high/low/close，索引为时间。")
        return

    # --- [2] 加载与处理数据 ---
    df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)

    # ATR 计算
    prev_close = df['close'].shift(1)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    df['atr14'] = tr.rolling(ATR_PERIOD).mean()

    # 趋势计算（1 分钟）：原始斜率经 3 周期 EMA 平滑，减少参数频繁切换
    df['ma60'] = df['close'].rolling(TREND_MA_PERIOD).mean()
    raw_slope = df['ma60'].diff(5) / df['ma60'].shift(5)
    df['ma60_slope'] = raw_slope.ewm(span=3, adjust=False).mean()
    df['atr14_avg'] = df['atr14'].rolling(ATR_LOOKBACK).mean().shift(1)
    # 大周期趋势（15 分钟）用于多周期共振
    if MTF_ENABLED:
        df_15 = df[['open', 'high', 'low', 'close']].resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        df_15['ma'] = df_15['close'].rolling(MTF_MA_BARS).mean()
        df_15['slope_15'] = df_15['ma'].diff(MTF_SLOPE_BARS) / df_15['ma'].shift(MTF_SLOPE_BARS)
        df_15 = df_15.dropna(subset=['slope_15'])
        # 映射到每根 1 分钟：该分钟所属 15 分钟 bar 的斜率
        def _map_slope_15(t):
            key = t.floor('15min')
            return df_15.loc[key, 'slope_15'] if key in df_15.index else np.nan
        df['slope_15'] = df.index.to_series().map(_map_slope_15)
        df['slope_15'] = df['slope_15'].fillna(0)
    # RSI（用于冷静期过滤）
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(100)
    df = df.dropna()

    # 有效费率 = 佣金 + 滑点（压力测试时叠加）
    effective_rate = COMMISSION_RATE + (STRESS_SLIPPAGE_RATE if STRESS_SLIPPAGE_ENABLED else 0.0)
    batch_money = batch_money_override if batch_money_override is not None else BATCH_MONEY
    beta_capital = beta_capital_override if beta_capital_override is not None else BETA_CAPITAL

    # --- [3] 模拟引擎 ---
    last_buy_price = df['close'].iloc[0]
    cash_alpha = 0
    hold_t0_volume = 0
    total_cost = 0
    hold_layers = 0
    # 单笔止盈时按“笔”记录持仓，每笔 {shares, cost, buy_price}；整体止盈时 positions 不用
    positions = []  # list of dict
    trades = []
    alpha_equity = []
    # 统计：30 万子弹打满的次数；打满后想买但买不了的 K 线次数
    exhaust_count = 0
    bars_would_buy_but_full = 0
    # 进一步统计：每一轮“打满”周期内的最大浮亏与最终盈利
    cycle_in_progress = False
    cycle_alpha_start = 0.0
    cycle_had_exhaust = False
    cycle_min_equity = 0.0
    exhaust_cycle_stats = []  # 每项: {pnl, max_drawdown}
    # 第 3 轮打满周期详细日志（从打满 9 层到平仓的每根 K 线）
    recording_third_cycle = False
    third_cycle_log = []
    third_cycle_start_time = None  # 该轮周期首次买入时间（从 trades 反推）
    n_atr_pause_bars = 0   # ATR 熔断导致未加仓的 K 线数
    n_force_stops = 0      # 触发单周期最大浮亏止损次数
    last_sell_bar = -9999     # 上次全量卖出的 K 线下标，用于冷静期
    dates_with_position = set()  # 有持仓的自然日（用于资金占用年化）
    trigger_log = []             # 前几笔买卖的触发条件明细，用于展示策略执行
    MAX_TRIGGER_LOG = 5          # 每类(买/卖)最多记录笔数

    if max_layers_override is not None:
        max_layers = max_layers_override
        part_money_list = [batch_money / max_layers] * max_layers
    elif UNLIMITED_FLOW:
        max_layers = MAX_LAYERS_UNLIMITED
        layer_weights_use = tuple((k // 2) + 1 for k in range(MAX_LAYERS_UNLIMITED))  # (1,1,2,2,3,3,...)
        part_money_list = [BASE_UNIT * w for w in layer_weights_use]
    else:
        max_layers = MAX_LAYERS
        _weight_sum = sum(LAYER_WEIGHTS)
        part_money_list = [BATCH_MONEY * w / _weight_sum for w in LAYER_WEIGHTS]

    for i in range(len(df)):
        curr_p = df['close'].iloc[i]
        atr = df['atr14'].iloc[i]
        slope = df['ma60_slope'].iloc[i]
        slope_15 = df['slope_15'].iloc[i] if (MTF_ENABLED and 'slope_15' in df.columns) else 0.0

        # 动态步长与止盈
        _floor = grid_step_floor_override if grid_step_floor_override is not None else GRID_STEP_FLOOR
        base_grid_step = max(_floor, (atr / curr_p) * ATR_GRID_FACTOR)
        sell_threshold = SELL_PROFIT_THRESHOLD
        batch_factor = 1.0

        same_direction = (slope > 0 and slope_15 > 0) or (slope < 0 and slope_15 < 0)
        if MTF_ENABLED and not same_direction:
            pass  # 保持中性
        elif slope > 0:
            base_grid_step *= UPTREND_GRID_FACTOR
            sell_threshold *= UPTREND_SELL_FACTOR
            batch_factor = UPTREND_BATCH_FACTOR
        elif slope < 0:
            base_grid_step *= DOWNTREND_GRID_FACTOR
            sell_threshold *= DOWNTREND_SELL_FACTOR
            batch_factor = DOWNTREND_BATCH_FACTOR

        # 层数加成网格步长：随层数增加额外间距，防止子弹过快耗尽
        grid_step = base_grid_step + (hold_layers * LAYER_STEP_BONUS)

        # ATR 熔断：当前 ATR > 近期均值×倍数 则本 K 线不新开层
        atr_avg = df['atr14_avg'].iloc[i]
        pause_buy = ATR_CIRCUIT_BREAKER_ENABLED and (
            pd.notna(atr_avg) and atr > atr_avg * ATR_CIRCUIT_BREAKER_RATIO
        )
        if pause_buy and hold_layers < max_layers and curr_p <= last_buy_price * (1 - grid_step):
            n_atr_pause_bars += 1

        # 时间窗口过滤：未启用时直接放行；启用时 9:30–9:45 不新开第一层、14:45 后只卖不买
        time_filter_block_buy = False
        if TIME_FILTER_ENABLED:
            h, m = df.index[i].hour, df.index[i].minute
            if (hold_layers == 0 and (h, m) >= (9, 30) and (h, m) < NO_FIRST_LAYER_UNTIL) or (h, m) >= NO_BUY_AFTER:
                time_filter_block_buy = True

        buy_triggered = curr_p <= last_buy_price * (1 - grid_step * buy_step_factor)

        def _cycle_done_clear():
            nonlocal last_sell_bar, hold_t0_volume, total_cost, hold_layers, last_buy_price
            nonlocal cycle_in_progress, cycle_had_exhaust, recording_third_cycle
            last_sell_bar = i
            hold_t0_volume = 0
            total_cost = 0
            hold_layers = 0
            positions.clear()
            last_buy_price = curr_p
            if cycle_in_progress and cycle_had_exhaust:
                final_pnl = cash_alpha - cycle_alpha_start
                max_drawdown = cycle_min_equity - cycle_alpha_start
                exhaust_cycle_stats.append({"pnl": float(final_pnl), "max_drawdown": float(max_drawdown)})
                if recording_third_cycle:
                    recording_third_cycle = False
            cycle_in_progress = False
            cycle_had_exhaust = False

        # 卖出逻辑优先：有持仓时先检查单周期最大浮亏止损，再止盈（整体或单笔）
        if hold_t0_volume > 0:
            unrealized_pnl = hold_t0_volume * curr_p - total_cost

            if ENABLE_FLOAT_LOSS_STOP and unrealized_pnl < -MAX_CYCLE_FLOAT_LOSS:
                # 单周期最大浮亏止损（强制全平）
                fee = hold_t0_volume * curr_p * effective_rate
                cash_alpha += (hold_t0_volume * curr_p - fee)
                trades.append({
                    'time': df.index[i], 'type': 'SELL', 'price': curr_p, 'shares': hold_t0_volume,
                    'reason': 'max_float_loss',
                })
                n_force_stops += 1
                _cycle_done_clear()

            elif SELL_BY_LOT and positions:
                # 单笔止盈：统一止盈阈值×涨多系数，无层数折扣
                sell_eff = sell_threshold * sell_threshold_factor
                to_remove = []
                for idx, lot in enumerate(positions):
                    if curr_p >= lot['buy_price'] * (1 + sell_eff):
                        sell_shares = lot['shares']
                        fee = sell_shares * curr_p * effective_rate
                        cash_alpha += (sell_shares * curr_p - fee)
                        trades.append({'time': df.index[i], 'type': 'SELL', 'price': curr_p, 'shares': sell_shares, 'reason': 'lot_profit'})
                        if len([x for x in trigger_log if x.get("type") == "SELL"]) < MAX_TRIGGER_LOG:
                            profit_pct_lot = (curr_p - lot['buy_price']) / lot['buy_price'] * 100
                            trigger_log.append({
                                "type": "SELL", "time": df.index[i], "buy_price": lot['buy_price'], "curr_p": curr_p,
                                "sell_threshold_pct": sell_eff * 100, "profit_pct": profit_pct_lot, "shares": sell_shares,
                            })
                        to_remove.append(idx)
                for idx in reversed(to_remove):
                    positions.pop(idx)
                hold_t0_volume = sum(p['shares'] for p in positions)
                total_cost = sum(p['cost'] for p in positions)
                hold_layers = len(positions)
                if not positions:
                    last_sell_bar = i
                    last_buy_price = curr_p
                    if cycle_in_progress and cycle_had_exhaust:
                        final_pnl = cash_alpha - cycle_alpha_start
                        max_drawdown = cycle_min_equity - cycle_alpha_start
                        exhaust_cycle_stats.append({"pnl": float(final_pnl), "max_drawdown": float(max_drawdown)})
                        if recording_third_cycle:
                            recording_third_cycle = False
                    cycle_in_progress = False
                    cycle_had_exhaust = False

        # 买入逻辑 (滚动锚点)，熔断/时间窗口/冷静期时不加仓
        if hold_layers < max_layers and buy_triggered and not pause_buy and not time_filter_block_buy:
            in_cooling = False
            if COOLING_ENABLED and hold_layers == 0 and hold_t0_volume == 0:
                rsi_now = df['rsi'].iloc[i] if pd.notna(df['rsi'].iloc[i]) else 50
                cooling_bars_effective = COOLING_BARS_SHORT if rsi_now < RSI_COOLING_THRESHOLD else COOLING_BARS
                if (i - last_sell_bar) < cooling_bars_effective:
                    in_cooling = True
                if COOLING_RSI_ENABLED and pd.notna(df['rsi'].iloc[i]) and df['rsi'].iloc[i] > COOLING_RSI_BELOW:
                    in_cooling = True
            if not in_cooling:
                if not cycle_in_progress and hold_t0_volume == 0 and hold_layers == 0:
                    cycle_in_progress = True
                    cycle_alpha_start = cash_alpha
                    cycle_had_exhaust = False
                    cycle_min_equity = cash_alpha
                money = part_money_list[hold_layers] * batch_factor
                shares = int(money / curr_p // 100) * 100
                if shares > 0:
                    _anchor = last_buy_price
                    trigger_price = _anchor * (1 - grid_step * buy_step_factor)
                    actual_drop_pct = (_anchor - curr_p) / _anchor * 100
                    fee = shares * curr_p * effective_rate
                    cost_lot = shares * curr_p
                    cash_alpha -= (cost_lot + fee)
                    total_cost += cost_lot
                    hold_t0_volume += shares
                    hold_layers += 1
                    if SELL_BY_LOT:
                        positions.append({'shares': shares, 'cost': cost_lot, 'buy_price': curr_p})
                    last_buy_price = curr_p
                    trades.append({'time': df.index[i], 'type': 'BUY', 'price': curr_p, 'shares': shares})
                    if len([x for x in trigger_log if x.get("type") == "BUY"]) < MAX_TRIGGER_LOG:
                        trigger_log.append({
                            "type": "BUY", "time": df.index[i], "anchor": _anchor, "curr_p": curr_p,
                            "grid_step_pct": grid_step * 100, "buy_step_factor": buy_step_factor,
                            "trigger_price": trigger_price, "actual_drop_pct": actual_drop_pct,
                            "layer": hold_layers, "atr_over_price_pct": (atr / curr_p) * 100, "shares": shares,
                        })
                    if hold_layers == max_layers:
                        exhaust_count += 1
                        cycle_had_exhaust = True
                        if exhaust_count == 3:
                            recording_third_cycle = True
                            third_cycle_start_time = trades[-9]["time"] if len(trades) >= 9 else df.index[i]
        elif hold_layers == max_layers and buy_triggered:
            bars_would_buy_but_full += 1

        # 记录有持仓的日期（用于资金占用年化）
        if hold_t0_volume > 0:
            try:
                dates_with_position.add(df.index[i].date())
            except Exception:
                dates_with_position.add(pd.Timestamp(df.index[i]).date())
        # 记录整体 Alpha 权益曲线，并在有打满周期时刷新该周期内的最小权益
        equity = cash_alpha + (hold_t0_volume * curr_p)
        if cycle_in_progress:
            if not cycle_had_exhaust:
                # 在未打满之前也跟踪权益低点，方便观察整个周期风险
                cycle_min_equity = min(cycle_min_equity, equity)
            else:
                cycle_min_equity = min(cycle_min_equity, equity)
        alpha_equity.append(equity)

        # 第 3 轮打满：逐 K 线记录时间、仓位、总成本、现价、浮盈浮亏、累计 Alpha 盈亏
        if recording_third_cycle:
            unrealized = (hold_t0_volume * curr_p - total_cost) if hold_t0_volume else 0.0
            cum_alpha = cash_alpha - cycle_alpha_start
            third_cycle_log.append({
                "time": df.index[i],
                "hold_volume": hold_t0_volume,
                "total_cost": total_cost,
                "price": curr_p,
                "unrealized_pnl": unrealized,
                "cumulative_alpha_pnl": cum_alpha,
            })

    # --- [4] 绩效计算 ---
    alpha_curve = np.array(alpha_equity)
    cum_max = np.maximum.accumulate(alpha_curve - alpha_curve[0] + INITIAL_CAPITAL)
    drawdowns = (alpha_curve - alpha_curve[0] + INITIAL_CAPITAL) / cum_max - 1
    max_dd = drawdowns.min()

    end_price = df['close'].iloc[-1]
    start_price = df['close'].iloc[0]
    beta_profit = (end_price - start_price) * (beta_capital / start_price)
    final_alpha = alpha_equity[-1]
    losing_exhaust_cycles = sum(1 for stat in exhaust_cycle_stats if stat["pnl"] < 0)

    # 流动仓年化：① 按回测区间自然日 ② 按资金占用日（仅对有持仓的天数年化）
    t0, t1 = df.index[0], df.index[-1]
    days_span = (t1 - t0).total_seconds() / 86400.0
    years_span = days_span / 365.0 if days_span > 0 else 1.0
    flow_annualized = (final_alpha / batch_money) / years_span if years_span > 0 else 0.0
    days_util = len(dates_with_position) if dates_with_position else 1
    flow_annualized_util = (final_alpha / batch_money) * (365.0 / days_util) if days_util > 0 else 0.0

    if return_metrics:
        return {
            "batch_money": batch_money,
            "beta_capital": beta_capital,
            "beta_profit": beta_profit,
            "total_profit": final_alpha + beta_profit,
            "layers": max_layers,
            "alpha": final_alpha,
            "flow_ann": flow_annualized,
            "flow_ann_util": flow_annualized_util,
            "max_dd": max_dd,
            "trades": len(trades),
            "exhaust_count": exhaust_count,
            "exhaust_cycles": len(exhaust_cycle_stats),
            "losing_cycles": losing_exhaust_cycles,
            "days_util": days_util,
        }

    print(f"\n📊 159201 优化平衡版回测结果")
    print("-" * 40)
    if STRESS_SLIPPAGE_ENABLED:
        print(f"【压力测试】有效费率 = 佣金+滑点 = {effective_rate*100:.2f}% 单边 (佣金 {COMMISSION_RATE*100:.2f}% + 滑点 {STRESS_SLIPPAGE_RATE*100:.2f}%)")
    if TIME_FILTER_ENABLED:
        print(f"【时间过滤】9:30–9:45 不新开第一层 | 14:45 后只卖不买")
    print(f"成交总数: {len(trades)} (买+卖)")
    print(f"打满({max_layers}层)次数: {exhaust_count} 次")
    print(f"打满后想买但无子弹的K线数: {bars_would_buy_but_full} 根 (急速下跌时被动等待)")
    print(f"【风控】ATR 熔断未加仓 K 线数: {n_atr_pause_bars} 根 | 单周期浮亏止损触发: {n_force_stops} 次")
    print(f"打满且完成平仓的轮次: {len(exhaust_cycle_stats)} 轮，其中亏损轮次: {losing_exhaust_cycles} 轮")
    for i, stat in enumerate(exhaust_cycle_stats, start=1):
        max_float_loss = -min(0.0, stat["max_drawdown"])
        print(f"  第{i}轮打满: 最大浮亏 {max_float_loss:,.2f} 元，最终盈利 {stat['pnl']:,.2f} 元")
    print(f"Alpha 套利净收益: {final_alpha:,.2f} 元 (已扣手续费)")
    print(f"流动仓({batch_money/10000:.0f}万)年化: 自然日 {flow_annualized*100:.2f}% | 资金占用 {flow_annualized_util*100:.2f}% (有仓 {days_util} 日, 回测 {days_span:.0f} 自然日)")
    print(f"Beta 持仓市值变动: {beta_profit:,.2f} 元 (固定仓{beta_capital/10000:.0f}万)")
    print(f"策略摊薄总收益: {final_alpha + beta_profit:,.2f} 元")
    print(f"Alpha 曲线最大回撤: {max_dd*100:.2%}")
    print("-" * 40)

    # --- 触发条件示例：展示实际执行时的买入/卖出触发 ---
    if trigger_log and not return_metrics:
        buys = [x for x in trigger_log if x.get("type") == "BUY"]
        sells = [x for x in trigger_log if x.get("type") == "SELL"]
        print(f"\n📌 策略触发条件示例（前{MAX_TRIGGER_LOG}笔买/卖）")
        print("买入触发: 当前价 ≤ 锚点×(1 - 网格步长×跌少系数) 即买")
        if buys:
            print(f"  {'时间':<20} | {'锚点':>8} | {'现价':>8} | {'步长%':>7} | {'触发价':>8} | {'实际跌幅%':>10} | 层 | 股数")
            for r in buys:
                t = str(r["time"])[:19] if hasattr(r["time"], "strftime") else str(r["time"])
                print(f"  {t:<20} | {r['anchor']:.4f} | {r['curr_p']:.4f} | {r['grid_step_pct']:.3f} | {r['trigger_price']:.4f} | {r['actual_drop_pct']:>10.3f} | {r['layer']} | {r['shares']}")
        print("卖出触发: 当前价 ≥ 该笔买入价×(1 + 止盈阈值×涨多系数) 即卖该笔")
        if sells:
            print(f"  {'时间':<20} | {'买入价':>8} | {'现价':>8} | {'止盈%':>7} | {'实际涨幅%':>10} | 股数")
            for r in sells:
                t = str(r["time"])[:19] if hasattr(r["time"], "strftime") else str(r["time"])
                print(f"  {t:<20} | {r['buy_price']:.4f} | {r['curr_p']:.4f} | {r['sell_threshold_pct']:.3f} | {r['profit_pct']:>10.3f} | {r['shares']}")
        print("-" * 40)

    # --- 诊断：卖光后到下次买入的间隔（验证“连续上涨导致无法触发网格”）---
    sell_times = [t["time"] for t in trades if t["type"] == "SELL"]
    buy_times = [t["time"] for t in trades if t["type"] == "BUY"]
    gaps_min = []
    for j, t_sell in enumerate(sell_times):
        next_buys = [t for t in buy_times if t > t_sell]
        if next_buys:
            next_buy = min(next_buys)
            gap_min = (next_buy - t_sell).total_seconds() / 60.0
            gaps_min.append(gap_min)
    if gaps_min:
        print(f"\n📌 卖光(止盈)后到下次触发买入的间隔（分钟）")
        print(f"    止盈卖光次数: {len(sell_times)} 次 | 有后续买入的: {len(gaps_min)} 次")
        print(f"    平均间隔: {np.mean(gaps_min):.0f} 分钟 | 最小: {np.min(gaps_min):.0f} 分钟 | 最大: {np.max(gaps_min):.0f} 分钟")
        print(f"    → 卖光后锚点抬高，只有价格从卖价回调一个网格步长才会再买；连续上涨时长期空仓、无法触发网格。")
    print("-" * 40)

    # 第 3 轮打满周期详细数据（时间、总仓位浮盈浮亏与累计 Alpha 盈亏）
    if third_cycle_log:
        start_time = third_cycle_start_time if third_cycle_start_time is not None else third_cycle_log[0]["time"]
        end_time = third_cycle_log[-1]["time"]
        n_bars = len(third_cycle_log)
        worst = min(third_cycle_log, key=lambda x: x["unrealized_pnl"])
        final_pnl = third_cycle_log[-1]["cumulative_alpha_pnl"]
        print(f"\n📌 第 3 轮打满周期明细（从打满 {max_layers} 层到平仓）")
        print(f"    周期开始(首笔买入): {start_time}")
        print(f"    打满至平仓 K 线数: {n_bars} 根")
        print(f"    平仓时间: {end_time}")
        print(f"    期间最大浮亏: {worst['unrealized_pnl']:,.2f} 元 (出现在 {worst['time']})")
        print(f"    最终盈利(Alpha): {final_pnl:,.2f} 元")
        print(f"    时间 | 持仓(股) | 总成本(元) | 现价 | 浮盈浮亏(元) | 累计Alpha盈亏(元)")
        print("-" * 75)
        def fmt(r):
            t = str(r["time"])[:19] if hasattr(r["time"], "strftime") else str(r["time"])
            return f"    {t} | {r['hold_volume']:>8} | {r['total_cost']:>12,.0f} | {r['price']:.3f} | {r['unrealized_pnl']:>12,.2f} | {r['cumulative_alpha_pnl']:>14,.2f}"
        for row in third_cycle_log[:15]:
            print(fmt(row))
        if n_bars > 32:
            print("    ...")
            print(fmt(worst))
            print("    ...")
        for row in third_cycle_log[-15:]:
            print(fmt(row))
        print("-" * 75)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(alpha_curve, label='Alpha Equity (Arbitrage Profit)', color='#2ca02c', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title('159201 Optimized Grid Strategy - Alpha Performance')
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Profit (CNY)')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig(OUTPUT_IMAGE, dpi=120)
    plt.close()
    print(f"图表已保存为 {OUTPUT_IMAGE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # 层数扫描：尝试 9,11,13,15,17,20 层，取资金占用年化 - 0.3*|回撤| 最大
        candidates = [9, 11, 13, 15, 17, 20]
        results = []
        for L in candidates:
            m = run_backtest(return_metrics=True, max_layers_override=L)
            m["score"] = m["flow_ann_util"] * 100 + 0.3 * m["max_dd"] * 100  # 年化% + 回撤惩罚(回撤为负)
            results.append(m)
        print("\n层数扫描 (单笔止盈0.5%, 关浮亏止损, 均权)")
        print("-" * 90)
        print(f"{'层数':>4} | {'Alpha(元)':>10} | {'自然日年化%':>10} | {'占用年化%':>10} | {'最大回撤%':>10} | {'成交数':>6} | {'打满':>4} | 综合得分")
        print("-" * 90)
        for r in results:
            print(f"{r['layers']:>4} | {r['alpha']:>10,.0f} | {r['flow_ann']*100:>10.2f} | {r['flow_ann_util']*100:>10.2f} | {r['max_dd']*100:>10.2f} | {r['trades']:>6} | {r['exhaust_count']:>4} | {r['score']:.2f}")
        best = max(results, key=lambda x: x["score"])
        print("-" * 90)
        print(f"综合最优: {best['layers']} 层 (得分 {best['score']:.2f}, 占用年化 {best['flow_ann_util']*100:.2f}%, 回撤 {best['max_dd']*100:.2f}%)")
        print("提示: 得分 = 占用年化% + 0.3×回撤%；可手动将 MAX_LAYERS / LAYER_WEIGHTS 改为上表最优后运行 python mac_backtest_159201.py 查看详情。")
    elif len(sys.argv) > 1 and sys.argv[1] == "asymmetric":
        # 跌少涨多策略扫描：buy_step_factor<1=更小跌幅就买，sell_threshold_factor>1=更高涨幅才卖
        buy_factors = [0.5, 0.7, 0.85, 1.0]
        sell_factors = [1.0, 1.2, 1.5, 1.8, 2.0]
        results = []
        for bf in buy_factors:
            for sf in sell_factors:
                m = run_backtest(return_metrics=True, buy_step_factor=bf, sell_threshold_factor=sf)
                m["buy_factor"] = bf
                m["sell_factor"] = sf
                m["score"] = m["flow_ann_util"] * 100 + 0.3 * m["max_dd"] * 100
                results.append(m)
        print("\n跌少涨多 策略扫描 (9层均权, 单笔止盈基准0.5%)")
        print("跌少=买入步长系数(越小越易买) 涨多=止盈系数(越大越晚卖)")
        print("-" * 95)
        print(f"{'跌少(buy)':>10} | {'涨多(sell)':>10} | {'Alpha(元)':>10} | {'占用年化%':>10} | {'最大回撤%':>10} | {'成交数':>6} | 综合得分")
        print("-" * 95)
        for r in results:
            print(f"{r['buy_factor']:>10.2f} | {r['sell_factor']:>10.2f} | {r['alpha']:>10,.0f} | {r['flow_ann_util']*100:>10.2f} | {r['max_dd']*100:>10.2f} | {r['trades']:>6} | {r['score']:.2f}")
        best = max(results, key=lambda x: x["score"])
        print("-" * 95)
        print(f"综合最优: 跌少={best['buy_factor']:.2f} 涨多={best['sell_factor']:.2f} (得分 {best['score']:.2f}, 占用年化 {best['flow_ann_util']*100:.2f}%, 回撤 {best['max_dd']*100:.2f}%)")
        print("提示: 运行 python mac_backtest_159201.py asymmetric 使用上述最优参数需在代码中设置 BUY_STEP_FACTOR / SELL_THRESHOLD_FACTOR。")
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep_grid":
        # GRID_STEP_FLOOR 扫描：多组步长下限，综合得分 = 占用年化% + 0.3×回撤% - 0.2×打满次数
        candidates_pct = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        results = []
        for pct in candidates_pct:
            floor = pct / 100.0
            m = run_backtest(return_metrics=True, grid_step_floor_override=floor)
            m["floor_pct"] = pct
            m["score"] = m["flow_ann_util"] * 100 + 0.3 * m["max_dd"] * 100 - 0.2 * m["exhaust_count"]
            results.append(m)
        print("\nGRID_STEP_FLOOR 步长下限扫描 (9层 涨多1.4)")
        print("得分 = 占用年化% + 0.3×回撤% - 0.2×打满次数")
        print("-" * 95)
        print(f"{'步长下限%':>10} | {'Alpha(元)':>10} | {'占用年化%':>10} | {'最大回撤%':>10} | {'成交数':>6} | {'打满':>4} | 综合得分")
        print("-" * 95)
        for r in results:
            print(f"{r['floor_pct']:>10.2f} | {r['alpha']:>10,.0f} | {r['flow_ann_util']*100:>10.2f} | {r['max_dd']*100:>10.2f} | {r['trades']:>6} | {r['exhaust_count']:>4} | {r['score']:.2f}")
        best = max(results, key=lambda x: x["score"])
        print("-" * 95)
        print(f"综合最优: 步长下限 {best['floor_pct']:.2f}% (得分 {best['score']:.2f}, 占用年化 {best['flow_ann_util']*100:.2f}%, 回撤 {best['max_dd']*100:.2f}%, 打满 {best['exhaust_count']} 次)")
        print("提示: 将 GRID_STEP_FLOOR 设为最优值(如 0.0010 表示 0.10%) 后运行 python mac_backtest_159201.py")
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # 资金池×层数对比：20万/9层、30万/13层、40万/18层、50万/23层，均权
        configs = [
            (200000, 9),
            (300000, 13),
            (400000, 18),
            (500000, 23),
        ]
        results = []
        for money, layers in configs:
            m = run_backtest(return_metrics=True, max_layers_override=layers, batch_money_override=money,
                            buy_step_factor=BUY_STEP_FACTOR, sell_threshold_factor=SELL_THRESHOLD_FACTOR)
            if m is not None:
                results.append(m)
        if results:
            print("\n📊 资金池×层数 数据对比（均权、同一数据与策略）")
            print("-" * 115)
            print(f"{'资金池':>10} | {'层数':>4} | {'Alpha(元)':>12} | {'自然日年化%':>10} | {'占用年化%':>10} | {'最大回撤%':>10} | {'成交数':>6} | {'打满':>4} | {'完成平仓轮次':>12} | {'亏损轮次':>6}")
            print("-" * 115)
            for r in results:
                print(f"{r['batch_money']/10000:>8.0f}万 | {r['layers']:>4} | {r['alpha']:>12,.0f} | {r['flow_ann']*100:>10.2f} | {r['flow_ann_util']*100:>10.2f} | {r['max_dd']*100:>10.2f} | {r['trades']:>6} | {r['exhaust_count']:>4} | {r['exhaust_cycles']:>12} | {r['losing_cycles']:>6}")
            print("-" * 115)
            print("说明: 同一 history_159201_1m.csv，单笔止盈 0.5%×涨多1.4，动态冷静期，ATR 熔断。年化与回撤均按该资金池规模计算。")
    elif len(sys.argv) > 1 and sys.argv[1] == "compare_capital":
        # 固定仓｜流动仓 组合对比：30万固定｜20万流动 vs 20万固定｜30万流动
        configs = [
            {"beta_capital": 300000, "batch_money": 200000, "layers": 9},
            {"beta_capital": 200000, "batch_money": 300000, "layers": 13},
        ]
        results = []
        for c in configs:
            m = run_backtest(return_metrics=True, max_layers_override=c["layers"], batch_money_override=c["batch_money"],
                            beta_capital_override=c["beta_capital"], buy_step_factor=BUY_STEP_FACTOR, sell_threshold_factor=SELL_THRESHOLD_FACTOR)
            if m is not None:
                results.append(m)
        if results:
            print("\n📊 固定仓｜流动仓 数据对比（同一策略：单笔止盈 0.5%×涨多1.4）")
            print("-" * 125)
            print(f"{'固定仓':>8} | {'流动仓':>8} | {'层数':>4} | {'Alpha(元)':>10} | {'Beta收益(元)':>12} | {'总收益(元)':>12} | {'流动仓年化%':>10} | {'最大回撤%':>8} | {'成交数':>6} | {'打满':>4} | {'完成平仓':>6}")
            print("-" * 125)
            for r in results:
                fix = r["beta_capital"] / 10000
                flow = r["batch_money"] / 10000
                print(f"{fix:>6.0f}万 | {flow:>6.0f}万 | {r['layers']:>4} | {r['alpha']:>10,.0f} | {r['beta_profit']:>12,.0f} | {r['total_profit']:>12,.0f} | {r['flow_ann_util']*100:>10.2f} | {r['max_dd']*100:>8.2f} | {r['trades']:>6} | {r['exhaust_count']:>4} | {r['exhaust_cycles']:>6}")
            print("-" * 125)
            print("说明: 总收益 = Alpha 套利 + Beta 持仓市值变动；流动仓年化按资金占用日折算。")
    else:
        run_backtest(buy_step_factor=BUY_STEP_FACTOR, sell_threshold_factor=SELL_THRESHOLD_FACTOR)
