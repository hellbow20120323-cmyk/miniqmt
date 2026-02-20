"""
159201 自由现金流 ETF — 实盘信号看板
与 mac_backtest_159201.py 同一套 ATR 动态网格 + 趋势自适应逻辑，输出 BUY/SELL 到 order_signal.json 供 Windows 执行。
"""
import json
import time
import os
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

# --- 路径：相对本脚本所在目录 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_FILE = os.path.join(_SCRIPT_DIR, 'shared_quote.json')
SIGNAL_FILE = os.path.join(_SCRIPT_DIR, 'order_signal.json')
STATE_FILE = os.path.join(_SCRIPT_DIR, 'dashboard_state.json')

# --- 标的 ---
STOCK_CODE = '159201.SZ'

# --- 与回测一致：30万固定｜20万流动，均权9层，统一止盈 0.5%×涨多1.4，动态冷静期，ATR 熔断 ---
ATR_PERIOD = 14
ATR_GRID_FACTOR = 0.38
GRID_STEP_FLOOR = 0.0012
LAYER_STEP_BONUS = 0.0001
SELL_PROFIT_THRESHOLD = 0.005
SELL_THRESHOLD_FACTOR = 1.4      # 涨多系数，与回测一致
BUY_STEP_FACTOR = 1.0
TREND_MA_PERIOD = 60
MAX_LAYERS = 9
BATCH_MONEY = 200000             # 流动仓 20 万（固定仓 30 万仅回测/持仓用）
COOLING_BARS = 15                # 默认冷静期 15 分钟
COOLING_BARS_SHORT = 5           # RSI<35 时缩短为 5 分钟
RSI_COOLING_THRESHOLD = 35
RSI_PERIOD = 14
ATR_CIRCUIT_BREAKER_ENABLED = True
ATR_CIRCUIT_BREAKER_RATIO = 2.0
ATR_LOOKBACK = 60
UPTREND_GRID_FACTOR = 1.2
UPTREND_SELL_FACTOR = 1.33
UPTREND_BATCH_FACTOR = 0.7
DOWNTREND_GRID_FACTOR = 1.0
DOWNTREND_SELL_FACTOR = 0.83
DOWNTREND_BATCH_FACTOR = 1.2

PART_MONEY = BATCH_MONEY / MAX_LAYERS

# --- 容错与恢复：数据过期不交易、状态先于信号持久化、原子写入、与真实持仓同步 ---
DATA_STALE_SECONDS = 300          # 行情超过 5 分钟未更新则不再发出新信号（网络/桥中断时避免误判）
STATE_TMP = STATE_FILE + ".tmp"
SIGNAL_TMP = SIGNAL_FILE + ".tmp"

def _load_state():
    default = {
        "last_buy_price": None,
        "hold_layers": 0,
        "total_cost": 0.0,
        "hold_t0_volume": 0,
        "last_sell_timestamp": None,
        "positions": [],
    }
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, 'r') as f:
            s = json.load(f)
    except Exception:
        return default
    # 校验 positions 结构，防止损坏或旧格式导致逻辑错误
    positions = s.get("positions", [])
    if not isinstance(positions, list):
        positions = []
    else:
        valid = []
        for p in positions:
            if isinstance(p, dict) and "shares" in p and "cost" in p and "buy_price" in p:
                sh, co, bp = p["shares"], p["cost"], p["buy_price"]
                if isinstance(sh, (int, float)) and isinstance(co, (int, float)) and isinstance(bp, (int, float)):
                    if sh > 0 and co >= 0 and bp > 0:
                        valid.append({"shares": int(sh), "cost": float(co), "buy_price": float(bp)})
        positions = valid
    s["positions"] = positions
    s["hold_layers"] = len(positions)
    s["hold_t0_volume"] = sum(p["shares"] for p in positions)
    s["total_cost"] = sum(p["cost"] for p in positions)
    return s

def _save_state(s):
    try:
        persist = {
            "last_buy_price": s.get("last_buy_price"),
            "hold_layers": s.get("hold_layers", 0),
            "total_cost": s.get("total_cost", 0.0),
            "hold_t0_volume": s.get("hold_t0_volume", 0),
            "last_sell_timestamp": s.get("last_sell_timestamp"),
            "positions": s.get("positions", []),
        }
        with open(STATE_TMP, 'w') as f:
            json.dump(persist, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(STATE_TMP, STATE_FILE)
    except Exception:
        pass

def _write_signal_atomic(signal_data):
    try:
        with open(SIGNAL_TMP, 'w') as f:
            json.dump(signal_data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(SIGNAL_TMP, SIGNAL_FILE)
    except Exception:
        pass

state = _load_state()
state["signals"] = state.get("signals", [])
state["status"] = "等待数据..."

console = Console()

# --- 将 shared 的 history 转为带 OHLC 的 DataFrame（至少需 high/low/close 以算 ATR） ---
def history_to_df(history):
    if not history or len(history) < TREND_MA_PERIOD + 5:
        return None
    rows = []
    for h in history:
        if isinstance(h, (int, float)):
            rows.append({"open": h, "high": h, "low": h, "close": h})
        elif isinstance(h, dict):
            rows.append({
                "open": h.get("open", h.get("close")),
                "high": h.get("high", h.get("close")),
                "low": h.get("low", h.get("close")),
                "close": h.get("close", 0),
            })
        else:
            return None
    df = pd.DataFrame(rows)
    if df.empty or "close" not in df.columns:
        return None
    df["open"] = df.get("open", df["close"])
    df["high"] = df.get("high", df["close"])
    df["low"] = df.get("low", df["close"])
    return df

# --- ATR + 趋势 + 动态步长/止盈/仓位系数（与回测同一套公式） ---
def calculate_atr_and_trend(df):
    if df is None or len(df) < ATR_PERIOD or len(df) < TREND_MA_PERIOD + 5:
        return None
    prev_close = df['close'].shift(1)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    df = df.copy()
    df['atr14'] = tr.rolling(ATR_PERIOD).mean()
    df['atr14_avg'] = df['atr14'].rolling(ATR_LOOKBACK).mean().shift(1)
    df['ma60'] = df['close'].rolling(TREND_MA_PERIOD).mean()
    raw_slope = df['ma60'].diff(5) / df['ma60'].shift(5)
    df['ma60_slope'] = raw_slope.ewm(span=3, adjust=False).mean()
    # RSI（用于动态冷静期）
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(100)
    df = df.dropna()
    if len(df) == 0:
        return None
    last = df.iloc[-1]
    curr_p = float(last['close'])
    atr = float(last['atr14'])
    slope = float(last['ma60_slope'])
    atr_avg = last.get('atr14_avg', np.nan)
    rsi_now = float(last['rsi']) if pd.notna(last.get('rsi')) else 50.0
    pause_buy = ATR_CIRCUIT_BREAKER_ENABLED and (pd.notna(atr_avg) and atr > atr_avg * ATR_CIRCUIT_BREAKER_RATIO)

    base_grid_step = max(GRID_STEP_FLOOR, (atr / curr_p) * ATR_GRID_FACTOR)
    sell_threshold = SELL_PROFIT_THRESHOLD
    batch_factor = 1.0
    trend_label = "中性"

    if slope > 0:
        base_grid_step *= UPTREND_GRID_FACTOR
        sell_threshold *= UPTREND_SELL_FACTOR
        batch_factor = UPTREND_BATCH_FACTOR
        trend_label = "上升"
    elif slope < 0:
        base_grid_step *= DOWNTREND_GRID_FACTOR
        sell_threshold *= DOWNTREND_SELL_FACTOR
        batch_factor = DOWNTREND_BATCH_FACTOR
        trend_label = "下降"

    return {
        "atr14": atr,
        "ma60": float(last['ma60']),
        "ma60_slope": slope,
        "base_grid_step": base_grid_step,
        "sell_threshold": sell_threshold,
        "batch_factor": batch_factor,
        "trend": trend_label,
        "curr_p": curr_p,
        "rsi": rsi_now,
        "pause_buy": pause_buy,
    }

# --- 信号输出：先持久化状态再写信号（避免崩溃后重复发单）；原子写入信号文件 ---
def execute_signal(direction, price, reason, shares=None):
    msg = f"检测到{direction}信号 | 价格:{price:.3f} | 原因:{reason}"
    state["signals"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if len(state["signals"]) > 8:
        state["signals"] = state["signals"][-8:]

    voice_msg = "买入159201" if direction == "BUY" else "卖出159201"
    os.system(f'say "{voice_msg}" &')

    signal_data = {
        "code": STOCK_CODE,
        "direction": direction,
        "price": price,
        "shares": shares,
        "timestamp": time.time(),
    }
    _write_signal_atomic(signal_data)

# --- 仪表盘布局 ---
def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=14),
        Layout(name="footer", size=10)
    )
    return layout

def generate_display(data, atr_info, last_buy_price, hold_layers, total_cost, hold_t0_volume):
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("项目", style="dim")
    table.add_column("数值", justify="right")
    table.add_column("说明", justify="center")

    curr_p = data.get('price', 0) if data else 0
    pos = data.get('position', {}) if data else {}

    if atr_info:
        next_buy = last_buy_price * (1 - atr_info['grid_step']) if last_buy_price else 0
        sell_eff = atr_info.get("sell_eff", atr_info["sell_threshold"])

        table.add_row("当前价格", f"{curr_p:.3f}", f"🕒 {data.get('time', '')}")
        table.add_row("ATR(14)", f"{atr_info['atr14']:.4f}", "波动率")
        table.add_row("动态网格步长", f"{atr_info['grid_step']*100:.2f}%", f"单笔止盈 ≥买入价×(1+{sell_eff*100:.2f}%)")
        table.add_row("趋势 (MA60斜率)", atr_info['trend'], f"batch×{atr_info['batch_factor']:.2f}")
        table.add_row("下一买点 (≤)", f"{next_buy:.3f}" if last_buy_price else "—", "跌破即触发买入")
        table.add_row("单笔止盈", f"≥买入价×{1+sell_eff:.4f}", "每笔达则卖该笔")
        table.add_row("T+0 层数 / 持仓", f"{hold_layers} 层 / {hold_t0_volume} 股", f"成本 {total_cost:,.0f}")
        table.add_row("真实持仓(桥)", f"{pos.get('volume', 0)} 股", f"可用: {pos.get('can_use_volume', 0)}")
    else:
        table.add_row("当前价格", f"{curr_p:.3f}", f"🕒 {data.get('time', '')}" if data else "—")
        table.add_row("指标", "—", "需至少约 65 根 K 线才能计算 ATR/趋势")
        if hold_t0_volume or hold_layers:
            table.add_row("T+0 层数 / 持仓", f"{hold_layers} 层 / {hold_t0_volume} 股", f"成本 {total_cost:,.0f}")

    return table

# --- 主循环：读行情 → 算 ATR/趋势 → 与回测一致的买卖逻辑 → 更新状态与 UI ---
def main():
    global state
    layout = make_layout()
    with Live(layout, refresh_per_second=1, screen=True) as live:
        while True:
            if os.path.exists(SHARED_FILE):
                try:
                    with open(SHARED_FILE, 'r') as f:
                        data = json.load(f)

                    history = data.get('history', [])
                    df = history_to_df(history)
                    atr_info = calculate_atr_and_trend(df)
                    curr_p = data.get('price')
                    if curr_p is None and atr_info:
                        curr_p = atr_info['curr_p']
                    if curr_p is None:
                        curr_p = 0

                    last_buy_price = state.get("last_buy_price")
                    hold_layers = state.get("hold_layers", 0)
                    total_cost = state.get("total_cost", 0.0)
                    hold_t0_volume = state.get("hold_t0_volume", 0)

                    # 行情新鲜度：过期则不发出新信号，避免网络/桥中断时用旧数据误触发
                    data_ok = True
                    try:
                        if os.path.exists(SHARED_FILE):
                            age = time.time() - os.path.getmtime(SHARED_FILE)
                            if age > DATA_STALE_SECONDS:
                                data_ok = False
                    except Exception:
                        data_ok = False

                    if last_buy_price is None and curr_p > 0:
                        last_buy_price = curr_p
                        state["last_buy_price"] = last_buy_price
                        _save_state(state)

                    # 兼容旧状态：无 positions 时用当前持仓合成一笔，并立即持久化便于恢复
                    positions = state.get("positions", [])
                    if not positions and hold_t0_volume > 0 and total_cost > 0:
                        avg = total_cost / hold_t0_volume
                        positions = [{"shares": hold_t0_volume, "cost": total_cost, "buy_price": avg}]
                        state["positions"] = positions
                        state["hold_layers"] = len(positions)
                        state["hold_t0_volume"] = hold_t0_volume
                        state["total_cost"] = total_cost
                        _save_state(state)
                    hold_layers = len(positions)
                    hold_t0_volume = sum(p["shares"] for p in positions)
                    total_cost = sum(p["cost"] for p in positions)
                    state["hold_layers"] = hold_layers
                    state["hold_t0_volume"] = hold_t0_volume
                    state["total_cost"] = total_cost

                    # 与真实持仓同步：若桥显示 0 持仓而本地认为有仓，说明已在外盘平仓/重启后不一致，以真实为准避免重复卖
                    real_volume = (data.get("position") or {}).get("volume", 0) or 0
                    if real_volume == 0 and hold_t0_volume > 0:
                        state["positions"] = []
                        state["hold_layers"] = 0
                        state["hold_t0_volume"] = 0
                        state["total_cost"] = 0.0
                        state["last_sell_timestamp"] = time.time()
                        positions = []
                        hold_layers = 0
                        hold_t0_volume = 0
                        total_cost = 0.0
                        _save_state(state)

                    if atr_info:
                        base_grid_step = atr_info['base_grid_step']
                        grid_step = base_grid_step + (hold_layers * LAYER_STEP_BONUS)
                        atr_info['grid_step'] = grid_step
                        sell_threshold = atr_info['sell_threshold']
                        batch_factor = atr_info['batch_factor']
                        sell_eff = sell_threshold * SELL_THRESHOLD_FACTOR
                        atr_info['sell_eff'] = sell_eff
                        atr_info['data_stale'] = not data_ok

                        rsi_now = atr_info.get('rsi', 50)
                        cooling_minutes = COOLING_BARS_SHORT if rsi_now < RSI_COOLING_THRESHOLD else COOLING_BARS
                        last_sell_ts = state.get("last_sell_timestamp")
                        in_cooling = (
                            hold_layers == 0
                            and last_sell_ts is not None
                            and (time.time() - last_sell_ts) < cooling_minutes * 60
                        )
                        pause_buy = atr_info.get("pause_buy", False)

                        # 仅当行情新鲜时才发出买卖信号，避免网络中断后误触发
                        if data_ok:
                            # 卖出：先持久化状态再发信号，崩溃恢复后不会重复卖
                            if positions:
                                to_remove = []
                                sell_shares_total = 0
                                for idx, lot in enumerate(positions):
                                    if curr_p >= lot["buy_price"] * (1 + sell_eff):
                                        to_remove.append(idx)
                                        sell_shares_total += lot["shares"]
                                for idx in reversed(to_remove):
                                    positions.pop(idx)
                                if sell_shares_total > 0:
                                    state["positions"] = positions
                                    state["hold_layers"] = len(positions)
                                    state["hold_t0_volume"] = sum(p["shares"] for p in positions)
                                    state["total_cost"] = sum(p["cost"] for p in positions)
                                    state["last_buy_price"] = curr_p
                                    if not positions:
                                        state["last_sell_timestamp"] = time.time()
                                    _save_state(state)
                                    execute_signal("SELL", curr_p, f"单笔止盈(涨幅≥{sell_eff*100:.2f}%)", shares=sell_shares_total)

                            # 买入：先持久化状态再发信号
                            elif hold_layers < MAX_LAYERS and curr_p <= last_buy_price * (1 - grid_step * BUY_STEP_FACTOR) and not in_cooling and not pause_buy:
                                money = PART_MONEY * batch_factor
                                shares = int(money / curr_p // 100) * 100
                                if shares > 0:
                                    state["last_buy_price"] = curr_p
                                    state["positions"] = positions + [{"shares": shares, "cost": shares * curr_p, "buy_price": curr_p}]
                                    state["hold_layers"] = hold_layers + 1
                                    state["hold_t0_volume"] = hold_t0_volume + shares
                                    state["total_cost"] = total_cost + shares * curr_p
                                    _save_state(state)
                                    execute_signal("BUY", curr_p, f"ATR网格触发(步长{grid_step*100:.2f}%)", shares=shares)

                    header_msg = "💎 159201 自由现金流 ETF | ATR 动态网格 + 趋势自适应 | 实盘信号"
                    if atr_info and atr_info.get("data_stale"):
                        header_msg += " | ⚠️ 行情已过期，暂停发单"
                    layout["header"].update(Panel(header_msg, style="bold green"))
                    layout["main"].update(
                        generate_display(
                            data, atr_info,
                            state.get("last_buy_price"),
                            state.get("hold_layers", 0),
                            state.get("total_cost", 0.0),
                            state.get("hold_t0_volume", 0),
                        )
                    )
                    layout["footer"].update(
                        Panel("\n".join(state["signals"]) or "暂无信号", title="📜 最近信号", border_style="yellow")
                    )

                except Exception as e:
                    layout["header"].update(Panel(f"❌ 错误: {str(e)}", style="bold red"))
            else:
                layout["header"].update(
                    Panel("⏳ 等待 shared_quote.json… 请由桥接或本地写入行情", style="bold yellow")
                )

            time.sleep(1)

if __name__ == "__main__":
    main()
