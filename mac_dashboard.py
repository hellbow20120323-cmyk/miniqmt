"""
159201 自由现金流 ETF — 实盘信号看板
与 mac_backtest_159201.py 同一套 ATR 动态网格 + 趋势自适应逻辑，输出 BUY/SELL 到 order_signal.json 供 Windows 执行。
"""
import json
import time
import os
import uuid
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

# --- 路径：相对本脚本所在目录；离线测试时可设环境变量 DASHBOARD_WORK_DIR 指向 test_offline ---
_SCRIPT_DIR = os.environ.get("DASHBOARD_WORK_DIR") or os.path.dirname(os.path.abspath(__file__))
SHARED_FILE = os.path.join(_SCRIPT_DIR, 'shared_quote.json')
SHARED_POOL_FILE = os.path.join(_SCRIPT_DIR, 'shared_pool.json')  # 双标共享资金池，见 开发文档_双标的共享资金池.md
SIGNAL_FILE = os.path.join(_SCRIPT_DIR, 'order_signal.json')
ORDER_RESULT_FILE = os.path.join(_SCRIPT_DIR, 'order_result.json')
STATE_FILE = os.path.join(_SCRIPT_DIR, 'dashboard_state.json')

# 物理池与迟滞（与 global_vault / 开发文档一致）
PHYSICAL_POOL = 300_000
POOL_90_PCT = 270_000   # 占用 > 90% 触发步长惩罚
POOL_85_PCT = 255_000   # 占用 < 85% 解除惩罚；>85% 禁止新开第一层
_step_penalty_active = False

# --- 部分成交：单边下跌补买 15 分钟超时，单边上涨只对齐不补卖 ---
PENDING_BUY_TIMEOUT_SEC = 900

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

# --- 静默期（仅内存）：发信号后锁定该层/该笔，直到真实持仓更新或超时，避免延迟期内重复发单 ---
PENDING_TIMEOUT_SEC = 120
pending_until_layers = None
pending_since = None
pending_sell_since = None
pending_sell_volume = 0

def _load_state():
    default = {
        "last_buy_price": None,
        "hold_layers": 0,
        "total_cost": 0.0,
        "hold_t0_volume": 0,
        "last_sell_timestamp": None,
        "positions": [],
        "last_sent_signal_id": None,
        "last_sent_signal_direction": None,
        "last_sent_signal_shares": None,
        "last_sent_signal_price": None,
        "last_sent_buy_prev_anchor": None,
        "last_sent_sell_removed_lots": [],
        "last_sent_was_topup": False,
        "last_applied_result_signal_id": None,
        "pending_buy_shares": 0,
        "pending_buy_price": None,
        "pending_buy_since": None,
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
            "last_sent_signal_id": s.get("last_sent_signal_id"),
            "last_sent_signal_direction": s.get("last_sent_signal_direction"),
            "last_sent_signal_shares": s.get("last_sent_signal_shares"),
            "last_sent_signal_price": s.get("last_sent_signal_price"),
            "last_sent_buy_prev_anchor": s.get("last_sent_buy_prev_anchor"),
            "last_sent_sell_removed_lots": s.get("last_sent_sell_removed_lots", []),
            "last_sent_was_topup": s.get("last_sent_was_topup", False),
            "last_applied_result_signal_id": s.get("last_applied_result_signal_id"),
            "pending_buy_shares": s.get("pending_buy_shares", 0) or 0,
            "pending_buy_price": s.get("pending_buy_price"),
            "pending_buy_since": s.get("pending_buy_since"),
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


def _load_shared_pool():
    """读取 shared_pool.json，返回 committed 总额与各标的 used/frozen/acc_alpha；无文件或异常返回空结构。"""
    out = {
        "used_159201": 0.0, "used_512890": 0.0,
        "frozen_159201": 0.0, "frozen_512890": 0.0,
        "committed": 0.0,
        "acc_alpha_159201": 0.0, "acc_alpha_512890": 0.0,
        "updated_at": None,
    }
    if not os.path.exists(SHARED_POOL_FILE):
        return out
    try:
        with open(SHARED_POOL_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k in out:
            if k in data and data[k] is not None:
                if k == "updated_at":
                    out[k] = data[k]
                else:
                    out[k] = float(data[k])
        out["committed"] = out["used_159201"] + out["used_512890"] + out["frozen_159201"] + out["frozen_512890"]
    except Exception:
        pass
    return out


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

# --- 信号输出：先持久化状态再写信号（避免崩溃后重复发单）；原子写入信号文件；双标时带 client_order_id/amount/layer_index ---
def execute_signal(direction, price, reason, shares=None, is_topup=False, amount=None, layer_index=None):
    msg = f"检测到{direction}信号 | 价格:{price:.3f} | 原因:{reason}"
    state["signals"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if len(state["signals"]) > 8:
        state["signals"] = state["signals"][-8:]

    voice_msg = "买入159201" if direction == "BUY" else "卖出159201"
    os.system(f'say "{voice_msg}" &')

    # 可读唯一标识：BUY_159201.SZ_L1_abc12def / SELL_159201.SZ_abc12def
    short_id = uuid.uuid4().hex[:8]
    if direction == "BUY":
        layer = state.get("hold_layers", 0)
        signal_id = f"BUY_{STOCK_CODE}_L{layer}_{short_id}"
    else:
        signal_id = f"SELL_{STOCK_CODE}_{short_id}"

    # 双标共享资金池：Executor 与 GlobalVault 需要 client_order_id / amount / layer_index
    if amount is None and direction == "BUY" and shares and price:
        amount = round(price * shares, 2)
    if layer_index is None and direction == "BUY":
        layer_index = state.get("hold_layers", 0)

    signal_data = {
        "signal_id": signal_id,
        "client_order_id": signal_id,
        "code": STOCK_CODE,
        "direction": direction,
        "price": price,
        "shares": shares,
        "timestamp": time.time(),
        "reason": reason,
    }
    if amount is not None:
        signal_data["amount"] = amount
    if layer_index is not None:
        signal_data["layer_index"] = int(layer_index)
    state["last_sent_signal_id"] = signal_data["signal_id"]
    state["last_sent_signal_direction"] = direction
    state["last_sent_signal_shares"] = shares
    state["last_sent_signal_price"] = price
    state["last_sent_was_topup"] = is_topup
    _write_signal_atomic(signal_data)
    _save_state(state)


def _apply_order_result(result):
    """根据 order_result 做部分成交对齐：跌时改仓位+设补单，涨时把未卖部分写回 positions。"""
    if not isinstance(result, dict):
        return
    sid = result.get("signal_id")
    if not sid or sid != state.get("last_sent_signal_id") or sid == state.get("last_applied_result_signal_id"):
        return
    requested = int(result.get("requested_shares") or 0)
    filled = int(result.get("filled_shares", requested) or requested)
    direction = (result.get("direction") or "").upper()
    price = float(result.get("price") or result.get("last_sent_signal_price") or 0)
    if price <= 0:
        price = state.get("last_sent_signal_price") or 0

    state["last_applied_result_signal_id"] = sid

    if filled >= requested:
        _save_state(state)
        return

    if direction == "BUY":
        is_topup = state.get("last_sent_was_topup", False)
        positions = list(state.get("positions", []))
        if is_topup:
            if filled > 0:
                positions.append({"shares": filled, "cost": filled * price, "buy_price": price})
            remaining = requested - filled
            if remaining > 0:
                state["pending_buy_shares"] = remaining
                state["pending_buy_price"] = price
                state["pending_buy_since"] = time.time()
            else:
                state["pending_buy_shares"] = 0
                state["pending_buy_price"] = None
                state["pending_buy_since"] = None
        else:
            if not positions:
                state["last_buy_price"] = state.get("last_sent_buy_prev_anchor") or state.get("last_buy_price")
            else:
                if filled > 0:
                    positions[-1] = {"shares": filled, "cost": filled * price, "buy_price": price}
                else:
                    positions.pop()
                    state["last_buy_price"] = state.get("last_sent_buy_prev_anchor") or state.get("last_buy_price")
                remaining = requested - filled
                if remaining > 0:
                    state["pending_buy_shares"] = remaining
                    state["pending_buy_price"] = price
                    state["pending_buy_since"] = time.time()
                else:
                    state["pending_buy_shares"] = 0
                    state["pending_buy_price"] = None
                    state["pending_buy_since"] = None

        state["positions"] = positions
        state["hold_layers"] = len(positions)
        state["hold_t0_volume"] = sum(p["shares"] for p in positions)
        state["total_cost"] = sum(p["cost"] for p in positions)

    elif direction == "SELL":
        removed = state.get("last_sent_sell_removed_lots", [])
        if not removed:
            _save_state(state)
            return
        total_removed = sum(lot["shares"] for lot in removed)
        total_cost_removed = sum(lot["cost"] for lot in removed)
        remaining = requested - filled
        if remaining <= 0:
            _save_state(state)
            return
        avg_price = total_cost_removed / total_removed if total_removed else price
        back_cost = total_cost_removed * (remaining / total_removed) if total_removed else remaining * price
        positions = state.get("positions", [])
        positions.append({"shares": remaining, "cost": back_cost, "buy_price": avg_price})
        state["positions"] = positions
        state["hold_layers"] = len(positions)
        state["hold_t0_volume"] = sum(p["shares"] for p in positions)
        state["total_cost"] = sum(p["cost"] for p in positions)
        state["pending_sell_since"] = None
        state["pending_sell_volume"] = 0

    _save_state(state)


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
        # 双标共享池：占用与两路 Alpha
        if "pool_committed" in atr_info:
            c = atr_info["pool_committed"]
            pct = (c / PHYSICAL_POOL * 100) if PHYSICAL_POOL else 0
            penalty = "步长×1.5" if atr_info.get("pool_penalty") else "正常"
            table.add_row("共享池占用", f"{c:,.0f} ({pct:.1f}%)", penalty)
            table.add_row("Alpha 159201 / 512890", f"{atr_info.get('pool_acc_alpha_159201', 0):.2f} / {atr_info.get('pool_acc_alpha_512890', 0):.2f}", "两路累计")
    else:
        table.add_row("当前价格", f"{curr_p:.3f}", f"🕒 {data.get('time', '')}" if data else "—")
        table.add_row("指标", "—", "需至少约 65 根 K 线才能计算 ATR/趋势")
        if hold_t0_volume or hold_layers:
            table.add_row("T+0 层数 / 持仓", f"{hold_layers} 层 / {hold_t0_volume} 股", f"成本 {total_cost:,.0f}")

    return table

# --- 主循环：读行情 → 算 ATR/趋势 → 与回测一致的买卖逻辑 → 更新状态与 UI ---
def main():
    global state, pending_until_layers, pending_since, pending_sell_since, pending_sell_volume
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

                    # 部分成交：读 order_result 做状态对齐（跌：改仓位+设补单；涨：未卖部分写回 positions）
                    if os.path.exists(ORDER_RESULT_FILE):
                        try:
                            with open(ORDER_RESULT_FILE, 'r') as f:
                                _apply_order_result(json.load(f))
                        except Exception:
                            pass
                    if pending_sell_since is None and (state.get("pending_sell_since") is not None or state.get("pending_sell_volume")):
                        pending_sell_since = state.get("pending_sell_since")
                        pending_sell_volume = state.get("pending_sell_volume") or 0
                    elif state.get("pending_sell_since") is None:
                        pending_sell_since = None
                        pending_sell_volume = 0

                    # 从 state 刷新（apply 可能已改 positions）
                    positions = state.get("positions", [])
                    hold_layers = len(positions)
                    hold_t0_volume = sum(p["shares"] for p in positions)
                    total_cost = sum(p["cost"] for p in positions)

                    # 补单超时放弃（15 分钟）
                    if state.get("pending_buy_shares") and state.get("pending_buy_since"):
                        if (time.time() - state["pending_buy_since"]) > PENDING_BUY_TIMEOUT_SEC:
                            state["pending_buy_shares"] = 0
                            state["pending_buy_price"] = None
                            state["pending_buy_since"] = None
                            _save_state(state)

                    # 与真实持仓同步：若桥显示 0 持仓而本地认为有仓，说明已在外盘平仓/重启后不一致，以真实为准避免重复卖
                    real_volume = (data.get("position") or {}).get("volume", 0) or 0
                    curr_p_for_layers = curr_p if curr_p and curr_p > 0 else 0
                    real_layers = int(real_volume * curr_p_for_layers / PART_MONEY) if curr_p_for_layers > 0 else 0

                    # 静默期解除：真实持仓已跟上或超时则清除
                    if pending_until_layers is not None and (
                        real_layers >= pending_until_layers or (time.time() - (pending_since or 0)) > PENDING_TIMEOUT_SEC
                    ):
                        pending_until_layers = None
                        pending_since = None
                    if pending_sell_since is not None and (
                        real_volume < pending_sell_volume or (time.time() - pending_sell_since) > PENDING_TIMEOUT_SEC
                    ):
                        pending_sell_since = None
                        pending_sell_volume = 0

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
                        # 双标共享池：占用 >90% 步长×1.5，<85% 解除（迟滞）
                        pool_data = _load_shared_pool()
                        committed = pool_data.get("committed", 0.0)
                        global _step_penalty_active
                        if committed > POOL_90_PCT:
                            _step_penalty_active = True
                        elif committed < POOL_85_PCT:
                            _step_penalty_active = False
                        if _step_penalty_active:
                            grid_step *= 1.5
                        atr_info['grid_step'] = grid_step
                        atr_info['pool_committed'] = committed
                        atr_info['pool_penalty'] = _step_penalty_active
                        atr_info['pool_used_159201'] = pool_data.get("used_159201", 0)
                        atr_info['pool_acc_alpha_159201'] = pool_data.get("acc_alpha_159201", 0)
                        atr_info['pool_acc_alpha_512890'] = pool_data.get("acc_alpha_512890", 0)
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
                            # 补单（单边下跌部分成交）：未消费信号时才发，价格回到补单价以下且 15 分钟内
                            pending_shares = state.get("pending_buy_shares") or 0
                            pending_price = state.get("pending_buy_price")
                            pending_ts = state.get("pending_buy_since")
                            if (pending_shares > 0 and pending_price is not None and pending_ts is not None
                                and (time.time() - pending_ts) <= PENDING_BUY_TIMEOUT_SEC
                                and curr_p <= pending_price and not os.path.exists(SIGNAL_FILE)):
                                execute_signal(
                                    "BUY", curr_p, "补单(部分成交回补)", shares=pending_shares, is_topup=True,
                                    amount=round(pending_price * pending_shares, 2), layer_index=hold_layers,
                                )
                                state["pending_buy_shares"] = 0
                                state["pending_buy_price"] = None
                                state["pending_buy_since"] = None
                                _save_state(state)
                            else:
                                # 卖出：先持久化状态再发信号，崩溃恢复后不会重复卖；静默期内若真实持仓未下降则不再发卖单
                                if positions:
                                    to_remove = []
                                    sell_shares_total = 0
                                    for idx, lot in enumerate(positions):
                                        if curr_p >= lot["buy_price"] * (1 + sell_eff):
                                            to_remove.append(idx)
                                            sell_shares_total += lot["shares"]
                                    if sell_shares_total > 0:
                                        in_sell_cooldown = (
                                            pending_sell_since is not None
                                            and (time.time() - pending_sell_since) <= PENDING_TIMEOUT_SEC
                                            and real_volume >= pending_sell_volume
                                        )
                                        if not in_sell_cooldown:
                                            state["last_sent_sell_removed_lots"] = [
                                                {"shares": positions[i]["shares"], "cost": positions[i]["cost"], "buy_price": positions[i]["buy_price"]}
                                                for i in to_remove
                                            ]
                                            for idx in reversed(to_remove):
                                                positions.pop(idx)
                                            state["positions"] = positions
                                            state["hold_layers"] = len(positions)
                                            state["hold_t0_volume"] = sum(p["shares"] for p in positions)
                                            state["total_cost"] = sum(p["cost"] for p in positions)
                                            state["last_buy_price"] = curr_p
                                            if not positions:
                                                state["last_sell_timestamp"] = time.time()
                                            _save_state(state)
                                            execute_signal("SELL", curr_p, f"单笔止盈(涨幅≥{sell_eff*100:.2f}%)", shares=sell_shares_total)
                                            pending_sell_since = time.time()
                                            pending_sell_volume = hold_t0_volume

                                # 买入：静默期内若真实持仓未达到目标层数则不再发买单；双标时新开第一层需物理池剩余≥15%
                                elif hold_layers < MAX_LAYERS and curr_p <= last_buy_price * (1 - grid_step * BUY_STEP_FACTOR) and not in_cooling and not pause_buy:
                                    pool_block_first_layer = (hold_layers == 0 and committed > POOL_85_PCT)
                                    in_buy_cooldown = (
                                        pending_until_layers is not None
                                        and (time.time() - (pending_since or 0)) <= PENDING_TIMEOUT_SEC
                                        and real_layers < pending_until_layers
                                    )
                                    if not in_buy_cooldown and not pool_block_first_layer:
                                        money = PART_MONEY * batch_factor
                                        shares = int(money / curr_p // 100) * 100
                                        if shares > 0:
                                            state["last_sent_buy_prev_anchor"] = last_buy_price
                                            state["last_buy_price"] = curr_p
                                            state["positions"] = positions + [{"shares": shares, "cost": shares * curr_p, "buy_price": curr_p}]
                                            state["hold_layers"] = hold_layers + 1
                                            state["hold_t0_volume"] = hold_t0_volume + shares
                                            state["total_cost"] = total_cost + shares * curr_p
                                            _save_state(state)
                                            execute_signal(
                                                "BUY", curr_p, f"ATR网格触发(步长{grid_step*100:.2f}%)", shares=shares,
                                                amount=round(money, 2), layer_index=hold_layers,
                                            )
                                            pending_until_layers = hold_layers + 1
                                            pending_since = time.time()

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
