"""
FastAPI 只读后端：展示实时数据与历史数据（K 线、历史信号）。
数据根目录：DASHBOARD_WORK_DIR 或项目根目录。
"""
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 数据根：与 mac_dashboard 一致
_DATA_ROOT = os.environ.get("DASHBOARD_WORK_DIR") or str(Path(__file__).resolve().parent.parent)

# ATR/趋势计算用常量（与 mac_dashboard 一致）
ATR_PERIOD = 14
TREND_MA_PERIOD = 60
ATR_GRID_FACTOR = 0.38
GRID_STEP_FLOOR = 0.0012
LAYER_STEP_BONUS = 0.0001
SELL_PROFIT_THRESHOLD = 0.005
SELL_THRESHOLD_FACTOR = 1.4
BUY_STEP_FACTOR = 1.0
UPTREND_GRID_FACTOR = 1.2
UPTREND_SELL_FACTOR = 1.33
UPTREND_BATCH_FACTOR = 0.7
DOWNTREND_GRID_FACTOR = 1.0
DOWNTREND_SELL_FACTOR = 0.83
DOWNTREND_BATCH_FACTOR = 1.2
ATR_CIRCUIT_BREAKER_ENABLED = True
ATR_CIRCUIT_BREAKER_RATIO = 2.0
ATR_LOOKBACK = 60
RSI_PERIOD = 14
PHYSICAL_POOL = 300_000
PART_MONEY = 200_000 / 9
# 行情过期阈值：与 mac_dashboard.DATA_STALE_SECONDS 一致，超过该秒数则视为暂停触发实盘信号
DATA_STALE_SECONDS = 300

app = FastAPI(title="miniqmt 数据展示")


# region agent log
DEBUG_LOG_PATH = "/Users/yuhao/Documents/miniqmt/.cursor/debug-fedda0.log"


def _debug_log(hypothesis_id: str, message: str, data: dict, location: str):
    try:
        entry = {
            "sessionId": "fedda0",
            "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": "initial",
            "hypothesisId": hypothesis_id,
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # 调试日志失败时静默忽略，避免影响主流程
        pass


# endregion


def _path(*parts: str) -> str:
    return os.path.join(_DATA_ROOT, *parts)


def _read_json(path: str, default=None):
    if default is None:
        default = {}
    p = _path(path)
    if not os.path.exists(p):
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _history_to_df(history):
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


def _calculate_atr_and_trend(df):
    if df is None or len(df) < ATR_PERIOD or len(df) < TREND_MA_PERIOD + 5:
        return None
    prev_close = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ),
    )
    df = df.copy()
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()
    df["atr14_avg"] = df["atr14"].rolling(ATR_LOOKBACK).mean().shift(1)
    df["ma60"] = df["close"].rolling(TREND_MA_PERIOD).mean()
    raw_slope = df["ma60"].diff(5) / df["ma60"].shift(5)
    df["ma60_slope"] = raw_slope.ewm(span=3, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(100)
    df = df.dropna()
    if len(df) == 0:
        return None
    last = df.iloc[-1]
    curr_p = float(last["close"])
    atr = float(last["atr14"])
    slope = float(last["ma60_slope"])
    atr_avg = last.get("atr14_avg", np.nan)
    rsi_now = float(last["rsi"]) if pd.notna(last.get("rsi")) else 50.0
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
        "atr14": float(atr),
        "ma60": float(last["ma60"]),
        "ma60_slope": float(slope),
        "grid_step": float(base_grid_step),
        "sell_threshold": float(sell_threshold),
        "sell_threshold_factor": SELL_THRESHOLD_FACTOR,
        "batch_factor": float(batch_factor),
        "trend": trend_label,
        "curr_p": float(curr_p),
        "rsi": float(rsi_now),
        "pause_buy": bool(pause_buy),
    }


def _symbol_quote_state_atr(symbol: str):
    if symbol == "159201":
        quote_path = "shared_quote_159201.json"
        state_path = "dashboard_state.json"
        name = "159201 自由现金流"
    elif symbol == "512890":
        quote_path = "shared_quote_512890.json"
        state_path = "dashboard_state_512890.json"
        name = "512890 红利低波"
    else:
        return None
    quote = _read_json(quote_path, {})
    state = _read_json(state_path, {})
    # 基于 shared_quote 文件修改时间判断行情是否过期
    quote_file = _path(quote_path)
    quote_mtime = os.path.getmtime(quote_file) if os.path.exists(quote_file) else None
    quote_stale = False
    if quote_mtime is not None:
        try:
            quote_stale = (time.time() - quote_mtime) > DATA_STALE_SECONDS
        except Exception:
            quote_stale = False
    history = quote.get("history", [])
    df = _history_to_df(history)
    atr_info = _calculate_atr_and_trend(df)
    if atr_info is not None:
        # 将行情过期标记下传给前端，便于展示“暂停触发信号”
        atr_info["data_stale"] = bool(quote_stale)
    last_buy_price = state.get("last_buy_price")
    hold_layers = state.get("hold_layers", 0)
    total_cost = state.get("total_cost", 0.0)
    hold_t0_volume = state.get("hold_t0_volume", 0)
    positions = state.get("positions", [])
    # 下一买/卖价与数量
    next_buy_price = None
    next_buy_shares = 0
    next_sell_price = None
    next_sell_shares = 0
    if atr_info and last_buy_price:
        pm = PART_MONEY
        bf = atr_info.get("batch_factor", 1.0)
        gs = atr_info.get("grid_step", 0)
        next_buy_price = last_buy_price * (1 - gs) if gs else None
        if next_buy_price and next_buy_price > 0:
            next_buy_shares = int(pm * bf / next_buy_price // 100) * 100
        sell_eff = atr_info.get("sell_threshold", SELL_PROFIT_THRESHOLD) * atr_info.get("sell_threshold_factor", 1.0)
        if positions:
            lot_sells = [(lot["buy_price"] * (1 + sell_eff), lot["shares"]) for lot in positions]
            next_sell_price, next_sell_shares = min(lot_sells, key=lambda x: x[0])
    return {
        "symbol": symbol,
        "name": name,
        "quote": quote,
        "state": state,
        "atr_info": atr_info,
        "quote_mtime": quote_mtime,
        "quote_stale": bool(quote_stale),
        "last_buy_price": last_buy_price,
        "hold_layers": hold_layers,
        "total_cost": total_cost,
        "hold_t0_volume": hold_t0_volume,
        "positions": positions,
        "next_buy_price": next_buy_price,
        "next_buy_shares": next_buy_shares,
        "next_sell_price": next_sell_price,
        "next_sell_shares": next_sell_shares,
    }


@app.get("/api/overview")
def api_overview():
    """汇总两标的的当前数据 + 资金池 + 最新信号与结果"""
    s159 = _symbol_quote_state_atr("159201")
    s512 = _symbol_quote_state_atr("512890")
    pool = _read_json("shared_pool.json", {})
    committed = float(pool.get("used_159201", 0) or 0) + float(pool.get("frozen_159201", 0) or 0)
    if "used_512890" in pool:
        committed += float(pool.get("used_512890", 0) or 0) + float(pool.get("frozen_512890", 0) or 0)
    signal = _read_json("order_signal.json", {})
    result = _read_json("order_result.json", {})
    for s in (s159, s512):
        if s and s.get("atr_info"):
            s["atr_info"] = {**s["atr_info"], "pool_committed": committed}
    # region agent log
    try:
        s159_q = (s159 or {}).get("quote") or {}
        s512_q = (s512 or {}).get("quote") or {}
        data = {
            "data_root": _DATA_ROOT,
            "quote_159201_time": s159_q.get("time"),
            "quote_159201_price": s159_q.get("price"),
            "quote_159201_history_len": len(s159_q.get("history") or []),
            "quote_159201_mtime": os.path.getmtime(_path("shared_quote_159201.json"))
            if os.path.exists(_path("shared_quote_159201.json"))
            else None,
            "quote_512890_time": s512_q.get("time"),
            "quote_512890_price": s512_q.get("price"),
            "quote_512890_history_len": len(s512_q.get("history") or []),
            "quote_512890_mtime": os.path.getmtime(_path("shared_quote_512890.json"))
            if os.path.exists(_path("shared_quote_512890.json"))
            else None,
        }
        _debug_log("H1_H4", "api_overview snapshot", data, "web/main.py:205")
    except Exception:
        # 调试日志失败时静默忽略
        pass
    # endregion
    return {
        "data_root": _DATA_ROOT,
        "symbols": {"159201": s159, "512890": s512},
        "pool": {**pool, "committed": committed},
        "order_signal": signal,
        "order_result": result,
    }


@app.get("/api/symbol/{symbol}")
def api_symbol(symbol: str):
    """单标的当前数据"""
    s = _symbol_quote_state_atr(symbol)
    if s is None:
        raise HTTPException(status_code=404, detail="symbol must be 159201 or 512890")
    pool = _read_json("shared_pool.json", {})
    committed = float(pool.get("used_159201", 0) or 0) + float(pool.get("frozen_159201", 0) or 0)
    if s.get("atr_info"):
        s["atr_info"] = {**s["atr_info"], "pool_committed": committed}
    return {"symbol": symbol, "data": s, "order_signal": _read_json("order_signal.json", {}), "order_result": _read_json("order_result.json", {})}


@app.get("/api/history/signals")
def api_history_signals(symbol: str | None = None):
    """历史信号列表，按时间倒序；可选 ?symbol=159201 或 512890 过滤"""
    ids_data = _read_json("executed_signals.json", {})
    ids = ids_data.get("signal_ids", [])
    if not isinstance(ids, list):
        ids = []
    done_dir = _path("order_signal_done")
    out = []
    for sid in reversed(ids):
        fn = os.path.join(done_dir, f"order_signal_{sid}.done")
        if not os.path.exists(fn):
            continue
        try:
            with open(fn, "r", encoding="utf-8") as f:
                sig = json.load(f)
        except Exception:
            continue
        code = (sig.get("code") or "").strip()
        if symbol:
            if symbol == "159201" and "159201" not in code:
                continue
            if symbol == "512890" and "512890" not in code:
                continue
        ts = sig.get("timestamp")
        if ts is not None:
            try:
                sig["_time_str"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                sig["_time_str"] = str(ts)
        out.append(sig)
    result = _read_json("order_result.json", {})
    return {"signals": out, "latest_result": result}


@app.get("/api/history/kline/{symbol}")
def api_history_kline(symbol: str, from_ts: str | None = None, to_ts: str | None = None):
    """历史 K 线，symbol=159201 或 512890；可选 from/to 截断"""
    if symbol == "159201":
        csv_path = _path("history_159201_1m.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="history_159201_1m.csv not found")
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if "close" not in df.columns:
            raise HTTPException(status_code=500, detail="CSV must have open/high/low/close")
        df = df.sort_index()
        times = df.index
    elif symbol == "512890":
        candidates = [
            "quote_512890_1min_5y.csv",
            "quote_512890_1min_3y.csv",
            "quote_512890_1min_2024.csv",
        ]
        csv_path = None
        for c in candidates:
            p = _path(c)
            if os.path.exists(p):
                csv_path = p
                break
        if not csv_path:
            raise HTTPException(status_code=404, detail="no 512890 quote CSV found")
        try:
            df = pd.read_csv(csv_path)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time")
                times = df["time"]
            else:
                df = df.set_index(pd.to_datetime(df.index)) if df.index.dtype != "datetime64[ns]" else df
                times = df.index
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if "close" not in df.columns:
            raise HTTPException(status_code=500, detail="CSV must have open/high/low/close")
    else:
        raise HTTPException(status_code=404, detail="symbol must be 159201 or 512890")
    rows = []
    for i in range(len(df)):
        t = times.iloc[i] if hasattr(times, "iloc") else times[i]
        if hasattr(t, "isoformat"):
            t_str = t.isoformat()
        else:
            t_str = str(t)
        row = {"time": t_str, "open": float(df["open"].iloc[i]), "high": float(df["high"].iloc[i]), "low": float(df["low"].iloc[i]), "close": float(df["close"].iloc[i])}
        if "volume" in df.columns:
            row["volume"] = int(df["volume"].iloc[i])
        rows.append(row)
    # 可选 from/to 过滤（按字符串比较或解析）
    if from_ts:
        rows = [r for r in rows if r["time"] >= from_ts]
    if to_ts:
        rows = [r for r in rows if r["time"] <= to_ts]
    return {"symbol": symbol, "kline": rows}


# 静态文件
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    if (STATIC_DIR / "index.html").exists():
        return FileResponse(STATIC_DIR / "index.html")
    return {"message": "miniqmt data API", "docs": "/docs"}
