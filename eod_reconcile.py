#!/usr/bin/env python3
"""
收盘对账小工具（Windows 端运行）：

- 从 QMT 读取当日成交（159201.SZ、512890.SH）；
- 从共享目录读取 Mac 端发单记录（order_signal_done/*.done + executed_signals.json）；
- 按 code+方向 在时间上匹配成交与信号，计算每笔的实际成交价 vs 理论发单价的滑点；
- 汇总输出今日滑点损耗（视为对 Alpha 的负贡献）和简单统计。

用法（在 Windows 上）：
    cd C:\\Mac\\Home\\Documents\\miniqmt
    python eod_reconcile.py

如需指定对账日期（非当天），可用：
    python eod_reconcile.py 20260226

说明：
- 依赖与 order_executor / bridge_producer 相同的 QMT & xtquant 环境；
- 仅使用 XtQuantTrader.query_stock_trades 读取当日成交；
- 信号来自 order_signal_done/order_signal_<signal_id>.done，范围限定为当日 timestamp。
"""
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from typing import Dict, List, Optional

from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
from xtquant import xtconstant

# --- 配置（保持与现有脚本一致）---
SHARED_DIR = r"C:\Mac\Home\Documents\miniqmt"
QMT_USERDATA_PATH = r"C:\国金证券QMT交易端\userdata_mini"
ACCOUNT_ID = "8883921646"
ACCOUNT_TYPE = "STOCK"
CODES = ["159201.SZ", "512890.SH"]
SYMBOL_NAME = {"159201.SZ": "159201 自由现金流", "512890.SH": "512890 红利低波"}

# 信号与成交的时间匹配窗口（秒）
MATCH_WINDOW_SEC = 120


def _ensure_shared_dir() -> str:
    if not os.path.exists(SHARED_DIR):
        raise SystemExit(f"共享目录 SHARED_DIR 不存在: {SHARED_DIR}")
    return SHARED_DIR


def _direction_from_trade(t) -> str:
    """将 xtquant 成交对象映射为 'BUY' / 'SELL'."""
    order_type = getattr(t, "order_type", None)
    if order_type == xtconstant.STOCK_BUY:
        return "BUY"
    if order_type == xtconstant.STOCK_SELL:
        return "SELL"
    # 兜底用 offset_flag 判断
    offset_flag = getattr(t, "offset_flag", 0)
    if offset_flag in (48, getattr(xtconstant, "OFFSET_FLAG_OPEN", 48)):
        return "BUY"
    return "SELL"


def _trade_ts(traded_time: int, trade_date: date) -> float:
    """
    将 xtquant 的 traded_time（如 93105 / 93105000）映射为 Unix 秒时间戳。
    xtquant 仅给出时分秒，这里默认组合到传入 trade_date 上。
    """
    s = str(int(traded_time)).zfill(9)
    h = int(s[:2])
    m = int(s[2:4])
    sec = int(s[4:6])
    dt = datetime.combine(trade_date, dtime(hour=h, minute=m, second=sec))
    return dt.timestamp()


@dataclass
class Signal:
    signal_id: str
    code: str
    direction: str  # 'BUY' / 'SELL'
    price: float    # 理论发单价
    shares: int
    ts: float       # Unix 秒


@dataclass
class Trade:
    code: str
    direction: str  # 'BUY' / 'SELL'
    price: float
    volume: int
    ts: float
    traded_id: str


def load_signals(target_date: date) -> List[Signal]:
    """从 SHARED_DIR 读取当日已执行信号."""
    base = _ensure_shared_dir()
    executed_path = os.path.join(base, "executed_signals.json")
    done_dir = os.path.join(base, "order_signal_done")
    if not os.path.exists(executed_path) or not os.path.exists(done_dir):
        return []
    try:
        with open(executed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = data.get("signal_ids", [])
        if not isinstance(ids, list):
            ids = []
    except Exception:
        ids = []

    signals: List[Signal] = []
    for sid in ids:
        fn = os.path.join(done_dir, f"order_signal_{sid}.done")
        if not os.path.exists(fn):
            continue
        try:
            with open(fn, "r", encoding="utf-8") as f:
                sig = json.load(f)
        except Exception:
            continue
        code = str(sig.get("code") or "").strip()
        if code not in CODES:
            continue
        direction = str(sig.get("direction") or "").upper()
        if direction not in ("BUY", "SELL"):
            continue
        ts = float(sig.get("timestamp") or 0)
        if ts <= 0:
            continue
        sig_dt = datetime.fromtimestamp(ts)
        if sig_dt.date() != target_date:
            continue
        price = float(sig.get("price") or 0)
        shares = int(sig.get("shares") or 0)
        signals.append(
            Signal(
                signal_id=sid,
                code=code,
                direction=direction,
                price=price,
                shares=shares,
                ts=ts,
            )
        )
    return signals


def load_trades(target_date: date) -> List[Trade]:
    """通过 XtQuantTrader 查询当日成交."""
    session_id = random.randint(100000, 999999)
    trader = XtQuantTrader(QMT_USERDATA_PATH, session_id)
    trader.start()
    res = trader.connect()
    if res != 0:
        raise SystemExit("连接 QMT 失败，请检查 QMT_USERDATA_PATH 与登录状态")
    acc = StockAccount(ACCOUNT_ID, ACCOUNT_TYPE)
    trades = trader.query_stock_trades(acc)
    out: List[Trade] = []
    if not trades:
        return out
    for t in trades:
        code = getattr(t, "stock_code", None) or getattr(t, "stock_code1", "")
        if code not in CODES:
            continue
        direction = _direction_from_trade(t)
        price = float(getattr(t, "traded_price", 0) or 0)
        vol = int(getattr(t, "traded_volume", 0) or 0)
        if vol <= 0 or price <= 0:
            continue
        traded_time = getattr(t, "traded_time", 0) or 0
        ts = _trade_ts(traded_time, target_date)
        traded_id = str(getattr(t, "traded_id", "") or "")
        out.append(
            Trade(
                code=code,
                direction=direction,
                price=price,
                volume=vol,
                ts=ts,
                traded_id=traded_id,
            )
        )
    return out


def match_trades(signals: List[Signal], trades: List[Trade]):
    """
    将成交按 code+方向 在时间上匹配到最近的一条信号（±MATCH_WINDOW_SEC 内）。

    返回：
    - matched: dict[signal_id] -> list[Trade]
    - unmatched_trades: List[Trade]
    """
    # 按 code+方向分桶，方便加速匹配
    signals_by_key: Dict[tuple, List[Signal]] = defaultdict(list)
    for s in signals:
        signals_by_key[(s.code, s.direction)].append(s)
    for key in signals_by_key:
        signals_by_key[key].sort(key=lambda s: s.ts)

    matched: Dict[str, List[Trade]] = defaultdict(list)
    unmatched_trades: List[Trade] = []

    for tr in trades:
        key = (tr.code, tr.direction)
        cand = signals_by_key.get(key) or []
        if not cand:
            unmatched_trades.append(tr)
            continue
        # 找到时间上最近的一条信号
        best_s: Optional[Signal] = None
        best_dt = None
        for s in cand:
            dt = abs(tr.ts - s.ts)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_s = s
        if best_s is None or best_dt is None or best_dt > MATCH_WINDOW_SEC:
            unmatched_trades.append(tr)
        else:
            matched[best_s.signal_id].append(tr)
    return matched, unmatched_trades


def summarize(target_date: date, signals: List[Signal], trades: List[Trade]):
    matched, unmatched_trades = match_trades(signals, trades)

    print("=" * 72)
    print(f"收盘对账 — {target_date.isoformat()}  (匹配窗口 ±{MATCH_WINDOW_SEC} 秒)")
    print("=" * 72)

    if not trades:
        print("QMT 当日无成交记录。")
        return

    if not signals:
        print("Mac 端当日无已执行的信号记录（executed_signals.json / order_signal_done）。")
        return

    total_slip_cost = 0.0
    total_volume = 0
    per_symbol_stats = {code: {"volume": 0, "slip_cost": 0.0, "n": 0} for code in CODES}

    print("\n[逐信号滑点统计]（只列出成功匹配到成交的信号）")
    for s in signals:
        trs = matched.get(s.signal_id)
        if not trs:
            continue
        vol = sum(t.volume for t in trs)
        vwap = sum(t.price * t.volume for t in trs) / vol if vol > 0 else 0.0
        # BUY: 成交价高于信号价为负贡献；SELL: 成交价低于信号价为负贡献
        if s.direction == "BUY":
            per_share_slip = vwap - s.price
        else:  # SELL
            per_share_slip = s.price - vwap
        slip_cost = per_share_slip * vol
        slip_bps = (per_share_slip / s.price * 10000) if s.price > 0 else 0.0

        total_slip_cost += slip_cost
        total_volume += vol
        stats = per_symbol_stats[s.code]
        stats["volume"] += vol
        stats["slip_cost"] += slip_cost
        stats["n"] += 1

        name = SYMBOL_NAME.get(s.code, s.code)
        tstr = datetime.fromtimestamp(s.ts).strftime("%H:%M:%S")
        print(
            f"- {name} {s.direction} signal_id={s.signal_id[:10]}… "
            f"@{s.price:.3f} -> VWAP={vwap:.3f}, 量={vol:6d} 股, "
            f"滑点={per_share_slip:+.4f} ({slip_bps:+.1f} bp), 成本/Alpha贡献={slip_cost:+.2f} 元 "
            f"({tstr})"
        )

    print("\n[按标的汇总]")
    for code in CODES:
        st = per_symbol_stats[code]
        if st["n"] == 0:
            continue
        avg_slip_per_share = st["slip_cost"] / st["volume"] if st["volume"] > 0 else 0.0
        ref_price = None
        # 参考价格：用当日所有该标的成交的 VWAP 近似
        vols = [t.volume for t in trades if t.code == code]
        vs = [t for t in trades if t.code == code]
        if vols:
            ref_price = sum(t.price * t.volume for t in vs) / sum(vols)
        bps = (avg_slip_per_share / ref_price * 10000) if ref_price else 0.0
        name = SYMBOL_NAME.get(code, code)
        print(
            f"- {name}: 匹配信号 {st['n']} 条, 成交量 {st['volume']} 股, "
            f"总滑点成本={st['slip_cost']:+.2f} 元, "
            f"均笔滑点≈{avg_slip_per_share:+.4f} 元/股 ({bps:+.1f} bp)"
        )

    print("\n[整体总结]")
    if total_volume > 0:
        avg_slip = total_slip_cost / total_volume
        print(
            f"  总滑点成本（视为 Alpha 损耗）: {total_slip_cost:+.2f} 元\n"
            f"  平均滑点: {avg_slip:+.4f} 元/股"
        )
    else:
        print("  无法计算整体滑点（总成交量为 0）")

    n_unmatched_sig = sum(1 for s in signals if s.signal_id not in matched)
    if n_unmatched_sig > 0 or unmatched_trades:
        print("\n[未匹配项提示]")
        if n_unmatched_sig > 0:
            print(f"  - 有 {n_unmatched_sig} 条 Mac 端信号未在 QMT 成交中找到匹配成交（可能为撤单/过期）。")
        if unmatched_trades:
            print(f"  - 有 {len(unmatched_trades)} 条 QMT 成交未匹配到 Mac 信号（可能为手工单或时间窗口外）。")
    print("\n说明：滑点成本为相对 Mac 发单价的偏差乘以成交量；视为当天 Alpha 的负贡献。")


def main():
    if len(sys.argv) >= 2:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y%m%d").date()
        except ValueError:
            raise SystemExit("日期格式错误，应为 YYYYMMDD，例如: python eod_reconcile.py 20260226")
    else:
        target_date = date.today()

    print(f"目标对账日期: {target_date.isoformat()}")
    _ensure_shared_dir()

    signals = load_signals(target_date)
    trades = load_trades(target_date)
    summarize(target_date, signals, trades)


if __name__ == "__main__":
    main()

