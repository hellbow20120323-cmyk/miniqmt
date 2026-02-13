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

# --- 1. 核心配置 ---
SHARED_FILE = '/Users/yuhao/Documents/miniqmt/shared_quote.json'
SIGNAL_FILE = '/Users/yuhao/Documents/miniqmt/order_signal.json'
STOCK_CODE = '512480.SH'

# 策略参数
BASE_COST = 1.87     # 初始参考成本
GRID_STEP = 0.01     # 1% 网格步长
RSI_PERIOD = 14
BOLL_PERIOD = 20

# 运行时状态
state = {
    "last_trade_price": BASE_COST,
    "signals": [],
    "status": "等待数据..."
}

console = Console()

# --- 2. 指标计算逻辑 ---
def calculate_indicators(prices):
    if len(prices) < BOLL_PERIOD:
        return None
    
    df = pd.DataFrame(prices, columns=['close'])
    
    # BOLL 计算
    df['ma'] = df['close'].rolling(window=BOLL_PERIOD).mean()
    df['std'] = df['close'].rolling(window=BOLL_PERIOD).std()
    df['upper'] = df['ma'] + (df['std'] * 2)
    df['lower'] = df['ma'] - (df['std'] * 2)
    
    # RSI 计算
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df.iloc[-1]

# --- 3. 信号执行与通知 ---
def execute_signal(direction, price, reason):
    msg = f"检测到{direction}信号 | 价格:{price:.3f} | 原因:{reason}"
    state["signals"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if len(state["signals"]) > 5: state["signals"].pop(0)
    
    # 语音播报
    voice_msg = "买入半导体" if direction == "BUY" else "卖出半导体"
    os.system(f'say "{voice_msg}" &') # 加 & 符号防止阻塞
    
    # 生成信号文件供 Windows 读取
    signal_data = {
        "code": STOCK_CODE,
        "direction": direction,
        "price": price,
        "timestamp": time.time()
    }
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(signal_data, f)

# --- 4. 仪表盘界面生成 ---
def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=12),
        Layout(name="footer", size=8)
    )
    return layout

def generate_display(data, indicators):
    # 主数据表
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("项目", style="dim")
    table.add_column("数值", justify="right")
    table.add_column("状态/指标", justify="center")

    if data and indicators is not None:
        curr_p = data['price']
        pos = data.get('position', {})
        
        # 计算网格距离
        dist_buy = (curr_p - state["last_trade_price"] * (1 - GRID_STEP)) / curr_p
        dist_sell = (curr_p - state["last_trade_price"] * (1 + GRID_STEP)) / curr_p

        table.add_row("当前价格", f"{curr_p:.3f}", f"🕒 {data['time']}")
        table.add_row("RSI (14)", f"{indicators['rsi']:.2f}", "🔥 超买" if indicators['rsi'] > 65 else "❄️ 超卖" if indicators['rsi'] < 35 else "⚖️ 中性")
        table.add_row("布林带", f"上:{indicators['upper']:.3f} / 下:{indicators['lower']:.3f}", "支撑位临近" if curr_p < indicators['lower'] else "压力位临近" if curr_p > indicators['upper'] else "带内震荡")
        table.add_row("真实持仓", f"{pos.get('volume', 0)} 股", f"可用: {pos.get('can_use_volume', 0)}")
        table.add_row("网格距离", f"距买入:{dist_buy:+.2%} / 距卖出:{dist_sell:+.2%}", "🎯 观察中")

    return table

# --- 5. 主循环 ---
def main():
    layout = make_layout()
    with Live(layout, refresh_per_second=1, screen=True) as live:
        while True:
            if os.path.exists(SHARED_FILE):
                try:
                    with open(SHARED_FILE, 'r') as f:
                        data = json.load(f)
                    
                    indicators = calculate_indicators(data.get('history', []))
                    
                    if indicators is not None:
                        curr_p = data['price']
                        # 策略逻辑过滤
                        # 买入：网格到位 + 布林下轨 + RSI超卖
                        if curr_p <= state["last_trade_price"] * (1 - GRID_STEP):
                            if curr_p <= indicators['lower'] and indicators['rsi'] < 35:
                                execute_signal("BUY", curr_p, "BOLL下轨+RSI超卖")
                                state["last_trade_price"] = curr_p

                        # 卖出：网格到位 + 布林上轨 + RSI超买
                        elif curr_p >= state["last_trade_price"] * (1 + GRID_STEP):
                            if curr_p >= indicators['upper'] and indicators['rsi'] > 65:
                                if data.get('position', {}).get('can_use_volume', 0) > 0:
                                    execute_signal("SELL", curr_p, "BOLL上轨+RSI超买")
                                    state["last_trade_price"] = curr_p

                    # 更新 UI
                    layout["header"].update(Panel(f"💎 半导体 ETF (512480) 增强网格策略监控 | 状态: 运行中", style="bold green"))
                    layout["main"].update(generate_display(data, indicators))
                    layout["footer"].update(Panel("\n".join(state["signals"]), title="📜 最近信号记录", border_style="yellow"))
                
                except Exception as e:
                    layout["header"].update(Panel(f"❌ 错误: {str(e)}", style="bold red"))
            
            time.sleep(1)

if __name__ == "__main__":
    main()
