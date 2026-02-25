# miniqmt 量化策略

Mac 策略决策 + Windows xtquant 执行的跨平台量化桥接。

## 项目结构

| 文件 | 说明 |
|------|------|
| `mac_dashboard.py` | 实时看板，读取 159201 行情生成买卖信号 |
| `mac_backtest_159201.py` | 159201 自由现金流 ETF 网格回测 |
| `mac_backtest_159201_sweep.py` | 159201 参数扫描 |
| `mac_backtest_159201_quick.py` | 159201 快速参数验证 |
| `mac_backtest.py` | 通用回测框架 |
| `mac_backtest_pro_600895.py` | 600895 专用回测 |
| `bridge_producer.py` | 桥接生产者（QMT → shared_quote_159201.json） |
| `init_flow_plan.py` | 沿用现有持仓、以当日收盘价重置成本基准的新建仓计划 |
| `init_fixed_flow_plan.py` | 将当前持仓按 80% 固定 / 20% 流动拆分并初始化网格 |
| `reset_build_plan.py` | 清空策略状态，以当前价为锚点重新开始建仓 |
| `global_vault.py` | 共享资金池实现（与 `shared_pool.json` 协同） |
| `策略文档_159201.md` | 159201 策略说明 |

## 159201 策略要点

- **ATR 动态网格** + **趋势自适应** + **仓位分级**
- 固定底仓 + 流动网格（可通过 `init_fixed_flow_plan.py` 初始化 80% 固定 / 20% 流动）
- 详见 `策略文档_159201.md`

## 回测

```bash
python3 mac_backtest_159201.py
```

## 实盘运行（单标 159201）

1. **Windows / QMT 端**
   - 启动 MiniQMT 并登录账户；
   - 运行 `bridge_producer.py`，确保持续生成/更新 `shared_quote_159201.json` 与 `shared_pool.json`。

2. **Mac 端看板 + 信号**

```bash
cd /path/to/miniqmt
DASHBOARD_WORK_DIR=$(pwd) python3 mac_dashboard.py
```

- 读取 `shared_quote_159201.json` 行情；
- 使用 `dashboard_state.json` 存储策略状态；
- 买卖信号输出到 `order_signal.json`，由 Windows 端执行器消费。

3. **初始化 / 重置 159201 状态**

- 沿用现有持仓、以当日收盘价重置成本基准（全部视为流动仓）：

```bash
python3 init_flow_plan.py
```

- 将当前持仓按 80% 固定 / 20% 流动拆分，并以当日收盘价为锚点初始化网格：

```bash
python3 init_fixed_flow_plan.py
```

- 清空策略状态，从当前价重新开始建仓（需手工确认真实持仓是否已平）：

```bash
python3 reset_build_plan.py
```

> 所有初始化/重置脚本均建议 **收盘后（15:05 之后）** 运行，且需确保 QMT 端真实持仓与期望状态一致，以免出现「策略状态与真实持仓不一致」。

## 数据

需自行准备 `history_159201_1m.csv` 等 1 分钟 K 线数据用于回测，格式含 open/high/low/close。
