# miniqmt 量化策略

Mac 策略决策 + Windows xtquant 执行的跨平台量化桥接。

## 项目结构

| 文件 | 说明 |
|------|------|
| `mac_dashboard.py` | 实时看板，读取行情生成买卖信号 |
| `mac_backtest_159201.py` | 159201 自由现金流 ETF 网格回测 |
| `mac_backtest_159201_sweep.py` | 159201 参数扫描 |
| `mac_backtest_159201_quick.py` | 159201 快速参数验证 |
| `mac_backtest.py` | 通用回测框架 |
| `mac_backtest_pro_600895.py` | 600895 专用回测 |
| `bridge_producer.py` | 桥接生产者 |
| `策略文档_159201.md` | 159201 策略说明 |

## 159201 策略要点

- **ATR 动态网格** + **趋势自适应** + **仓位分级**
- 预设：`default`（~22 笔/年）或 `high_freq`（~432 笔/年）
- 详见 `策略文档_159201.md`

## 运行

```bash
python3 mac_backtest_159201.py
```

## 数据

需自行准备 `history_159201_1m.csv` 等 1 分钟 K 线数据，格式含 open/high/low/close。
