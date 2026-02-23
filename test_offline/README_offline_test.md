# 离线测试：BUY 信号与结果审计

## 测试文件说明

- **shared_quote.json**：模拟下跌行情（历史 K 线递减，MA60 斜率 < 0），当前价 1.052，用于触发「跌破支撑」买入。
- **dashboard_state.json**：初始状态：`last_buy_price=1.06`、空仓，使 `1.052 <= 1.06*(1-grid_step)` 能触发第一层 BUY。

## 运行 Dashboard（指向测试目录）

在 **miniqmt 项目根目录** 执行（不要 cd 到 test_offline）：

```bash
cd /path/to/miniqmt
export DASHBOARD_WORK_DIR="$(pwd)/test_offline"
.venv/bin/python mac_dashboard.py
```

运行几秒后按 Ctrl+C 结束，信号会写在 `test_offline/order_signal.json`。

## 结果审计

检查 `test_offline/order_signal.json`：

| 审计项 | 要求 | 检查方法 |
|--------|------|----------|
| **唯一性** | signal_id 为可读唯一标识 | 含前缀 `BUY_159201.SZ_L1_` 且后接 8 位十六进制（如 `BUY_159201.SZ_L1_a1b2c3d4`） |
| **方向** | 严格为 BUY | `direction` 字段值为 `"BUY"` |
| **整百股数** | 股数为 100 的整数倍 | `shares % 100 === 0`（如 21300、2200），不得出现 21265 等散单 |

### 快速检查命令

```bash
cd test_offline
# 存在且为合法 JSON
test -f order_signal.json && python3 -c "
import json
with open('order_signal.json') as f:
    d = json.load(f)
sid = d.get('signal_id', '')
dir_ok = d.get('direction') == 'BUY'
shares = d.get('shares', 0)
shares_ok = isinstance(shares, int) and shares > 0 and shares % 100 == 0
print('signal_id:', sid)
print('BUY_159201.SZ_L1_ 前缀:', sid.startswith('BUY_159201.SZ_L1_'))
print('direction==BUY:', dir_ok)
print('shares 整百:', shares_ok, '(', shares, ')')
print('审计通过:', dir_ok and shares_ok and sid.startswith('BUY_159201.SZ_L1_'))
"
```
