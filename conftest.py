# pytest 配置与共享 fixture：保证在 miniqmt 目录下可导入 mac_dashboard / global_vault / order_executor
from pathlib import Path
import sys

# 测试文件所在目录（miniqmt）加入 path，便于同目录下的模块被导入
_miniqmt_dir = Path(__file__).resolve().parent
if str(_miniqmt_dir) not in sys.path:
    sys.path.insert(0, str(_miniqmt_dir))
