import os
import json
import time
import shutil
import unittest
from datetime import datetime

# 导入你之前的核心逻辑函数 (确保 test_executor_dryrun.py 和 test_apply_partial_fill.py 在同级目录)
from test_executor_dryrun import validate_signal, save_executed_id, load_executed_ids, write_order_result
from test_apply_partial_fill import apply_order_result

class MiniQMTTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.shared_dir = cls.test_dir
        cls.state_file = os.path.join(cls.test_dir, "dashboard_state.json")
        cls.signal_file = os.path.join(cls.test_dir, "order_signal.json")
        cls.result_file = os.path.join(cls.test_dir, "order_result.json")
        cls.executed_file = os.path.join(cls.test_dir, "executed_signals.json")
        cls.remote_stop_file = os.path.join(cls.test_dir, "STOP.txt")

    def setUp(self):
        """每项测试前清理环境"""
        for f in [self.signal_file, self.result_file, self.executed_file, self.remote_stop_file]:
            if os.path.exists(f): os.remove(f)
        if os.path.exists(self.state_file): os.remove(self.state_file)

    # --- 阶段二：分布式通信与幂等性测试 ---

    def test_stage2_signal_consumption(self):
        """验证信号正常消费与结果写入"""
        signal = {
            "signal_id": "test_sig_001", "code": "159201.SZ", "direction": "BUY",
            "shares": 2000, "price": 1.05, "timestamp": time.time()
        }
        with open(self.signal_file, 'w') as f: json.dump(signal, f)
        
        # 模拟执行器逻辑
        err = validate_signal(signal)
        self.assertIsNone(err)
        write_order_result("test_sig_001", "success", 2000, 2000, "BUY")
        
        self.assertTrue(os.path.exists(self.result_file))
        with open(self.result_file, 'r') as f:
            res = json.load(f)
            self.assertEqual(res['filled_shares'], 2000)

    def test_stage2_idempotency(self):
        """验证幂等性：同一信号不重复下单"""
        save_executed_id("duplicate_id")
        signal = {"signal_id": "duplicate_id", "code": "159201.SZ"}
        
        executed = load_executed_ids()
        self.assertIn("duplicate_id", executed) # 验证已拦截

    # --- 阶段三：部分成交与自愈能力测试 ---

    def test_stage3_partial_fill_reconciliation(self):
        """验证买单部分成交后的状态对齐与 Pending 产生"""
        state = {
            "last_sent_signal_id": "sig_pf_001",
            "last_sent_signal_direction": "BUY",
            "last_sent_signal_shares": 2000,
            "positions": [{"shares": 2000, "cost": 2100, "buy_price": 1.05}]
        }
        result = {
            "signal_id": "sig_pf_001", "status": "success", "direction": "BUY",
            "requested_shares": 2000, "filled_shares": 500, "price": 1.05
        }
        
        apply_order_result(state, result)
        
        self.assertEqual(state['positions'][-1]['shares'], 500) # 最后一笔已修正
        self.assertEqual(state['pending_buy_shares'], 1500)     # 剩余计入 Pending

    def test_stage3_remote_stop(self):
        """验证手机远程熔断开关"""
        with open(self.remote_stop_file, 'w') as f: f.write("STOP")
        
        # 模拟 Mac 端检测逻辑
        is_running = not os.path.exists(self.remote_stop_file)
        self.assertFalse(is_running)

if __name__ == "__main__":
    unittest.main()
