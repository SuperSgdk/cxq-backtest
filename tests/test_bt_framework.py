"""通用入口的手算验证，全部行情为明确标注的人工测试数据。"""

import datetime as dt
import json
import unittest
from pathlib import Path

import pandas as pd

from cxq_bt.costs import AShareCosts
from cxq_bt.framework import (BacktestConfig, TargetWeightStrategy, load_market_data,
                              load_target_weights, run_backtest)

ZERO = AShareCosts(0, 0, 0, 0, 0)
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def market(prices, *, codes=("TEST_A",)):
    dates = pd.bdate_range("2024-01-02", periods=len(prices))
    return pd.DataFrame([dict(date=str(date.date()), code=code, open=price, high=price,
                             low=price, close=price, volume=1000, up_limit=price * 1.1,
                             down_limit=price * .9, is_st=0, is_suspended=0)
                         for date, price in zip(dates, prices) for code in codes])


class TestFramework(unittest.TestCase):
    def test_收盘决定下一交易日成交并手算费用(self):
        data = market([10, 10, 11])
        signals = {dt.date(2024, 1, 2): {"TEST_A": 1}, dt.date(2024, 1, 3): {"TEST_A": 0}}
        costs = AShareCosts(.0005, 5, .0005, .00001, .0005)
        result = run_backtest(data, TargetWeightStrategy(signals), BacktestConfig(100000, costs))
        self.assertEqual([t["date"] for t in result.trades], ["2024-01-03", "2024-01-04"])
        self.assertEqual([t["shares"] for t in result.trades], [9900, 9900])
        self.assertAlmostEqual(result.trades[0]["price"], 10.005)
        self.assertAlmostEqual(result.trades[1]["price"], 10.9945)
        # 独立手算：买入99049.5，费用50.515245；卖出108845.55，费用109.9340055。
        self.assertAlmostEqual(result.summary["final_equity"], 100000 - 99049.5 - 50.515245 + 108845.55 - 109.9340055, places=7)
        cash = 100000
        for trade in result.trades:
            cash += -trade["amount"] - trade["fee"] if trade["side"] == "buy" else trade["amount"] - trade["fee"]
        self.assertAlmostEqual(result.equity[-1]["cash"], cash, places=8)

    def test_受限仓仍计资产且不能变成现金(self):
        data = market([10, 10, 10], codes=("TEST_A", "TEST_B"))
        mask = (data.date == "2024-01-04") & (data.code == "TEST_A")
        data.loc[mask, "is_suspended"] = 1
        data.loc[mask, "volume"] = 0
        signals = {"2024-01-02": {"TEST_A": 1}, "2024-01-03": {"TEST_B": 1}}
        result = run_backtest(data, TargetWeightStrategy(signals), BacktestConfig(10000, ZERO))
        self.assertEqual(result.summary["final_equity"], 10000)
        self.assertEqual(result.equity[-1]["cash"], 0)
        self.assertEqual(result.equity[-1]["positions"], {"TEST_A": 1000})
        self.assertTrue(any("停牌" in o["reason"] for o in result.orders))

    def test_策略只能看到决定日及以前的行情(self):
        class Inspect:
            def on_close(self, context):
                for code in context.codes:
                    frame = context.history(code)
                    assert max(frame.date.dt.date) <= context.date
                    frame.loc[:, "close"] = 999
                return None
        result = run_backtest(market([10, 10, 10]), Inspect(), BacktestConfig(10000, ZERO))
        self.assertEqual(result.summary["final_equity"], 10000)

    def test_没有信号保持持仓而不是自行清仓(self):
        result = run_backtest(market([10, 10, 11]), TargetWeightStrategy({"2024-01-02": {"TEST_A": 1}}), BacktestConfig(10000, ZERO))
        self.assertEqual(result.summary["trade_count"], 1)
        self.assertEqual(result.equity[-1]["positions"], {"TEST_A": 1000})

    def test_开盘涨停拒买不补概率成交(self):
        data = market([10, 11, 11])
        data.loc[data.date == "2024-01-03", "up_limit"] = 11
        result = run_backtest(data, TargetWeightStrategy({"2024-01-02": {"TEST_A": 1}}), BacktestConfig(10000, ZERO))
        self.assertEqual(result.summary["trade_count"], 0)
        self.assertEqual(result.equity[-1]["cash"], 10000)

    def test_滑点不能穿越价格边界(self):
        data = market([10, 10.99, 11])
        data.loc[data.date == "2024-01-03", "up_limit"] = 11
        result = run_backtest(data, TargetWeightStrategy({"2024-01-02": {"TEST_A": 1}}), BacktestConfig(10000, AShareCosts(0, 0, 0, 0, .01)))
        self.assertEqual(result.summary["trade_count"], 0)
        self.assertTrue(any("滑点" in o["reason"] for o in result.orders))

    def test_禁止读取未来日线极值参与开盘成交(self):
        data = market([10, 10, 10])
        data["high"] = data["high"].astype(float)
        # 合法日线高价低于滑点成交价，不能拿收盘后才知道的极值裁剪开盘成交。
        data.loc[data.date == "2024-01-03", "high"] = 10.02
        result = run_backtest(data, TargetWeightStrategy({"2024-01-02": {"TEST_A": 1}}), BacktestConfig(10000, AShareCosts(0, 0, 0, 0, .01)))
        self.assertEqual(result.trades[0]["price"], 10.1)

    def test_资金费率非法必须报错(self):
        for cash in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                BacktestConfig(cash)
        with self.assertRaises(ValueError):
            BacktestConfig(costs=AShareCosts(commission_rate=-.1))

    def test_缺行缺状态重复行情必须报错(self):
        data = market([10, 10], codes=("TEST_A", "TEST_B"))
        for wrong in (data.iloc[:-1], data.drop(columns="is_st"), pd.concat([data, data.iloc[:1]])):
            with self.assertRaises(ValueError):
                load_market_data(wrong)

    def test_完整目标组合清掉缺席持仓(self):
        signals = {"2024-01-02": {"TEST_A": 1}, "2024-01-03": {"TEST_B": 1}}
        result = run_backtest(market([10, 10, 10], codes=("TEST_A", "TEST_B")), TargetWeightStrategy(signals), BacktestConfig(10000, ZERO))
        self.assertEqual(result.equity[-1]["positions"], {"TEST_B": 1000})

    def test_期末计划不伪造成交且JSON没有NaN(self):
        result = run_backtest(market([10, 10]), TargetWeightStrategy({"2024-01-03": {"TEST_A": 1}}), BacktestConfig(10000, ZERO))
        self.assertEqual(result.summary["trade_count"], 0)
        self.assertTrue(any(o["status"] == "unexecuted_end" for o in result.orders))
        json.dumps(result.to_dict(), allow_nan=False)

    def test_自带样例能独立运行并记录参数指纹(self):
        result = run_backtest(EXAMPLES / "market_handcalc.csv", TargetWeightStrategy(load_target_weights(EXAMPLES / "signals_handcalc.csv")), BacktestConfig(100000))
        self.assertEqual([t["date"] for t in result.trades], ["2024-01-03", "2024-01-05", "2024-01-05", "2024-01-09"])
        self.assertEqual(len(result.settings["input_sha256"]), 64)
        self.assertEqual(result.settings["initial_cash"], 100000)


if __name__ == "__main__":
    unittest.main()
