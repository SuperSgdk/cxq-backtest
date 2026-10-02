# -*- coding: utf-8 -*-
r"""第 1 层 · 费用模型单测 —— 全部手算，不使用任何历史数据。

运行：.venvScriptspython.exe -X utf8 -B tests	est_bt_costs.py
"""

import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.costs import COSTS, AShareCosts, AShareCommission  # noqa: E402


class TestBuyFee(unittest.TestCase):
    """买入费用 = 佣金(最低5元) + 过户费。"""

    def test_10k(self):
        # 手算：10000 * 0.0005 = 5.0，正好等于下限 → 佣金 5.0
        #       10000 * 0.00001 = 0.1
        #       合计 5.1
        self.assertAlmostEqual(COSTS.buy_fee(10_000.0), 5.1, places=9)

    def test_min_commission_floor(self):
        # 手算：1000 * 0.0005 = 0.5 < 5 → 佣金取 5.0
        #       1000 * 0.00001 = 0.01  合计 5.01
        self.assertAlmostEqual(COSTS.buy_fee(1_000.0), 5.01, places=9)

    def test_99049_5(self):
        # 手算（接口约定）：99049.5 * 0.0005 = 49.52475（> 5，不触发下限）
        #                 99049.5 * 0.00001 = 0.990495
        #                 合计 50.515245
        self.assertAlmostEqual(COSTS.buy_fee(99_049.5), 50.515245, places=9)

    def test_no_stamp_duty_on_buy(self):
        # 买入不收印花税：把印花税率改大，买入费用必须完全不变
        c = AShareCosts(stamp_duty_rate=0.5)
        self.assertAlmostEqual(c.buy_fee(100_000.0), COSTS.buy_fee(100_000.0),
                               places=12)

    def test_zero_and_negative(self):
        self.assertEqual(COSTS.buy_fee(0.0), 0.0)
        self.assertEqual(COSTS.buy_fee(-1.0), 0.0)


class TestSellFee(unittest.TestCase):
    """卖出费用 = 佣金(最低5元) + 印花税 + 过户费。"""

    def test_108845_55(self):
        # 手算（接口约定）：108845.55 * 0.0005 = 54.422775（佣金）
        #                 108845.55 * 0.0005 = 54.422775（印花税）
        #                 108845.55 * 0.00001 = 1.0884555（过户费）
        #                 合计 109.9340055
        self.assertAlmostEqual(COSTS.sell_fee(108_845.55), 109.9340055, places=9)

    def test_stamp_duty_only_on_sell(self):
        # 同一金额，卖出费用 − 买入费用 必须正好等于一个印花税
        amount = 100_000.0
        self.assertAlmostEqual(COSTS.sell_fee(amount) - COSTS.buy_fee(amount),
                               amount * COSTS.stamp_duty_rate, places=12)

    def test_sell_fee_above_buy_fee_always(self):
        for amount in (100.0, 1_000.0, 10_000.0, 99_049.5, 1_000_000.0):
            self.assertGreater(COSTS.sell_fee(amount), COSTS.buy_fee(amount))


class TestSlippage(unittest.TestCase):
    def test_buy_price(self):
        # 手算：10.00 * 1.0005 = 10.005
        self.assertAlmostEqual(COSTS.buy_price(10.00), 10.005, places=12)

    def test_sell_price(self):
        # 手算：11.00 * 0.9995 = 10.9945
        self.assertAlmostEqual(COSTS.sell_price(11.00), 10.9945, places=12)

    def test_buy_more_expensive_sell_cheaper(self):
        self.assertGreater(COSTS.buy_price(10.0), 10.0)
        self.assertLess(COSTS.sell_price(10.0), 10.0)


class TestRoundTrip(unittest.TestCase):
    """端到端手算：接口说明 的 9,635.60。"""

    def test_net_profit_9635_6007(self):
        # 手算：D1 买 9900 股 @ 10.005 = 99049.5，费用 50.515245
        #       D4 卖 9900 股 @ 10.9945 = 108845.55，费用 109.9340055
        #       净盈亏 = (108845.55 - 109.9340055) - (99049.5 + 50.515245)
        #              = 108735.6159945 - 99100.015245
        #              = 9635.6007495
        buy_cost = COSTS.buy_cost(9900, 10.00)
        sell_net = COSTS.sell_proceeds(9900, 11.00)
        self.assertAlmostEqual(buy_cost, 99_100.015245, places=6)
        self.assertAlmostEqual(sell_net, 108_735.6159945, places=6)
        self.assertAlmostEqual(sell_net - buy_cost, 9635.6007495, places=6)


class TestBacktraderCommission(unittest.TestCase):
    """适配器必须和纯函数算出**完全一样**的费用（防 percabs 那个 100 倍坑）。"""

    def test_matches_pure_function(self):
        comm = AShareCommission()
        for shares, price in ((100, 10.0), (9900, 10.005), (10000, 100.0)):
            amount = shares * price
            self.assertAlmostEqual(comm._getcommission(shares, price, False),
                                   COSTS.buy_fee(amount), places=9,
                                   msg="买入费用对不上 %d@%s" % (shares, price))
            self.assertAlmostEqual(comm._getcommission(-shares, price, False),
                                   COSTS.sell_fee(amount), places=9,
                                   msg="卖出费用对不上 %d@%s" % (shares, price))

    def test_percabs_is_true(self):
        # 🔴 这条是防回归：percabs=False 会让费率被除以 100
        comm = AShareCommission()
        self.assertTrue(comm.p.percabs,
                        "percabs 必须是 True，否则费用缩小 100 倍")
        # 再确认一次：broker 看到的费率就是万分之五，没有被 /100
        self.assertAlmostEqual(float(comm.p.commission), 0.0005, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
