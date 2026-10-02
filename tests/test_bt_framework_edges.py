# -*- coding: utf-8 -*-
"""通用入口的手算边界；行情全部是测试构造，不代表真实历史业绩。"""

import math
import unittest

import pandas as pd

from cxq_bt.costs import AShareCosts
from cxq_bt.framework import BacktestConfig, TargetWeightStrategy, load_market_data, run_backtest


DATES = tuple(str(value.date()) for value in pd.bdate_range("2024-01-02", periods=5))
ZERO_COSTS = AShareCosts(commission_rate=0, min_commission=0, stamp_duty_rate=0,
                        transfer_fee_rate=0, slippage_rate=0)


def market(days=3, codes=("TEST_A",)):
    """每股 10 元；一万元恰好可买 1000 股，便于独立手算。"""
    return pd.DataFrame([
        dict(date=date, code=code, open=10.0, high=10.1, low=9.9, close=10.0,
             volume=100_000, up_limit=11.0, down_limit=9.0, is_st=0, is_suspended=0)
        for date in DATES[:days] for code in codes
    ])


def change(data, day, code="TEST_A", **values):
    index = (data.date == DATES[day]) & (data.code == code)
    for column, value in values.items():
        data.loc[index, column] = value


class Plans:
    def __init__(self, plans):
        self.plans = plans

    def on_close(self, context):
        return self.plans.get(str(context.date))


def simulate(data, plans, **settings):
    options = dict(initial_cash=10_000, costs=ZERO_COSTS, max_positions=2)
    options.update(settings)
    return run_backtest(data, Plans(plans), BacktestConfig(**options))


class TestFrameworkTradingEdges(unittest.TestCase):
    def test_open_execution_does_not_use_future_ohlc_or_daily_volume(self):
        base = market(days=2)
        changed = base.copy(deep=True)
        # 只改变执行日收盘后才完整知道的信息；开盘价及已知交易状态相同。
        change(changed, 1, high=10.8, low=9.2, close=10.7, volume=0)
        plans = {DATES[0]: {"TEST_A": 1.0}}
        first = simulate(base, plans)
        second = simulate(changed, plans)
        self.assertEqual(first.trades, second.trades)
        self.assertEqual(first.trades[0]["date"], DATES[1])
        self.assertEqual(first.trades[0]["shares"], 1000)
        self.assertEqual(first.trades[0]["price"], 10.0)
        self.assertEqual(second.equity[-1]["equity"], 10_700)

    def test_open_limit_up_cannot_use_later_break_to_fill(self):
        data = market(days=2)
        change(data, 1, open=11.0, high=11.0, low=10.0, close=10.2)
        result = simulate(data, {DATES[0]: {"TEST_A": 1.0}})
        self.assertEqual(result.trades, [])
        self.assertTrue(any("开盘涨停" in order["reason"] for order in result.orders))
        self.assertEqual(result.equity[-1]["cash"], 10_000)

    def test_limit_down_holding_still_occupies_slot(self):
        data = market(codes=("TEST_A", "TEST_B"))
        # 已有 A 在下一次调仓开盘跌停。即使当天后来反弹，也不能按日级数据猜成交。
        change(data, 2, open=9.0, low=9.0, high=10.0, close=10.0)
        result = simulate(data, {DATES[0]: {"TEST_A": 1}, DATES[1]: {"TEST_B": 1}},
                          max_positions=1)
        self.assertEqual([(trade["code"], trade["side"]) for trade in result.trades],
                         [("TEST_A", "buy")])
        self.assertEqual(result.equity[-1]["positions"], {"TEST_A": 1000})
        self.assertTrue(any("持仓上限" in order["reason"] for order in result.orders))

    def test_unsold_value_is_not_available_cash(self):
        data = market(codes=("TEST_A", "TEST_B"))
        change(data, 2, open=9.0, low=9.0, high=10.0, close=10.0)
        result = simulate(data, {DATES[0]: {"TEST_A": 0.5}, DATES[1]: {"TEST_B": 1}},
                          initial_cash=20_000)
        # A 跌停卖不出。B 只能用原来的一万元现金买 1000 股，不能挪用 A 的九千元。
        self.assertEqual(result.equity[-1]["positions"], {"TEST_A": 1000, "TEST_B": 1000})
        self.assertEqual(result.equity[-1]["cash"], 0)

    def test_none_holds_and_empty_plan_exits(self):
        result = simulate(market(days=4), {DATES[0]: {"TEST_A": 1}, DATES[2]: {}})
        self.assertEqual(result.equity[2]["positions"], {"TEST_A": 1000})
        self.assertEqual(result.equity[3]["positions"], {})
        self.assertEqual([(trade["date"], trade["side"]) for trade in result.trades],
                         [(DATES[1], "buy"), (DATES[3], "sell")])
        self.assertEqual(result.equity[-1]["cash"], 10_000)

    def test_final_empty_plan_records_unexecuted_exit(self):
        result = simulate(market(days=3), {DATES[0]: {"TEST_A": 1}, DATES[2]: {}})
        exits = [order for order in result.orders if order["status"] == "unexecuted_end"]
        self.assertEqual([order["code"] for order in exits], ["TEST_A"])
        self.assertEqual(result.equity[-1]["positions"], {"TEST_A": 1000})

    def test_stop_has_priority_and_only_one_direction_per_stock(self):
        data = market()
        change(data, 2, open=7.9, high=8.0, low=7.9, close=7.9, down_limit=7.0)
        result = simulate(data, {DATES[0]: {"TEST_A": 1}, DATES[1]: {"TEST_A": 1}},
                          hard_stop_pct=-0.2)
        executed = [trade for trade in result.trades if trade["date"] == DATES[2]]
        self.assertEqual([(trade["side"], trade["shares"]) for trade in executed], [("sell", 1000)])
        self.assertEqual(result.equity[-1]["cash"], 7900)

    def test_stop_triggers_at_exact_threshold(self):
        data = market()
        change(data, 2, open=8.0, high=8.0, low=8.0, close=8.0, down_limit=7.0)
        result = simulate(data, {DATES[0]: {"TEST_A": 1}, DATES[1]: {"TEST_A": 1}},
                          hard_stop_pct=-0.2)
        self.assertEqual(result.equity[-1]["positions"], {})
        self.assertEqual(result.equity[-1]["cash"], 8000)

    def test_minimum_commission_is_paid_before_allocating_shares(self):
        costs = AShareCosts(commission_rate=0, min_commission=5, stamp_duty_rate=0,
                            transfer_fee_rate=0, slippage_rate=0)
        result = simulate(market(days=2), {DATES[0]: {"TEST_A": 1}},
                          initial_cash=1000, costs=costs)
        # 100 股需 1000 + 5 元；1000 元账户不能先买入再透支费用。
        self.assertEqual(result.trades, [])
        self.assertEqual(result.equity[-1]["cash"], 1000)
        self.assertTrue(any("资金不足" in order["reason"] for order in result.orders))

    def test_large_commission_cannot_make_sell_proceeds_negative(self):
        costs = AShareCosts(commission_rate=0, min_commission=1000, stamp_duty_rate=0,
                            transfer_fee_rate=0, slippage_rate=0)
        data = market()
        change(data, 2, open=0.5, high=0.5, low=0.5, close=0.5,
               down_limit=0.4, up_limit=0.6)
        result = simulate(data, {DATES[0]: {"TEST_A": 1}, DATES[1]: {}},
                          initial_cash=12_000, costs=costs)
        # 先以 11000 元加 1000 元佣金买入；跌价后 550 元卖款不足以支付 1000 元费用。
        self.assertEqual(result.equity[-1]["positions"], {"TEST_A": 1100})
        self.assertEqual(result.equity[-1]["cash"], 0)
        self.assertEqual(len(result.trades), 1)
        self.assertTrue(any("不足以支付费用" in order["reason"] for order in result.orders))

    def test_slippage_cannot_fill_beyond_daily_price_limit(self):
        costs = AShareCosts(commission_rate=0, min_commission=0, stamp_duty_rate=0,
                            transfer_fee_rate=0, slippage_rate=0.01)
        data = market(days=2)
        change(data, 1, open=10.95, high=11.0, low=10.9, close=10.95)
        result = simulate(data, {DATES[0]: {"TEST_A": 1}}, costs=costs)
        self.assertEqual(result.trades, [])
        self.assertTrue(any("越过涨跌停边界" in order["reason"] for order in result.orders))


class TestFrameworkStrategyEdges(unittest.TestCase):
    def test_all_ohlc_prices_must_use_the_given_daily_limit_range(self):
        data = market(days=2)
        change(data, 1, high=12.0, close=12.0)
        # 开盘 10 元虽合法，最高与收盘 12 元却越过 11 元涨停价；同价口径不能同时成立。
        with self.assertRaises(ValueError):
            load_market_data(data)

    def test_context_only_exposes_past_and_returned_history_is_a_copy(self):
        observed = []

        class Inspect:
            def on_close(self, context):
                history = context.history("TEST_A")
                observed.append((str(context.date), len(history), str(history.date.max().date())))
                history.loc[:, "close"] = 999
                if not (context.history("TEST_A").close == 10).all():
                    raise AssertionError("修改历史副本污染了下一次读取")
                with self_outer.assertRaises(TypeError):
                    context.positions["TEST_A"] = 100
                return None

        self_outer = self
        run_backtest(market(days=4), Inspect(), BacktestConfig(costs=ZERO_COSTS))
        self.assertEqual(observed, [(date, index + 1, date) for index, date in enumerate(DATES[:4])])

    def test_target_signal_outside_data_window_is_rejected(self):
        strategy = TargetWeightStrategy({DATES[4]: {"TEST_A": 1}})
        with self.assertRaisesRegex(ValueError, "超出行情日历"):
            run_backtest(market(days=2), strategy, BacktestConfig(costs=ZERO_COSTS))

    def test_unknown_stock_is_rejected_even_at_zero_weight(self):
        with self.assertRaisesRegex(ValueError, "没有行情"):
            simulate(market(days=2), {DATES[0]: {"UNKNOWN": 0}})

    def test_invalid_weights_and_position_limit_are_rejected(self):
        cases = [{"TEST_A": -0.1}, {"TEST_A": 1.1}, {"TEST_A": math.nan},
                 {"TEST_A": math.inf}, {"TEST_A": 0.6, "TEST_B": 0.6}]
        for weights in cases:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                simulate(market(days=2, codes=("TEST_A", "TEST_B")), {DATES[0]: weights})
        with self.assertRaisesRegex(ValueError, "max_positions"):
            simulate(market(days=2, codes=("TEST_A", "TEST_B")),
                     {DATES[0]: {"TEST_A": 0.5, "TEST_B": 0.5}}, max_positions=1)

    def test_invalid_money_and_fee_settings_are_rejected(self):
        for amount in (0, -1, math.nan, math.inf):
            with self.subTest(initial_cash=amount), self.assertRaises(ValueError):
                BacktestConfig(initial_cash=amount)
        for fee in (-0.1, math.nan, math.inf, 1.0):
            with self.subTest(commission_rate=fee), self.assertRaises(ValueError):
                BacktestConfig(costs=AShareCosts(commission_rate=fee))


if __name__ == "__main__":
    unittest.main()
