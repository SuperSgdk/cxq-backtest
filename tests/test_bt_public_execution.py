"""公开排名调仓配方的执行回归。全部行情是明确的手算构造，非历史业绩。"""
import datetime as dt
import unittest

import pandas as pd

from cxq_bt.engine import EngineConfig, run_engine
from cxq_bt.execution import BreakLookupResult, as_of, confirmed_break_fill
from cxq_bt.sizing import Holding, plan_day
from tests.test_bt_engine import mk


DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
BENCH = mk(DAYS, [3000] * 4)


def minute_frame(break_minute=None, reopen=10.7):
    """标签为区间开始；09:35 的 low 在 09:36 才能确认。"""
    stamps = pd.date_range("2024-01-04 09:30", periods=21, freq="min")
    frame = pd.DataFrame({"trade_time": stamps, "open": 11.0,
                          "high": 11.0, "low": 11.0, "close": 11.0,
                          "volume": 1000.0})
    if break_minute is not None:
        offset = break_minute - 30
        frame.loc[offset, ["low", "close"]] = [10.6, 10.7]
        if offset + 1 < len(frame):
            frame.loc[offset + 1, ["open", "low", "close"]] = [reopen, 10.6, reopen]
    return frame


class TestPublicMinuteTiming(unittest.TestCase):
    def test_uses_next_open_after_bar_is_complete(self):
        result = confirmed_break_fill(minute_frame(35), 11)
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.at, dt.time(9, 36))
        self.assertEqual(result.confirmed_at, dt.time(9, 36))
        self.assertEqual(result.price, 10.7)
        # 被确认 bar 的开盘价是11元；它不能当成看见 low 后的买价。
        self.assertNotEqual(result.price, 11)

    def test_0950_bar_low_cannot_confirm_before_deadline(self):
        self.assertEqual(confirmed_break_fill(minute_frame(50), 11).status, "no_break")

    def test_0949_break_can_execute_at_0950(self):
        result = confirmed_break_fill(minute_frame(49), 11)
        self.assertEqual((result.status, result.at), ("filled", dt.time(9, 50)))

    def test_missing_minute_is_unknown_instead_of_no_break(self):
        for frame in (minute_frame().iloc[:-1], minute_frame().drop(index=2)):
            self.assertEqual(confirmed_break_fill(frame, 11).status, "missing")

    def test_resealed_next_open_or_zero_volume_confirmed_bar_does_not_fill(self):
        self.assertEqual(confirmed_break_fill(minute_frame(35, reopen=11), 11).status, "no_fill")
        frame = minute_frame(35)
        frame.loc[5, "volume"] = 0
        self.assertEqual(confirmed_break_fill(frame, 11).status, "no_fill")

    def test_next_minute_unfinished_volume_cannot_change_open_fill(self):
        before = minute_frame(35)
        after = before.copy()
        after.loc[6, "volume"] = 0  # 09:36分钟总量要等09:37才知道。
        self.assertEqual(confirmed_break_fill(before, 11), confirmed_break_fill(after, 11))
        self.assertEqual(confirmed_break_fill(after, 11).at, dt.time(9, 36))

    def test_as_of_never_exposes_current_unfinished_ohlcv(self):
        frame = minute_frame(35)
        at_0935 = as_of(frame, dt.time(9, 35))
        self.assertEqual(at_0935.trade_time.max().time(), dt.time(9, 34))
        self.assertEqual(at_0935.low.min(), 11)
        self.assertEqual(as_of(frame, dt.time(9, 36)).low.min(), 10.6)
        self.assertTrue(as_of(frame, dt.time(9, 30)).empty)

    def test_future_low_does_not_modify_previous_confirmation(self):
        before = minute_frame(35)
        after = before.copy()
        after.loc[20, "low"] = 0.01
        self.assertEqual(confirmed_break_fill(before, 11), confirmed_break_fill(after, 11))


class TestPublicPriceSeparation(unittest.TestCase):
    def test_sell_reference_is_independent_of_later_buy_reference(self):
        plan = plan_day(holdings=[Holding("A", 2000)], cash=0,
                        prices={"A": 11, "B": 10},
                        valuation_prices={"A": 11, "B": 10},
                        buy_prices={"A": 10.7, "B": 10},
                        sell_prices={"A": 11, "B": 10}, picks=[("A", 5), ("B", 5)])
        self.assertEqual(plan.equity, 22000)
        sale = plan.sells[0]
        self.assertEqual(sale.code, "A")
        self.assertEqual(sale.ref_price, 11)
        self.assertAlmostEqual(sale.amount / sale.shares, 10.9945)
        self.assertGreaterEqual(plan.cash_after, 0)

    def test_stuck_asset_counts_towards_equity_and_exposure(self):
        plan = plan_day(holdings=[Holding("A", 1000)], cash=10000,
                        prices={"B": 10}, valuation_prices={"A": 10, "B": 10},
                        can_sell={"A": False}, picks=[("B", 5)],
                        market_score=1, exposure_fn=lambda _: 0.5)
        self.assertEqual(plan.equity, 20000)
        self.assertEqual(plan.cash_after, 10000)
        self.assertEqual(plan.buys, [])  # 受限资产已占满一万元暴露额度。
        self.assertEqual(plan.final_shares, {"A": 1000})

    def test_stuck_asset_cannot_be_spent_as_cash(self):
        plan = plan_day(holdings=[Holding("A", 1000)], cash=0,
                        prices={"B": 10}, valuation_prices={"A": 10, "B": 10},
                        can_sell={"A": False}, picks=[("B", 5)])
        self.assertEqual(plan.equity, 10000)
        self.assertEqual(plan.buys, [])
        self.assertEqual(plan.cash_after, 0)

    def test_unknown_holding_valuation_stops_instead_of_zero(self):
        with self.assertRaisesRegex(ValueError, "缺估值价"):
            plan_day(holdings=[Holding("A", 1000)], cash=10000,
                     prices={"B": 10}, picks=[("B", 5)])


class TestPublicExecutionEndToEnd(unittest.TestCase):
    def run_waiting(self, lookup, *, max_holdings=1, prob=0):
        frames = {"A.SH": mk(DAYS, [9.8, 10, 10, 10], ups=[11, 10, 11, 11]),
                  "B.SH": mk(DAYS, [10, 10, 10, 10])}
        return run_engine(frames=frames, benchmark_frame=BENCH,
                          picks={DAYS[0]: [("A.SH", 5), ("B.SH", 4)]},
                          cfg=EngineConfig(max_holdings=max_holdings, hard_stop_pct=None,
                                           sealed_fill_prob=prob), break_lookup=lookup)

    def test_waiting_failure_cannot_backfill_0930_replacement(self):
        result = self.run_waiting(lambda *_: BreakLookupResult("no_break"))
        self.assertEqual(result.trades, [])
        self.assertEqual(result.final_value, 100000)
        self.assertTrue(any("不回填09:30替补" in b.reason for b in result.blocked))

    def test_failed_wait_does_not_increase_other_candidates_budget(self):
        result = self.run_waiting(lambda *_: BreakLookupResult("no_break"), max_holdings=2)
        self.assertEqual({t.code for t in result.trades}, {"B.SH"})
        self.assertLess(result.trades[0].shares, 5000)  # B只获得4/9的原始预算，不能拿走A的5/9。
        self.assertGreaterEqual(min(result.cash), 0)

    def test_missing_minute_never_enters_probability_model(self):
        for lookup in (None, lambda *_: None,
                       lambda *_: BreakLookupResult("missing", reason="缺分钟")):
            result = self.run_waiting(lookup, prob=1)
            self.assertEqual(result.trades, [])
            self.assertEqual(result.sealed_fills, [])

    def test_explicit_queue_experiment_fills_at_limit_without_crossing_it(self):
        result = self.run_waiting(lambda *_: BreakLookupResult("no_break"), prob=1)
        buy = result.trades[0]
        self.assertEqual(buy.price, 10)
        self.assertEqual(buy.at, dt.time(9, 50))
        self.assertIn("实验", buy.price_source)
        self.assertEqual(result.sealed_fills[0][2], 1)
        self.assertGreaterEqual(min(result.cash), 0)

    def test_day_high_low_does_not_rewrite_0930_slippage(self):
        # 09:30已知开盘10元；日后最高价10元不能把10.005的模型成交倒改成10元。
        flat = mk(DAYS, [10] * 4, highs=[10] * 4, lows=[10] * 4)
        roomy = mk(DAYS, [10] * 4)
        results = [run_engine(frames={"A.SH": frame}, benchmark_frame=BENCH,
                              picks={DAYS[0]: [("A.SH", 5)]},
                              cfg=EngineConfig(hard_stop_pct=None))
                   for frame in (flat, roomy)]
        self.assertEqual(results[0].trades, results[1].trades)
        self.assertAlmostEqual(results[0].trades[0].price, 10.005, places=10)

    def test_0930_trim_uses_11_not_0945_buy_price(self):
        frames = {"A.SH": mk(DAYS, [9.8, 10, 11, 11], ups=[11, 11, 11, 12]),
                  "B.SH": mk(DAYS, [10, 10, 10, 10])}
        result = run_engine(frames=frames, benchmark_frame=BENCH,
                            picks={DAYS[0]: [("A.SH", 5)],
                                   DAYS[1]: [("A.SH", 5), ("B.SH", 5)]},
                            cfg=EngineConfig(hard_stop_pct=None),
                            break_lookup=lambda *_: (dt.time(9, 45), 10.7))
        sales = [t for t in result.trades if t.side == "sell"]
        self.assertEqual(len(sales), 1)
        self.assertEqual(sales[0].reference_price, 11)
        self.assertAlmostEqual(sales[0].price, 10.9945)
        self.assertEqual(sales[0].at, dt.time(9, 30))
        # 每日现金与已成交持仓按收盘价对账，精度到分以上。
        cash = 100000.0
        held = {}
        for day, equity, actual_cash in zip(result.dates, result.equity, result.cash):
            for trade in [t for t in result.trades if t.date == day]:
                if trade.side == "buy":
                    cash -= trade.amount + trade.fee
                    held[trade.code] = held.get(trade.code, 0) + trade.shares
                else:
                    cash += trade.amount - trade.fee
                    held[trade.code] -= trade.shares
            value = cash + sum(sh * float(frames[c].loc[pd.Timestamp(day), "close"])
                               for c, sh in held.items())
            self.assertAlmostEqual(actual_cash, cash, places=6)
            self.assertAlmostEqual(equity, value, places=6)

    def test_later_fill_keeps_locked_quantity_and_audits_time(self):
        result = self.run_waiting(lambda *_: BreakLookupResult(
            "filled", dt.time(9, 36), 9.8, dt.time(9, 36)))
        buy = result.trades[0]
        self.assertEqual(buy.shares, 9900)  # 开盘10元预留的股数，不能用9.8元增加股数。
        self.assertEqual(buy.at, dt.time(9, 36))
        self.assertEqual(buy.decision_date, dt.date(2024, 1, 2))
        self.assertEqual(buy.reference_price, 9.8)
        self.assertGreaterEqual(min(result.cash), 0)

    def test_suspended_holding_keeps_previous_close_valuation(self):
        stock = mk([DAYS[0], DAYS[1], DAYS[3]], [9.8, 10, 10])
        result = run_engine(frames={"A.SH": stock}, benchmark_frame=BENCH,
                            picks={DAYS[0]: [("A.SH", 5)], DAYS[1]: []},
                            cfg=EngineConfig(hard_stop_pct=None))
        self.assertAlmostEqual(result.equity[2], result.equity[1], places=6)
        self.assertEqual(len(result.trades), 1)


if __name__ == "__main__":
    unittest.main()
