# -*- coding: utf-8 -*-
r"""第 3 层 · 交易规则拦截单测 —— 全部手算场景，不使用任何历史数据。

运行：.venvScriptspython.exe -X utf8 -B tests	est_bt_rules.py
"""

import datetime as dt
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.rules import (  # noqa: E402
    BLOCK_SEALED_DOWN, BLOCK_SEALED_UP, BLOCK_ST, BLOCK_SUSPENDED,
    BLOCK_SOLD_TODAY, BLOCK_T1, BLOCK_T_BOARD_LATE, DEADLINE,
    EXIT_STOP_LOSS, OPEN_WINDOW_MINUTES, STATUS_SEALED_UP, STATUS_TURNOVER,
    STOP_LOSS_DEFAULT, BarSnapshot, RuleEngine, stop_loss_reason,
)

D1 = dt.date(2024, 1, 2)
D2 = dt.date(2024, 1, 3)


def bar(**kw):
    kw.setdefault("code", "600000.SH")
    return BarSnapshot(**kw)


class TestDeadline(unittest.TestCase):
    def test_deadline_is_0950(self):
        # 实验参数应按调用场景核验。
        self.assertEqual(OPEN_WINDOW_MINUTES, 20)
        self.assertEqual(DEADLINE, dt.time(9, 50))


class TestBuyFilters(unittest.TestCase):

    def setUp(self):
        self.e = RuleEngine()

    def test_st_blocked(self):
        v = self.e.check_buy(bar(is_st=True), D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_ST)

    def test_st_has_highest_priority(self):
        """ST 优先级最高：同时是一字板，报的原因也必须是 ST（接口说明）。"""
        b = bar(is_st=True, up_limit=11.0, open=11.0, high=11.0,
                low=11.0, close=11.0, limit_status=STATUS_SEALED_UP)
        self.assertEqual(self.e.check_buy(b, D1).reason, BLOCK_ST)

    def test_suspended_blocked(self):
        v = self.e.check_buy(bar(is_suspended=True), D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_SUSPENDED)

    def test_sealed_up_by_authoritative_field(self):
        """乙：limit_list_ths.status == 一字板。"""
        v = self.e.check_buy(bar(limit_status=STATUS_SEALED_UP), D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_SEALED_UP)

    def test_sealed_up_by_ohlc_fallback(self):
        """甲兜底：没有权威字段时，开=高=低=收=涨停价 判一字板。"""
        b = bar(up_limit=11.0, open=11.0, high=11.0, low=11.0, close=11.0)
        v = self.e.check_buy(b, D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_SEALED_UP)

    def test_authoritative_field_wins_over_ohlc(self):
        """权威字段说「换手板」→ 即使 OHLC 长得像一字板，也放行（乙优先）。"""
        b = bar(limit_status=STATUS_TURNOVER, up_limit=11.0,
                open=11.0, high=11.0, low=11.0, close=11.0)
        self.assertTrue(self.e.check_buy(b, D1))

    def test_unknown_up_limit_does_not_guess(self):
        """没有涨停价 → 不敢判一字板（规则层绝不猜值）。"""
        b = bar(open=11.0, high=11.0, low=11.0, close=11.0)
        self.assertTrue(self.e.check_buy(b, D1))

    def test_t_board_breaks_early_is_buyable(self):
        """T字板 09:35 开板 → 在 20 分钟窗口内 → 可买。"""
        b = bar(up_limit=11.0, open=11.0, high=11.0, low=10.5, close=11.0,
                first_break_time=dt.time(9, 35))
        self.assertTrue(self.e.check_buy(b, D1))

    def test_t_board_breaks_exactly_at_deadline_is_buyable(self):
        """边界：正好 09:50 开板 → 可买（含边界）。"""
        b = bar(up_limit=11.0, open=11.0, high=11.0, low=10.5, close=11.0,
                first_break_time=dt.time(9, 50))
        self.assertTrue(self.e.check_buy(b, D1))

    def test_t_board_breaks_one_second_late_is_blocked(self):
        """边界：09:50:01 开板 → 买不进。"""
        b = bar(up_limit=11.0, open=11.0, high=11.0, low=10.5, close=11.0,
                first_break_time=dt.time(9, 50, 1))
        v = self.e.check_buy(b, D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_T_BOARD_LATE)

    def test_t_board_breaks_late_is_blocked(self):
        b = bar(up_limit=11.0, open=11.0, high=11.0, low=10.5, close=11.0,
                first_break_time=dt.time(13, 20))
        self.assertEqual(self.e.check_buy(b, D1).reason, BLOCK_T_BOARD_LATE)

    def test_t_board_by_authoritative_field(self):
        """人工显式T字板状态，核验输入状态判定；不依赖供应商数据样本。"""
        late = bar(limit_status="T字板", up_limit=11.0, open=10.5,
                   high=11.0, low=10.4, close=11.0,
                   first_break_time=dt.time(10, 30))
        self.assertEqual(self.e.check_buy(late, D1).reason, BLOCK_T_BOARD_LATE)

        early = bar(limit_status="T字板", up_limit=11.0, open=10.5,
                    high=11.0, low=10.4, close=11.0,
                    first_break_time=dt.time(9, 35))
        self.assertTrue(self.e.check_buy(early, D1))

    def test_status_only_meaningful_on_lianban_pool(self):
        """人工缺失状态场景：不能仅凭缺失状态断言股票可交易。"""
        # 权威字段缺失（非连扳池），但行情显示是一字板 -> 甲兜底必须拦住
        b = bar(limit_status=None, up_limit=11.0, open=11.0,
                high=11.0, low=11.0, close=11.0)
        self.assertEqual(self.e.check_buy(b, D1).reason, BLOCK_SEALED_UP)

    def test_normal_stock_buyable(self):
        b = bar(up_limit=11.0, open=10.2, high=10.8, low=10.1, close=10.5)
        self.assertTrue(self.e.check_buy(b, D1))

    def test_not_buy_back_same_day_after_sell(self):
        """当天卖掉的票，当天不再买回。"""
        b = bar(up_limit=11.0, open=10.2, high=10.8, low=10.1, close=10.5)
        self.e.record_sell("600000.SH", D1)
        v = self.e.check_buy(b, D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_SOLD_TODAY)
        # 第二天就可以买了
        self.assertTrue(self.e.check_buy(b, D2))


class TestSellFilters(unittest.TestCase):

    def setUp(self):
        self.e = RuleEngine()

    def test_t1_blocks_same_day_sell(self):
        """★ 核心规则：当日买入不可卖。"""
        b = bar(open=10.2, high=10.8, low=10.1, close=10.5)
        self.e.record_buy("600000.SH", D1)
        v = self.e.check_sell(b, D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_T1)

    def test_t1_allows_next_day_sell(self):
        b = bar(open=10.2, high=10.8, low=10.1, close=10.5)
        self.e.record_buy("600000.SH", D1)
        self.assertTrue(self.e.check_sell(b, D2), "次日必须能卖")

    def test_t1_has_highest_priority_on_sell(self):
        """同时 T+1 且停牌 → 报 T+1。"""
        b = bar(is_suspended=True)
        self.e.record_buy("600000.SH", D1)
        self.assertEqual(self.e.check_sell(b, D1).reason, BLOCK_T1)

    def test_suspended_blocks_sell(self):
        """停牌 = 冻结，卖不出，等复牌。"""
        v = self.e.check_sell(bar(is_suspended=True), D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_SUSPENDED)

    def test_sealed_down_blocks_sell(self):
        """封死跌停 → 挂单也没人接，只能顺延。"""
        b = bar(down_limit=9.0, open=9.0, high=9.0, low=9.0, close=9.0)
        v = self.e.check_sell(b, D1)
        self.assertFalse(v)
        self.assertEqual(v.reason, BLOCK_SEALED_DOWN)

    def test_down_limit_opened_is_sellable(self):
        """跌停打开过（最高价高于跌停价）→ 能卖。"""
        b = bar(down_limit=9.0, open=9.0, high=9.3, low=9.0, close=9.1)
        self.assertTrue(self.e.check_sell(b, D1))

    def test_normal_sell_ok(self):
        self.assertTrue(self.e.check_sell(bar(open=10.2, close=10.5), D1))

    def test_record_sell_clears_t1(self):
        """卖出后 T+1 约束解除（下次买入会重新记录买入日）。"""
        self.e.record_buy("600000.SH", D1)
        self.e.record_sell("600000.SH", D2)
        self.assertIsNone(self.e.buy_date("600000.SH"))
        self.assertTrue(self.e.check_sell(bar(), D2))


class TestFullDayFlow(unittest.TestCase):
    """把一天的流程走一遍：买入 → 当日想卖被拦 → 次日卖出。"""

    def test_buy_then_same_day_sell_blocked_then_next_day_ok(self):
        e = RuleEngine()
        normal = bar(up_limit=11.0, open=10.2, high=10.8, low=10.1, close=10.5)

        self.assertTrue(e.check_buy(normal, D1), "D1 应该能买")
        e.record_buy("600000.SH", D1)

        v = e.check_sell(normal, D1)
        self.assertFalse(v, "D1 当日不能卖")
        self.assertEqual(v.reason, BLOCK_T1)

        self.assertTrue(e.check_sell(normal, D2), "D2 应该能卖")
        e.record_sell("600000.SH", D2)
        self.assertFalse(e.check_buy(normal, D2), "D2 卖完不能当天买回")


class TestFilterBuyable(unittest.TestCase):
    def test_batch_verdicts(self):
        e = RuleEngine()
        bars = [
            bar(code="A", up_limit=11.0, open=10.0, high=10.5, low=9.9, close=10.2),
            bar(code="B", is_st=True),
            bar(code="C", limit_status=STATUS_SEALED_UP),
            bar(code="D", is_suspended=True),
        ]
        r = e.filter_buyable(bars, D1)
        self.assertTrue(r["A"])
        self.assertEqual(r["B"].reason, BLOCK_ST)
        self.assertEqual(r["C"].reason, BLOCK_SEALED_UP)
        self.assertEqual(r["D"].reason, BLOCK_SUSPENDED)


class TestHardStopLoss(unittest.TestCase):
    """人工开仓成本与开盘价场景，检查配置止损及缺值边界。"""

    E = 10.005          # 开仓价 = 10.00 参考价 + 0.05% 滑点（真实成交价）

    # ── 触发 ──
    def test_trigger_at_minus_6_pct(self):
        # 9.40 / 10.005 - 1 = -0.0604698 = -6.05% <= -5% -> 触发
        r = stop_loss_reason(self.E, 9.40)
        self.assertIsNotNone(r)
        self.assertTrue(r.startswith(EXIT_STOP_LOSS))
        self.assertIn("-6.05%", r)

    def test_trigger_at_exact_boundary(self):
        # 精确 -5%：10.005 × 0.95 = 9.50475 -> 触发（判据是 <=，含边界）
        self.assertIsNotNone(stop_loss_reason(self.E, 9.50475))

    def test_not_trigger_just_above_boundary(self):
        """🔴 9.505 看着像「跌了 5%」，实际 9.505/10.005-1 = -4.9975%。

        判据是原始浮点比较，**不许四舍五入成 -5% 就拦下**。
        """
        self.assertIsNone(stop_loss_reason(self.E, 9.505))
        self.assertIsNone(stop_loss_reason(self.E, 9.51))

    def test_not_trigger_when_flat_or_up(self):
        self.assertIsNone(stop_loss_reason(self.E, 10.005), "平盘不该止损")
        self.assertIsNone(stop_loss_reason(self.E, 11.0), "盈利不该止损")
        self.assertIsNone(stop_loss_reason(self.E, 10.00))

    # ── 可配置 / 可关闭 ──
    def test_default_is_minus_5_pct(self):
        self.assertEqual(STOP_LOSS_DEFAULT, -0.05)

    def test_none_means_disabled(self):
        """🔴 传 None = 整个功能关闭；**不是**设成 0（设 0 会退化成「开盘不涨就砍」）。"""
        self.assertIsNone(stop_loss_reason(self.E, 1.0, None))
        self.assertIsNone(stop_loss_reason(self.E, 0.01, None))

    def test_custom_threshold(self):
        # 阈值 -3%：
        #   9.75 / 10.005 - 1 = -2.55%  >  -3% -> 不触发
        #   9.70 / 10.005 - 1 = -3.05%  <= -3% -> 触发
        self.assertIsNone(stop_loss_reason(self.E, 9.75, -0.03))
        self.assertIsNotNone(stop_loss_reason(self.E, 9.70, -0.03))

    # ── 数据铁律：不知道就不猜 ──
    def test_never_guesses_when_entry_price_missing(self):
        """不知道开仓价 -> 绝不触发（宁可不止损，也不许凭空造一个成本价）。"""
        for bad in (None, 0.0, -1.0, float("nan"), float("inf"), "x"):
            self.assertIsNone(stop_loss_reason(bad, 5.0), "entry=%r 不该触发" % (bad,))

    def test_never_guesses_when_open_missing(self):
        """当天没有开盘价（停牌/缺数据）-> 顺延，不触发。"""
        for bad in (None, 0.0, -1.0, float("nan"), "x"):
            self.assertIsNone(stop_loss_reason(self.E, bad), "open=%r 不该触发" % (bad,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
