# -*- coding: utf-8 -*-
"""仓位与整手测试。人工手算目标权重、现金约束、阈值和持仓名额。"""

import random
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.sizing import (  # noqa: E402
    DEFAULT_MIN_WEIGHT, DROP_LOW_WEIGHT, DROP_NO_LOT, LOT, MAX_HOLDINGS,
    REASON_CLOSE, REASON_TRIM,
    DayPlan, Holding, allocate_weights, plan_day, round_lot, target_shares,
    tiered_exposure,
)


class TestConstants(unittest.TestCase):
    def test_max_holdings_is_3(self):
        # 组件参数由调用者按场景核验。
        self.assertEqual(MAX_HOLDINGS, 3)

    def test_lot_is_100(self):
        self.assertEqual(LOT, 100)


class TestAllocateWeights(unittest.TestCase):
    """纯按评分归一化 —— 不再做区间截断。"""

    def test_user_example_4_6(self):
        """人工4:6评分场景，目标权重分别为40%和60%"""
        self.check([4, 6], [0.4, 0.6])

    def test_user_example_95_61(self):
        """95 : 61 -> 95/156 : 61/156"""
        self.check([95, 61], [95 / 156, 61 / 156])

    def test_no_clamping_unlike_old_rule(self):
        """🔴 这条钉住「区间表已作废」：9:1 就是 90%:10%，不截断成 60%:40%。"""
        w = allocate_weights([9, 1])
        self.assertAlmostEqual(w[0], 0.9, places=12)
        self.assertAlmostEqual(w[1], 0.1, places=12)

    def test_three_stocks(self):
        """8:1:1 -> 80% : 10% : 10%（旧规则会被截断成 45%/27.5%/27.5%）"""
        self.check([8, 1, 1], [0.8, 0.1, 0.1])

    def test_single_stock_full(self):
        self.assertEqual(allocate_weights([1.0]), [1.0])
        self.assertEqual(allocate_weights([99999.0]), [1.0])

    def test_all_zero_equal(self):
        self.check([0, 0, 0], [1 / 3, 1 / 3, 1 / 3])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            allocate_weights([1.0, -1.0])

    def test_sum_is_one_property(self):
        rng = random.Random(20260911)
        for _ in range(3000):
            n = rng.randint(1, MAX_HOLDINGS)
            scores = [rng.choice([0.0, 0.1, 4.0, 61.0, 95.0, rng.random() * 100])
                      for _ in range(n)]
            w = allocate_weights(scores)
            self.assertAlmostEqual(sum(w), 1.0, places=12,
                                   msg="scores=%s -> %s" % (scores, w))
            order = sorted(range(n), key=lambda i: -scores[i])
            for a, b in zip(order, order[1:]):
                self.assertGreaterEqual(w[a] + 1e-12, w[b])

    def check(self, scores, expected):
        got = allocate_weights(scores)
        self.assertEqual(len(got), len(expected))
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=12,
                                   msg="scores=%s 得到 %s" % (scores, got))


class TestTieredExposure(unittest.TestCase):
    """市场环境评分 -> 总仓位系数。分档数值由调用方给，引擎不猜。"""

    def setUp(self):
        # 🔴 档位要**覆盖到 0**，否则低分因为没有匹配档位就返回 0（空仓），
        #    flat_below 就失去意义了 —— 这是第一版测试踩到的坑。
        self.fn = tiered_exposure(((80, 1.00), (60, 0.80), (40, 0.60), (0, 0.30)),
                                  flat_below=20)

    def test_high_is_full(self):
        self.assertAlmostEqual(self.fn(85), 1.00)
        self.assertAlmostEqual(self.fn(80), 1.00, msg="等于档位线算这一档")

    def test_step_down(self):
        self.assertAlmostEqual(self.fn(70), 0.80)
        self.assertAlmostEqual(self.fn(60), 0.80)
        self.assertAlmostEqual(self.fn(45), 0.60)
        self.assertAlmostEqual(self.fn(40), 0.60)

    def test_low_tier_still_trades(self):
        self.assertAlmostEqual(self.fn(30), 0.30)

    def test_flat_below_threshold(self):
        self.assertAlmostEqual(self.fn(20), 0.30, msg="等于 flat_below 不算低于")
        self.assertAlmostEqual(self.fn(19.99), 0.0)
        self.assertAlmostEqual(self.fn(5), 0.0)

    def test_score_below_all_tiers_returns_zero(self):
        """档位没覆盖到 0 时，低分因为没有匹配档位而返回 0（空仓）。"""
        fn = tiered_exposure(((80, 1.0), (60, 0.8), (40, 0.6)))
        self.assertAlmostEqual(fn(20), 0.0, msg="没有任何档位匹配 -> 空仓")

    def test_no_flat_line(self):
        fn2 = tiered_exposure(((80, 1.0), (0, 0.5)))
        self.assertAlmostEqual(fn2(3), 0.5)

    def test_tiers_given_out_of_order(self):
        fn3 = tiered_exposure(((40, 0.6), (80, 1.0), (60, 0.8)))
        self.assertAlmostEqual(fn3(85), 1.0)
        self.assertAlmostEqual(fn3(70), 0.8)
        self.assertAlmostEqual(fn3(45), 0.6)


class TestRoundLot(unittest.TestCase):
    def test_9900(self):
        # 手算：int(100000/10.005/100)*100 = int(99.95)*100 = 9900
        self.assertEqual(round_lot(100_000.0, 10.005), 9900)

    def test_exact(self):
        self.assertEqual(round_lot(100_000.0, 10.00), 10_000)

    def test_not_enough_for_one_lot(self):
        self.assertEqual(round_lot(1_000.0, 33.33), 0)

    def test_down_round_never_overbuys(self):
        rng = random.Random(7)
        for _ in range(2000):
            amount = rng.random() * 500_000
            price = 0.5 + rng.random() * 200
            s = round_lot(amount, price)
            self.assertEqual(s % LOT, 0)
            self.assertLessEqual(s * price, amount + 1e-6)

    def test_degenerate(self):
        self.assertEqual(round_lot(0.0, 10.0), 0)
        self.assertEqual(round_lot(1000.0, 0.0), 0)
        self.assertEqual(round_lot(-100.0, 10.0), 0)
        self.assertEqual(round_lot(None, 10.0), 0)
        self.assertEqual(round_lot(1000.0, float("nan")), 0)


class TestMinWeightDrop(unittest.TestCase):
    """单票下限 + 买不到一手 -> 剔除 -> 重新归一化。"""

    def test_low_weight_dropped_and_rest_goes_full(self):
        """手算：92 : 8，下限 20%
            权重 [0.92, 0.08] -> 0.08 < 0.20 剔除第二只
            重新归一化 [1.00] -> 第一只满仓
            equity=100,000 价格 10 -> 10000 股；买 10000 股要 100,050+51.03 > 100,000
            砍一手 -> 9900 股 @ 10.005 = 99,049.5 + 50.515245 = 99,100.015245
        """
        plan = plan_day(holdings=[], cash=100_000.0,
                        prices={"A": 10.0, "B": 10.0},
                        picks=[("A", 92.0), ("B", 8.0)],
                        min_weight=0.20)
        self.assertEqual(plan.target_codes, ["A"])
        self.assertEqual([b.code for b in plan.buys], ["A"])
        self.assertEqual(plan.buys[0].shares, 9900)
        self.assertEqual(len(plan.dropped), 1)
        self.assertEqual(plan.dropped[0][0], "B")
        self.assertIn("权重低于下限", plan.dropped[0][1])

    def test_both_kept_when_above_floor(self):
        """手算：61 : 39，总分 100 -> 权重就是 [0.61, 0.39]（都在下限 20% 之上）

            equity=100,000，价格 10.00
              A: 100000 * 0.61 = 61,000 -> 6100 股
              B: 100000 * 0.39 = 39,000 -> 3900 股

            买入支出：
              A: 6100 × 10.005 = 61,030.5
                 费用 = max(30.51525,5) + 0.610305 = 31.125555  -> 61,061.63
              B: 3900 × 10.005 = 39,019.5
                 费用 = max(19.50975,5) + 0.390195 = 19.899945  -> 39,039.40
              合计 100,101.03 > 100,000 现金  -> 必须砍一手

            砍**权重最低**的那笔（B 的 0.39 < A 的 0.61）-> B 降到 3800 股
              B: 3800 × 10.005 = 38,019
                 费用 = max(19.0095,5) + 0.38019 = 19.38969    -> 38,038.39
              合计 99,100.02 <= 100,000  ✅

        ⚠️ 这条手算我写错过两次：
           ① 把权重当成 61/156（那是 95:61 那组的算法）
           ② 忘了现金约束（6100+3900 要 100,101 元，买不起）
           留着提醒「手算也会错，而且会错两次」。
        """
        plan = plan_day(holdings=[], cash=100_000.0,
                        prices={"A": 10.0, "B": 10.0},
                        picks=[("A", 61.0), ("B", 39.0)],
                        min_weight=0.20)
        got = {b.code: b.shares for b in plan.buys}
        self.assertEqual(got, {"A": 6100, "B": 3800}, "B 被砍一手（权重更低）")
        self.assertEqual(plan.dropped, ())

    def test_dropped_when_cannot_afford_one_lot(self):
        """🔴 手算（买不到一手 -> 剔除）：
            equity=100,000，两只票价格都是 500 元，下限 20%
            A: 100000*0.608974 = 60,897 -> /500 = 121 股 -> 整手 100 股 ✅
            B: 100000*0.391026 = 39,103 -> /500 =  78 股 -> 整手   0 股 ❌ 剔除
            重新归一化 A=1.00 -> 100000/500 = 200 股
              200 x 500.25 = 100,050 + 51.0255 = 100,101.03 > 100,000 -> 砍一手
              -> 100 股 (50,025 + 25.51275 = 50,050.51)
        """
        plan = plan_day(holdings=[], cash=100_000.0,
                        prices={"A": 500.0, "B": 500.0},
                        picks=[("A", 61.0), ("B", 39.0)],
                        min_weight=0.20)
        self.assertEqual(plan.target_codes, ["A"])
        self.assertEqual([b.code for b in plan.buys], ["A"])
        self.assertEqual(plan.buys[0].shares, 100)
        self.assertEqual(plan.dropped[0][0], "B")
        self.assertEqual(plan.dropped[0][1], DROP_NO_LOT)

    def test_all_dropped_gives_empty_target(self):
        """本金太小 -> 一只都买不起 -> 空仓（target 为空，也不报错）。"""
        plan = plan_day(holdings=[], cash=500.0,
                        prices={"A": 500.0}, picks=[("A", 95.0)])
        self.assertEqual(plan.target_codes, [])
        self.assertEqual(plan.buys, [])


class TestScoreThreshold(unittest.TestCase):
    """评分阈值：低于阈值不买。数值待上游策略定，这里只验机制。"""

    def test_below_threshold_excluded(self):
        """手算：阈值 60，清单 78 / 76 / 62 / 50
            50 < 60 -> 刷掉；前 3 名是 78 / 76 / 62，都过阈值 -> 买 3 只
        """
        picks = [("A", 78.0), ("B", 76.0), ("C", 62.0), ("D", 50.0)]
        prices = {c: 10.0 for c in "ABCD"}
        plan = plan_day(holdings=[], cash=100_000.0, prices=prices,
                        picks=picks, score_threshold=60.0, min_weight=0.0)
        self.assertEqual(plan.target_codes, ["A", "B", "C"])
        self.assertNotIn("D", plan.target_codes)

    def test_threshold_leaves_two(self):
        """手算：阈值 75，清单 78 / 76 / 62 -> 62 被刷掉 -> 只买 2 只。"""
        picks = [("A", 78.0), ("B", 76.0), ("C", 62.0)]
        prices = {c: 10.0 for c in "ABC"}
        plan = plan_day(holdings=[], cash=100_000.0, prices=prices,
                        picks=picks, score_threshold=75.0, min_weight=0.0)
        self.assertEqual(plan.target_codes, ["A", "B"])

    def test_threshold_leaves_one(self):
        """手算：阈值 77，清单 78 / 76 / 62 -> 只剩 A -> 单票满仓。"""
        picks = [("A", 78.0), ("B", 76.0), ("C", 62.0)]
        prices = {c: 10.0 for c in "ABC"}
        plan = plan_day(holdings=[], cash=100_000.0, prices=prices,
                        picks=picks, score_threshold=77.0)
        self.assertEqual(plan.target_codes, ["A"])
        self.assertEqual(plan.target_weights["A"], 1.0)

    def test_no_threshold_means_all_pass(self):
        picks = [("A", 78.0), ("B", 1.0)]
        plan = plan_day(holdings=[], cash=100_000.0,
                        prices={"A": 10.0, "B": 10.0}, picks=picks,
                        min_weight=0.0)
        self.assertEqual(plan.target_codes, ["A", "B"])


class TestMarketExposure(unittest.TestCase):
    """市场环境评分 -> 总仓位系数；低于空仓线 -> 清仓且不买（用户选甲）。"""

    def setUp(self):
        self.fn = tiered_exposure(((80, 1.00), (60, 0.80), (40, 0.60)),
                                  flat_below=20)

    def test_single_stock_full_when_env_high(self):
        """环境 85 -> 系数 1.0 -> 单票满仓（手算同 接口说明：9900 股）"""
        plan = plan_day(holdings=[], cash=100_000.0, prices={"A": 10.0},
                        picks=[("A", 95.0)], market_score=85.0,
                        exposure_fn=self.fn)
        self.assertAlmostEqual(plan.exposure, 1.00)
        self.assertEqual(plan.buys[0].shares, 9900)

    def test_single_stock_80pct_when_env_mid(self):
        """手算：环境 70 -> 系数 0.80
            equity=100,000 * 0.80 = 80,000 -> /10 = 8000 股
            8000 x 10.005 = 80,040 + 40.04004 = 80,080.04 <= 100,000 ✅
        """
        plan = plan_day(holdings=[], cash=100_000.0, prices={"A": 10.0},
                        picks=[("A", 95.0)], market_score=70.0,
                        exposure_fn=self.fn)
        self.assertAlmostEqual(plan.exposure, 0.80)
        self.assertEqual(plan.buys[0].shares, 8000)

    def test_single_stock_60pct_when_env_low(self):
        """手算：环境 45 -> 系数 0.60 -> 100,000*0.6/10 = 6000 股"""
        plan = plan_day(holdings=[], cash=100_000.0, prices={"A": 10.0},
                        picks=[("A", 95.0)], market_score=45.0,
                        exposure_fn=self.fn)
        self.assertAlmostEqual(plan.exposure, 0.60)
        self.assertEqual(plan.buys[0].shares, 6000)

    def test_flat_clears_holdings_and_does_not_buy(self):
        """🔴 用户选甲：环境低于空仓线 -> 手上有的也在当天开盘清掉，且不买。

        手算：持有 A 1000 股、B 2000 股，价格都是 10.00，现金 5,000
              环境 15 < 20 -> exposure = 0 -> 两只全清，不买
              A: 1000 x 9.995 = 9,995   费用 max(4.9975,5)+4.9975+0.09995 = 10.09745
              B: 2000 x 9.995 = 19,990  费用 max(9.995,5)+9.995+0.1999 = 20.1899
        """
        plan = plan_day(holdings=[Holding("A", 1000), Holding("B", 2000)],
                        cash=5_000.0,
                        prices={"A": 10.0, "B": 10.0},
                        picks=[("A", 95.0), ("B", 90.0)],
                        market_score=15.0, exposure_fn=self.fn)
        self.assertTrue(plan.flat)
        self.assertAlmostEqual(plan.exposure, 0.0)
        self.assertEqual(plan.buys, [], "空仓时不许买")
        self.assertEqual(sorted(s.code for s in plan.sells), ["A", "B"])
        self.assertTrue(all(s.reason == "环境空仓" for s in plan.sells))
        self.assertEqual(plan.final_shares, {})
        self.assertEqual(plan.holdings_after, 0)

    def test_flat_cannot_sell_suspended(self):
        """空仓时停牌（无价）的票卖不掉 -> 留着，不报错。"""
        plan = plan_day(holdings=[Holding("A", 1000), Holding("S", 500)],
                        cash=1_000.0, prices={"A": 10.0},
                        valuation_prices={"A": 10.0, "S": 10.0},
                        picks=[("A", 95.0)],
                        market_score=10.0, exposure_fn=self.fn)
        self.assertTrue(plan.flat)
        self.assertEqual([s.code for s in plan.sells], ["A"])
        self.assertEqual(plan.final_shares, {"S": 500})
        self.assertEqual(plan.stuck_codes, ("S",))

    def test_no_env_params_means_full(self):
        plan = plan_day(holdings=[], cash=100_000.0, prices={"A": 10.0},
                        picks=[("A", 95.0)])
        self.assertAlmostEqual(plan.exposure, 1.0)
        self.assertFalse(plan.flat)

    def test_bad_exposure_raises(self):
        with self.assertRaises(ValueError):
            plan_day(holdings=[], cash=1000.0, prices={"A": 10.0},
                     picks=[("A", 1.0)], market_score=50.0,
                     exposure_fn=lambda s: 1.5)


class TestRotationAndLimits(unittest.TestCase):
    """换股（该组件约定：推荐全是新票就换）+ 3 只硬上限。"""

    def test_swap_when_all_picks_are_new(self):
        """目标清单全部换成新标的时，旧持仓按交易约束尝试退出。

        持 A/B/C 三只，推荐 D/E/F -> 清掉 A/B/C，买入 D/E/F，仍然 3 只。
        """
        held = [Holding(c, 1000) for c in "ABC"]
        prices = {c: 10.0 for c in "ABCDEF"}
        picks = [("D", 95.0), ("E", 90.0), ("F", 85.0)]
        plan = plan_day(holdings=held, cash=10_000.0, prices=prices,
                        picks=picks, min_weight=0.0)
        self.assertEqual(plan.target_codes, ["D", "E", "F"])
        self.assertEqual(sorted(s.code for s in plan.sells), ["A", "B", "C"])
        self.assertEqual(plan.holdings_after, 3)
        self.assertLessEqual(plan.holdings_after, MAX_HOLDINGS)

    def test_keep_when_still_on_the_list(self):
        """还在榜单上就留着，只做加仓/减仓（掉出榜单才卖 —— 无 3 天硬上限）。"""
        held = [Holding("A", 1000)]
        plan = plan_day(holdings=held, cash=90_000.0,
                        prices={"A": 10.0, "B": 10.0},
                        picks=[("A", 95.0), ("B", 90.0)], min_weight=0.0)
        self.assertIn("A", plan.target_codes)
        self.assertNotIn("A", [s.code for s in plan.sells if s.reason == "清仓"])

    def test_holdings_over_limit_raises(self):
        hs = [Holding("S%d" % i, 100) for i in range(4)]
        with self.assertRaises(ValueError):
            plan_day(holdings=hs, cash=100_000.0,
                     prices={"S%d" % i: 10.0 for i in range(4)},
                     picks=[("S%d" % i, 9.0 - i) for i in range(4)])

    def test_more_picks_than_slots_takes_top3(self):
        picks = [("S%d" % i, float(10 - i)) for i in range(6)]
        plan = plan_day(holdings=[], cash=400_000.0,
                        prices={"S%d" % i: 10.0 for i in range(6)},
                        picks=picks, min_weight=0.0)
        self.assertEqual(plan.target_codes, ["S0", "S1", "S2"])
        self.assertEqual(plan.holdings_after, 3)


class TestRegressionAfterIndependentReview(unittest.TestCase):
    """独立复核抓出的 bug 的回归测试（口径改版后重算）。"""

    def test_suspended_holding_still_occupies_a_slot(self):
        """🔴 BUG-1 回归：卖不掉的持仓（停牌 -> 无价格）必须继续占名额。"""
        held = [Holding(c, 1000) for c in "ABCD"] if MAX_HOLDINGS >= 4 else \
               [Holding(c, 1000) for c in "ABC"]
        stuck = "A"
        prices = {c: 10.0 for c in "BCDEFG"}          # stuck 无价
        picks = [("D", 10.0), ("E", 9.0), ("F", 8.0)]
        plan = plan_day(holdings=held, cash=100_000.0, prices=prices,
                        valuation_prices={**prices, **{h.code: 10.0 for h in held}},
                        picks=picks, min_weight=0.0)
        self.assertLessEqual(plan.holdings_after, MAX_HOLDINGS)
        self.assertEqual(plan.final_shares.get(stuck), 1000, "停牌的必须还在")
        self.assertIn(stuck, plan.stuck_codes)
        self.assertEqual(len(plan.target_codes), MAX_HOLDINGS - 1,
                         "停牌占掉一个名额")

    def test_never_exceeds_limit_when_prices_missing(self):
        """🔴 BUG-1 回归（属性测试）：随机抹掉价格（模拟停牌）后仍不许超上限。"""
        rng = random.Random(20260911)
        universe = ["S%02d" % i for i in range(12)]
        for trial in range(3000):
            n_held = rng.randint(0, MAX_HOLDINGS)
            held_codes = rng.sample(universe, n_held)
            holdings = [Holding(c, rng.choice([100, 1000, 5000])) for c in held_codes]
            cash = rng.choice([0.0, 10_000.0, 100_000.0, 500_000.0])
            prices = {c: round(rng.uniform(2.0, 80.0), 2) for c in universe}
            valuations = dict(prices)  # 估值保留，删除的只是当天成交价。
            for c in universe:
                if rng.random() < 0.25:
                    prices.pop(c, None)
            picks = [(c, round(rng.uniform(0.0, 100.0), 3))
                     for c in rng.sample(universe, rng.randint(0, 8))]
            plan = plan_day(holdings=holdings, cash=cash, prices=prices,
                            valuation_prices=valuations, picks=picks, min_weight=0.0)
            ctx = "trial=%d held=%s cash=%s" % (trial, held_codes, cash)
            self.assertLessEqual(plan.holdings_after, MAX_HOLDINGS, ctx)
            self.assertGreaterEqual(plan.cash_after, -1e-6, "超支 " + ctx)

    def test_trim_proceeds_fund_other_buys(self):
        """🔴 BUG-2 回归：减仓回款必须计入可用现金。

        手算：持有 A 2000 股 @10.00，现金 0，推荐 A/B 各 50 分 -> 权重 [0.5, 0.5]
            equity = 0 + 2000*10 = 20,000
            A 目标 1000 股 -> 减仓 1000 股 @9.995 = 9,995
              费用 = 佣金 max(4.9975,5)=5.0 + 印花税 4.9975 + 过户费 0.09995
                   = 10.09745  净回款 9,984.90255
            B 目标 1000 股 -> 买 1000 股要 10,005 + 5.10255 = 10,010.10 > 9,984.90
              砍一手 -> 900 股：9,004.5 + 5.090045 = 9,009.590045 ✅
            cash_after = 9,984.90255 - 9,009.590045 = 975.312505
        """
        plan = plan_day(holdings=[Holding("A", 2000)], cash=0.0,
                        prices={"A": 10.0, "B": 10.0},
                        picks=[("A", 50.0), ("B", 50.0)], min_weight=0.0)
        self.assertEqual(len(plan.buys), 1, "减仓回款必须能支持买入 B")
        self.assertEqual(plan.buys[0].code, "B")
        self.assertEqual(plan.buys[0].shares, 900)
        self.assertAlmostEqual(plan.cash_after, 975.312505, places=5)

    def test_sell_codes_blocks_buying_even_if_not_held(self):
        """🔴 BUG-3 回归：sell_codes 里的票即使没持有，当天也不许买进。"""
        plan = plan_day(holdings=[], cash=100_000.0, prices={"A": 10.0},
                        picks=[("A", 90.0)], sell_codes=["A"])
        self.assertEqual(plan.buys, [])
        self.assertEqual(plan.target_codes, [])

    def test_cash_shortage_cuts_lowest_weight_first(self):
        """现金不足时优先砍权重最低的那笔（不是金额最大的）。"""
        plan = plan_day(holdings=[], cash=15_000.0,
                        prices={"A": 10.0, "B": 10.0},
                        picks=[("A", 90.0), ("B", 10.0)], min_weight=0.0)
        got = {b.code: b.shares for b in plan.buys}
        self.assertGreaterEqual(plan.cash_after, 0.0)
        self.assertGreater(got["A"], got["B"], "高分的那只应该买得更多")

    def test_zero_share_holding_not_counted(self):
        hs = [Holding("S%d" % i, 0 if i < 3 else 100) for i in range(6)]
        plan = plan_day(holdings=hs, cash=100_000.0,
                        prices={"S%d" % i: 10.0 for i in range(6)},
                        picks=[("S0", 9.0)])
        self.assertLessEqual(plan.holdings_after, MAX_HOLDINGS)

    def test_nan_price_treated_as_no_price(self):
        plan = plan_day(holdings=[], cash=100_000.0,
                        prices={"A": float("nan"), "B": None, "C": 10.0},
                        picks=[("A", 9.0), ("B", 8.0), ("C", 7.0)])
        self.assertEqual([b.code for b in plan.buys], ["C"])


class TestBug2StuckSlotLeak(unittest.TestCase):
    """人工卖出受限场景：已有持仓继续占名额。

    强制退出计划未能成交时，不能空出名额再买入其他股票突破持仓上限。"""

    def test_stuck_and_in_sell_codes_still_occupies_slot(self):
        held = [Holding(c, 1000) for c in "ABC"]
        prices = {"D": 10.0, "E": 10.0, "F": 10.0}      # A/B/C 全无价
        picks = [("D", 10.0), ("E", 9.0), ("F", 8.0)]
        plan = plan_day(holdings=held, cash=100_000.0, prices=prices,
                        valuation_prices={**prices, **{h.code: 10.0 for h in held}},
                        picks=picks, sell_codes=["A"], min_weight=0.0)
        self.assertLessEqual(plan.holdings_after, MAX_HOLDINGS,
                             "卖不掉 + 被点名清仓 的票漏掉了名额")
        self.assertEqual(plan.final_shares.get("A"), 1000, "卖不掉的必须还在")
        self.assertIn("A", plan.stuck_codes)

    def test_stuck_set_is_decided_only_by_price_availability(self):
        """stuck 只由「当天有没有价」决定，与 to_close 无关。"""
        held = [Holding(c, 1000) for c in "ABC"]
        plan = plan_day(holdings=held, cash=1_000.0, prices={"D": 10.0},
                        valuation_prices={c: 10.0 for c in "ABCD"},
                        picks=[("D", 10.0)], sell_codes=["A", "B"],
                        min_weight=0.0)
        self.assertEqual(plan.stuck_codes, ("A", "B", "C"))

    def test_stopped_out_but_unsellable_does_not_free_a_slot(self):
        """硬止损票当天跌停封死（无价）-> 顺延，但**名额不能放出去**。"""
        held = [Holding(c, 1000) for c in "ABC"]
        prices = {"D": 10.0, "E": 10.0, "F": 10.0}
        plan = plan_day(holdings=held, cash=100_000.0, prices=prices,
                        valuation_prices={**prices, **{h.code: 10.0 for h in held}},
                        picks=[("D", 10.0), ("E", 9.0), ("F", 8.0)],
                        sell_codes=["A"],
                        sell_reasons={"A": "硬止损 -9.00%（跌停封死，顺延）"},
                        min_weight=0.0)
        self.assertEqual(plan.buys, [], "三个名额全被占住，一只都不许买")
        self.assertEqual(plan.holdings_after, MAX_HOLDINGS)


class TestSellReasons(unittest.TestCase):
    """出场原因覆盖 —— 把「硬止损」带进成交流水（引擎 -> plan_day -> Trade.reason）。"""

    def test_reason_override_applied(self):
        plan = plan_day(holdings=[Holding("A", 1000)], cash=1_000.0,
                        prices={"A": 9.0}, picks=[], sell_codes=["A"],
                        sell_reasons={"A": "硬止损 -6.05%（开盘 9.400 / 开仓 10.005）"})
        self.assertEqual(len(plan.sells), 1)
        self.assertEqual(plan.sells[0].reason,
                         "硬止损 -6.05%（开盘 9.400 / 开仓 10.005）")

    def test_default_reason_is_close_when_no_override(self):
        plan = plan_day(holdings=[Holding("A", 1000)], cash=1_000.0,
                        prices={"A": 9.0}, picks=[], sell_codes=["A"])
        self.assertEqual(plan.sells[0].reason, REASON_CLOSE)

    def test_override_does_not_leak_into_trim(self):
        """只覆盖「强制清仓」那一笔；同一份计划里的**减仓**仍是「减仓」。"""
        plan = plan_day(
            holdings=[Holding("A", 1000), Holding("B", 1000)],
            cash=0.0,
            prices={"A": 10.0, "B": 1000.0},
            picks=[("B", 10.0)],
            sell_codes=["A"],
            sell_reasons={"A": "硬止损 -6.05%"},
            market_score=50.0, exposure_fn=lambda s: 0.5,
        )
        by_code = {s.code: s.reason for s in plan.sells}
        self.assertEqual(by_code.get("A"), "硬止损 -6.05%")
        self.assertEqual(by_code.get("B"), REASON_TRIM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
