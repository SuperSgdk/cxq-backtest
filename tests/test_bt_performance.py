# -*- coding: utf-8 -*-
"""绩效指标测试。全部为人工净值和交易序列，预期值按公式独立推导。
核验边界、样本标准差和可选第三方库差异。"""

import math
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.performance import (  # noqa: E402
    DEFAULT_HORIZONS, TRADING_DAYS_PER_YEAR,
    DecayCurve, PerformanceAnalyzer, PerformanceReport, Trade, Turnover,
    active_returns, annual_return, annual_volatility, calmar_ratio, cross_check,
    decay_curve, evaluate, from_trades, information_ratio, max_drawdown,
    max_drawdown_from_returns, sharpe_from_annual, sharpe_ratio, sortino_ratio,
    to_returns, total_return, tracking_error, turnover,
)

SQRT252 = math.sqrt(252)


# ══════════════════ 手算样本 ══════════════════

# 人工样本A：四个权益点构成三个收益期，涨跌再涨。
EQ_A = [1.0, 1.1, 1.05, 1.2]
R_A = [1.0 / 10.0, -1.0 / 22.0, 1.0 / 7.0]

#: 样本 B：均值为 0 的四天，用来把 Sharpe / Sortino 的 0 钉死
R_B = [0.01, -0.01, 0.01, -0.01]

#: 样本 C：252 天每天 +0.001（年化系数正好是 1，年化收益 = 累计收益）
R_C = [0.001] * 252

#: 样本 D：策略 + 基准（算超额收益 / IR / 跟踪误差），长度必须与收益一一对应
R_P = [0.02, -0.01, 0.03, 0.00]
R_BM = [0.01, -0.02, 0.01, 0.01]

#: 样本 A 的基准：EQ_A 是 4 个净值点 -> 只有 3 个收益，基准也必须是 3 个
R_BM3 = [0.01, -0.02, 0.01]

#: 样本 E：金额口径净值（算换手率）。
#: 注意交易日数 n = 净值点数 - 1：EQ_T 是 3 个点 -> n = 2，EQ_E 是 4 个点 -> n = 3
EQ_T = [100000.0, 105000.0, 110000.0]
EQ_E = [100000.0, 110000.0, 105000.0, 120000.0]

#: 样本 F：持仓周期衰减曲线用的价格序列（11 个交易日）
DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
        "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15",
        "2024-01-16"]
PX = [10.0, 10.5, 11.0, 10.0, 12.0, 13.0, 13.0, 13.0, 13.0, 13.0, 14.0]
PRICES = {"600000.SH": dict(zip(DAYS, PX))}


class PerfTest(unittest.TestCase):
    """公共断言工具。"""

    def assertClose(self, got, want, rel=1e-12, msg=""):
        """相对误差比较（浮点只能做到 1e-15 量级，绝对误差卡不住大数字）。"""
        if want == 0.0:
            self.assertLessEqual(abs(got), 1e-15, msg or "%r 应该等于 0" % (got,))
            return
        diff = abs(got - want)
        self.assertLessEqual(
            diff, rel * abs(want),
            msg or "%r != %r（相对误差 %.3g）" % (got, want, diff / abs(want)))

    def assertNan(self, got, msg=""):
        self.assertTrue(math.isnan(got), msg or "%r 应该是 nan" % (got,))


# ══════════════════ 1. 净值 -> 收益 ══════════════════

class TestToReturns(PerfTest):

    def test_length_is_points_minus_one(self):
        # 期初 1 个点不产生收益：4 个点 -> 3 个收益
        self.assertEqual(len(to_returns(EQ_A)), 3)

    def test_values(self):
        # 手算：1.1/1.0 - 1 = 1/10；1.05/1.1 - 1 = -1/22；1.2/1.05 - 1 = 1/7
        got = to_returns(EQ_A)
        for g, w in zip(got, R_A):
            self.assertClose(g, w)

    def test_rejects_single_point(self):
        with self.assertRaises(ValueError):
            to_returns([1.0])

    def test_rejects_non_positive_equity(self):
        # 净值归零/为负一定是数据接错了，不许静默算下去
        with self.assertRaises(ValueError):
            to_returns([1.0, 0.0, 1.2])


# ══════════════════ 2. 累计收益 / 年化收益 ══════════════════

class TestReturn(PerfTest):

    def test_total_return_A(self):
        # 手算：1.1 * (21/22) * (8/7) = 1.2  ->  R = 0.2
        self.assertClose(total_return(R_A), 0.2)

    def test_total_return_B(self):
        # 手算：1.01 * 0.99 * 1.01 * 0.99 = 0.99980001  ->  R = -0.00019999
        self.assertClose(total_return(R_B), -0.00019999)

    def test_annual_return_252_days_equals_total(self):
        # 手算：n = 252 时年化系数 252/252 = 1，(1+R)**1 - 1 = R
        #       精确值 1.001**252 - 1 = 0.286434044376187746
        #       252 次浮点连乘会积掉 ~1.2e-13 的相对误差，所以这一条放到 1e-11
        self.assertClose(annual_return(R_C), 0.286434044376187746, rel=1e-11)
        self.assertEqual(annual_return(R_C), total_return(R_C),
                         "n = 252 时年化收益必须与累计收益逐位相同")

    def test_annual_return_A(self):
        # 手算：(1.2) ** (252/3) - 1 = 1.2 ** 84 - 1 = 4479448.738824572412728
        self.assertClose(annual_return(R_A), 4479448.7388245724)

    def test_annual_return_B(self):
        # 手算：0.99980001 ** 63 - 1 = -0.01252157450152481
        self.assertClose(annual_return(R_B), -0.01252157450152481)

    def test_rejects_total_loss(self):
        with self.assertRaises(ValueError):
            annual_return([-1.0, 0.5])


# ══════════════════ 3. 年化波动率 ══════════════════

class TestVolatility(PerfTest):

    def test_volatility_B(self):
        # 手算：mean = 0，平方偏差和 = 4 * 1e-4，ddof=1 -> 4e-4/3
        #       std = 0.01*sqrt(4/3)，AV = 0.01*sqrt(4/3)*sqrt(252) = 0.01*sqrt(336)
        #       = 0.183303027798233600
        self.assertClose(annual_volatility(R_B), 0.183303027798233600)

    def test_volatility_A(self):
        # 手算（ddof=1，n=3）：mean = 0.065800865800865801
        #       平方偏差和 = 0.019485012.../2 -> std = 0.098704146689373719
        #       AV = 0.098704146689373719 * sqrt(252) = 1.566879753065533112
        self.assertClose(annual_volatility(R_A), 1.566879753065533112)

    def test_ddof0_differs(self):
        # ddof=0 会小 sqrt((n-1)/n) 倍：n=3 时 std_ddof0 = std_ddof1 * sqrt(2/3)
        self.assertClose(annual_volatility(R_A, ddof=0),
                         0.080591598295262370 * SQRT252)

    def test_single_sample_is_nan(self):
        self.assertNan(annual_volatility([0.01]))


# ══════════════════ 4. Sharpe ══════════════════

class TestSharpe(PerfTest):

    def test_sharpe_A(self):
        # 手算：0.065800865800865801 / 0.098704146689373719 * sqrt(252)
        #       = 10.582699884516705835
        self.assertClose(sharpe_ratio(R_A), 10.582699884516705835)

    def test_sharpe_A_with_risk_free(self):
        # 手算（rf = 0.001 日频）：mean - rf = 0.064800865800865801
        #       std 不变 -> 0.064800865800865801 / 0.098704146689373719 * sqrt(252)
        #       = 10.421870695482274319
        self.assertClose(sharpe_ratio(R_A, risk_free=0.001),
                         10.421870695482274319)

    def test_sharpe_B_is_zero(self):
        # 手算：mean([0.01,-0.01,0.01,-0.01]) 在浮点下正好是 0.0 -> Sharpe = 0
        self.assertEqual(sharpe_ratio(R_B), 0.0)

    def test_sharpe_B_with_risk_free(self):
        # 手算（rf = 0.001）：mean - rf = -0.001
        #       -0.001 / 0.011547005383792515 * sqrt(252) = -1.374772708486752002
        self.assertClose(sharpe_ratio(R_B, risk_free=0.001),
                         -1.374772708486752002)

    def test_ddof1_vs_ddof0(self):
        # 🔴 本模块 ddof=1（样本标准差），与 empyrical stats.py:653、quantstats
        #    stats.py:884 一致。ddof=0 的 Sharpe 恰好是 ddof=1 的 sqrt(n/(n-1)) 倍：
        #    n = 3 -> sqrt(3/2) = 1.224744871391589049
        #    手算 ddof=0 的值 = 0.065800865800865801 / 0.080591598295262370 * sqrt(252)
        #                    = 12.961107409038197170
        s1 = sharpe_ratio(R_A, ddof=1)
        s0 = sharpe_ratio(R_A, ddof=0)
        self.assertClose(s0, 12.961107409038197170)
        self.assertClose(s0 / s1, math.sqrt(1.5))
        self.assertNotAlmostEqual(s1, s0, places=6,
                                  msg="ddof 不同必须给出不同的 Sharpe")

    def test_geometric_vs_arithmetic(self):
        # 接口约定 第 6 层原文口径：(年化收益 - rf) / 年化波动率
        # 手算：4479448.7388245724 / 1.566879753065533112 = 2858833.761851043677
        ar = annual_return(R_A)
        av = annual_volatility(R_A)
        geo = sharpe_from_annual(ar, av)
        self.assertClose(geo, 2858833.761851043677)
        # 两个口径必须**不相等**：算术版是 mean*252 那一路，几何版是复利那一路
        self.assertNotAlmostEqual(geo, sharpe_ratio(R_A), places=3,
                                  msg="几何版和算术版本来就不同，别把它们当成一个数")

    def test_zero_volatility_follows_numpy(self):
        # 波动为 0 且均值为正 -> empyrical 走 numpy 除法给 inf（不是抛异常）
        self.assertEqual(sharpe_ratio([0.001] * 5), float("inf"))
        # 均值为 0、波动也为 0 -> 0/0 = nan
        self.assertNan(sharpe_ratio([0.0] * 5))


# ══════════════════ 5. Sortino ══════════════════

class TestSortino(PerfTest):

    def test_sortino_A(self):
        # 手算：只有 -1/22 是负的 -> mean(min(r,0)**2) = (1/22)**2 / 3 = 1/1452
        #       downside = 0.026243194054073898
        #       Sortino = 0.065800865800865801 * sqrt(252) / 0.026243194054073898
        #               = 39.802943179045010343
        self.assertClose(sortino_ratio(R_A), 39.802943179045010343)

    def test_sortino_B_is_zero(self):
        # 手算：mean = 0 -> 分子 0，Sortino = 0
        self.assertEqual(sortino_ratio(R_B), 0.0)

    def test_sortino_A_equals_empyrical_formula(self):
        # 与 empyrical 等价写法对拍：
        #   average_annual_return = mean * 252
        #   annualized_downside_risk = sqrt(mean(min(r,0)**2)) * sqrt(252)
        mean_ = sum(R_A) / 3.0
        downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in R_A) / 3.0)
        self.assertClose(sortino_ratio(R_A),
                         (mean_ * TRADING_DAYS_PER_YEAR) / (downside * SQRT252))

    def test_no_downside_is_inf(self):
        # 从来没亏过：empyrical 给 inf、quantstats 给 nan，本模块跟 empyrical
        self.assertEqual(sortino_ratio([0.01, 0.02, 0.03]), float("inf"))


# ══════════════════ 6. 最大回撤 ══════════════════

class TestMaxDrawdown(PerfTest):

    def test_A_exact_fraction(self):
        # 手算：历史最高 1.0, 1.1, 1.1, 1.2；只有第 3 天回撤
        #       MDD = 1 - 1.05/1.1 = 1/22 = 0.045454545454545455
        self.assertClose(max_drawdown(EQ_A), 1.0 / 22.0, rel=1e-15)

    def test_first_day_loss_counts(self):
        # 🔴 期初净值也是历史最高点：第一天就亏 5% 必须被算进去
        #    手算：1 - 0.95/1.0 = 0.05（empyrical / quantstats 同口径）
        self.assertClose(max_drawdown([1.0, 0.95, 0.969]), 0.05)

    def test_from_returns_matches_equity(self):
        # 收益率 [-0.05, 0.02] -> 净值 [1.0, 0.95, 0.969] -> MDD = 0.05
        self.assertClose(max_drawdown_from_returns([-0.05, 0.02]), 0.05)

    def test_B(self):
        # 手算：净值 1.0, 1.01, 0.9999, 1.009899, 0.99980001
        #       0.99980001 = 1.01**2 * 0.99**2，所以 0.99980001/1.01 = 0.989901
        #       MDD = 1 - 0.989901 = 0.010099
        self.assertClose(max_drawdown_from_returns(R_B), 0.010099)

    def test_monotone_rise_has_no_drawdown(self):
        self.assertEqual(max_drawdown([1.0, 1.1, 1.2, 1.3]), 0.0)

    def test_sign_convention_is_positive(self):
        # 本模块返回正数（接口约定 写的是 max(1 - 净值/历史最高净值)），
        # empyrical / quantstats 返回负数，对拍时取相反数（见 TestCrossCheck）
        self.assertGreater(max_drawdown(EQ_A), 0.0)


# ══════════════════ 7. Calmar ══════════════════

class TestCalmar(PerfTest):

    def test_B(self):
        # 手算：AR = 0.99980001**63 - 1 = -0.01252157450152481
        #       MDD = 0.010099
        #       Calmar = -0.01252157450152481 / 0.010099 = -1.2398826123
        self.assertClose(calmar_ratio(R_B), -1.2398826123, rel=1e-9)

    def test_A(self):
        # 手算：4479448.7388245724 / (1/22) = 98547872.2541405931
        self.assertClose(calmar_ratio(R_A), 98547872.2541405931)

    def test_no_drawdown_is_nan(self):
        # ⚠️ MDD = 0 时：empyrical 给 nan（stats.py:526-531），quantstats 给 inf
        #    （stats.py:1677 直接除 0）。本模块跟 empyrical。
        self.assertNan(calmar_ratio([0.01, 0.02, 0.03]))


# ══════════════════ 8. 换手率 ══════════════════

class TestTurnover(PerfTest):

    def test_two_days_exact_60(self):
        # 手算：3 个净值点 -> n = 2 个交易日
        #       平均权益 = (100000 + 105000 + 110000) / 3 = 105000
        #       累计 = 50000 / 105000 = 10/21 = 0.476190476190476190
        #       年化 = 10/21 * 252/2 = 1260/21 = 60.0（正好整数）
        tv = turnover(EQ_T, [50000.0])
        self.assertClose(tv.total, 10.0 / 21.0, rel=1e-15)
        self.assertClose(tv.annualized, 60.0)
        self.assertClose(tv.avg_equity, 105000.0)
        self.assertClose(tv.trade_amount, 50000.0)
        self.assertEqual(tv.n_periods, 2)

    def test_four_days_buy_and_sell(self):
        # 手算：4 个净值点 -> n = 3 个交易日（首点是期初，不构成一期）
        #       平均权益 = (100000+110000+105000+120000)/4 = 108750
        #       累计 = 100000 / 108750 = 80/87 = 0.919540229885057471
        #       年化 = 80/87 * 252/3 = 80/87 * 84 = 2240/29 = 77.241379310344827586
        #       单边 = 一半 = 1120/29 = 38.620689655172413793
        tv = turnover(EQ_E, [50000.0, 50000.0])
        self.assertClose(tv.total, 80.0 / 87.0, rel=1e-15)
        self.assertClose(tv.annualized, 2240.0 / 29.0, rel=1e-15)
        self.assertClose(tv.one_way_annualized, 1120.0 / 29.0, rel=1e-15)
        self.assertEqual(tv.n_periods, 3)

    def test_no_trade_is_zero(self):
        self.assertEqual(turnover(EQ_E, []).annualized, 0.0)

    def test_direction_does_not_matter(self):
        # 买卖都按正数传：负数（卖）取绝对值后必须得到同一个结果
        self.assertClose(turnover(EQ_E, [-50000.0, 50000.0]).total,
                         turnover(EQ_E, [50000.0, 50000.0]).total, rel=0.0)

    def test_returns_dataclass(self):
        self.assertIsInstance(turnover(EQ_E, []), Turnover)


# ══════════════════ 9. 基准 / 超额 / IR ══════════════════

class TestBenchmark(PerfTest):

    def test_active_returns(self):
        # 手算：[0.02-0.01, -0.01-(-0.02), 0.03-0.01, 0-0.01]
        #       = [0.01, 0.01, 0.02, -0.01]
        got = active_returns(R_P, R_BM)
        for g, w in zip(got, [0.01, 0.01, 0.02, -0.01]):
            self.assertClose(g, w, rel=1e-15)
        self.assertClose(sum(got), 0.03, rel=1e-15)

    def test_information_ratio(self):
        # 手算：mean = 0.03/4 = 0.0075
        #       平方偏差 = [0.0025**2, 0.0025**2, 0.0125**2, 0.0175**2]
        #       方差 = 4.75e-4 / 3，std = 0.012583057392117916
        #       IR = 0.0075 / 0.012583057392117916 = 0.596039560679269713
        self.assertClose(information_ratio(R_P, R_BM), 0.596039560679269713)

    def test_information_ratio_is_daily_not_annual(self):
        # 两个库（quantstats / empyrical.excess_sharpe）都**不年化**，
        # 年化要自己乘 sqrt(252)：0.596039560679269713 * 15.874507866387544
        #                          = 9.461834694681
        self.assertClose(information_ratio(R_P, R_BM) * SQRT252,
                         9.461834694681, rel=1e-9)

    def test_tracking_error(self):
        # 手算：0.012583057392117916 * sqrt(252) = 0.199749843554381789
        self.assertClose(tracking_error(R_P, R_BM), 0.199749843554381789)

    def test_length_mismatch_raises(self):
        # 不插值、不补零、不猜（接口约定 口径）
        with self.assertRaises(ValueError):
            active_returns([0.01, 0.02], [0.01])

    def test_zero_tracking_error_is_zero(self):
        # quantstats stats.py:2671-2673：跟踪误差为 0 时 IR 记 0
        self.assertEqual(information_ratio(R_P, R_P), 0.0)


# ══════════════════ 10. 汇总入口 evaluate ══════════════════

class TestEvaluate(PerfTest):

    def test_report_fields_on_A(self):
        rpt = evaluate(EQ_A)
        self.assertIsInstance(rpt, PerformanceReport)
        self.assertEqual(rpt.n_periods, 3)
        self.assertClose(rpt.total_return, 0.2)
        self.assertClose(rpt.annual_return, 4479448.7388245724)
        self.assertClose(rpt.annual_volatility, 1.566879753065533112)
        self.assertClose(rpt.max_drawdown, 1.0 / 22.0, rel=1e-15)
        self.assertClose(rpt.sharpe, 10.582699884516705835)
        self.assertClose(rpt.sortino, 39.802943179045010343)
        self.assertClose(rpt.calmar, 98547872.2541405931)
        self.assertClose(rpt.sharpe_geometric, 2858833.761851043677)
        self.assertClose(rpt.equity_start, 1.0)
        self.assertClose(rpt.equity_end, 1.2)

    def test_turnover_wired_in(self):
        # 金额口径净值 + 两笔各 5 万 -> 年化换手 2240/29 = 77.241379310344827586
        rpt = evaluate(EQ_E, trade_amounts=[50000.0, 50000.0])
        self.assertClose(rpt.turnover, 2240.0 / 29.0, rel=1e-15)
        self.assertClose(rpt.turnover_total, 80.0 / 87.0, rel=1e-15)

    def test_benchmark_block(self):
        # 手算累计收益：策略 1.1*(21/22)*(8/7) = 1.2 -> 0.2
        #               基准 1.01*0.98*1.01 = 0.999698 -> -0.000302
        #               超额 = 0.2 - (-0.000302) = 0.200302
        # 手算超额序列 = [1/10-1/100, -1/22+2/100, 1/7-1/100]
        #              = [9/100, -7/275, 93/700]
        #       mean = 152/770 / 3 = 76/1155 = 0.065800865800865801
        #       方差(ddof=1) = 3577773/533610000 -> std = 0.081883125465599243
        #       IR = 0.065800865800865801 / 0.081883125465599243 = 0.803594946171297231
        #       跟踪误差 = std * sqrt(252) = 1.299854319328053366
        # 手算年化（252/3 = 84 次方）：策略 1.2**84 - 1 = 4479448.7388245724
        #                             基准 0.999698**84 - 1 = -0.025052671348
        #                             超额 = 4479448.763877244
        rpt = evaluate(EQ_A, benchmark_returns=R_BM3, benchmark_name="沪深300")
        self.assertClose(rpt.excess_return, 0.200302)
        self.assertClose(rpt.excess_return_annualized, 4479448.763877244, rel=1e-9)
        self.assertClose(rpt.tracking_error, 1.299854319328053366)
        self.assertClose(rpt.information_ratio, 0.803594946171297231)
        self.assertClose(rpt.information_ratio_annualized, 12.756674294, rel=1e-8)
        self.assertEqual(rpt.benchmark_name, "沪深300")

    def test_no_benchmark_leaves_nan(self):
        rpt = evaluate(EQ_A)
        self.assertNan(rpt.excess_return)
        self.assertNan(rpt.information_ratio)

    def test_to_dict_and_format(self):
        rpt = evaluate(EQ_E, benchmark_returns=R_BM3, benchmark_name="中证1000",
                       trade_amounts=[50000.0, 50000.0])
        d = rpt.to_dict()
        self.assertEqual(len(d), 17)
        self.assertClose(d["max_drawdown"], rpt.max_drawdown, rel=0.0)
        text = rpt.format()
        self.assertIn("最大回撤", text)
        self.assertIn("中证1000", text)
        self.assertIn("Information Ratio", text)


# ══════════════════ 11. 三方对拍 ══════════════════

class TestCrossCheck(PerfTest):
    """使用人工净值核对自算与可选empyrical、quantstats。

    浮点缩放路径可能略有差异，nan/inf边界按各指标定义判断，
    不把全位相等当成普遍成立的保证。"""

    def test_all_rows_agree(self):
        rows = cross_check(EQ_A)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertLessEqual(row.rel_diff, 1e-9,
                                 "%s 三方对不上：自算 %r / empyrical %r / quantstats %r"
                                 % (row.name, row.self_value,
                                    row.empyrical_value, row.quantstats_value))

    def test_worst_row_is_only_float_noise(self):
        # 容差只处理人工场景的浮点缩放路径差异，不改变指标公式。
        worst = max(cross_check(EQ_A), key=lambda r: r.rel_diff)
        self.assertIn(worst.name, ("Max Drawdown", "Calmar"))
        self.assertLessEqual(worst.rel_diff, 1e-13)

    def test_with_benchmark(self):
        # 样本 D 是 4 个收益，基准也用 4 个（长度必须一致）
        eq_d = [1.0]
        for r in R_P:
            eq_d.append(eq_d[-1] * (1.0 + r))
        rows = cross_check(eq_d, benchmark_returns=R_BM)
        self.assertEqual(len(rows), 7)
        ir = [r for r in rows if r.name.startswith("Information Ratio")][0]
        # 自算 == empyrical.excess_sharpe == quantstats.information_ratio
        self.assertClose(ir.self_value, 0.596039560679269713)
        self.assertClose(ir.empyrical_value, ir.self_value)
        self.assertClose(ir.quantstats_value, ir.self_value)

    def test_empyrical_has_no_information_ratio(self):
        # 🔴 记录一个真实的接口差异：empyrical 的 IR 叫 excess_sharpe
        import empyrical
        import quantstats as qs
        self.assertFalse(hasattr(empyrical, "information_ratio"))
        self.assertTrue(hasattr(empyrical, "excess_sharpe"))
        self.assertTrue(hasattr(qs.stats, "information_ratio"))

    def test_sign_convention_against_libraries(self):
        # 自算正数、两个库负数，绝对值必须一致（符号是口径差异，不是数值差异）
        import empyrical
        import pandas as pd
        import quantstats as qs
        s = pd.Series(R_A, index=pd.date_range("2024-01-02", periods=3, freq="B"))
        mine = max_drawdown_from_returns(R_A)
        self.assertClose(-float(empyrical.max_drawdown(s)), mine)
        self.assertClose(-float(qs.stats.max_drawdown(s)), mine)
        self.assertLess(float(empyrical.max_drawdown(s)), 0.0)

    def test_ddof_mismatch_would_break_the_check(self):
        # 反证：如果 Sharpe 用 ddof=0，和 empyrical 的相对差会大得离谱（~18%），
        # 这条用来证明"对拍能抓住 ddof 用错"，不是摆设
        import empyrical
        import pandas as pd
        s = pd.Series(R_A, index=pd.date_range("2024-01-02", periods=3, freq="B"))
        emp = float(empyrical.sharpe_ratio(s))
        self.assertClose(sharpe_ratio(R_A, ddof=1), emp)
        wrong = sharpe_ratio(R_A, ddof=0)
        self.assertGreater(abs(wrong - emp) / abs(emp), 0.1)


# ══════════════════ 12. 持仓周期衰减曲线 ══════════════════

class TestDecayCurve(PerfTest):
    """样本 F：价格 10, 10.5, 11, 10, 12, 13, 13, 13, 13, 13, 14（D0..D10）。

    买入日当天不算，所以 T+k = 买入日之后第 k 个交易日的价格。
    """

    TRADES = [
        # 第 1 笔：D0 买 10，D3 卖 10（持有 3 个交易日，收益 0）
        Trade("600000.SH", DAYS[0], 10.0, DAYS[3], 10.0),
        # 第 2 笔：D2 买 11，D5 卖 13（持有 3 个交易日，收益 13/11-1 = 2/11）
        Trade("600000.SH", DAYS[2], 11.0, DAYS[5], 13.0),
    ]

    def test_curvature_by_hand(self):
        # 手算（T+k 用买入日之后第 k 个交易日的价格）：
        #   第 1 笔（买 D0@10）：T+1 10.5/10-1 = 1/20   T+2 11/10-1 = 1/10
        #                        T+3 10/10-1  = 0      T+5 13/10-1 = 3/10
        #                        T+10 14/10-1 = 2/5
        #   第 2 笔（买 D2@11）：T+1 10/11-1 = -1/11    T+2 12/11-1 = 1/11
        #                        T+3 13/11-1 = 2/11    T+5 13/11-1 = 2/11
        #                        T+10 需要 D12，不存在 -> 这一档不计入
        #   平均：T+1 (1/20-1/11)/2 = -9/440      T+2 (1/10+1/11)/2 = 21/220
        #         T+3 (0+2/11)/2 = 1/11          T+5 (3/10+2/11)/2 = 53/220
        #         T+10 = 2/5（只有 1 笔）
        curve = from_trades(self.TRADES, PRICES)
        self.assertIsInstance(curve, DecayCurve)
        self.assertEqual(curve.horizons, (1, 2, 3, 5, 10))
        self.assertEqual(curve.horizons, DEFAULT_HORIZONS)
        self.assertClose(curve.means[1], -9.0 / 440.0, rel=1e-15)
        self.assertClose(curve.means[2], 21.0 / 220.0, rel=1e-15)
        self.assertClose(curve.means[3], 1.0 / 11.0, rel=1e-15)
        self.assertClose(curve.means[5], 53.0 / 220.0, rel=1e-15)
        self.assertClose(curve.means[10], 0.4, rel=1e-15)
        self.assertEqual(curve.counts, {1: 2, 2: 2, 3: 2, 5: 2, 10: 1})
        self.assertEqual(curve.n_entries, 2)
        self.assertEqual(curve.best_horizon, 10)

    def test_buy_day_not_counted(self):
        # 🔴 接口约定 抓过的真 bug：hold_days 把买入日也算进去 / 或干脆不计数。
        #    这一条把口径钉死：买在 D0@10，T+1 必须用 D1 = 10.5 的价格（5%），
        #    而不是买入日当天（0%）。
        curve = decay_curve(PRICES, [("600000.SH", DAYS[0], 10.0)])
        self.assertClose(curve.means[1], 0.05, rel=1e-15)
        self.assertNotEqual(curve.means[1], 0.0,
                            "T+1 用成了买入日当天 -> 买入日被当成了第 1 天")
        self.assertClose(curve.means[2], 0.1, rel=1e-15)
        self.assertClose(curve.means[3], 0.0, rel=1e-15)
        self.assertClose(curve.means[5], 0.3, rel=1e-15)
        self.assertClose(curve.means[10], 0.4, rel=1e-15)
        self.assertEqual(curve.counts[10], 1)

    def test_immature_samples_are_skipped_not_zero(self):
        # 买在 D9：之后只有 D10 一天，所以只有 T+1 有样本，其余档位根本不出现
        # （不是记 0 —— 那是拿"没到期"冒充"没赚钱"）
        curve = decay_curve(PRICES, [("600000.SH", DAYS[9], 10.0)])
        self.assertEqual(list(curve.means.keys()), [1])
        self.assertClose(curve.means[1], 0.4, rel=1e-15)
        self.assertEqual(curve.n_entries, 1)

    def test_realized_stats(self):
        # 手算：两笔各持有 3 个交易日 -> mean_hold_days = 3.0
        #       收益 = [0, 2/11] -> mean = 1/11 = 0.09090909090909091
        #       胜率 = 1/2（0 不算赢）
        curve = from_trades(self.TRADES, PRICES)
        r = curve.realized
        self.assertEqual(r.n, 2)
        self.assertClose(r.mean_return, 1.0 / 11.0, rel=1e-15)
        self.assertClose(r.win_rate, 0.5, rel=1e-15)
        self.assertClose(r.mean_hold_days, 3.0, rel=1e-15)

    def test_accepts_dict_and_tuple(self):
        # 便捷入口认三种写法：Trade / dict / 5 元组
        as_dict = {"code": "600000.SH", "buy_date": DAYS[0], "buy_price": 10.0,
                   "sell_date": DAYS[3], "sell_price": 10.0}
        as_tuple = ("600000.SH", DAYS[0], 10.0, DAYS[3], 10.0)
        curve = from_trades([as_dict, as_tuple], PRICES)
        self.assertEqual(curve.n_entries, 2)
        self.assertClose(curve.means[1], 0.05, rel=1e-15)

    def test_missing_buy_date_raises(self):
        with self.assertRaises(ValueError):
            decay_curve(PRICES, [("600000.SH", "2023-01-01", 10.0)])

    def test_missing_code_raises(self):
        with self.assertRaises(ValueError):
            decay_curve(PRICES, [("000001.SZ", DAYS[0], 10.0)])

    def test_bad_horizons_raise(self):
        with self.assertRaises(ValueError):
            decay_curve(PRICES, [("600000.SH", DAYS[0], 10.0)], horizons=(0, 1))

    def test_bad_price_raises(self):
        with self.assertRaises(ValueError):
            decay_curve(PRICES, [("600000.SH", DAYS[0], 0.0)])

    def test_to_dict_and_format(self):
        curve = from_trades(self.TRADES, PRICES)
        d = curve.to_dict()
        self.assertEqual(d["best_horizon"], 10)
        self.assertEqual(d["counts"][10], 1)
        self.assertIn("T+1", curve.format())
        self.assertIn("已实现", curve.format())


# ══════════════════ 13. backtrader 挂载点 ══════════════════

class TestPerformanceAnalyzer(PerfTest):
    """第 6 层挂到 backtrader 上（接口说明：bt.Analyzer + empyrical）。

        场景：5 根平盘后每天 +1 元的 bar（开 = 收），第 1 根 bar 买 100 股，
        第 2 根 bar 开盘 11.0 成交（backtrader 默认下一根 bar 开盘成交），之后不动。
        手算权益（现金 100000 - 1100 = 98900，持仓按当日收盘估值）：
            bar0 100000（还没买）-> bar1 98900 + 100*11 = 100000
            bar2 98900 + 100*12 = 100100
            bar3 98900 + 100*13 = 100200
            bar4 98900 + 100*14 = 100300
        手算累计收益 = 100300/100000 - 1 = 0.003，且曲线单调不降 -> MDD = 0
        """

    @classmethod
    def setUpClass(cls):
        import backtrader as bt
        import pandas as pd
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        bars = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": [100000] * 5},
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                                  "2024-01-05", "2024-01-08"]),
        )

        class BuyOnce(bt.Strategy):
            def __init__(self):
                self.done = False

            def next(self):
                if not self.done:
                    self.buy(size=100)
                    self.done = True

        cer = bt.Cerebro()
        cer.adddata(bt.feeds.PandasData(dataname=bars))
        cer.broker.setcash(100000.0)
        cer.addstrategy(BuyOnce)
        cer.addanalyzer(PerformanceAnalyzer, _name="perf")
        cls.strat = cer.run()[0]

    def test_report_attached(self):
        rpt = self.strat.analyzers.perf.report
        self.assertIsInstance(rpt, PerformanceReport)

    def test_n_periods(self):
        # 5 根 bar -> 5 个净值点 -> 4 个收益
        self.assertEqual(self.strat.analyzers.perf.report.n_periods, 4)

    def test_total_return_is_three_per_mille(self):
        # 手算：100300 / 100000 - 1 = 0.003
        self.assertClose(self.strat.analyzers.perf.report.total_return, 0.003)

    def test_no_drawdown(self):
        self.assertEqual(self.strat.analyzers.perf.report.max_drawdown, 0.0)

    def test_get_analysis_returns_dict(self):
        d = self.strat.analyzers.perf.get_analysis()
        self.assertIsInstance(d, dict)
        self.assertIn("total_return", d)
        self.assertIn("information_ratio", d)
        self.assertClose(d["total_return"], 0.003)


if __name__ == "__main__":
    unittest.main(verbosity=2)
