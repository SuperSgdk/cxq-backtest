# -*- coding: utf-8 -*-
"""引擎组件测试。人工手算行情验证收盘决定、下一日执行与现金持仓记账。"""

import datetime as dt
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import backtrader as bt
import pandas as pd

warnings.filterwarnings("ignore")     # backtrader 内部还在用 datetime.utcnow()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.engine import (EngineConfig, ParquetFeed, _break_queries,
                           make_break_lookup, qfq_frame, run_engine)  # noqa: E402

# ── 交易日──
D0 = "2024-01-02"   # 决策日
D1 = "2024-01-03"   # 执行日
D2 = "2024-01-04"
D3 = "2024-01-05"
D4 = "2024-01-08"
D5 = "2024-01-09"
DAYS = [D0, D1, D2, D3, D4, D5]


def mk(dates, opens, closes=None, *, highs=None, lows=None, ups=None,
       downs=None, is_st=None, adj=1.0):
    """造一根引擎帧（结构与 ParquetFeed 要的列一致）。

    high/low 默认给 open/close 的 ±2% —— 这是为了让"滑点"有发挥空间：
    成交价 = 开盘价 ×(1±0.05%)，必须落在当日 high/low 之内，
    否则 backtrader 会把滑点砍掉（slip_out=False），手算就对不上了。
    """
    o = [float(x) for x in opens]
    c = [float(x) for x in closes] if closes is not None else list(o)
    n = len(o)
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df.index.name = "trade_date"
    df["open"] = o
    df["close"] = c
    df["high"] = ([float(x) for x in highs] if highs is not None
                  else [max(a, b) * 1.02 for a, b in zip(o, c)])
    df["low"] = ([float(x) for x in lows] if lows is not None
                 else [min(a, b) * 0.98 for a, b in zip(o, c)])
    df["volume"] = 1_000_000.0
    df["openinterest"] = 0.0
    df["is_st"] = list(is_st) if is_st is not None else 0
    df["up_limit"] = [float(x) for x in ups] if ups is not None else float("nan")
    df["down_limit"] = [float(x) for x in downs] if downs is not None else float("nan")
    df["raw_open"] = o
    df["raw_up_limit"] = ([float(x) for x in ups] if ups is not None
                          else float("nan"))
    df["adj_ratio"] = float(adj)
    assert len(df) == n
    return df


BENCH = mk(DAYS, [3000] * 6, [3000, 3010, 3020, 3030, 3040, 3050])


class TestArrayFeed(unittest.TestCase):
    """NumPy 数据源必须逐根、逐字段保持原始引擎帧语义。"""

    def test_all_lines_and_sparse_dates_are_preserved(self):
        dates = [D0, D2]
        frame = mk(dates, [10.0, 11.0], closes=[10.5, 10.8],
                   highs=[10.7, 11.2], lows=[9.8, 10.6],
                   ups=[11.0, 12.1], downs=[9.0, 9.9], is_st=[0, 1], adj=0.8)
        rows = []

        class Capture(bt.Strategy):
            def next(self):
                data = self.data
                rows.append((bt.num2date(data.datetime[0]).date(), data.open[0],
                             data.high[0], data.low[0], data.close[0], data.volume[0],
                             data.openinterest[0], data.is_st[0], data.up_limit[0],
                             data.down_limit[0], data.raw_open[0], data.raw_up_limit[0],
                             data.adj_ratio[0]))

        cerebro = bt.Cerebro()
        cerebro.adddata(ParquetFeed(dataname=frame))
        cerebro.addstrategy(Capture)
        cerebro.run(runonce=False)

        self.assertEqual([row[0].isoformat() for row in rows], dates)
        self.assertEqual(rows[0][1:], (10.0, 10.7, 9.8, 10.5, 1_000_000.0, 0.0,
                                       0.0, 11.0, 9.0, 10.0, 11.0, 0.8))
        self.assertEqual(rows[1][1:], (11.0, 11.2, 10.6, 10.8, 1_000_000.0, 0.0,
                                       1.0, 12.1, 9.9, 11.0, 12.1, 0.8))

    def test_missing_column_fails_loudly(self):
        frame = mk([D0], [10.0]).drop(columns=["adj_ratio"])
        cerebro = bt.Cerebro()
        cerebro.adddata(ParquetFeed(dataname=frame))
        with self.assertRaisesRegex(ValueError, "adj_ratio"):
            cerebro.run(runonce=False)


class TestBreakPrefetch(unittest.TestCase):
    """分钟线并行预取只提前读取，不改变开板判据。"""

    def test_query_set_uses_next_trade_day_and_only_sealed_open(self):
        sealed = mk([D0, D1, D2], [9.8, 10.0, 10.2],
                    ups=[11.0, 10.0, 11.2])
        ordinary = mk([D0, D1, D2], [19.8, 20.0, 20.2],
                      ups=[21.0, 22.0, 22.2])
        got = _break_queries({"A.SH": sealed, "B.SH": ordinary}, BENCH.iloc[:3],
                             {D0: [("A.SH", 5.0), ("B.SH", 4.0)]})
        self.assertEqual(got, [("A.SH", dt.date(2024, 1, 3), 10.0)])

    def test_preloaded_result_is_reused_and_duplicates_are_read_once(self):
        bars = pd.DataFrame({
            "trade_time": pd.to_datetime(["2024-01-03 09:30", "2024-01-03 09:31", "2024-01-03 09:32"]),
            "open": [10.0, 9.95, 9.94], "high": [10.0, 10.0, 10.0],
            "low": [10.0, 9.90, 9.90], "close": [10.0, 9.95, 9.94],
        })
        calls = []

        class Table:
            def to_pandas(self, **_kwargs):
                return bars.copy()

            @property
            def num_rows(self):
                return len(bars)

        class Repo:
            def minute_path(self, _code, _freq):
                return Path(__file__)

        def read_table(*args, **kwargs):
            calls.append((args, kwargs))
            return Table()

        query = ("A.SH", dt.date(2024, 1, 3), 10.0)
        with mock.patch("cxq_bt.engine.pq.read_table", side_effect=read_table):
            lookup = make_break_lookup(Repo(), preload=[query, query], workers=2)
            result = lookup(*query)
            self.assertEqual((result.status, result.at, result.price),
                             ("filled", dt.time(9, 32), 9.94))
        self.assertEqual(len(calls), 1)


class TestCoreHandCalc(unittest.TestCase):
    """A. 核心对拍 —— 引擎必须算出 接口说明 写死的 **9635.6007**。"""

    def setUp(self):
        # 600000：01-03 开盘 10.00 买入 → 01-08 开盘 11.00 卖出
        self.stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20],
                        [9.80, 10.00, 10.30, 10.60, 11.00, 11.20])
        self.picks = {D0: [("600000.SH", 92.5)],   # 01-02 收盘定：买
                      D3: []}                       # 01-05 收盘定：清仓

    def run_it(self):
        return run_engine(frames={"600000.SH": self.stock},
                          benchmark_frame=BENCH, picks=self.picks)

    # ── 手算过程（全部来自 接口约定，一个数字都不改）──
    # 买入：参考价 10.00 -> 滑点后 10.005
    #   shares = int(100000 / 10.005 / 100) * 100 = 9900
    #   金额 = 9900 × 10.005 = 99,049.5
    #   佣金 = max(99049.5×0.0005, 5) = 49.524750
    #   过户费 = 99049.5×0.00001 = 0.990495          -> 买入费用 50.515245
    #   剩余现金 = 100000 − 99049.5 − 50.515245 = 899.984755
    # 卖出：参考价 11.00 -> 滑点后 10.9945
    #   金额 = 9900 × 10.9945 = 108,845.55
    #   佣金 54.422775 + 印花税 54.422775 + 过户费 1.0884555 = 109.9340055
    #   净盈亏 = (108845.55 − 109.9340055) − (99049.5 + 50.515245) = 9635.6007495

    def test_buy_hand_calc(self):
        r = self.run_it()
        buys = [t for t in r.trades if t.side == "buy"]
        self.assertEqual(len(buys), 1)
        b = buys[0]
        self.assertEqual(b.date, dt.date(2024, 1, 3))     # T+1 日成交
        self.assertEqual(b.shares, 9900)
        self.assertAlmostEqual(b.price, 10.005, places=6)
        self.assertAlmostEqual(b.amount, 99_049.5, places=6)
        self.assertAlmostEqual(b.fee, 50.515245, places=6)

    def test_sell_hand_calc(self):
        r = self.run_it()
        sells = [t for t in r.trades if t.side == "sell"]
        self.assertEqual(len(sells), 1)
        s = sells[0]
        self.assertEqual(s.date, dt.date(2024, 1, 8))
        self.assertEqual(s.shares, 9900)
        self.assertAlmostEqual(s.price, 10.9945, places=6)
        self.assertAlmostEqual(s.amount, 108_845.55, places=6)
        self.assertAlmostEqual(s.fee, 109.9340055, places=6)

    def test_net_pnl_matches_golden_number(self):
        r = self.run_it()
        self.assertAlmostEqual(r.pnl, 9_635.6007495, places=4)   # 接口约定 基准：净盈亏
        # 期末总资产 = 本金 + 净盈亏 = 100,000 + 9,635.6007495
        self.assertAlmostEqual(r.final_value, 109_635.6007495, places=4)
        self.assertEqual(len(r.trades), 2)

    def test_equity_curve_uses_close_price(self):
        """净值 = 现金 + 持仓 × **当日收盘价**（接口说明 净值口径）。"""
        r = self.run_it()
        self.assertEqual([d.isoformat() for d in r.dates], DAYS)
        expect = [
            100_000.0,                                   # 01-02 还没买
            899.984755 + 9900 * 10.00,                   # 01-03 收盘 10.00
            899.984755 + 9900 * 10.30,                   # 01-04 收盘 10.30
            899.984755 + 9900 * 10.60,                   # 01-05 收盘 10.60
            109_635.6007495,                             # 01-08 已清仓 = 本金 + 净盈亏
            109_635.6007495,                             # 01-09 空仓不变
        ]
        self.assertAlmostEqual(r.equity[-1], expect[-1], places=4)
        for got, want in zip(r.equity, expect):
            self.assertAlmostEqual(got, want, places=4)

    def test_missing_pick_day_means_no_trade(self):
        """上游没给清单的交易日 = 不做任何交易（持仓不动），并记账备查。"""
        r = self.run_it()
        miss = [d.isoformat() for d in r.missing_days]
        self.assertEqual(miss, [D1, D2, D4, D5])
        self.assertEqual(len(r.trades), 2)      # 只有清单覆盖的两天动了手


class TestRulesIntercept(unittest.TestCase):
    """B. 交易规则拦截 —— 买不进 / 卖不掉的场景。"""

    def test_sealed_up_cannot_buy(self):
        """一字板（开盘=涨停）+ 09:50 前没开板 -> 一分钱都不许动。"""
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20],
                   ups=[float("nan"), 10.00] + [float("nan")] * 4)
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 92.5)]},
                       break_lookup=lambda code, d, up: None)
        self.assertEqual(len(r.trades), 0)
        self.assertAlmostEqual(r.final_value, 100_000.0, places=6)
        self.assertTrue(any("一字板" in b.reason for b in r.blocked))
        self.assertTrue(all(abs(v - 100_000.0) < 1e-6 for v in r.equity))

    def test_t_board_break_buys_at_break_price(self):
        """涨停开盘但 09:45 开板 -> 按**开板价**买（接口说明 #4）。

                手算：09:30按10元锁定9900股；09:45可执行价9.98 -> 滑点后9.98499。
                    金额 = 9900 × 9.98499 = 98,851.401
                    费用 = 98,851.401 × 0.00051 = 50.41421451
                    剩余现金 = 100000 − 98851.401 − 50.41421451 = 1098.18478549
                """
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20],
                   ups=[float("nan"), 10.00] + [float("nan")] * 4)
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 92.5)]},
                       break_lookup=lambda code, d, up: (dt.time(9, 45), 9.98))
        self.assertEqual(len(r.trades), 1)
        b = r.trades[0]
        self.assertEqual(b.shares, 9900)  # 09:30锁定股数，晚确认不扩预算
        self.assertAlmostEqual(b.price, 9.98499, places=6)
        self.assertAlmostEqual(b.amount, 98_851.401, places=4)
        self.assertAlmostEqual(b.fee, 50.41421451, places=6)
        self.assertAlmostEqual(r.cash[1], 1098.18478549, places=4)

    def test_t_board_break_too_late_cannot_buy(self):
        """09:50 之后才开板 -> 买不进。"""
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20],
                   ups=[float("nan"), 10.00] + [float("nan")] * 4)
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 92.5)]},
                       break_lookup=lambda code, d, up: (dt.time(10, 12), 9.98))
        self.assertEqual(len(r.trades), 0)
        self.assertTrue(any("09:50 后才开板" in b.reason for b in r.blocked))

    def test_st_is_never_bought(self):
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20],
                   is_st=[0, 1, 0, 0, 0, 0])
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 92.5)]})
        self.assertEqual(len(r.trades), 0)
        self.assertTrue(any("ST" in b.reason for b in r.blocked))

    def test_sealed_down_sell_postponed_to_next_day(self):
        """跌停封死卖不出 -> 顺延到下一个交易日再卖（接口说明 #7）。"""
        # 01-08 全天封死跌停（开=高=低=收=跌停价 9.00）
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 9.00, 11.20],
                   closes=[9.80, 10.00, 10.30, 10.60, 9.00, 11.20],
                   highs=[9.996, 10.20, 10.506, 10.812, 9.00, 11.424],
                   lows=[9.604, 9.80, 10.094, 10.388, 9.00, 10.976],
                   downs=[float("nan")] * 4 + [9.00, float("nan")])
        picks = {D0: [("600000.SH", 92.5)],   # 01-02 定：买
                 D3: [],                       # 01-05 定：清仓 -> 01-08 开盘卖（封死跌停，卖不掉）
                 D4: []}                       # 01-08 定：继续清 -> 01-09 开盘卖成功
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks=picks)
        sells = [t for t in r.trades if t.side == "sell"]
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0].date, dt.date(2024, 1, 9))     # 顺延了一天
        self.assertTrue(any(b.date == dt.date(2024, 1, 8) and "跌停" in b.reason
                            for b in r.blocked))

    def test_suspended_means_no_bar(self):
        """停牌 = 日线缺行 -> 买不进（不是看字段）。"""
        stock = mk([D0, D1, D3, D4, D5],              # 缺 01-04，模拟停牌
                   [9.80, 10.00, 10.50, 11.00, 11.20])
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks={D2: [("600000.SH", 92.5)]})   # 01-04 定，下一根 bar 才有行情
        self.assertEqual(len(r.trades), 1)
        self.assertEqual(r.trades[0].date, dt.date(2024, 1, 5))


class TestSizingAndLimits(unittest.TestCase):
    """C. 仓位硬约束在引擎里也成立。"""

    def test_at_most_three_holdings(self):
        codes = ["A.SH", "B.SH", "C.SH", "D.SH"]
        frames = {}
        for i, c in enumerate(codes):
            frames[c] = mk(DAYS, [9.80 + i, 10.00 + i, 10.20 + i, 10.50 + i,
                                  11.00 + i, 11.20 + i])
        picks = {D0: [(c, 90 - 10 * i) for i, c in enumerate(codes)]}
        r = run_engine(frames=frames, benchmark_frame=BENCH, picks=picks)
        buys = [t for t in r.trades if t.side == "buy"]
        self.assertEqual(len(buys), 3)                 # 只买前 3 名
        self.assertNotIn("D.SH", {b.code for b in buys})
        self.assertGreaterEqual(min(r.cash), -1e-9)    # 现金永不为负

    def test_cash_never_negative_all_days(self):
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20])
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 92.5)], D3: []})
        self.assertGreaterEqual(min(r.cash), -1e-9)

    def test_score_threshold_blocks_low_score(self):
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20])
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 50.0)]},
                       cfg=EngineConfig(score_threshold=60.0))
        self.assertEqual(len(r.trades), 0)

    def test_exposure_zero_flattens(self):
        """环境评分低到空仓线 -> 清仓且不买（用户选甲，接口说明 #19）。"""
        stock = mk(DAYS, [9.80, 10.00, 10.20, 10.50, 11.00, 11.20])
        picks = {D0: [("600000.SH", 92.5)], D3: [("600000.SH", 92.5)]}
        r = run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                       picks=picks,
                       market_scores={D3: 0.0},
                       cfg=EngineConfig(exposure_fn=lambda s: 0.0 if s < 10 else 1.0))
        sells = [t for t in r.trades if t.side == "sell"]
        self.assertEqual(len(sells), 1)              # 手上那只被清掉
        self.assertEqual(sells[0].date, dt.date(2024, 1, 8))


class TestQfqFrame(unittest.TestCase):
    """D. 前复权口径 —— 手算缩放比例。"""

    def _daily(self):
        return pd.DataFrame({
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open":   [10.0, 10.0, 10.0],
            "high":   [10.5, 10.5, 10.5],
            "low":    [ 9.5,  9.5,  9.5],
            "close":  [10.0, 11.0, 12.0],
            "vol":    [100.0, 100.0, 100.0],
            "up_limit":   [11.0, 11.0, 13.2],
            "down_limit": [ 9.0,  9.0, 10.8],
            "is_st":  [0, 0, 0],
            "adj_factor": [1.0, 1.0, 2.0],
        })

    def test_ratio_is_factor_over_last(self):
        # ratio = adj_factor / adj_factor(最后一天) = [0.5, 0.5, 1.0]
        out = qfq_frame(self._daily())
        self.assertAlmostEqual(out["open"].iloc[0], 5.0, places=9)     # 10 × 0.5
        self.assertAlmostEqual(out["close"].iloc[2], 12.0, places=9)   # 12 × 1.0
        self.assertAlmostEqual(out["up_limit"].iloc[0], 5.5, places=9) # 11 × 0.5
        self.assertAlmostEqual(out["raw_up_limit"].iloc[0], 11.0, places=9)
        self.assertAlmostEqual(out["adj_ratio"].iloc[0], 0.5, places=9)
        # 日线 vol 是**手** -> 换股
        self.assertAlmostEqual(out["volume"].iloc[0], 10_000.0, places=6)

    def test_no_qfq_keeps_raw(self):
        out = qfq_frame(self._daily(), qfq=False)
        self.assertAlmostEqual(out["open"].iloc[0], 10.0, places=9)
        self.assertAlmostEqual(out["adj_ratio"].iloc[0], 1.0, places=9)

    def test_missing_factor_refuses_to_guess(self):
        d = self._daily()
        d.loc[1, "adj_factor"] = float("nan")
        with self.assertRaises(ValueError):
            qfq_frame(d)

    def test_no_factor_column_refuses_to_guess(self):
        """人工缺少 adj_factor 列时必须报错，不能默认为无需复权。"""
        d = self._daily().drop(columns=["adj_factor"])
        with self.assertRaises(ValueError):
            qfq_frame(d)


class TestHardStopLossEndToEnd(unittest.TestCase):
    """人工硬止损端到端场景，检查触发、交易限制与账户结果。"""

    #: D1 开盘 10.00 买入 -> 真实成交价 10.005（含 0.05% 滑点），9900 股
    #:   金额 = 9900 × 10.005 = 99,049.5
    #:   费用 = max(49.52475, 5) + 0.990495 = 50.515245
    #:   剩余现金 = 100,000 − 99,049.5 − 50.515245 = 899.984755
    CASH_AFTER_BUY = 899.984755
    SHARES = 9900

    def _run(self, opens, *, closes=None, downs=None, cfg=None, picks=None):
        stock = mk(DAYS, opens, closes=closes, downs=downs)
        return run_engine(frames={"600000.SH": stock}, benchmark_frame=BENCH,
                          picks=picks if picks is not None
                          else {D0: [("600000.SH", 92.5)]},
                          cfg=cfg)

    def test_stop_loss_sells_at_open_next_day(self):
        """D1 开盘 10.00 买 -> D2 开盘 9.40 = 相对开仓价 -6.05% -> 止损卖出。

        手算：9.40 / 10.005 − 1 = −6.0470% <= −5%  -> 触发
              卖价 = 9.40 × (1 − 0.0005) = 9.3953
              金额 = 9900 × 9.3953 = 93,013.47
              费用 = 46.506735(佣) + 46.506735(印花) + 0.9301347(过户) = 93.9436047
              回款 = 93,013.47 − 93.9436047 = 92,919.5263953
              期末 = 899.984755 + 92,919.5263953 = 93,819.5111503
        """
        r = self._run([9.80, 10.00, 9.40, 9.30, 9.20, 9.10])
        sells = [t for t in r.trades if t.side == "sell"]
        self.assertEqual(len(sells), 1)
        s = sells[0]
        self.assertEqual(s.date, dt.date(2024, 1, 4), "D2（买入日的下一个交易日）")
        self.assertEqual(s.shares, self.SHARES)
        self.assertAlmostEqual(s.price, 9.3953, places=6)
        self.assertAlmostEqual(s.amount, 93_013.47, places=4)
        self.assertAlmostEqual(s.fee, 93.9436047, places=6)
        self.assertTrue(s.reason.startswith("硬止损"), "原因要能归因: %r" % s.reason)
        self.assertIn("-6.05%", s.reason)
        self.assertAlmostEqual(r.final_value, 93_819.5111503, places=4)

    def test_stop_loss_runs_on_days_upstream_gave_no_list(self):
        """人工无清单场景：已启用的硬止损仍应检查持仓。

        止损不能依赖当天存在策略目标；缺清单不等于关闭风控。"""
        r = self._run([9.80, 10.00, 9.40, 9.30, 9.20, 9.10])
        # 止损发生在 D2，而 D2 正是一个「上游没给清单」的日子
        self.assertIn(D2, [d.isoformat() for d in r.missing_days])
        self.assertEqual([t.side for t in r.trades], ["buy", "sell"])
        self.assertEqual(r.trades[1].date, dt.date(2024, 1, 4))

    def test_no_stop_when_drop_is_under_threshold(self):
        """D2 开盘 9.55 -> 9.55/10.005−1 = −4.55% > −5% -> **不卖**，继续持有。

        手算：期末 = 899.984755 + 9900 × 9.55 = 899.984755 + 94,545 = 95,444.984755
        """
        r = self._run([9.80, 10.00, 9.55, 9.55, 9.55, 9.55])
        self.assertEqual([t.side for t in r.trades], ["buy"], "不该有任何卖出")
        self.assertAlmostEqual(r.final_value, 95_444.984755, places=4)

    def test_can_be_disabled_with_none(self):
        """传 None = 关掉止损；不是设成 0（0 会退化成「开盘不涨就砍」）。"""
        r = self._run([9.80, 10.00, 9.40, 9.30, 9.20, 9.10],
                      cfg=EngineConfig(hard_stop_pct=None))
        self.assertEqual([t.side for t in r.trades], ["buy"])
        # 手算：期末 = 899.984755 + 9900 × 9.10 = 90,989.984755
        self.assertAlmostEqual(r.final_value, 90_989.984755, places=4)

    def test_custom_threshold_minus_3_pct(self):
        """阈值改成 -3%：9.70/10.005−1 = −3.05% <= −3% -> 触发。"""
        r = self._run([9.80, 10.00, 9.70, 9.70, 9.70, 9.70],
                      cfg=EngineConfig(hard_stop_pct=-0.03))
        sells = [t for t in r.trades if t.side == "sell"]
        self.assertEqual(len(sells), 1)
        self.assertAlmostEqual(sells[0].price, 9.70 * (1 - 0.0005), places=6)

    def test_stop_loss_deferred_when_sealed_limit_down(self):
        """开盘封死跌停 -> 卖不出 -> **顺延**到下一交易日再止损，且不许漏掉。

        手算：D2 开盘 9.40 == 跌停价 9.40 -> 被拦，不成交
              D3 开盘 9.30 -> 9.30/10.005−1 = −7.05% <= −5% -> 卖
              卖价 = 9.30 × 0.9995 = 9.29535
              金额 = 9900 × 9.29535 = 92,023.965
              费用 = 46.0119825 + 46.0119825 + 0.92023965 = 92.94420465
              回款 = 92,023.965 − 92.94420465 = 91,931.02079535
              期末 = 899.984755 + 91,931.02079535 = 92,831.00555035
        """
        downs = [float("nan"), float("nan"), 9.40,
                 float("nan"), float("nan"), float("nan")]
        r = self._run([9.80, 10.00, 9.40, 9.30, 9.20, 9.10], downs=downs)
        sells = [t for t in r.trades if t.side == "sell"]
        self.assertEqual(len(sells), 1, "顺延之后必须还是卖掉了")
        self.assertEqual(sells[0].date, dt.date(2024, 1, 5), "顺延到 D3")
        self.assertAlmostEqual(sells[0].price, 9.29535, places=6)
        self.assertAlmostEqual(r.final_value, 92_831.00555035, places=4)
        self.assertTrue(any(b.date == dt.date(2024, 1, 4) and "跌停" in b.reason
                            for b in r.blocked), "D2 被拦要记账")

    def test_stop_loss_does_not_rebuy_same_day(self):
        """止损票**当天不许买回**（与「上游点名清仓」同一条纪律）。"""
        picks = {D0: [("600000.SH", 92.5)],
                 D1: [("600000.SH", 99.0)]}      # D1 收盘又把它推荐回来
        r = self._run([9.80, 10.00, 9.40, 9.40, 9.40, 9.40], picks=picks)
        buys = [t for t in r.trades if t.side == "buy"]
        sells = [t for t in r.trades if t.side == "sell"]
        self.assertEqual(len(sells), 1, "D2 止损卖出")
        self.assertEqual(len(buys), 1, "D2 当天不许买回；只有 D1 那一笔买")


if __name__ == "__main__":
    unittest.main(verbosity=2)
