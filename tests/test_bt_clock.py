# -*- coding: utf-8 -*-
"""主时钟和最小周期闸门测试。人工数据验证不同起点下的调度不会漏记交易日。"""

import datetime as dt
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")     # backtrader 内部还在用 datetime.utcnow()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.engine import (EngineConfig, clock_frame, run_engine)  # noqa: E402

# ── 交易日──
L0 = "2023-12-28"   # 引子（基准独有，任何个股都没有 bar）
L1 = "2023-12-29"   # 引子
D0 = "2024-01-02"   # 决策日
D1 = "2024-01-03"   # 执行日
D2 = "2024-01-04"
D3 = "2024-01-05"
D4 = "2024-01-08"
D5 = "2024-01-09"
DAYS = [D0, D1, D2, D3, D4, D5]
LEAD = [L0, L1]


def mk(dates, opens, closes=None, *, adj=1.0):
    """造一根引擎帧（结构与 ParquetFeed 要的列一致）—— 与 test_bt_engine.mk 同口径。"""
    o = [float(x) for x in opens]
    c = [float(x) for x in closes] if closes is not None else list(o)
    n = len(o)
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df.index.name = "trade_date"
    df["open"] = o
    df["close"] = c
    df["high"] = [max(a, b) * 1.02 for a, b in zip(o, c)]
    df["low"] = [min(a, b) * 0.98 for a, b in zip(o, c)]
    df["volume"] = 1_000_000.0
    df["openinterest"] = 0.0
    df["is_st"] = 0
    df["up_limit"] = float("nan")
    df["down_limit"] = float("nan")
    df["raw_open"] = o
    df["raw_up_limit"] = float("nan")
    df["adj_ratio"] = float(adj)
    assert len(df) == n
    return df


#: 基准（沪深300 位置）：主体 6 根
BENCH = mk(DAYS, [3000] * 6, [3000, 3010, 3020, 3030, 3040, 3050])
#: 带引子的基准：run_backtest 实际会喂给引擎的那种（引子只给基准收益当"前值"）
BENCH_LEAD = mk(LEAD + DAYS, [2990, 2995] + [3000] * 6,
                [2990, 2995, 3000, 3010, 3020, 3030, 3040, 3050])
#: 老票：6 根全有，开盘价一直 10.00（不会碰 -5% 硬止损）
OLD = mk(DAYS, [10.00] * 6, [10.00] * 6)
# 人工晚起点数据源只有最后一根bar，用于检查最小周期闸门。
LATE = mk([D3, D4, D5], [10.00, 10.10, 10.20], [10.00, 10.10, 10.20])

CFG = EngineConfig(exposure_fn=None)      # 不做环境调整，手算才干净


def dates_of(idx):
    return [pd.Timestamp(x).date() for x in idx]


class TestClockFrameHandCalc(unittest.TestCase):
    """A. clock_frame：引子必须被剪掉，区间两端都是闭区间。"""

    def test_lead_in_dropped(self):
        c = clock_frame(BENCH_LEAD, D0, D5)
        self.assertEqual(dates_of(c.index), dates_of(DAYS))

    def test_empty_when_window_outside(self):
        # 窗口整个落在引子之前 -> 空帧（调用方 run_backtest 会当场报错，不静默跑空）
        self.assertEqual(len(clock_frame(BENCH_LEAD, "2023-01-03", "2023-06-30")), 0)

    def test_end_inclusive(self):
        c = clock_frame(BENCH_LEAD, D0, D0)
        self.assertEqual(dates_of(c.index), [dt.date(2024, 1, 2)])

    def test_no_lead_in_untouched(self):
        c = clock_frame(BENCH, D0, D5)
        self.assertEqual(len(c), 6)


class TestEngineCoversWholeClock(unittest.TestCase):
    """B. 净值点数 == 时钟 bar 数（有票没上市也不许缩短）。"""

    def test_late_stock_does_not_shorten_curve(self):
        """🔴 回归：晚上市的票不许把净值曲线往后拖（原来 6 点会变成 3 点）。"""
        r = run_engine(frames={"600000.SH": OLD, "301583.SZ": LATE},
                       benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 4.0), ("301583.SZ", 4.0)]},
                       cfg=CFG)
        self.assertEqual(len(r.equity), 6)                      # 原来只有 3
        self.assertEqual(r.dates, dates_of(DAYS))
        self.assertEqual(r.dates[0], dt.date(2024, 1, 2))       # 曲线从回测首日开始
        # 晚票在 D1（执行日）还没有 bar -> 记 blocked，不猜价格、不崩
        late_blocked = [b for b in r.blocked if b.code == "301583.SZ"]
        self.assertEqual(len(late_blocked), 1)
        self.assertEqual(late_blocked[0].date, dt.date(2024, 1, 3))
        self.assertIn("买不进", late_blocked[0].reason)

    def test_old_stock_still_traded_hand_calc(self):
        """同一天里，有 bar 的票照常成交（手算：10.00 -> 滑点 10.005 -> 9900 股）。"""
        r = run_engine(frames={"600000.SH": OLD, "301583.SZ": LATE},
                       benchmark_frame=BENCH,
                       picks={D0: [("600000.SH", 4.0), ("301583.SZ", 4.0)]},
                       cfg=CFG)
        buys = [t for t in r.trades if t.side == "buy"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0].code, "600000.SH")
        self.assertEqual(buys[0].date, dt.date(2024, 1, 3))
        self.assertEqual(buys[0].shares, 9900)
        self.assertAlmostEqual(buys[0].price, 10.005, places=6)

    def test_late_stock_tradable_after_its_first_bar(self):
        """晚票出第一根 bar 之后，照样能按清单买入（手算：D2 定 -> D3 开盘 10.005）。"""
        r = run_engine(frames={"301583.SZ": LATE}, benchmark_frame=BENCH,
                       picks={D2: [("301583.SZ", 4.0)]}, cfg=CFG)
        self.assertEqual(len(r.equity), 6)
        buys = [t for t in r.trades if t.side == "buy"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0].date, dt.date(2024, 1, 5))
        self.assertEqual(buys[0].shares, 9900)
        self.assertAlmostEqual(buys[0].price, 10.005, places=6)

    def test_equity_point_per_clock_bar(self):
        """契约：净值点数 == 基准帧 bar 数（这里 6 根 -> 6 点，含没清单的日子）。"""
        r = run_engine(frames={}, benchmark_frame=BENCH, picks={}, cfg=CFG)
        self.assertEqual(len(r.equity), 6)
        self.assertEqual(len(r.missing_days), 6)      # 6 天上游都没给清单
        # 🔴 报告里那个被 bug 弄成 41 的字段就是它 = 净值点数（期初资金当第一天的前值）
        self.assertEqual(r.report.n_periods, 6)


class TestClockIsWhatYouPass(unittest.TestCase):
    """C. run_engine 只认传进来的基准帧 —— 引子要么剪掉，要么自己认账。"""

    def test_lead_in_counts_as_clock_bars(self):
        # 不剪就多了 2 个空转净值点（这正是 run_backtest 必须剪的原因）
        r = run_engine(frames={}, benchmark_frame=BENCH_LEAD, picks={}, cfg=CFG)
        self.assertEqual(len(r.equity), 8)
        self.assertEqual(r.dates[0], dt.date(2023, 12, 28))

    def test_clipped_clock_matches_backtest_window(self):
        r = run_engine(frames={}, benchmark_frame=clock_frame(BENCH_LEAD, D0, D5),
                       picks={}, cfg=CFG)
        self.assertEqual(len(r.equity), 6)
        self.assertEqual(r.dates[0], dt.date(2024, 1, 2))


class TestBenchmarkReturnsAlignment(unittest.TestCase):
    """D. 基准收益要与策略收益等长（引子只用来提供 dates[0] 的前值）。"""

    def test_bench_returns_length_matches(self):
        r = run_engine(frames={"600000.SH": OLD}, benchmark_frame=clock_frame(BENCH_LEAD, D0, D5),
                       picks={D0: [("600000.SH", 4.0)]}, cfg=CFG,
                       bench_returns_frame=BENCH_LEAD)
        self.assertEqual(len(r.equity), 6)
        self.assertEqual(len(r.benchmark_returns), 6)
        # 手算首日：引子最后一根 2995 -> 区间首日 3000，收益 = 3000/2995 - 1
        self.assertAlmostEqual(r.benchmark_returns[0], 3000.0 / 2995.0 - 1.0, places=12)

    def test_bench_returns_first_day_without_lead_in(self):
        # 不给 bench_returns_frame -> 退回用时钟帧本身，首日没有前值 -> 记 0（不猜）
        r = run_engine(frames={}, benchmark_frame=BENCH, picks={}, cfg=CFG)
        self.assertEqual(len(r.benchmark_returns), 6)
        self.assertAlmostEqual(r.benchmark_returns[0], 0.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
