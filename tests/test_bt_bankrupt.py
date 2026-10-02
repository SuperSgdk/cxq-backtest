# -*- coding: utf-8 -*-
"""现金不足与破产标注测试。全部为人工手算场景，不依赖历史行情。
验证整手和费用限制，以及账户不能继续交易时的统计处理。"""

import datetime as dt
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")     # backtrader 内部还在用 datetime.utcnow()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.engine import EngineConfig, run_engine  # noqa: E402

# ── 交易日──
D0 = "2024-01-02"   # 决策日（期初那一点）
D1 = "2024-01-03"   # 执行日：D0 的清单在这里按开盘价成交
D2 = "2024-01-04"
D3 = "2024-01-05"
D4 = "2024-01-08"
D5 = "2024-01-09"
DAYS = [D0, D1, D2, D3, D4, D5]


def mk(dates, opens, closes=None):
    """造一根引擎帧（结构与 ParquetFeed 要的列一致）—— 与 test_bt_clock.mk 同口径。"""
    o = [float(x) for x in opens]
    c = [float(x) for x in closes] if closes is not None else list(o)
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df.index.name = "trade_date"
    df["open"] = o
    df["close"] = c
    df["high"] = [max(a, b) * 1.02 for a, b in zip(o, c)]
    df["low"] = [min(a, b) * 1.02 for a, b in zip(o, c)]
    df["volume"] = 1_000_000.0
    df["openinterest"] = 0.0
    df["is_st"] = 0
    df["up_limit"] = float("nan")
    df["down_limit"] = float("nan")
    df["raw_open"] = o
    df["raw_up_limit"] = float("nan")
    df["adj_ratio"] = 1.0
    assert len(df) == len(o)
    return df


#: 基准 = 主时钟（6 根 bar -> 净值必须 6 点）
BENCH = mk(DAYS, [3000] * 6, [3000, 3010, 3020, 3030, 3040, 3050])
#: 价格全程不动：5.00 买入、5.00 收盘（净值只反映"买完剩下的现金 + 持仓市值"）
FLAT = mk(DAYS, [5.00] * 6, [5.00] * 6)
#: 价格路径 5.00 / 5.00 / 4.00 / 5.00 / 4.00 / 5.00（跌 -> 回 -> 再跌 -> 回）
DIP = mk(DAYS, [5.00, 5.00, 4.00, 5.00, 4.00, 5.00],
         [5.00, 5.00, 4.00, 5.00, 4.00, 5.00])

#: 只在 D0 给清单 -> D1 建仓一笔，之后**再没有清单**（引擎只跑硬止损，不主动平仓）
PICKS = {D0: [("600000.SH", 4.0)]}

#: 手算锚点（上面 docstring 里逐项算过）
CASH_AFTER_BUY = 494.7449975
EQ_FLAT = 994.7449975          # 收盘 5.00
EQ_DIP = 894.7449975           # 收盘 4.00


def cfg(cash=1000.0, floor=1000.0):
    """手算场景统一配置。

    `hard_stop_pct=None`（**关掉硬止损**）：本文件测的是"爆仓标注"，
    价格路径必须完全决定净值 —— 留着止损的话 D2 的 -20% 会在开盘把票砍掉，
    净值就变成"卖出回款"而不是"持仓市值"，手算要多绕一层（且与标注无关）。
    `exposure_fn=None`：不做市场环境调整 -> 目标仓位恒 1.0，手算才干净。
    """
    return EngineConfig(cash=cash, bankrupt_floor=floor,
                        exposure_fn=None, hard_stop_pct=None)


class TestHandCalcPath(unittest.TestCase):
    """A. 先把"净值路径"钉死 —— 后面所有标注断言都以这些手算数为准。"""

    def test_flat_path_hand_calc(self):
        r = run_engine(frames={"600000.SH": FLAT}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg())
        self.assertEqual(len(r.equity), 6)
        self.assertAlmostEqual(r.equity[0], 1000.0, places=6)        # 期初那一点 = 现金
        for v in r.equity[1:]:
            self.assertAlmostEqual(v, EQ_FLAT, places=6)
        buys = [t for t in r.trades if t.side == "buy"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0].shares, 100)                        # 200 股买不起 -> 砍到 100
        self.assertAlmostEqual(buys[0].price, 5.0025, places=6)      # 5.00 × 1.0005
        self.assertAlmostEqual(buys[0].fee, 5.0050025, places=6)     # 佣金(最低 5) + 过户费

    def test_dip_path_hand_calc(self):
        r = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg())
        want = [1000.0, EQ_FLAT, EQ_DIP, EQ_FLAT, EQ_DIP, EQ_FLAT]
        for got, w in zip(r.equity, want):
            self.assertAlmostEqual(got, w, places=6)
        # 全程只有 D1 那一笔买入（D1 之后没有清单 -> 引擎不主动平仓，也不止损）
        self.assertEqual(len(r.trades), 1)


class TestBankruptBoundary(unittest.TestCase):
    """B. 严格小于的边界 —— 净值恰好 == floor 时**不**标注。"""

    def test_equal_to_floor_never_flags(self):
        r = run_engine(frames={}, benchmark_frame=BENCH, picks={},
                       cfg=cfg(cash=1000.0, floor=1000.0))
        self.assertEqual(r.equity, [1000.0] * 6)
        self.assertIsNone(r.bankrupt_on)          # 1000 < 1000 == False

    def test_floor_zero_never_flags(self):
        r = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(floor=0.0))
        self.assertIsNone(r.bankrupt_on)

    def test_floor_above_everything_flags_first_point(self):
        # 反过来卡另一端：floor 高于任何净值 -> 第一个净值点就标注
        r = run_engine(frames={"600000.SH": FLAT}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(floor=1e12))
        self.assertEqual(r.bankrupt_on, dt.date(2024, 1, 2))


class TestBankruptFlagsFirstLossyPoint(unittest.TestCase):
    """C. 初始 1000 / floor 1000 -> 第一个**成交后**的净值点就标注。"""

    def test_flags_on_first_trading_day(self):
        r = run_engine(frames={"600000.SH": FLAT}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(cash=1000.0, floor=1000.0))
        # D0 是期初（1000.00，不 < 1000）——不标注；D1 买入后扣掉 5.005 元费用就破了线
        self.assertAlmostEqual(r.equity[0], 1000.0, places=6)
        self.assertAlmostEqual(r.equity[1], EQ_FLAT, places=6)
        self.assertEqual(r.bankrupt_on, dt.date(2024, 1, 3))

    def test_not_flagged_before_any_trade(self):
        # 没有清单 -> 没有任何成交 -> 净值恒等于 1000 -> 永不标注
        r = run_engine(frames={"600000.SH": FLAT}, benchmark_frame=BENCH,
                       picks={}, cfg=cfg(cash=1000.0, floor=1000.0))
        self.assertEqual(r.trades, [])
        self.assertIsNone(r.bankrupt_on)


class TestBankruptCrossingDay(unittest.TestCase):
    """D. 亏破线 -> 标注那一天；回到线上不清掉；再破也不覆盖（只记第一次）。"""

    def test_flags_on_the_day_it_crosses(self):
        r = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(cash=1000.0, floor=900.0))
        # D1 = 994.7449975（> 900，不标）-> D2 = 894.7449975（< 900，**标 D2**）
        self.assertAlmostEqual(r.equity[1], EQ_FLAT, places=6)
        self.assertAlmostEqual(r.equity[2], EQ_DIP, places=6)
        self.assertEqual(r.bankrupt_on, dt.date(2024, 1, 4))

    def test_recovery_above_floor_does_not_clear_it(self):
        r = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(cash=1000.0, floor=900.0))
        # D3 回到 994.7449975（> 900）—— 已记的 D2 **不许**被清掉
        self.assertAlmostEqual(r.equity[3], EQ_FLAT, places=6)
        self.assertGreater(r.equity[3], 900.0)
        self.assertEqual(r.bankrupt_on, dt.date(2024, 1, 4))

    def test_only_first_crossing_is_recorded(self):
        r = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(cash=1000.0, floor=900.0))
        # D4 再破一次（894.7449975 < 900）—— 记的仍是第一次的 D2
        self.assertAlmostEqual(r.equity[4], EQ_DIP, places=6)
        self.assertEqual(r.bankrupt_on, dt.date(2024, 1, 4))


class TestBankruptDoesNotChangeAnything(unittest.TestCase):
    """E. 🔴 标注**不许**改变回测本身：这笔钱、这些成交、这些净值点一个都不能动。"""

    def test_curve_and_trades_identical_across_floors(self):
        low = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                         picks=PICKS, cfg=cfg(floor=0.0))          # 永不标注
        high = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                          picks=PICKS, cfg=cfg(floor=1e12))        # 第一个点就标注
        self.assertIsNone(low.bankrupt_on)
        self.assertEqual(high.bankrupt_on, dt.date(2024, 1, 2))
        self.assertEqual(low.equity, high.equity)
        self.assertEqual(low.dates, high.dates)
        self.assertEqual(low.cash, high.cash)
        self.assertEqual([(t.date, t.code, t.side, t.shares, t.price)
                          for t in low.trades],
                         [(t.date, t.code, t.side, t.shares, t.price)
                          for t in high.trades])

    def test_curve_still_covers_every_clock_bar(self):
        """🔴 核心契约：标注**不是**提前退出 —— 净值点数照旧 == 区间交易日数。"""
        r = run_engine(frames={"600000.SH": DIP}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(cash=1000.0, floor=1000.0))
        self.assertEqual(r.bankrupt_on, dt.date(2024, 1, 3))       # 第 2 天就"爆仓"
        self.assertEqual(len(r.equity), 6)                        # 但曲线还是 6 点
        self.assertEqual(r.dates, [pd.Timestamp(x).date() for x in DAYS])
        self.assertEqual(r.report.n_periods, 6)
        # 标注日必须落在净值曲线区间内（不能是 None 之后的幽灵日期）
        self.assertIn(r.bankrupt_on, r.dates)

    def test_bankrupt_on_surfaces_in_format(self):
        r = run_engine(frames={"600000.SH": FLAT}, benchmark_frame=BENCH,
                       picks=PICKS, cfg=cfg(cash=1000.0, floor=1000.0))
        self.assertIn("2024-01-03", r.format())
        plain = run_engine(frames={"600000.SH": FLAT}, benchmark_frame=BENCH,
                           picks=PICKS, cfg=cfg(floor=0.0))
        self.assertIn("未触发", plain.format())


if __name__ == "__main__":
    unittest.main(verbosity=2)
