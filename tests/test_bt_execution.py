# -*- coding: utf-8 -*-
r"""第 4 层 · 成交时点单测 —— 手算场景，不用历史数据。

运行：.venvScriptspython.exe -X utf8 -B tests	est_bt_execution.py
"""

import datetime as dt
import inspect
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt import execution as ex  # noqa: E402

DAY = "2024-01-02"


def day_frame(rows):
    """rows = [(时刻, o, h, l, c), ...] -> 带 (trade_date, trade_time) 索引的分钟线。"""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(DAY), pd.Timestamp(DAY + " " + t)) for t, *_ in rows],
        names=["trade_date", "trade_time"])
    frame = pd.DataFrame(
        [{"open": o, "high": h, "low": lo, "close": c} for _, o, h, lo, c in rows],
        index=idx)
    # 手算行情明确逐分钟补齐：沿用上一根构造 OHLC，绝非补真实历史缺口。
    labels = pd.date_range(idx[0][1], idx[-1][1], freq="min")
    complete = pd.MultiIndex.from_tuples([(pd.Timestamp(DAY), t) for t in labels],
                                         names=["trade_date", "trade_time"])
    return frame.reindex(complete).ffill()


# ── 手算用的三类日线 ──
# 1) 常规票：开盘 10.00，之后波动，涨停价 11.00
NORMAL = day_frame([
    ("09:30:00", 10.00, 10.00, 10.00, 10.00),   # 集合竞价那一笔
    ("09:31:00", 10.05, 10.10, 10.00, 10.08),
    ("09:32:00", 10.08, 10.12, 10.05, 10.10),
    ("09:35:00", 10.10, 10.20, 10.05, 10.15),
    ("09:55:00", 10.15, 10.30, 10.10, 10.25),
])

# 2) 一字板：开=高=低=收=涨停价 11.00，全天不开
SEALED = day_frame([
    ("09:30:00", 11.00, 11.00, 11.00, 11.00),
    ("09:31:00", 11.00, 11.00, 11.00, 11.00),
    ("09:35:00", 11.00, 11.00, 11.00, 11.00),
    ("09:55:00", 11.00, 11.00, 11.00, 11.00),
])

# 3) T字板：09:30 涨停，09:35 开板（low 10.80 < 11.00），之后回封
T_BOARD = day_frame([
    ("09:30:00", 11.00, 11.00, 11.00, 11.00),
    ("09:34:00", 11.00, 11.00, 11.00, 11.00),
    ("09:35:00", 11.00, 11.00, 10.80, 10.90),   # <- 首次开板
    ("09:36:00", 10.90, 11.00, 10.85, 11.00),
    ("09:55:00", 11.00, 11.00, 10.95, 11.00),
])

# 4) 开板太晚：09:55 才开
LATE_BREAK = day_frame([
    ("09:30:00", 11.00, 11.00, 11.00, 11.00),
    ("09:40:00", 11.00, 11.00, 11.00, 11.00),
    ("09:55:00", 11.00, 11.00, 10.80, 10.90),   # <- 晚于 09:50
])


class TestTimeConstants(unittest.TestCase):
    def test_exec_is_0930(self):
        self.assertEqual(ex.EXEC_TIME, dt.time(9, 30))

    def test_t_board_window_ends_0950(self):
        self.assertEqual(ex.T_BOARD_WINDOW_END, dt.time(9, 50))

    def test_decision_cutoff_is_prev_close_not_0925(self):
        """🔴 决策在**前一天收盘后**，不是 T+1 日 09:25。"""
        self.assertIn("T 日收盘后", ex.DECISION_CUTOFF_LABEL)
        self.assertNotIn("9:25", ex.DECISION_CUTOFF_LABEL)
        self.assertNotIn("09:25", ex.DECISION_CUTOFF_LABEL)


class TestDataGate(unittest.TestCase):
    """数据闸门：只能看到 <= t 的 bar。"""

    def test_as_of_filters_future(self):
        v = ex.as_of(NORMAL, dt.time(9, 31))
        self.assertEqual(len(v), 1, "09:31时只能看到已完成的09:30根")
        times = [x.time() for x in v.index.get_level_values("trade_time")]
        self.assertEqual(times, [dt.time(9, 30)])

    def test_as_of_never_exposes_later_bars(self):
        """🔴 反证：闸门之后，结果里的最大时刻必须 <= t。"""
        for t in (dt.time(9, 30), dt.time(9, 32), dt.time(9, 40),
                  dt.time(9, 55), dt.time(15, 0)):
            v = ex.as_of(NORMAL, t)
            if len(v) == 0:
                continue
            latest = max(x.time() for x in v.index.get_level_values("trade_time"))
            completed = (dt.datetime.combine(dt.date(2024, 1, 2), latest)
                         + dt.timedelta(minutes=1)).time()
            self.assertLessEqual(completed, t, "闸门提前暴露未完成bar：t=%s bar=%s" % (t, latest))

    def test_as_of_returns_copy_not_view(self):
        """闸门返回副本 —— 改它不会污染原表。"""
        v = ex.as_of(NORMAL, dt.time(9, 31))
        v.iloc[0, v.columns.get_loc("open")] = 999.0
        self.assertAlmostEqual(float(NORMAL.iloc[0]["open"]), 10.00)

    def test_as_of_empty_before_open(self):
        self.assertEqual(len(ex.as_of(NORMAL, dt.time(9, 29))), 0)


class TestOpenPrice(unittest.TestCase):
    def test_takes_0930_bar_open(self):
        """09:30 那根 bar 是集合竞价，它的 open 就是开盘价。"""
        self.assertAlmostEqual(ex.open_price(NORMAL), 10.00)

    def test_sealed_open_is_limit(self):
        self.assertAlmostEqual(ex.open_price(SEALED), 11.00)

    def test_none_when_no_0930_bar(self):
        df = day_frame([("10:00:00", 1.0, 1.0, 1.0, 1.0)])
        self.assertIsNone(ex.open_price(df))

    def test_none_on_empty(self):
        self.assertIsNone(ex.open_price(NORMAL.iloc[0:0]))
        self.assertIsNone(ex.open_price(None))


class TestFirstBreak(unittest.TestCase):
    def test_finds_first_break(self):
        info = ex.first_break(T_BOARD, 11.00)
        self.assertIsNotNone(info)
        self.assertEqual(info.at, dt.time(9, 35))
        self.assertAlmostEqual(info.open_, 11.00)
        self.assertAlmostEqual(info.close, 10.90)
        self.assertAlmostEqual(info.low, 10.80)

    def test_none_when_never_breaks(self):
        self.assertIsNone(ex.first_break(SEALED, 11.00))

    def test_respects_deadline(self):
        """09:55 才开板 -> 09:50 的窗口里找不到。"""
        self.assertIsNone(ex.first_break(LATE_BREAK, 11.00))
        info = ex.first_break(LATE_BREAK, 11.00, deadline=dt.time(10, 0))
        self.assertIsNotNone(info)
        self.assertEqual(info.at, dt.time(9, 55))

    def test_ignores_tiny_float_noise(self):
        """低点只差 0.001（浮点噪声）不算开板 —— 容差是半个报价单位。"""
        df = day_frame([("09:30:00", 11.00, 11.00, 11.00, 11.00),
                        ("09:31:00", 11.00, 11.00, 10.999, 11.00)])
        self.assertIsNone(ex.first_break(df, 11.00))


class TestResolveBuyFill(unittest.TestCase):
    """用人工日级和分钟事件检查成交时间、缺数处理与价格模型。"""

    def test_first_break_requires_touching_limit_first(self):
        """人工未触及涨停的行情不能因为low低于涨停价被当成开板事件。"""
        self.assertIsNone(ex.first_break(NORMAL, 11.00),
                          "普通票最高价 10.30 < 涨停价 11.00，不该有开板信号")

    def test_normal_stock_fills_at_open(self):
        r = ex.resolve_buy_fill(code="A", day_bars=NORMAL, up_limit=11.00)
        self.assertTrue(r.filled)
        self.assertAlmostEqual(r.price, 10.00)
        self.assertEqual(r.source, "开盘价")

    def test_sealed_board_cannot_buy(self):
        """一字板开盘买不进去。"""
        r = ex.resolve_buy_fill(code="A", day_bars=SEALED, up_limit=11.00)
        self.assertFalse(r.filled)
        self.assertIsNone(r.price)
        self.assertIn("未开板", r.reason)

    def test_t_board_fills_at_break(self):
        """T字板 09:35 开板 -> 买得进，完成09:35 bar后按09:36 open近似成交。"""
        r = ex.resolve_buy_fill(code="A", day_bars=T_BOARD, up_limit=11.00,
                                break_price_mode="open")
        self.assertTrue(r.filled)
        self.assertAlmostEqual(r.price, 10.90)
        self.assertEqual(r.break_at, dt.time(9, 36))
        self.assertIn("开板", r.reason)

    def test_t_board_close_mode(self):
        """禁止用确认 bar 的 close 作为任意成交价。"""
        with self.assertRaises(ValueError):
            ex.resolve_buy_fill(code="A", day_bars=T_BOARD, up_limit=11.00,
                                break_price_mode="close")

    def test_t_board_low_mode(self):
        with self.assertRaises(ValueError):
            ex.resolve_buy_fill(code="A", day_bars=T_BOARD, up_limit=11.00,
                                break_price_mode="low")

    def test_late_break_cannot_buy(self):
        """开板晚于 09:50 -> 视为当天买不进。"""
        r = ex.resolve_buy_fill(code="A", day_bars=LATE_BREAK, up_limit=11.00)
        self.assertFalse(r.filled)
        self.assertIn("未开板", r.reason)

    def test_no_open_price_cannot_buy(self):
        r = ex.resolve_buy_fill(code="A", day_bars=NORMAL.iloc[0:0], up_limit=11.0)
        self.assertFalse(r.filled)
        self.assertIn("没有 09:30 开盘价", r.reason)

    def test_authoritative_status_sealed_wins(self):
        """乙优先：权威字段说一字板 -> 即使行情不像，也按一字板处理。"""
        class FakeBar:
            limit_status = "一字板"
        r = ex.resolve_buy_fill(code="A", day_bars=T_BOARD, up_limit=11.00,
                                bar=FakeBar())
        self.assertTrue(r.filled, "一字板但 09:35 开板了 -> 能买")
        self.assertEqual(r.break_at, dt.time(9, 36))


class TestResolveSellFill(unittest.TestCase):
    def test_normal_sells_at_open(self):
        r = ex.resolve_sell_fill(code="A", day_bars=NORMAL)
        self.assertTrue(r.filled)
        self.assertAlmostEqual(r.price, 10.00)

    def test_blocked_by_rule_engine(self):
        r = ex.resolve_sell_fill(code="A", day_bars=NORMAL, blocked=True,
                                 block_reason="T+1 当日买入不可卖")
        self.assertFalse(r.filled)
        self.assertEqual(r.reason, "T+1 当日买入不可卖")

    def test_open_sealed_down_limit_defers(self):
        """开盘就是跌停 -> 卖不出，顺延。"""
        df = day_frame([("09:30:00", 9.00, 9.00, 9.00, 9.00)])
        r = ex.resolve_sell_fill(code="A", day_bars=df, down_limit=9.00)
        self.assertFalse(r.filled)
        self.assertIn("顺延", r.reason)

    def test_no_price_cannot_sell(self):
        r = ex.resolve_sell_fill(code="A", day_bars=NORMAL.iloc[0:0])
        self.assertFalse(r.filled)


class TestExecutionLayerRedLine(unittest.TestCase):
    """🔴 执行层红线：不许改变「买不买」。

        该组件约定：买不买是**前一天复盘**定死的；执行层只回答「能不能成交、什么价」。
        这条红线做进**函数签名** —— resolve_* 收不到也返回不了买入清单。
        """

    def test_resolve_buy_fill_has_no_picks_parameter(self):
        params = set(inspect.signature(ex.resolve_buy_fill).parameters)
        for forbidden in ("picks", "target_codes", "plan", "selected"):
            self.assertNotIn(forbidden, params,
                             "执行层不许拿到买入清单（发现参数 %s）" % forbidden)

    def test_resolve_sell_fill_has_no_picks_parameter(self):
        params = set(inspect.signature(ex.resolve_sell_fill).parameters)
        for forbidden in ("picks", "target_codes", "plan", "selected"):
            self.assertNotIn(forbidden, params)

    def test_fill_result_carries_no_decision(self):
        """返回值只有「成交与否 + 价格 + 原因」，没有任何可用来改清单的字段。"""
        fields = set(ex.FillResult.__dataclass_fields__)
        for forbidden in ("picks", "target_codes", "should_buy", "action"):
            self.assertNotIn(forbidden, fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
