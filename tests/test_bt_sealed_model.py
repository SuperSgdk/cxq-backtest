# -*- coding: utf-8 -*-
"""tests/test_bt_sealed_model.py —— 一字板排队模型（数据驱动）：**手算场景，不依赖历史数据**。"""

import datetime as dt
import unittest

from cxq_bt import engine
from cxq_bt.sealed_model import DEFAULT_QUEUE_PCT, PMAX, TIGHT_X, fill_prob


class TestFillProb(unittest.TestCase):
    """p = clamp( 竞价成交额/封单额 / q , 0, PMAX )，q = 你的队列分位。"""

    def test_手算_一(self):
        """x = 0.15、q = 0.35 -> 0.15/0.35 = 0.42857..."""
        self.assertAlmostEqual(fill_prob(0.15e8, 1.0e8, 0.35), 0.15 / 0.35, places=10)

    def test_手算_二_超上限要截断(self):
        """x = 0.50、q = 0.35 -> 1.43 -> 截到 PMAX（**从不假设 100% 一定成交**）。"""
        self.assertEqual(fill_prob(0.5e8, 1.0e8, 0.35), PMAX)

    def test_竞价零成交就是买不进(self):
        self.assertEqual(fill_prob(0.0, 1.0e8, 0.35), 0.0)

    def test_缺任一数据都不猜(self):
        self.assertEqual(fill_prob(None, 1.0e8), 0.0)
        self.assertEqual(fill_prob(1.0e8, None), 0.0)
        self.assertEqual(fill_prob(1.0e8, 0), 0.0)          # 封单 0 = 数据异常 -> 不成交

    def test_队列分位越小越容易成交(self):
        """q 小 = 排得靠前 = 同样 x 下概率更高（单调）。"""
        a, s = 0.10e8, 1.0e8
        self.assertLess(fill_prob(a, s, 0.5), fill_prob(a, s, 0.2))

    def test_默认队列假设参数(self):
        self.assertEqual(DEFAULT_QUEUE_PCT, 0.35)
        self.assertLess(TIGHT_X, DEFAULT_QUEUE_PCT)          # 「封得死」的界要在 q 之内


class _Strat:
    def __init__(self, fn=None, prob=0.0):
        self.cfg = engine.EngineConfig(sealed_fill_prob=prob, sealed_fill_fn=fn)
        self.sealed_fills = []


class TestEngineHook(unittest.TestCase):
    """引擎：数据驱动模型优先于固定概率；模型抛异常绝不许影响主流程。"""

    D = dt.date(2025, 4, 8)
    WHY = "一字板，09:50 前未开板 -> 买不进"

    def setUp(self):
        self._orig = engine.sealed_draw
        engine.sealed_draw = lambda *a: 0.0        # 抽到 0 -> 只要 p>0 就成交

    def tearDown(self):
        engine.sealed_draw = self._orig

    def test_模型说能买就按涨停价成交(self):
        st = _Strat(fn=lambda code, d: 0.4)
        px, why = engine.CxqEngine._sealed_fallback(st, "000001.SZ", self.D, 10.55, self.WHY)
        self.assertEqual(px, 10.55)
        self.assertEqual(st.sealed_fills, [(self.D, "000001.SZ", 0.4)])

    def test_模型说不能买就买不进(self):
        st = _Strat(fn=lambda code, d: 0.0)
        px, why = engine.CxqEngine._sealed_fallback(st, "000001.SZ", self.D, 10.55, self.WHY)
        self.assertIsNone(px)
        self.assertEqual(why, self.WHY)

    def test_模型异常退回买不进而不是崩(self):
        def boom(code, d):
            raise RuntimeError("数据源炸了")
        st = _Strat(fn=boom)
        px, why = engine.CxqEngine._sealed_fallback(st, "000001.SZ", self.D, 10.55, self.WHY)
        self.assertIsNone(px)

    def test_没给模型时用固定概率(self):
        st = _Strat(fn=None, prob=1.0)
        px, why = engine.CxqEngine._sealed_fallback(st, "000001.SZ", self.D, 10.55, self.WHY)
        self.assertEqual(px, 10.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
