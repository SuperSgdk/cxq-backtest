# -*- coding: utf-8 -*-
"""实验一字板随机成交组件测试。人工场景仅检查参数与执行行为。
模型值不是实测成交概率，测试结果不是历史收益。"""

import datetime as dt
import unittest

from cxq_bt import engine


class TestSealedDraw(unittest.TestCase):
    """确定性抽样：同一 (code, 日期, 种子) 永远同一个数（回测必须可复现）。"""

    D = dt.date(2025, 4, 8)

    def test_同一输入永远同一个数(self):
        a = engine.sealed_draw("000001.SZ", self.D, 20260917)
        b = engine.sealed_draw("000001.SZ", self.D, 20260917)
        self.assertEqual(a, b)

    def test_落在0到1之间(self):
        for code in ("000001.SZ", "600519.SH", "300750.SZ", "000002.SZ"):
            v = engine.sealed_draw(code, self.D, 20260917)
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)

    def test_换代码换日期换种子都会变(self):
        base = engine.sealed_draw("000001.SZ", self.D, 20260917)
        self.assertNotEqual(base, engine.sealed_draw("000002.SZ", self.D, 20260917))
        self.assertNotEqual(base, engine.sealed_draw("000001.SZ", dt.date(2025, 4, 9), 20260917))
        self.assertNotEqual(base, engine.sealed_draw("000001.SZ", self.D, 1))

    def test_分布大致均匀(self):
        """1000 个不同日期的抽样，落在 [0,0.5) 的比例应在 0.4~0.6 之间（手算可核）。"""
        n = hit = 0
        d = dt.date(2025, 1, 1)
        for i in range(1000):
            v = engine.sealed_draw("000001.SZ", d + dt.timedelta(days=i), 7)
            n += 1
            hit += 1 if v < 0.5 else 0
        self.assertTrue(0.4 <= hit / n <= 0.6, "实际 %.3f" % (hit / n))


class _FakeStrat:
    """只带 `cfg` 与 `sealed_fills` 的假策略 —— 直接调 `_sealed_fallback`。"""

    def __init__(self, prob):
        self.cfg = engine.EngineConfig(sealed_fill_prob=prob)
        self.sealed_fills = []


class TestSealedFallback(unittest.TestCase):
    D = dt.date(2025, 4, 8)
    WHY = "一字板，09:50 前未开板 -> 买不进"

    def setUp(self):
        self._orig = engine.sealed_draw

    def tearDown(self):
        engine.sealed_draw = self._orig

    def test_p等于0时与历史行为完全一致(self):
        """默认口径：一个字都不改 —— 仍旧「买不进」。"""
        engine.sealed_draw = lambda *a: 0.0          # 哪怕抽到 0 也不成交
        st = _FakeStrat(0.0)
        px, why = engine.CxqEngine._sealed_fallback(st, "000001.SZ", self.D, 10.0, self.WHY)
        self.assertIsNone(px)
        self.assertEqual(why, self.WHY)
        self.assertEqual(st.sealed_fills, [])

    def test_p等于1时按涨停价成交并记账(self):
        engine.sealed_draw = lambda *a: 0.999
        st = _FakeStrat(1.0)
        px, why = engine.CxqEngine._sealed_fallback(st, "000001.SZ", self.D, 10.55, self.WHY)
        self.assertEqual(px, 10.55)                  # 一字板只能按**涨停价**成交
        self.assertEqual(why, "")
        self.assertEqual(st.sealed_fills, [(self.D, "000001.SZ", 1.0)])

    def test_抽不中就不成交(self):
        engine.sealed_draw = lambda *a: 0.9
        st = _FakeStrat(0.5)
        px, why = engine.CxqEngine._sealed_fallback(st, "000001.SZ", self.D, 10.55, self.WHY)
        self.assertIsNone(px)
        self.assertEqual(why, self.WHY)
        self.assertEqual(st.sealed_fills, [])

    def test_抽中就成交_临界值手算(self):
        """p=0.3，抽到 0.3 -> **不算成交**（严格小于）；抽到 0.299999 -> 成交。"""
        st = _FakeStrat(0.3)
        engine.sealed_draw = lambda *a: 0.3
        self.assertIsNone(engine.CxqEngine._sealed_fallback(
            st, "000001.SZ", self.D, 10.55, self.WHY)[0])
        engine.sealed_draw = lambda *a: 0.299999
        self.assertEqual(engine.CxqEngine._sealed_fallback(
            st, "000001.SZ", self.D, 10.55, self.WHY)[0], 10.55)
        self.assertEqual(len(st.sealed_fills), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
