# -*- coding: utf-8 -*-
"""最短持有期组件测试。人工场景验证冻结持仓仍占名额，风控退出优先。"""

import datetime as dt
import unittest

from cxq_bt import engine
from cxq_bt.engine import EngineConfig
from cxq_bt.sizing import Holding, plan_day


class TestFrozenInPlanDay(unittest.TestCase):
    """plan_day 层：冻结票不动、不换、仍占名额。"""

    def _plan(self, frozen=(), picks=(("C", 5.0), ("D", 4.0), ("E", 3.5))):
        return plan_day(
            holdings=[Holding("A", 1000),
                      Holding("B", 1000)],
            cash=0.0,
            prices={"A": 11.0, "B": 11.0, "C": 11.0, "D": 11.0, "E": 11.0},
            picks=list(picks),
            max_holdings=3,
            frozen_codes=frozen,
        )

    def test_不冻结时两只都清掉(self):
        """现状：持仓 A/B 都不在目标集合 -> 两笔清仓，名额腾出来买 C/D/E 里的前 3。"""
        p = self._plan()
        self.assertEqual(sorted(s.code for s in p.sells), ["A", "B"])

    def test_冻结A则A不清仓(self):
        """手算：冻结 A -> 只剩 B 要清；名额 = 3 - 0(stuck) - 1(frozen) = 2 -> 买 C/D。"""
        p = self._plan(frozen={"A"})
        self.assertEqual([s.code for s in p.sells], ["B"])
        self.assertEqual(sorted(b.code for b in p.buys), ["C", "D"])
        # A 仍在最终持仓里，且没被加减仓
        self.assertGreater(p.final_shares.get("A", 0), 0)

    def test_冻结两只则只剩一个名额(self):
        """手算：冻结 A/B -> 名额 = 3 - 0(stuck) - 2(frozen) = 1 -> **只买 C**。

        这里必须给现金（0 现金时什么都买不了，那条是另一回事）。
        """
        p = plan_day(
            holdings=[Holding("A", 1000), Holding("B", 1000)],
            cash=100_000.0,
            prices={"A": 11.0, "B": 11.0, "C": 11.0, "D": 11.0, "E": 11.0},
            picks=[("C", 5.0), ("D", 4.0), ("E", 3.5)],
            max_holdings=3, frozen_codes={"A", "B"},
        )
        self.assertEqual([s.code for s in p.sells], [])
        self.assertEqual([b.code for b in p.buys], ["C"])

    def test_冻结的票也不许加仓(self):
        """冻结票即使今天还在榜单里，也不加减仓（保持不动）。"""
        p = self._plan(frozen={"A"}, picks=(("A", 5.0), ("C", 4.0), ("D", 3.0)))
        self.assertEqual([s.code for s in p.sells], ["B"])      # B 照清，A 不动
        self.assertNotIn("A", [b.code for b in p.buys])
        self.assertEqual(p.final_shares.get("A"), 1000)          # 不加不减，保持原样
        self.assertNotIn("A", p.target_codes)                    # 没被当成换仓目标

    def test_上游点名优先于冻结(self):
        """冻结 ≠ 免责：A 在 sell_codes 里（上游点名/硬止损）-> 照卖。"""
        p = plan_day(
            holdings=[Holding("A", 1000)],
            cash=0.0, prices={"A": 11.0, "C": 11.0},
            picks=[("C", 5.0)], max_holdings=3,
            sell_codes={"A"}, frozen_codes={"A"},
        )
        self.assertEqual([s.code for s in p.sells], ["A"])


class TestHoldDaysLedger(unittest.TestCase):
    """引擎台账：开仓记 0、每交易日 +1、清仓删掉。"""

    def _strat(self, cfg):
        st = engine.CxqEngine.__new__(engine.CxqEngine)   # 不跑 backtrader 初始化
        st.cfg = cfg
        st._shares, st._entry_price, st._hold_days = {}, {}, {}
        return st

    def test_开仓记0_加仓不重置(self):
        st = self._strat(EngineConfig())
        st._apply_fill("A", "buy", 100, 10.0)
        self.assertEqual(st._hold_days["A"], 0)
        st._hold_days["A"] = 3
        st._apply_fill("A", "buy", 100, 10.5)          # 加仓
        self.assertEqual(st._hold_days["A"], 3)        # 🔴 加仓不重置持有期

    def test_清仓后忘掉(self):
        st = self._strat(EngineConfig())
        st._apply_fill("A", "buy", 100, 10.0)
        st._apply_fill("A", "sell", 100, 11.0)
        self.assertNotIn("A", st._hold_days)

    def test_冻结集合按阈值算(self):
        st = self._strat(EngineConfig(min_hold_days=2))
        st._shares = {"A": 100, "B": 100, "C": 100}
        st._hold_days = {"A": 0, "B": 1, "C": 2}
        self.assertEqual(st._frozen_codes(), {"A", "B"})   # 满 2 天的不再冻结

    def test_关闭时是空集(self):
        st = self._strat(EngineConfig(min_hold_days=0))
        st._shares, st._hold_days = {"A": 100}, {"A": 0}
        self.assertEqual(st._frozen_codes(), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
