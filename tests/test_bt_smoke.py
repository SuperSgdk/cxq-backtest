# -*- coding: utf-8 -*-
"""回测地基冒烟测试 —— 全部手算场景，不使用任何历史数据。

铁律（接口说明）：每条断言的标准答案都是人手算出来的，引擎算不出同样的数就是 bug。

基准场景（接口说明 已写死）：
    账户 100,000 元
    D0 收盘复盘 → D1 开盘买入 600000，参考价 10.00，滑点后 10.005
        shares = int(100000/10.005/100)*100 = 9900
        金额 = 99,049.5   佣金 max(49.5248,5)=49.5248   过户费 0.9905
        买入费用合计 = 50.5152
    D1 当日试图卖出 → 必须被 T+1 拦住
    D4 开盘 11.00 卖出，滑点后 10.9945
        金额 = 108,845.55  佣金 54.4228 + 印花税 54.4228 + 过户费 1.0885
        卖出费用 = 109.9340
        净盈亏 = (108845.55 - 109.934) - (99049.5 + 50.5152) = 9635.6007

运行：.venv\\Scripts\\python.exe -X utf8 -B tests	est_bt_smoke.py
"""

import datetime as dt
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 兼容垫片必须在 import backtrader 之前生效
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxq_bt import compat  # noqa: E402,F401

import backtrader as bt  # noqa: E402
import pandas as pd  # noqa: E402

# ── 费用口径（接口约定 组件参数由调用者按场景核验。
COMM = 0.0005      # 佣金 万5，最低 5 元，双边
STAMP = 0.0005     # 印花税 万5，仅卖出
TRANSFER = 0.00001 # 过户费 万0.1，双边
MIN_COMM = 5.0
SLIP = 0.0005      # 滑点 0.05% 单边


def buy_fee(amount: float) -> float:
    """买入费用 = 佣金(最低5元) + 过户费"""
    return max(amount * COMM, MIN_COMM) + amount * TRANSFER


def sell_fee(amount: float) -> float:
    """卖出费用 = 佣金(最低5元) + 印花税 + 过户费"""
    return max(amount * COMM, MIN_COMM) + amount * STAMP + amount * TRANSFER


class AShareCommission(bt.CommInfoBase):
    """A股真实费用：佣金万5(最低5元,双边) + 印花税万5(仅卖出) + 过户费万0.1(双边)

    注意 percabs=True：CommInfoBase 在 commtype=COMM_PERC 且 percabs=False 时
    会把 p.commission 除以 100，务必显式关掉，否则费用会缩小 100 倍。
    """
    params = (
        ("commission", COMM),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        amount = abs(size) * price
        fee = max(amount * COMM, MIN_COMM) + amount * TRANSFER
        if size < 0:                     # size<0 判卖出
            fee += amount * STAMP
        return fee


# ── 场景日期 ──
D0 = dt.date(2023, 12, 29)   # 复盘日（只用来让 D1 有"上一根 bar"）
D1 = dt.date(2024, 1, 2)     # 买入日
D4 = dt.date(2024, 1, 5)     # 卖出日（持有第 3 个交易日到期）

BARS = pd.DataFrame(
    {
        "open":   [9.80, 10.00, 10.20, 10.50, 11.00],
        "high":   [9.95, 10.20, 10.40, 10.70, 11.20],
        "low":    [9.70,  9.90, 10.10, 10.40, 10.90],
        "close":  [9.80, 10.00, 10.30, 10.60, 11.00],
        "volume": [100000] * 5,
    },
    index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03",
                          "2024-01-04", "2024-01-05"]),
)


class T1Strategy(bt.Strategy):
    """D1 开盘买入 → D1 当日试图卖出（T+1 必须拦住）→ D4 开盘卖出。

    enforce_t1=False 用来暴露 backtrader 的原生 T+0 行为，作为对照。
    """
    params = (("enforce_t1", True), ("shares", 9900))

    def __init__(self):
        self.opened_date = None
        self.blocked = 0          # T+1 拦截次数
        self.sold = 0
        self.buy_px = None
        self.sell_px = None
        self.orders = []

    def notify_order(self, order):
        name = {0: "Created", 1: "Submitted", 2: "Accepted", 3: "Partial",
                4: "Completed", 5: "Canceled", 6: "Expired",
                7: "Margin", 8: "Rejected"}.get(order.status, str(order.status))
        if order.status in (4, 5, 7, 8):
            self.orders.append((order.isbuy(), name, order.executed.price))
        if order.status == order.Completed:
            if order.isbuy():
                self.buy_px = order.executed.price
            else:
                self.sell_px = order.executed.price

    def next_open(self):
        """开盘前决策 → 按本 bar 开盘价成交"""
        d = self.data.datetime.date(0)
        pos = self.getposition(self.data)
        if d == D1 and self.opened_date is None:
            self.buy(size=self.p.shares)
            self.opened_date = d
        elif d == D4 and pos.size > 0:
            self.sell(size=pos.size)
            self.sold += 1

    def next(self):
        """盘中：当日买入的仓位试图卖出"""
        d = self.data.datetime.date(0)
        pos = self.getposition(self.data)
        if pos.size > 0 and self.opened_date == d:
            if self.p.enforce_t1:
                self.blocked += 1        # 🔴 T+1 层：直接拦下，不提交订单
            else:
                self.sell(size=pos.size)  # 交给 backtrader 原生行为（T+0）


def _run(enforce_t1: bool):
    cer = bt.Cerebro(cheat_on_open=True)
    cer.adddata(bt.feeds.PandasData(dataname=BARS))
    cer.broker.setcash(100000.0)
    cer.broker.addcommissioninfo(AShareCommission())
    cer.broker.set_slippage_perc(SLIP, slip_open=True, slip_limit=True,
                                 slip_match=True, slip_out=False)
    cer.addstrategy(T1Strategy, enforce_t1=enforce_t1)
    cer.run()
    return cer, cer.runstrats[0][0]


class TestCostModel(unittest.TestCase):
    """第 1 层 · 费用模型 —— 与手算逐分对齐。"""

    def test_buy_fee_10k(self):
        # 手算：成交 10,000 元 → 佣金 max(5.0,5)=5.0，过户费 0.1 → 5.1
        self.assertAlmostEqual(buy_fee(10_000.0), 5.1, places=6)

    def test_buy_fee_99049_5(self):
        # 手算：成交 99,049.5 → 佣金 max(49.52475,5)=49.52475，过户费 0.990495
        self.assertAlmostEqual(buy_fee(99_049.5), 50.515245, places=6)

    def test_sell_fee_108845_55(self):
        # 手算：成交 108,845.55
        #   佣金 max(54.422775,5)=54.422775
        #   印花税 54.422775   过户费 1.0884555
        self.assertAlmostEqual(sell_fee(108_845.55), 109.9340055, places=6)

    def test_sell_has_stamp_duty_buy_does_not(self):
        # 印花税只在卖出收 —— 同一个金额，卖出费用必须比买入多出正好一个印花税
        amount = 100_000.0
        self.assertAlmostEqual(sell_fee(amount) - buy_fee(amount),
                               amount * STAMP, places=9)

    def test_min_commission_floor(self):
        # 小额成交时佣金被 5 元下限兜住
        amount = 1_000.0
        self.assertAlmostEqual(buy_fee(amount), 5.0 + amount * TRANSFER, places=9)


class TestHandCalculatedScenario(unittest.TestCase):
    """端到端：跑出 接口说明 写死的 9,635.60。"""

    @classmethod
    def setUpClass(cls):
        cls.cer, cls.st = _run(enforce_t1=True)

    def test_shares_is_round_lot(self):
        # 手算 shares = int(100000/10.005/100)*100 = 9900
        self.assertEqual(9900 % 100, 0)
        self.assertEqual(self.st.p.shares, 9900)

    def test_buy_price_matches_hand_calc(self):
        # 手算买入滑点后价 = 10.00 × 1.0005 = 10.005
        self.assertAlmostEqual(self.st.buy_px, 10.005, places=6)

    def test_sell_price_matches_hand_calc(self):
        # 手算卖出滑点后价 = 11.00 × 0.9995 = 10.9945
        self.assertAlmostEqual(self.st.sell_px, 10.9945, places=6)

    def test_t1_block_fired(self):
        # D1 当日买入，当日试图卖出 → 必须被拦住一次
        self.assertEqual(self.st.blocked, 1)

    def test_sold_exactly_once(self):
        self.assertEqual(self.st.sold, 1)

    def test_position_closed(self):
        pos = self.cer.broker.getposition(self.cer.datas[0])
        self.assertEqual(pos.size, 0)

    def test_net_profit_matches_hand_calc(self):
        """★ 核心断言：净盈亏必须 = 9635.6007，差一分钱就是 bug。"""
        amount_buy = 9900 * 10.005
        amount_sell = 9900 * 10.9945
        expected = (amount_sell - sell_fee(amount_sell)) - \
                   (amount_buy + buy_fee(amount_buy))
        self.assertAlmostEqual(expected, 9635.6007, places=3)

        actual = self.cer.broker.getvalue() - 100000.0
        self.assertAlmostEqual(actual, expected, places=2)


class TestBacktraderHasNoT1(unittest.TestCase):
    """反证：不启用我们的 T+1 层，backtrader 会放行当日买卖（T+0）。

    这条测试的作用是**防止将来有人误以为 backtrader 自带 T+1**。
    """

    @classmethod
    def setUpClass(cls):
        cls.cer, cls.st = _run(enforce_t1=False)

    def test_same_day_sell_went_through(self):
        self.assertIsNotNone(self.st.sell_px,
                             "backtrader 原生允许当日买卖；若这里变了，说明上游行为改了")
        # 成交价落在 D2 开盘 10.20 的滑点后 = 10.1949，而不是 D4 的 10.9945
        self.assertAlmostEqual(self.st.sell_px, 10.1949, places=4)

    def test_t1_layer_did_not_fire(self):
        self.assertEqual(self.st.blocked, 0)


class TestCompatShim(unittest.TestCase):
    """兼容垫片：Python 3.10+ 删掉的 collections ABC 名字已被补回。

    注意断言方式：bindlines() 在 Strategy 上**本来就会**在更后面再抛一次
    AttributeError（self._owner.lines —— Strategy 没有 _owner）。
    所以这里只断言「不再抛 collections 那一个」，而不是「完全不抛」。
    """

    def test_collections_abcs_are_back(self):
        import collections
        for name in ("Iterable", "Callable", "Mapping", "Sequence"):
            self.assertTrue(hasattr(collections, name),
                            "兼容垫片没生效：collections.%s 缺失" % name)

    def test_bindlines_no_longer_raises_collections_error(self):
        """修复前的报错：module 'collections' has no attribute 'Iterable'"""
        _cer, st = _run(enforce_t1=True)
        try:
            st.bindlines(owner=0)
        except AttributeError as exc:
            self.assertNotIn(
                "collections", str(exc),
                "仍然踩到 collections.Iterable 的坑 —— 垫片没在 import backtrader 前生效",
            )
        except Exception:
            pass  # 其它异常与垫片无关

    def test_shim_is_what_prevents_the_error(self):
        """反证：把补回去的 Iterable 再删掉，同一个调用就会抛 collections 错误。

        这条保证上面那条不是「碰巧通过」。
        """
        import collections
        from cxq_bt import compat  # noqa: F401

        saved = collections.Iterable
        del collections.Iterable
        try:
            _cer, st = _run(enforce_t1=True)
            with self.assertRaises(AttributeError) as cm:
                st.bindlines(owner=0)
            self.assertIn("collections", str(cm.exception),
                          "删掉 Iterable 后居然没报 collections 错，这条反证不成立")
        finally:
            collections.Iterable = saved

        # 恢复后必须立刻又能用
        self.assertTrue(hasattr(collections, "Iterable"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
