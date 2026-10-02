# -*- coding: utf-8 -*-
"""A 股费用与滑点模型。

默认买卖佣金 0.05%，每笔最低 5 元；卖出印花税 0.05%；买卖过户费
0.001%；买卖各使用 0.05% 滑点。所有参数可配置，应按账户与历史时期
核验，不能把默认值当成普遍适用的官方费率。

人工手算：初始资金 100,000，参考价 10 买入 9,900 股，滑点后成交价
10.005，买入费用 50.5152；参考价 11 卖出，滑点后成交价 10.9945，
卖出费用 109.9340，净盈亏约 9,635.60。这是人工测试，不是历史业绩。

AShareCosts 提供纯函数；AShareCommission 将相同规则接入 Backtrader。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AShareCosts", "COSTS", "AShareCommission"]


@dataclass(frozen=True)
class AShareCosts:
    """A股真实费用 + 滑点。全部是**纯函数**，可脱离 backtrader 单独测试。"""

    # ── 费率 ──
    commission_rate: float = 0.0005      # 佣金 万5
    min_commission: float = 5.0          # 佣金最低 5 元
    stamp_duty_rate: float = 0.0005      # 印花税 万5（仅卖出）
    transfer_fee_rate: float = 0.00001   # 过户费 万0.1（双边）
    slippage_rate: float = 0.0005        # 滑点 0.05%（单边）

    # ── 费用 ──
    def buy_fee(self, amount: float) -> float:
        """买入费用 = 佣金(最低5元) + 过户费。"""
        if amount <= 0:
            return 0.0
        return (max(amount * self.commission_rate, self.min_commission)
                + amount * self.transfer_fee_rate)

    def sell_fee(self, amount: float) -> float:
        """卖出费用 = 佣金(最低5元) + 印花税 + 过户费。"""
        if amount <= 0:
            return 0.0
        return (max(amount * self.commission_rate, self.min_commission)
                + amount * self.stamp_duty_rate
                + amount * self.transfer_fee_rate)

    # ── 滑点 ──
    def buy_price(self, ref_price: float) -> float:
        """买入参考价 → 滑点后成交价（买贵）。"""
        return ref_price * (1.0 + self.slippage_rate)

    def sell_price(self, ref_price: float) -> float:
        """卖出参考价 → 滑点后成交价（卖贱）。"""
        return ref_price * (1.0 - self.slippage_rate)

    # ── 组合 ──
    def buy_cost(self, shares: int, ref_price: float) -> float:
        """买入总支出（成交金额 + 费用）。"""
        amount = shares * self.buy_price(ref_price)
        return amount + self.buy_fee(amount)

    def sell_proceeds(self, shares: int, ref_price: float) -> float:
        """卖出净收入（成交金额 − 费用）。"""
        amount = shares * self.sell_price(ref_price)
        return amount - self.sell_fee(amount)


#: 全局默认口径（接口约定）
COSTS = AShareCosts()


# ══════════════════ backtrader 适配器 ══════════════════
from . import compat as _compat  # noqa: E402  必须早于 backtrader
import backtrader as bt  # noqa: E402


class AShareCommission(bt.CommInfoBase):
    """把 AShareCosts 接到 backtrader 的 broker 上。

    🔴 **必须写 percabs=True** —— backtrader/comminfo.py:156-157：

        if self._commtype == self.COMM_PERC and not self.p.percabs:
            self.p.commission /= 100.0

    percabs 默认 False → 传进来的万分之五会被当成十万分之五用，
    **费用直接缩小 100 倍**，而且不会报错，只是结果全错。
    """

    params = (
        ("commission", COSTS.commission_rate),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),          # 🔴 见类文档，少这一行费用差 100 倍
    )

    def __init__(self, costs=None, **kwargs):
        self._costs = costs if costs is not None else COSTS
        super().__init__(**kwargs)

    def _getcommission(self, size, price, pseudoexec):
        """backtrader 回调：size>0 判买入，size<0 判卖出。"""
        amount = abs(size) * price
        if size < 0:
            return self._costs.sell_fee(amount)
        return self._costs.buy_fee(amount)
