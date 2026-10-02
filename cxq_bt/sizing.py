# -*- coding: utf-8 -*-
"""仓位、现金与整手计划函数。

根据外部分数或目标、仓位上限、最小权重和可用现金计算计划。分数
使用比例分配，不含私有评分生成逻辑。买不到一手的标的按规则剔除，
受限持仓保留资产属性，不能先当成已卖现金。阈值与市场仓位系数由
调用者配置，应按输入尺度解释。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .costs import COSTS, AShareCosts

__all__ = [
    "MAX_HOLDINGS", "LOT", "DEFAULT_MIN_WEIGHT",
    "REASON_OPEN", "REASON_ADD", "REASON_TRIM", "REASON_CLOSE", "REASON_FLAT",
    "allocate_weights", "round_lot", "target_shares", "tiered_exposure",
    "Holding", "BuyOrder", "SellOrder", "DayPlan", "plan_day",
]

#: 🔴 由 **dataclass 机制隐式调用**的钩子（`Holding(...)` 构造后自动跑）——
#:    静态扫描看不到，**别删**。理由见 cxq_bt/rules.py 的 DUNDER_HOOKS 说明。
DUNDER_HOOKS = ("__post_init__",)

#: 单日最多持有几只票 —— 组件参数由调用者按场景核验。
MAX_HOLDINGS = 3

#: A股最小交易单位（1 手 = 100 股）
LOT = 100

# 整手后不足一手的目标剔除；能够买入的目标仍按约定权重参与分配。
DEFAULT_MIN_WEIGHT = 0.0

REASON_OPEN = "建仓"
REASON_ADD = "加仓"
REASON_TRIM = "减仓"
REASON_CLOSE = "清仓"
REASON_FLAT = "环境空仓"

#: 被剔除的原因
DROP_LOW_WEIGHT = "权重低于下限 %.0f%%"
DROP_NO_LOT = "整手化后买不到 1 手"


# ══════════════════ 权重 ══════════════════

def allocate_weights(scores: Sequence[float]) -> list:
    """**纯按评分归一化**（当前组件配置）。

        4 分 : 6 分 -> [0.4, 0.6]
        95 分 : 61 分 -> [95/156, 61/156] = [0.6090, 0.3910]

        不再做区间截断 —— 接口说明 里那张权重区间表（[40%,60%] 等）已作废。
        """
    n = len(scores)
    if n == 0:
        return []
    for s in scores:
        if s is None or (isinstance(s, float) and math.isnan(s)) or s < 0:
            raise ValueError("打分必须是 >= 0 的有限数，收到 %r" % (s,))
    total = float(sum(scores))
    if total <= 0:
        return [1.0 / n] * n          # 全 0 分 -> 等权（不伪造打分，只是等分）
    return [float(s) / total for s in scores]


def round_lot(amount, price, lot: int = LOT) -> int:
    """把金额换算成**整手**股数（向下取整，永不超买）。None / NaN / 非正数 -> 0。"""
    if amount is None or price is None:
        return 0
    try:
        amount = float(amount)
        price = float(price)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(amount) or not math.isfinite(price):
        return 0
    if price <= 0 or amount <= 0:
        return 0
    return int(amount / price / lot) * lot


def target_shares(equity: float, weight: float, price: float,
                  lot: int = LOT) -> int:
    """目标股数 = 整手化(总权益 * 权重 / 价格)。"""
    return round_lot(equity * weight, price, lot)


def _price_of(prices: Mapping, code: str) -> Optional[float]:
    """取有效参考价。None / NaN / <=0 一律视为「没有价格」。"""
    px = prices.get(code)
    if px is None:
        return None
    try:
        px = float(px)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(px) or px <= 0:
        return None
    return px


# ══════════════════ 市场环境 -> 总仓位系数 ══════════════════

def tiered_exposure(tiers: Sequence, flat_below: Optional[float] = None
                    ) -> Callable[[float], float]:
    """工厂：把「分档表」变成一个 exposure_fn（市场环境评分 -> 目标总仓位系数）。

        参数
        ----
        tiers : [(最低环境评分, 仓位系数), ...]，按评分**降序**。
                例：((80, 1.00), (60, 0.80), (40, 0.60))
        flat_below : 低于这个评分 -> 返回 0（空仓）。None = 不设空仓线。

        手算示例（tiers=((80,1.0),(60,0.8),(40,0.6)), flat_below=20）：
            85 -> 1.0（满仓）
            70 -> 0.8
            45 -> 0.6
            20 -> 0.6   （等于 flat_below，不算低于）
            15 -> 0.0   （低于 20 -> 空仓）
             5 -> 0.0

        🔴 具体分档数值由用户/上游策略定，引擎不猜。
           这里只提供机制，数值放在调用方。
        """
    normalized = tuple(sorted(((float(lo), float(k)) for lo, k in tiers),
                              key=lambda x: -x[0]))

    def fn(score) -> float:
        s = float(score)
        if flat_below is not None and s < float(flat_below):
            return 0.0
        for lo, k in normalized:
            if s >= lo:
                return k
        return 0.0

    return fn


# ══════════════════ 数据契约 ══════════════════

@dataclass(frozen=True)
class Holding:
    """当前持仓。"""
    code: str
    shares: int

    def __post_init__(self):
        if self.shares < 0:
            raise ValueError("%s 持仓股数不能为负：%d" % (self.code, self.shares))


@dataclass(frozen=True)
class BuyOrder:
    code: str
    shares: int
    ref_price: float
    reason: str
    amount: float          # 成交金额（不含费用）
    fee: float             # 买入费用


@dataclass(frozen=True)
class SellOrder:
    code: str
    shares: int
    ref_price: float
    reason: str
    amount: float
    fee: float


@dataclass
class DayPlan:
    """一天的交易计划 —— 引擎按这个执行。"""
    sells: list = field(default_factory=list)
    buys: list = field(default_factory=list)
    target_codes: list = field(default_factory=list)
    target_weights: dict = field(default_factory=dict)
    equity: float = 0.0
    cash_before: float = 0.0
    cash_after: float = 0.0
    #: 执行完这份计划后，每只票的最终股数（0 表示已清空）
    final_shares: dict = field(default_factory=dict)
    #: 卖不掉的持仓（无价格 / 停牌）—— 它们继续占名额
    stuck_codes: tuple = ()
    #: 被剔除的候选 [(code, 原因), ...]
    dropped: tuple = ()
    #: 本次采用的总仓位系数（来自市场环境评分）
    exposure: float = 1.0
    #: 是否因「环境评分低于空仓线」而清仓
    flat: bool = False

    @property
    def holdings_after(self) -> int:
        """执行完这份计划后，持有几只票 —— **必须 <= MAX_HOLDINGS**。"""
        return sum(1 for s in self.final_shares.values() if s > 0)


# ══════════════════ 选目标 + 定权重 ══════════════════

def _select_targets(ordered: list, equity: float, exposure: float,
                    prices: Mapping, lot: int, costs: AShareCosts,
                    min_weight: float, tol: float = 1e-12):
    """反复剔除「权重低于下限」或「整手化后买不到 1 手」的票，直到稳定。

    返回 (存活的 [(code, score), ...], {code: weight}, [(code, 原因), ...])

    手算示例 1（权重下限，两只票，下限 20%）：
        scores = [92, 8] -> 权重 [0.92, 0.08]
        0.08 < 0.20 -> 剔除第二只
        重新归一化 -> [1.00] -> 第一只满仓

    手算示例 2（买不到一手）：
        权益 100,000，exposure 1.0，下限 20%
        两只票 scores = [61, 39]，价格都是 500 元
        权重 [0.61, 0.39]
        第二只：100000*0.39 = 39,000 / 500 = 78 股 -> 整手 0 股 -> 剔除
        重新归一化 -> [1.00] -> 第一只：100000/500 = 200 股
    """
    alive = list(ordered)
    dropped: list = []
    while alive:
        weights = allocate_weights([s for _, s in alive])
        drop_codes = set()
        for (code, _), w in zip(alive, weights):
            if w < min_weight - tol:
                drop_codes.add(code)
                dropped.append((code, DROP_LOW_WEIGHT % (min_weight * 100)))
            elif target_shares(equity * exposure, w, prices[code], lot) == 0:
                drop_codes.add(code)
                dropped.append((code, DROP_NO_LOT))
        if not drop_codes:
            return alive, {c: w for (c, _), w in zip(alive, weights)}, dropped
        alive = [(c, s) for c, s in alive if c not in drop_codes]
    return [], {}, dropped


# ══════════════════ 核心：一天的交易计划 ══════════════════

def plan_day(
    *,
    holdings: Iterable,
    cash: float,
    prices: Mapping,
    picks: Sequence,
    sell_codes: Iterable = (),
    sell_reasons: Optional[Mapping] = None,
    max_holdings: int = MAX_HOLDINGS,
    lot: int = LOT,
    costs: AShareCosts = COSTS,
    score_threshold: Optional[float] = None,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    frozen_codes: Iterable = (),
    buy_blocked_codes: Iterable = (),
    market_score: Optional[float] = None,
    exposure_fn: Optional[Callable[[float], float]] = None,
    valuation_prices: Optional[Mapping] = None,
    buy_prices: Optional[Mapping] = None,
    sell_prices: Optional[Mapping] = None,
    can_buy: Optional[Mapping] = None,
    can_sell: Optional[Mapping] = None,
) -> DayPlan:
    """算出 T+1 开盘要做哪些买卖。

            参数
            ----
            holdings : 当前持仓（Holding 列表）。**超过 max_holdings 只直接报错**
            cash     : 可用现金
            prices   : code -> 买入参考价（T+1 开盘价）。缺失/NaN/非正 -> 按「没有价格」处理
            picks    : 当日推荐清单 [(code, score), ...]，**按分数降序取前 N 名**
            sell_codes : 外部判断要**强制清仓**的票，两个来源：
                         ① 上游 sell_list（点名要清）；② 引擎硬止损（跌破阈值）。
                         **不管当前有没有持有，当天都不许买进**
            sell_reasons : code -> 出场原因（可选）。只覆盖**强制清仓**那一笔的 reason 字符串，
                         用来把「硬止损 -6.05%…」「上游清仓」带进成交流水；
                         缺省 = REASON_CLOSE（"清仓"）。**不影响减仓与建仓原因**。
            max_holdings : 硬上限，默认 3
            score_threshold : **评分阈值**。分数 < 它的票不买。None = 不设阈值。
                              🔴 具体数值待上游策略评分体系确定后填入，引擎不猜。
            min_weight  : **单票权重下限**。低于它的票剔除，仓位留给其它票。
            frozen_codes : **冻结票**：
                        持仓未满 `min_hold_days` 个交易日的票 -> **保持不动**
                        （不因掉出榜单而清仓、也不加减仓），但仍然**占一个名额**。
                        例外：它在 `sell_codes` 里（上游点名 / 硬止损）-> 照卖，风控优先。
            buy_blocked_codes : 仅禁止建仓/加仓；已有持仓仍按评分参与保留、减仓与清仓。
            market_score : **市场环境评分**（上游策略产出）。None = 不做环境调整。
            exposure_fn  : 市场环境评分 -> 目标总仓位系数（0~1）。
                           用 tiered_exposure() 构造。None = 恒为 1.0（满仓）。
                           **返回 0 表示空仓** -> 清掉所有持仓且不买。

            规则落实
            --------
            - 候选 = 推荐清单里「分数 >= 阈值」且「有价格」且「不在 sell_codes 里」的票
            - 目标集合 = 候选按分数降序取前 N 只，N = 硬上限 − 卖不掉的持仓数
            - 权重 = 纯按评分比例，再做「下限 / 买不到一手」剔除并重新归一化
            - 持有但不在目标集合 -> **清仓**（没价格的除外，那种卖不掉）
            - 目标集合里本来没有的 -> **建仓**
            - 目标集合里本来就有的 -> 只做 **加仓/减仓**
            - 实际完成卖出后的净资金可以用于买入；调用者须先确认卖出权限
            - 买入总额再乘 market 的 exposure 系数
            - 最终只数恒 <= max_holdings


    本函数输出计划，不证明订单成交；实际结果须由执行层记录。"""
    # ── 0. 归一化入参 ──
    # 旧 prices 是便捷输入；新调用方可明确分开三类价格与两类权限。
    # 估值有效不代表可成交；缺持仓估值时明确停止，不能把资产默认为零。
    valuations = dict(prices if valuation_prices is None else valuation_prices)
    buy_refs = dict(prices if buy_prices is None else buy_prices)
    sell_refs = dict(prices if sell_prices is None else sell_prices)
    buy_allowed = {} if can_buy is None else dict(can_buy)
    sell_allowed = {} if can_sell is None else dict(can_sell)
    buy_blocked = set(buy_blocked_codes)
    buy_blocked.update(c for c, allowed in buy_allowed.items() if not allowed)
    held_map: dict = {}
    for h in holdings:
        if h.shares > 0:
            held_map[h.code] = held_map.get(h.code, 0) + h.shares
    if len(held_map) > max_holdings:
        raise ValueError(
            "持仓已 %d 只，超过硬上限 %d 只：%s"
            % (len(held_map), max_holdings, sorted(held_map)))

    if cash < 0:
        raise ValueError("现金不能为负：%r" % (cash,))

    to_close = set(sell_codes)
    # 🔴 冻结票（未满最短持有期）：不清仓、不加减仓，但**占名额**
    frozen = {c for c in frozen_codes if c in held_map and c not in to_close}

    # ── 1. 市场环境系数 ──
    exposure = 1.0
    if market_score is not None and exposure_fn is not None:
        exposure = float(exposure_fn(market_score))
        if not math.isfinite(exposure) or exposure < 0 or exposure > 1:
            raise ValueError("exposure_fn 必须返回 0~1，收到 %r" % (exposure,))

    # 总权益只用估值价；受限持仓仍计入资产及风险暴露。
    equity = float(cash)
    for code, shares in held_map.items():
        px = _price_of(valuations, code)
        if px is None:
            raise ValueError("持仓 %s 缺估值价；停牌需传入上一有效收盘价" % code)
        equity += shares * px

    # 卖出受限持仓仍占名额，强制退出计划不能提前释放持仓席位。
    stuck = {c for c in held_map if not sell_allowed.get(c, True)
             or _price_of(sell_refs, c) is None}
    # 待售价格与估值价格不能反向改变买入预算。受限及冻结资产先占用暴露额度。
    unavailable_value = sum(held_map[c] * valuations[c] for c in stuck | frozen)
    allocation_value = max(0.0, equity * exposure - unavailable_value)
    target_prices = dict(valuations)
    target_prices.update({c: p for c, p in buy_refs.items() if _price_of(buy_refs, c) is not None})

    # ── 2. 环境空仓：清掉所有卖得掉的持仓，且不买（用户选甲） ──
    if exposure <= 0.0:
        sells = []
        for code in sorted(held_map):
            shares = held_map[code]
            px = _price_of(sell_refs, code)
            if code in stuck or px is None:
                continue                       # 卖不掉（停牌/跌停）-> 顺延
            sp = costs.sell_price(px)
            amount = shares * sp
            fee = costs.sell_fee(amount)
            sells.append(SellOrder(code, shares, px, REASON_FLAT, amount, fee))
        final_shares = dict(held_map)
        for s in sells:
            final_shares[s.code] = final_shares.get(s.code, 0) - s.shares
        final_shares = {c: n for c, n in final_shares.items() if n > 0}
        return DayPlan(sells=sells, buys=[], target_codes=[], target_weights={},
                       equity=equity, cash_before=cash,
                       cash_after=cash + sum(s.amount - s.fee for s in sells),
                       final_shares=final_shares,
                       stuck_codes=tuple(sorted(stuck)),
                       exposure=0.0, flat=True)

    # ── 3. 候选排序：过滤 禁买 / 无价 / 低于阈值 ──
    ranked = sorted(picks, key=lambda kv: (-float(kv[1]), kv[0]))
    ordered: list = []
    seen: set = set()
    for code, score in ranked:
        if code in buy_blocked and code not in held_map:
            continue
        if code in seen or code in to_close or code in frozen or code in stuck:
            continue                           # 冻结票不参与换仓（保持不动）
        if score_threshold is not None and float(score) < float(score_threshold):
            continue                           # 低于阈值 -> 不买（组件参数由调用者按场景核验。
        if _price_of(target_prices, code) is None:
            continue                           # 没有价格就不参与（不猜价格）
        seen.add(code)
        ordered.append((code, float(score)))

    # 卖出受限持仓仍占名额，强制退出计划不能提前释放持仓席位。
    slots = max(0, max_holdings - len(stuck) - len(frozen))
    target_all = ordered[:slots]

    # ── 4. 剔除（下限 / 买不到一手）+ 定权重 ──
    target, target_weights, dropped = _select_targets(
        target_all, allocation_value, 1.0, target_prices, lot, costs, min_weight)
    target_codes = [c for c, _ in target]

    # ── 5. 先算「卖」：清仓不在目标集合里的 + 外部指定要清的 ──
    sells: list = []
    freed = 0.0
    for code in sorted((set(held_map) - set(target_codes) - frozen) | to_close):
        shares = held_map.get(code, 0)
        if shares <= 0:
            continue
        px = _price_of(sell_refs, code)
        if code in stuck or px is None:
            continue                           # 卖不掉（停牌），顺延
        sp = costs.sell_price(px)
        amount = shares * sp
        fee = costs.sell_fee(amount)
        # 出场原因：引擎可以为某些票指定（如「硬止损 -6.05%…」），
        # 其余一律 REASON_CLOSE（清仓）。只在**这一次强制清仓**上生效，
        # 不影响减仓（REASON_TRIM）与建仓/加仓。
        reason = (sell_reasons or {}).get(code, REASON_CLOSE)
        sells.append(SellOrder(code, shares, px, reason, amount, fee))
        freed += amount - fee

    if not target:
        final_shares = dict(held_map)
        for s in sells:
            final_shares[s.code] = final_shares.get(s.code, 0) - s.shares
        final_shares = {c: n for c, n in final_shares.items() if n > 0}
        return DayPlan(sells=sells, equity=equity, cash_before=cash,
                       cash_after=cash + freed, final_shares=final_shares,
                       stuck_codes=tuple(sorted(stuck)), dropped=tuple(dropped),
                       exposure=exposure)

    # ── 6. 逐票算差额：正=买（建仓/加仓），负=卖（减仓）──
    buys: list = []
    trims: list = []
    for code, w in target_weights.items():
        want = target_shares(allocation_value, w, target_prices[code], lot)
        have = held_map.get(code, 0)
        diff = want - have
        if diff > 0:
            if code in buy_blocked or _price_of(buy_refs, code) is None:
                continue
            reason = REASON_ADD if have > 0 else REASON_OPEN
            px = costs.buy_price(buy_refs[code])
            amount = diff * px
            buys.append(BuyOrder(code, diff, buy_refs[code], reason,
                                 amount, costs.buy_fee(amount)))
        elif diff < 0:
            px = costs.sell_price(sell_refs[code])
            amount = abs(diff) * px
            fee = costs.sell_fee(amount)
            trims.append(SellOrder(code, abs(diff), sell_refs[code],
                                   REASON_TRIM, amount, fee))
            freed += amount - fee              # 🔴 减仓回款同样当日可用

    sells.extend(trims)

    # ── 7. 现金约束：买不起就砍，**优先砍权重最低的** ──
    buys = _fit_to_cash(buys, cash + freed, costs, lot, target_weights)

    cash_after = cash + freed - sum(b.amount + b.fee for b in buys)

    # ── 8. 结算每只票的最终股数 ──
    final_shares = dict(held_map)
    for s in sells:
        final_shares[s.code] = final_shares.get(s.code, 0) - s.shares
    for b in buys:
        final_shares[b.code] = final_shares.get(b.code, 0) + b.shares
    final_shares = {c: n for c, n in final_shares.items() if n > 0}

    return DayPlan(sells=sells, buys=buys, target_codes=target_codes,
                   target_weights=target_weights, equity=equity,
                   cash_before=cash, cash_after=cash_after,
                   final_shares=final_shares,
                   stuck_codes=tuple(sorted(stuck)), dropped=tuple(dropped),
                   exposure=exposure)


def _fit_to_cash(buys: list, cash: float, costs: AShareCosts, lot: int,
                 weights: Mapping) -> list:
    """买入总额超过可用现金时缩量，直到买得起。

    **优先砍权重最低的那一笔**（= 打分最低的那只），这样高确信度的仓位保留最多。
    """
    def total(bs):
        return sum(b.amount + b.fee for b in bs)

    guard = 0
    max_iter = 500000
    while buys and total(buys) > cash + 1e-9:
        guard += 1
        if guard > max_iter:
            raise RuntimeError(
                "缩量无法收敛（迭代 %d 次仍买不起）：buys=%s cash=%.4f"
                % (guard, [(b.code, b.shares) for b in buys], cash))

        reducible = [i for i, b in enumerate(buys) if b.shares - lot > 0]
        if reducible:
            i = min(reducible,
                    key=lambda k: (weights.get(buys[k].code, 0.0), buys[k].code))
            b = buys[i]
            ns = b.shares - lot
            px = costs.buy_price(b.ref_price)
            amt = ns * px
            buys[i] = BuyOrder(b.code, ns, b.ref_price, b.reason,
                               amt, costs.buy_fee(amt))
        else:
            i = min(range(len(buys)),
                    key=lambda k: (weights.get(buys[k].code, 0.0), buys[k].code))
            buys.pop(i)
    return buys
