# -*- coding: utf-8 -*-
"""A 股交易规则判定组件。

规则只读取 BarSnapshot，不联网。买入检查 ST、停牌、一字板和可选
分钟开板窗口；卖出检查 T+1、停牌和跌停限制。明确的 limit_status
优先，缺少状态时用行情字段判定。

公开日级入口只能使用执行时已知的信息。需要全天 OHLC 或分钟线的
额外方法，不得在当天开盘提前读取收盘后才完成的数据。分钟窗口是
模型约束，不保证真实排队成交。"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "PRICE_TOL", "OPEN_WINDOW_MINUTES", "DEADLINE",
    "BLOCK_ST", "BLOCK_SUSPENDED", "BLOCK_SEALED_UP", "BLOCK_T_BOARD_LATE",
    "BLOCK_T1", "BLOCK_SEALED_DOWN", "BLOCK_SOLD_TODAY",
    "STOP_LOSS_DEFAULT", "EXIT_STOP_LOSS", "stop_loss_reason",
    "STATUS_SEALED_UP", "STATUS_TURNOVER",
    "BarSnapshot", "Verdict", "RuleEngine",
]

# 框架与解释器会按名称隐式调用这些入口，不能只按直接调用次数判断是否删除。
DUNDER_HOOKS = ("__bool__",)

#: 价格比较容差。A股报价最小变动 0.01 元，用半个 tick 判等最稳。
PRICE_TOL = 0.005

# 实验分钟成交窗口长度，须按调用场景核验。
OPEN_WINDOW_MINUTES = 20

#: 交易截止时刻 = 09:30 + 20 分钟
DEADLINE = dt.time(9, 30 + OPEN_WINDOW_MINUTES)          # 09:50

# ── 拦截原因（常量，方便测试断言和流水归因） ──
BLOCK_ST = "ST票不买"
BLOCK_SUSPENDED = "停牌"
BLOCK_SEALED_UP = "一字板买不进"
BLOCK_T_BOARD_LATE = "T字板开板晚于 %s" % DEADLINE.strftime("%H:%M")
BLOCK_T1 = "T+1 当日买入不可卖"
BLOCK_SEALED_DOWN = "跌停封死卖不出"
BLOCK_SOLD_TODAY = "当日卖出当日不买回"

# 状态字段缺失需要行情判断，空值不自动表示股票可交易。
STATUS_SEALED_UP = "一字板"
STATUS_TURNOVER = "换手板"
STATUS_T_BOARD = "T字板"


def _eq(a: Optional[float], b: Optional[float], tol: float = PRICE_TOL) -> bool:
    """价格判等（任一为 None → False，绝不默认相等）。"""
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _f(v) -> Optional[float]:
    """-> float；None / NaN / inf / 非数 -> None（数据铁律：不猜值）。"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class BarSnapshot:
    """一只票在某一交易日的交易约束快照 —— **全部由第 5 层提供**。

    任何字段缺失都用 None / False 表示，规则层**绝不猜值**。
    """

    code: str

    # ── 基础过滤 ──
    is_st: bool = False
    is_suspended: bool = False

    # ── 价格（None = 当天没有该数据，规则层会跳过对应判定） ──
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    up_limit: Optional[float] = None
    down_limit: Optional[float] = None

    # ── 一字板权威字段（limit_list_ths.status）──
    limit_status: Optional[str] = None

    # ── 分钟级时刻（第 4 层/数据适配器提供）──
    #: 当日**首次打开涨停**的时刻。None = 全天没打开过（或没有分钟数据）
    first_break_time: Optional[dt.time] = None
    #: 当日**首次封上涨停**的时刻
    first_limit_time: Optional[dt.time] = None


@dataclass(frozen=True)
class Verdict:
    """判定结果。reason 为 None 表示放行。"""
    ok: bool
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


ALLOW = Verdict(True, None)
def _block(reason: str) -> Verdict:
    return Verdict(False, reason)


# ══════════════════ 硬止损（出场规则，不是"拦截"）══════════════════

# 组件默认止损为开盘价相对开仓价下跌5%；调用者可配置或关闭。
STOP_LOSS_DEFAULT = -0.05

#: 出场原因前缀（流水归因用）
EXIT_STOP_LOSS = "硬止损"


def stop_loss_reason(entry_price: Optional[float],
                     open_price: Optional[float],
                     hard_stop_pct: Optional[float] = STOP_LOSS_DEFAULT
                     ) -> Optional[str]:
    """硬止损是否触发。触发返回原因字符串，否则 None。

        口径（与上游策略 `_sell_phase` 逐字一致）::

            (open / entry_price - 1.0) <= hard_stop_pct

        三条边界，**一律不猜、不触发**：

        1. `hard_stop_pct is None`  -> 该功能关闭（用户可显式关掉）
        2. `entry_price` 缺失或 <= 0 -> 不知道开仓价就不敢判（数据铁律：不推测）
        3. `open_price` 缺失或 <= 0  -> 当天没开盘价（停牌/缺数据）-> 顺延，不触发

        ⚠️ 本函数**只管判、不管卖**：T+1（当日买入不可卖）、跌停封死卖不出
           由 :meth:`RuleEngine.check_sell` 先拦；调用方必须**先过 check_sell**，
           再过本函数。顺序反了就会在买入当天把自己止损掉。

        手算示例（entry 固定 10.005 = 10.00 开盘价 + 0.05% 滑点）::

            open 9.40   ->  9.40 /10.005 - 1 = -6.0475% <= -5% -> 触发
            open 9.50   ->  9.50 /10.005 - 1 = -5.0475% <= -5% -> 触发
            open 9.50475->  精确 -5.0000%                       -> 触发（含边界）
            open 9.505  ->  9.505/10.005 - 1 = -4.9975% >  -5% -> **不触发**
            open 9.51   ->  9.51 /10.005 - 1 = -4.9475% >  -5% -> 不触发

        ⚠️ 9.505 这个例子是刻意留的：它看着"像 5%"，实际只有 4.9975%，
           **不许四舍五入成 5% 就放行/拦下** —— 判据是原始浮点比较，不是显示值。
        """
    if hard_stop_pct is None:
        return None                                   # 功能关闭
    e = _f(entry_price)
    o = _f(open_price)
    if e is None or e <= 0:
        return None                                   # 不知道开仓价 -> 不猜
    if o is None or o <= 0:
        return None                                   # 当天没开盘价 -> 顺延
    chg = o / e - 1.0
    if chg <= float(hard_stop_pct):
        return "%s %.2f%%（开盘 %.3f / 开仓 %.3f）" % (EXIT_STOP_LOSS, chg * 100.0, o, e)
    return None


class RuleEngine:
    """第 3 层 · 交易规则拦截器。

    无状态判定靠 :class:`BarSnapshot`；有状态的部分（T+1 台账）由本类持有。

    典型用法（引擎每天的循环里）::

        engine = RuleEngine()
        v = engine.check_buy(bar, today)
        if v:
            ...下单...
            engine.record_buy(code, today)
    """

    def __init__(self, deadline: dt.time = DEADLINE):
        self.deadline = deadline
        self._buy_date: dict = {}     # code -> 最近一次买入日
        self._sold_date: dict = {}    # code -> 最近一次卖出日

    # ══════════════ 台账 ══════════════

    def record_buy(self, code: str, d: dt.date) -> None:
        """记录买入日 —— T+1 拦截的依据。**成交后必须调用**。"""
        self._buy_date[code] = d

    def record_sell(self, code: str, d: dt.date) -> None:
        """记录卖出日 —— 防止当日卖出又当日买回。"""
        self._sold_date[code] = d
        self._buy_date.pop(code, None)   # 清仓后不再受 T+1 约束

    def buy_date(self, code: str):
        return self._buy_date.get(code)

    # ══════════════ 买入判定 ══════════════

    def check_buy(self, bar: BarSnapshot, today: Optional[dt.date] = None) -> Verdict:
        """买入前置检查。优先级：ST > 停牌 > 一字板 > T字板开板太晚 > 当日卖出过。"""
        if bar.is_st:
            return _block(BLOCK_ST)              # ST 硬过滤，优先级最高（接口约定）
        if bar.is_suspended:
            return _block(BLOCK_SUSPENDED)
        if self.is_sealed_up(bar):
            return _block(BLOCK_SEALED_UP)
        if self.is_t_board_too_late(bar):
            return _block(BLOCK_T_BOARD_LATE)
        if today is not None and self._sold_date.get(bar.code) == today:
            return _block(BLOCK_SOLD_TODAY)
        return ALLOW

    # ══════════════ 卖出判定 ══════════════

    def check_sell(self, bar: BarSnapshot, today: Optional[dt.date] = None) -> Verdict:
        """卖出前置检查。优先级：T+1 > 停牌 > 跌停封死。"""
        if today is not None and self._buy_date.get(bar.code) == today:
            return _block(BLOCK_T1)              # 当日买入不可卖
        if bar.is_suspended:
            return _block(BLOCK_SUSPENDED)       # 停牌 = 冻结，等复牌
        if self.is_sealed_down(bar):
            return _block(BLOCK_SEALED_DOWN)     # 卖不出 → 顺延
        return ALLOW

    # ══════════════ 一字板 ══════════════

    def is_sealed_up(self, bar: BarSnapshot) -> bool:
        """一字板判定。**乙（权威字段）优先，甲（行情）兜底**。"""
        # 乙：limit_list_ths.status == "一字板"
        if bar.limit_status is not None:
            return str(bar.limit_status).strip() == STATUS_SEALED_UP
        # 甲：开 = 高 = 低 = 收 = 涨停价（且涨停价已知）
        if bar.up_limit is None:
            return False                          # 不知道涨停价 → 不敢判一字板
        return (_eq(bar.open, bar.up_limit) and _eq(bar.high, bar.up_limit)
                and _eq(bar.low, bar.up_limit) and _eq(bar.close, bar.up_limit))

    def is_t_board_too_late(self, bar: BarSnapshot) -> bool:
        """实验T字板窗口判定：先确认状态或开盘触及涨停，再比较明确的首次开板时间。

        时间必须来自已完成的分钟事件；状态缺失不提供事件时刻。
        这个组件约束不证明真实排队能成交。"""
        is_t = (bar.limit_status is not None
                and str(bar.limit_status).strip() == STATUS_T_BOARD)
        if not is_t:
            if bar.up_limit is None or bar.open is None:
                return False
            if not _eq(bar.open, bar.up_limit):
                return False                      # 开盘没涨停 → 不是 T 字板
        if bar.first_break_time is None:
            return False                          # 全天没开板 → 已由 is_sealed_up 拦掉
        return bar.first_break_time > self.deadline

    # ══════════════ 跌停封死 ══════════════

    def is_sealed_down(self, bar: BarSnapshot) -> bool:
        """封死跌停（开=高=低=收=跌停价）→ 挂单也没人接，只能顺延。"""
        if bar.down_limit is None:
            return False
        return (_eq(bar.open, bar.down_limit) and _eq(bar.high, bar.down_limit)
                and _eq(bar.low, bar.down_limit) and _eq(bar.close, bar.down_limit))

    # ══════════════ 批量辅助 ══════════════

    def filter_buyable(self, bars, today: Optional[dt.date] = None) -> dict:
        """对一批快照做买入过滤。返回 {code: Verdict}，方便一次性看谁被拦了、为什么。"""
        return {b.code: self.check_buy(b, today) for b in bars}
