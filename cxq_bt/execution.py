# -*- coding: utf-8 -*-
"""成交时间与分钟事件观察组件。

常规模式按下一交易日开盘价执行。分钟标签表示该分钟开始，当根 low
须等 bar 完成才可观察；观察后使用下一合法 bar 的 open 近似成交，
不能回头按已观察事件所在 bar 成交。缺数与有效数据下未观察到开板
分开处理；日线不能替代分钟证据。"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .rules import PRICE_TOL

#: 🔴 由 **Python 解释器隐式调用**的魔术方法 —— 静态扫描看不到，**别删**。
#:    理由见 cxq_bt/rules.py 的 DUNDER_HOOKS 说明。
DUNDER_HOOKS = ("__bool__",)

__all__ = [
    "EXEC_TIME", "T_BOARD_WINDOW_END", "DECISION_CUTOFF_LABEL",
    "as_of", "open_price", "first_break", "BreakInfo", "FillResult",
    "resolve_buy_fill", "resolve_sell_fill", "BreakLookupResult", "confirmed_break_fill",
]

#: 常规成交时刻（T+1 日开盘）
EXEC_TIME = dt.time(9, 30)

#: 一字板特例窗口终点 —— **只是特例窗口，不是所有交易的截止**
T_BOARD_WINDOW_END = dt.time(9, 50)

#: 决策截止的文字描述（写进报告用，避免有人再理解成 09:25）
DECISION_CUTOFF_LABEL = "T 日收盘后（复盘跑完那一刻）"


# ══════════════════ 数据闸门 ══════════════════

def _time_series(df: pd.DataFrame, time_col: str = "trade_time") -> pd.Series:
    """取 bar 时刻。列和索引层都找（主库是带 index 写的）。"""
    if time_col in df.columns:
        return pd.to_datetime(df[time_col])
    names = list(df.index.names or [])
    if time_col in names:
        vals = df.index.get_level_values(time_col)
        return pd.to_datetime(pd.Series(vals, index=df.index))
    raise ValueError(
        "找不到时间列 %r（列=%s，索引层=%s）"
        % (time_col, list(df.columns)[:8], names))


def as_of(df: pd.DataFrame, t: dt.time, time_col: str = "trade_time") -> pd.DataFrame:
    """开始标签的一分钟 bar 只在完成时刻 <= t 时返回，结果是副本。

    09:35 表示 [09:35,09:36)，该根完整 OHLC 和成交量在09:36才可见。
    本接口用于完整 bar 的可见性；实时开盘价应通过独立开盘事件获得，
    不能为取 open 而提前放出同根尚未知的 high/low/close/volume。
    """
    times = _time_series(df, time_col)
    cutoff = times.dt.normalize() + pd.Timedelta(hours=t.hour, minutes=t.minute,
                                                  seconds=t.second, microseconds=t.microsecond)
    mask = times + pd.Timedelta(minutes=1) <= cutoff
    return df.loc[mask].copy()


def open_price(df: pd.DataFrame, time_col: str = "trade_time") -> Optional[float]:
    """取输入中明确标记的 09:30 open，不能读取该根尚未知的其他字段。

    调用方须先规范分钟时间标签及竞价事件。区间结束标签不能未经转换
    就当作区间开始；本接口不对供应商数据的标签或竞价编码做推测。
    """
    if df is None or len(df) == 0:
        return None
    times = _time_series(df, time_col)
    hit = df.loc[times.dt.time == EXEC_TIME]
    if len(hit) == 0:
        return None
    px = float(hit.iloc[0]["open"])
    if not math.isfinite(px) or px <= 0:
        return None
    return px


# ══════════════════ 一字板特例 ══════════════════

@dataclass(frozen=True)
class BreakInfo:
    """首次开板信息。"""
    at: dt.time
    open_: float
    close: float
    low: float


@dataclass(frozen=True)
class BreakLookupResult:
    """分钟查询状态。缺数不能冒充有效行情里的未开板。"""
    status: str                 # filled / no_break / no_fill / missing
    at: Optional[dt.time] = None
    price: Optional[float] = None
    confirmed_at: Optional[dt.time] = None
    reason: str = ""


def confirmed_break_fill(df: pd.DataFrame, up_limit: float,
                         deadline: dt.time = T_BOARD_WINDOW_END,
                         time_col: str = "trade_time") -> BreakLookupResult:
    """一分钟近似：标签是区间开始，完成后用下一根合法 bar 的开盘价。

    例如 09:35 表示 [09:35,09:36)，low 到 09:36 才可见；成交最早是
    09:36 的 open。本函数要求 09:30 到候选成交时刻连续、唯一的分钟标签。
    若数据源把时间标在区间末尾或 09:30 单独代表竞价，调用前必须转换，
    不能直接套用。此模型没有逐笔或排队证据，不能声称还原真实成交。
    """
    if df is None or len(df) == 0:
        return BreakLookupResult("missing", reason="缺分钟行情，无法确认开板")
    times = _time_series(df, time_col)
    stamps = pd.DatetimeIndex(times)
    if stamps.hasnans or not stamps.is_monotonic_increasing or stamps.has_duplicates:
        return BreakLookupResult("missing", reason="分钟标签缺失、重复或未排序")
    if len(set(stamps.date)) != 1:
        return BreakLookupResult("missing", reason="分钟查询必须只含一个交易日")
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return BreakLookupResult("missing", reason="分钟行情缺 OHLC 字段")
    base = stamps[0].normalize() + pd.Timedelta(hours=9, minutes=30)
    stop = stamps[0].normalize() + pd.Timedelta(hours=deadline.hour, minutes=deadline.minute)
    window = df.loc[(stamps >= base) & (stamps <= stop)]
    labels = stamps[(stamps >= base) & (stamps <= stop)]
    if not len(labels) or labels[0] != base:
        return BreakLookupResult("missing", reason="缺 09:30 分钟行情")
    for i, stamp in enumerate(labels):
        if stamp != base + pd.Timedelta(minutes=i):
            return BreakLookupResult("missing", reason="分钟行情不连续，不能确认窗口状态")
        row = window.iloc[i]
        values = [float(row[c]) for c in ("open", "high", "low", "close")]
        if not all(math.isfinite(v) and v > 0 for v in values):
            return BreakLookupResult("missing", reason="分钟行情有无效价格")
        if i == 0 and abs(values[0] - float(up_limit)) > PRICE_TOL:
            return BreakLookupResult("no_fill", reason="开盘未封涨停，不适用开板模型")
        # 最后一根 09:50 的 low 到 09:51 才知道，不能用于截止前确认。
        known_at = stamp + pd.Timedelta(minutes=1)
        if known_at > stop:
            break
        if values[2] >= float(up_limit) - PRICE_TOL:
            continue
        if i + 1 >= len(labels) or labels[i + 1] != known_at:
            return BreakLookupResult("missing", reason="确认后缺下一根分钟开盘价")
        following = window.iloc[i + 1]
        price = float(following["open"])
        if not math.isfinite(price) or price <= 0:
            return BreakLookupResult("missing", reason="下一分钟开盘价异常")
        # 只能检查已完成确认 bar 的量；下一根的整分钟量开盘时尚未知。
        # 仅 OHLC 输入无法证明真实成交容量，下一 open 仍是明确的模型近似。
        for volume_column in ("volume", "vol"):
            if volume_column in row and (not math.isfinite(float(row[volume_column]))
                                          or float(row[volume_column]) <= 0):
                return BreakLookupResult("no_fill", reason="已完成确认分钟没有有效成交量")
        if price >= float(up_limit) - PRICE_TOL:
            return BreakLookupResult("no_fill", reason="确认后的分钟开盘已回封涨停")
        return BreakLookupResult("filled", known_at.time(), price, known_at.time(),
                                 "完成分钟开板确认后，下一分钟开盘近似成交")
    if labels[-1] < stop:
        return BreakLookupResult("missing", reason="分钟窗口覆盖不足，不能认定未开板")
    return BreakLookupResult("no_break", reason="完整窗口内未开板（以完成 bar 确认）")


def first_break(df: pd.DataFrame, up_limit: float,
                deadline: dt.time = T_BOARD_WINDOW_END,
                time_col: str = "trade_time") -> Optional[BreakInfo]:
    """在 09:30~deadline 里找**首次打开涨停**的那根 bar。

    判据：该 bar 的 low 跌破涨停价（low < up_limit - 容差）。
    容差用 rules.PRICE_TOL（半个报价单位 0.005），避免浮点误差误判。

    🔴 **前提：这只票开盘就在涨停**（否则谈不上「开板」）。
       普通票的最低价本来就远低于涨停价，不加这个闸门会算出
       「09:30 开板」这种**假信号** —— 未先触及涨停时，不能把低于涨停价当成开板事件。
    """
    if df is None or len(df) == 0 or up_limit is None:
        return None
    times = _time_series(df, time_col)
    window = (times.dt.time >= EXEC_TIME) & (times.dt.time <= deadline)
    sub = df.loc[window]
    if len(sub) == 0:
        return None
    sub_t = times.loc[window]
    # 🔴 前提闸门：第一根 bar 必须**碰到过**涨停价，否则不算开板
    first_high = pd.to_numeric(sub["high"], errors="coerce").iloc[0]
    if first_high != first_high or abs(float(first_high) - float(up_limit)) > PRICE_TOL:
        return None
    lows = pd.to_numeric(sub["low"], errors="coerce")
    broke = lows < (float(up_limit) - PRICE_TOL)
    if not broke.any():
        return None
    row = sub.loc[broke].iloc[0]
    at = sub_t.loc[broke].iloc[0]
    return BreakInfo(at=at.time() if hasattr(at, "time") else at,
                     open_=float(row["open"]),
                     close=float(row["close"]),
                     low=float(row["low"]))


# ══════════════════ 成交结果 ══════════════════

@dataclass(frozen=True)
class FillResult:
    """一只票在 T+1 日的成交结果。"""
    code: str
    filled: bool
    price: Optional[float] = None
    reason: str = ""
    source: str = ""          # 价格来源：开盘价 / 开板价
    break_at: Optional[dt.time] = None

    def __bool__(self) -> bool:
        return self.filled


def _open_bar(day_bars: pd.DataFrame):
    """取 09:30 那根 bar（没有则 None）。"""
    if day_bars is None or len(day_bars) == 0:
        return None
    times = _time_series(day_bars)
    hit = day_bars.loc[times.dt.time == EXEC_TIME]
    return None if len(hit) == 0 else hit.iloc[0]


def _is_sealed_up(bar, day_bars: pd.DataFrame,
                  up_limit: Optional[float]) -> bool:
    """一字板判定：乙（权威字段）优先，甲（行情）兜底 —— 与第 3 层同口径。"""
    if up_limit is None:
        return False
    r = _open_bar(day_bars)
    if r is None:
        return False
    # 开盘权限只看开盘价；该分钟的 high/low/close 此时尚未可见。
    return abs(float(r["open"]) - up_limit) <= PRICE_TOL


def resolve_buy_fill(*, code: str, day_bars: pd.DataFrame,
                     up_limit: Optional[float] = None,
                     bar=None,
                     break_price_mode: str = "open") -> FillResult:
    """给一只票定 T+1 日的**买入成交结果**。

    🔴 注意本函数的签名：它**收不到买入清单，也返回不了买入清单** ——
       这是刻意的。执行层不许改变「买不买」，那是前一天复盘定死的。
       本函数只回答：「如果决定买，今天能不能买到、按什么价成交。」

    break_price_mode : 开板价取哪一列（"open" / "close" / "low"）
    """
    px = open_price(day_bars)
    if px is None:
        return FillResult(code, False, None, "没有 09:30 开盘价", "")

    if not _is_sealed_up(bar, day_bars, up_limit):
        return FillResult(code, True, px, "开盘价成交", "开盘价")

    # 一字板 -> 看窗口内有没有开板
    if up_limit is None:
        return FillResult(code, False, None, "一字板但不知道涨停价", "")
    if break_price_mode != "open":
        raise ValueError("公开执行模型仅允许确认后下一分钟 open，不能用该分钟 low/close 挑价")
    info = confirmed_break_fill(day_bars, up_limit)
    if info.status != "filled":
        return FillResult(code, False, None, info.reason, "", info.at)
    return FillResult(code, True, info.price, info.reason, "确认后下一分钟开盘价", info.at)


def resolve_sell_fill(*, code: str, day_bars: pd.DataFrame,
                      down_limit: Optional[float] = None,
                      blocked: bool = False,
                      block_reason: str = "") -> FillResult:
    """给一只票定 T+1 日的**卖出成交结果**。

    卖出同样按 09:30 开盘价；封死跌停 / 停牌卖不出 -> 顺延（由调用方次日再试）。
    blocked 由第 3 层的 RuleEngine.check_sell 给出。
    """
    if blocked:
        return FillResult(code, False, None, block_reason or "卖出被拦截", "")
    px = open_price(day_bars)
    if px is None:
        return FillResult(code, False, None, "没有 09:30 开盘价（停牌？）", "")
    if down_limit is not None and abs(px - down_limit) <= PRICE_TOL:
        return FillResult(code, False, None, "开盘封死跌停 -> 顺延", "")
    return FillResult(code, True, px, "开盘价成交", "开盘价")
