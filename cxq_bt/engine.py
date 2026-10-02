# -*- coding: utf-8 -*-
"""基于 Backtrader 的可组合回测引擎组件。

接收外部策略每日清单，负责交易限制、资金、费用和记账；公开版的
目标权重入口是 cxq_bt.framework，见 docs/strategy_api.md。

收盘决定在下一交易日执行，不用次日尚未完成的日内极值和收盘价判断
开盘。分钟标签表示一分钟开始，low 到 bar 完成才可观察，之后使用
下一合法 bar 的 open 近似成交；缺数与已观察但未开板分开记录。
名额、股数与预算在09:30固定，失败预算留现金。买入权限、卖出权限
与估值分别处理，受限持仓不能提前当现金。核心应用一次滑点，broker
不再叠加，避免日内极值裁剪开盘成交。

额外一字板队列方法是实验近似，不提供实际盘口成交概率或盈利保证。"""

from __future__ import annotations

import datetime as dt
import concurrent.futures
import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
# 框架与解释器会按名称隐式调用这些入口，不能只按直接调用次数判断是否删除。
import pyarrow.parquet as pq

from . import compat as _compat  # noqa: F401  兼容垫片必须早于 backtrader
import backtrader as bt

from . import execution as _ex
from .costs import COSTS, AShareCosts, AShareCommission
from .datafeed import ParquetRepo
from .performance import evaluate
from .rules import (BarSnapshot, RuleEngine, Verdict,
                    STOP_LOSS_DEFAULT, stop_loss_reason)
from .sizing import (DEFAULT_MIN_WEIGHT, LOT, MAX_HOLDINGS, Holding, plan_day)

__all__ = [
    "EngineConfig", "Trade", "Blocked", "EngineResult",
    "ParquetFeed", "CxqEngine", "FRAMEWORK_CALLBACKS",
    "qfq_frame", "benchmark_frame", "clock_frame", "prepare_frame", "prepare_frames",
    "make_break_lookup", "run_engine", "run_backtest", "date_key",
]

# 框架与解释器会按名称隐式调用这些入口，不能只按直接调用次数判断是否删除。
FRAMEWORK_CALLBACKS = ("next_open", "next", "notify_order",
                       "prenext", "prenext_open")

# 分钟实验窗口终点；公开日级入口不使用盘中开板窗口。
T_BOARD_WINDOW_END = _ex.T_BOARD_WINDOW_END


def sealed_draw(code: str, when, seed: int) -> float:
    """一字板排队成交的**确定性抽样** `[0, 1)`。

    🔴 为什么不用 `random.random()`：回测必须**可复现** —— 同一份数据、同一套参数，
    跑两次的净值曲线要**逐点相同**（本项目的对拍文化）。
    所以用 `sha256(code|日期|种子)` 取前 16 个十六进制位 -> 一个固定的小数。

    手算可核：`sealed_draw("000001.SZ", date(2025,4,8), 1)` 每次都是同一个数；
    换 `code` / `日期` / `种子` 中的任何一个，结果就变（近似均匀分布在 [0,1)）。
    """
    key = "%s|%s|%d" % (str(code), str(when)[:10], int(seed))
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(h, 16) / float(1 << 64)


# ══════════════════ 工具 ══════════════════

def date_key(v) -> dt.date:
    """把 str / datetime / date / Timestamp 统一成 datetime.date。"""
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return pd.Timestamp(v).date()


def _f(v) -> Optional[float]:
    """-> float，NaN / None / 非数 -> None。"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


#: 🔴 「≥N 板不买」的查询函数：
#:   (code, date) -> 连板数 or None。由调用方注册（引擎不自己读盘，保持包纯粹）。
#:   None = 判不出 -> **不过滤**（保守：宁可不滤，也不误杀）。
_BOARD_QUERY: Optional[Callable] = None
_SKIP_BOARDS_GTE: Optional[int] = None


def register_board_filter(query, min_boards: Optional[int]) -> None:
    """注册可选的连板高度过滤，只在买入候选阶段生效。

    query 为 None 或 min_boards 为空时不启用；不改变上游评分。"""
    global _BOARD_QUERY, _SKIP_BOARDS_GTE
    _BOARD_QUERY = query
    _SKIP_BOARDS_GTE = int(min_boards) if min_boards else None


def _eq(a: Optional[float], b: Optional[float], tol: float = 0.005) -> bool:
    """价格相等判定（容差半个报价单位，与 rules.PRICE_TOL 同口径）。"""
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


# ══════════════════ 配置 ══════════════════

@dataclass(frozen=True)
class EngineConfig:
    """回测口径（全部来自 接口说明，不在这里发明数字）。"""
    cash: float = 100_000.0                       # 初始资金
    #: 爆仓标注线：**只标注、不停跑**（🔴 不许拿它当提前退出条件）。
    #: 当日收盘总资产 < 它 -> 记 result.bankrupt_on（**只记第一次**）。
    #: 语义：A 股整手（100 股）交易下，净值跌破它基本等于「连 1 手都买不起」，
    #:       曲线从那天起会退化成一条**死账户横线**，不再携带策略信息量（接口约定）。
    #: 它**不改变**任何成交、任何净值点 —— 「净值点数 == 区间交易日数」这条自检照旧成立。
    #: 🔴 口径是**严格小于**：净值恰好 == 它时**不**标注。
    bankrupt_floor: float = 1000.0
    lot: int = LOT                                # 一手 = 100 股
    max_holdings: int = MAX_HOLDINGS              # 单日最多 3 只
    #: 单票权重下限 —— **0.0 = 不设下限**（组件参数由调用者按场景核验。
    #: 剔除理由只有「执行上买不起」（整手化 0 股）这一条，见 sizing._select_targets。
    min_weight: float = DEFAULT_MIN_WEIGHT
    # 评分阈值必须对应调用方的输入尺度；比例权重接口不自动生成评分。
    score_threshold: Optional[float] = 3.0
    #: 市场环境评分 -> 总仓位系数。用 调用方提供的环境仓位系数函数 构造
    #: （口径来自外部策略 sentiment 的 5 档情绪周期）。None = 不做环境调整（恒满仓）。
    exposure_fn: Optional[Callable] = None
    # 硬止损是可配置退出规则，需同时通过卖出权限；不得由旧研究结果证明其有效性。
    skip_sealed_open: bool = False
    # 可选买入过滤只影响执行可行性，不生成或替换调用者的策略评分。
    only_sealed_open: bool = False
    # 实验参数须与执行假设一起验证，不能从研究成绩推导默认可成交性。
    sealed_fill_prob: float = 0.0
    #: 🔴 **数据驱动**的一字板可成交概率：`fn(code, buy_date) -> p(0~1)`。
    #: 传了它就以它为准（`sealed_fill_prob` 退化成"它返回空时的兜底"）。
    #: 生产用法见 `cxq_bt/sealed_model.py`（用 09:25 竞价成交额 / 昨日封单额 = 队列消化比例）。
    #: ⚠️ 它**只看成交机制、不读分数** —— 选股仍然是外部策略的事。
    sealed_fill_fn: Optional[Callable] = None
    #: 成交抽样的随机种子。固定 -> 同一份数据跑两次结果**逐位相同**（可复现）。
    sealed_fill_seed: int = 20260917
    hard_stop_pct: Optional[float] = STOP_LOSS_DEFAULT
    # 实验参数须与执行假设一起验证，不能从研究成绩推导默认可成交性。
    min_hold_days: int = 0
    costs: AShareCosts = COSTS
    qfq: bool = True                              # 统一前复权
    benchmark_name: str = "沪深300"


# ══════════════════ 数据：不复权 -> 前复权 ══════════════════

def qfq_frame(daily: pd.DataFrame, *, qfq: bool = True) -> pd.DataFrame:
    """日线（不复权）-> 引擎用的日线帧。

    前复权口径：qfq(t) = 不复权价(t) × adj_factor(t) ÷ adj_factor(基准日)，
    基准日 = **本帧最后一个交易日**（= 回测结束日，固定不变）。

    ⚠️ 为什么"手数/涨跌停"也用它：前复权对同一只票是**整体缩放**，
       缩放后「一手多少钱」与「涨停价」彼此仍自洽（不会出现
       "用真实价算手数、用复权价算净值"这种两套尺子）。
       绝对价位与真实历史不同 —— 这是前复权的固有性质，写在这里备查。

    产出列：open/high/low/close/volume/openinterest
            + is_st / up_limit / down_limit / raw_open / raw_up_limit / adj_ratio
    """
    if daily is None or len(daily) == 0:
        raise ValueError("日线为空，无法构建回测帧")

    d = daily.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.sort_values("trade_date").reset_index(drop=True)

    if "adj_factor" not in d.columns:
        # 🔴 复权所需列必须显式包含 adj_factor，不能假定默认投影已经提供，
        #    d.get() 返回 None -> to_numeric(None) 变成标量 -> 后面 .isna() 直接炸。
        #    这里明确报错，别让它悄悄退回"不复权"。
        raise ValueError("日线里没有 adj_factor 列 —— 前复权算不了（列=%s）"
                         % list(d.columns)[:12])
    factor = pd.to_numeric(d["adj_factor"], errors="coerce").astype(float)
    if factor.isna().any() or (factor <= 0).any():
        raise ValueError("adj_factor 有缺失或非正数，拒绝默默按 1.0 处理")
    ratio = (factor / float(factor.iloc[-1])) if qfq else pd.Series(1.0, index=d.index)

    out = pd.DataFrame(index=pd.DatetimeIndex(d["trade_date"]))
    out.index.name = "trade_date"

    # 🔴 一律用 to_numpy()：out 的索引是日期、源 Series 的索引是 0..n，
    #    直接赋 Series 会被 pandas **按标签对齐**成整列 NaN（踩过）
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float) * ratio.to_numpy()
    out["raw_open"] = pd.to_numeric(d["open"], errors="coerce").to_numpy(dtype=float)

    for c in ("up_limit", "down_limit"):
        raw = (pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
               if c in d.columns else np.full(len(d), float("nan")))
        out[c] = raw * ratio.to_numpy()
    out["raw_up_limit"] = (pd.to_numeric(d["up_limit"], errors="coerce").to_numpy(dtype=float)
                           if "up_limit" in d.columns
                           else np.full(len(d), float("nan")))

    out["adj_ratio"] = ratio.values
    #: 日线 vol 单位是**手**（接口约定）-> 换成股
    vol = (pd.to_numeric(d["vol"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
           if "vol" in d.columns else np.zeros(len(d)))
    out["volume"] = vol * 100.0
    out["openinterest"] = 0.0
    out["is_st"] = (pd.to_numeric(d["is_st"], errors="coerce").fillna(0).to_numpy(dtype=int)
                    if "is_st" in d.columns else np.zeros(len(d), dtype=int))
    return out


def benchmark_frame(index_daily: pd.DataFrame) -> pd.DataFrame:
    """指数日线 -> 与个股帧同结构的帧（只当"日历 + 基准"，不参与交易）。"""
    d = index_daily.copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.sort_values("trade_date").reset_index(drop=True)
    out = pd.DataFrame(index=pd.DatetimeIndex(d["trade_date"]))
    out.index.name = "trade_date"
    for c in ("open", "high", "low", "close"):
        out[c] = (pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
                  if c in d.columns else np.full(len(d), float("nan")))
    out["volume"] = 0.0
    out["openinterest"] = 0.0
    out["is_st"] = 0
    for c in ("up_limit", "down_limit", "raw_open", "raw_up_limit"):
        out[c] = float("nan")
    out["adj_ratio"] = 1.0
    return out


def clock_frame(benchmark_frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """基准帧裁剪为主时钟帧，仅保留 [start,end] 区间。

    计算基准首日收益可能需要区间前的价格，但这些前值不能作为回测主时钟，
    否则权益点数、年化与回撤窗口会多出区间外交易日。前值由基准收益函数处理。"""
    if benchmark_frame is None or len(benchmark_frame) == 0:
        return benchmark_frame
    idx = pd.DatetimeIndex(benchmark_frame.index)
    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    keep = pd.Series(True, index=idx)
    if lo is not None:
        keep &= (idx >= lo)
    if hi is not None:
        keep &= (idx <= hi)
    return benchmark_frame.loc[keep.to_numpy()]


# ══════════════════ backtrader 数据源 ══════════════════

class ParquetFeed(bt.feed.DataBase):
    """将已准备的 DataFrame 以 NumPy 数组方式接入 Backtrader。

    start 时取得列数组，之后按下标读取，减少逐个 pandas 单元格访问。
    保留日期与 line 字段语义，传入物为 qfq_frame 或 benchmark_frame 的结果。"""
    lines = ("is_st", "up_limit", "down_limit", "raw_open", "raw_up_limit", "adj_ratio")

    _COLUMNS = ("open", "high", "low", "close", "volume", "openinterest",
                "is_st", "up_limit", "down_limit", "raw_open", "raw_up_limit",
                "adj_ratio")

    def start(self):
        super().start()
        frame = self.p.dataname
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("ParquetFeed.dataname 必须是 pandas.DataFrame")
        missing = [name for name in self._COLUMNS if name not in frame.columns]
        if missing:
            raise ValueError("引擎帧缺列：%s" % ", ".join(missing))

        index = pd.DatetimeIndex(frame.index)
        self._feed_row = -1
        self._feed_dates = np.fromiter(
            (bt.date2num(value.to_pydatetime()) for value in index),
            dtype=float, count=len(index))
        # 预先绑定 line 对象，_load 热循环里不再逐字段 getattr。
        self._feed_columns = tuple(
            (getattr(self.lines, name), frame[name].to_numpy(dtype=float, copy=False))
            for name in self._COLUMNS)

    def _load(self):
        self._feed_row += 1
        if self._feed_row >= len(self._feed_dates):
            return False
        row = self._feed_row
        self.lines.datetime[0] = self._feed_dates[row]
        for line, values in self._feed_columns:
            line[0] = values[row]
        return True


# ══════════════════ 一字板开板查询（分钟线） ══════════════════

def make_break_lookup(repo: ParquetRepo, freq: str = "1min", *,
                      preload: Optional[Iterable[Tuple[str, object, float]]] = None,
                      workers: int = 8) -> Callable:
    """构造分钟开板查询函数，返回数据状态与可执行时间信息。

    preload 可提供 (code,date,raw_up_limit) 集合，线程只做只读文件查询，
    缓存由主线程按稳定顺序写入。预取只调整读取过程，不改变查询结果。
    调用者须使用开始标签及已完成bar语义，不能提前读取盘中未来信息。"""
    cache: Dict[Tuple[str, dt.date, float], _ex.BreakLookupResult] = {}

    def read_one(code: str, date, raw_up_limit: Optional[float]):
        up = _f(raw_up_limit)
        if up is None or up <= 0:
            return _ex.BreakLookupResult("missing", reason="缺原始涨停价格")
        p = repo.minute_path(code, freq)
        if not p.exists():
            return _ex.BreakLookupResult("missing", reason="缺分钟文件")
        ts = pd.Timestamp(date).normalize()
        t = pq.read_table(p, filters=[("trade_date", "==", ts)])
        bars = t.to_pandas(ignore_metadata=True)
        if len(bars) == 0:
            return _ex.BreakLookupResult("missing", reason="当天没有分钟行情")
        return _ex.confirmed_break_fill(bars, up, deadline=T_BOARD_WINDOW_END)

    def cache_key(code: str, date, raw_up_limit: float):
        return (str(code), date_key(date), float(raw_up_limit))

    if preload:
        # dict.fromkeys：去重且保留稳定顺序；线程只做只读文件查询，结果由主线程写缓存。
        queries = list(dict.fromkeys(
            cache_key(code, date, up) for code, date, up in preload))
        if queries:
            count = min(max(1, int(workers)), len(queries))
            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                values = pool.map(lambda item: read_one(*item), queries)
                cache.update(zip(queries, values))

    def lookup(code: str, date, raw_up_limit: Optional[float]):
        up = _f(raw_up_limit)
        if up is None or up <= 0:
            return None
        key = cache_key(code, date, up)
        if key not in cache:
            cache[key] = read_one(*key)
        return cache[key]

    return lookup


def _break_queries(frames: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame,
                   picks: Mapping) -> List[Tuple[str, dt.date, float]]:
    """从已准备好的日线找出必然需要分钟线判开板的查询。

    只看决策日下一交易日的 ``raw_open == raw_up_limit``；这是
    :meth:`CxqEngine._resolve_buy_price` 调分钟查询的同一前置判据。这里可以多算，
    但不能少算；ST、环境空仓等后续过滤不参与预取判据。
    """
    days = [value.date() for value in pd.DatetimeIndex(benchmark.index)]
    next_day = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    out: List[Tuple[str, dt.date, float]] = []
    for decide_raw, rows in (picks or {}).items():
        trade_day = next_day.get(date_key(decide_raw))
        if trade_day is None:
            continue
        stamp = pd.Timestamp(trade_day)
        for code, _score in rows:
            frame = frames.get(str(code))
            if frame is None or stamp not in frame.index:
                continue
            row = frame.loc[stamp]
            raw_open = _f(row.get("raw_open"))
            raw_up = _f(row.get("raw_up_limit"))
            if raw_open is not None and raw_up is not None and _eq(raw_open, raw_up):
                out.append((str(code), trade_day, raw_up))
    return out


# ══════════════════ 结果 ══════════════════

@dataclass(frozen=True)
class Trade:
    """一笔真实成交（由 broker 回调记录，价格含滑点）。"""
    date: dt.date
    code: str
    side: str            # "buy" / "sell"
    shares: int
    price: float
    amount: float        # 成交金额（不含费用）
    fee: float
    reason: str = ""
    at: dt.time = _ex.EXEC_TIME
    reference_price: Optional[float] = None
    decision_date: Optional[dt.date] = None
    price_source: str = ""


@dataclass(frozen=True)
class Blocked:
    """一笔被规则拦掉的操作（买不进 / 卖不掉）。"""
    date: dt.date
    code: str
    reason: str


@dataclass
class EngineResult:
    dates: List[dt.date] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)      # 每日**收盘**总资产
    cash: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    blocked: List[Blocked] = field(default_factory=list)
    missing_days: List[dt.date] = field(default_factory=list)   # 上游没给清单的交易日
    benchmark_returns: List[float] = field(default_factory=list)
    initial_cash: float = 0.0
    #: 首次「当日收盘总资产 < cfg.bankrupt_floor」的日期（只记第一次；回到线上也不清）。
    #: 🔴 它的作用是**标注**这条曲线从哪天起没有策略信息量，不是回测的终止条件。
    bankrupt_on: Optional[dt.date] = None
    #: 上面那条线用的阈值（从 cfg 抄一份）—— 让结果/报告自己说清"是按多少判的"，
    #: 免得换了 cfg 之后旧结果被误读。
    bankrupt_floor: float = 0.0
    #: 一字板「排队成交模型」实际成交的笔数明细：[(日期, 代码, 概率), ...]
    #: 🔴 单独记一本账 —— 报告里要写清"这里面有几笔是模型成交的"，不许混进正常成交。
    sealed_fills: List[Tuple] = field(default_factory=list)
    report: object = None

    @property
    def final_value(self) -> float:
        return self.equity[-1] if self.equity else self.initial_cash

    @property
    def pnl(self) -> float:
        return self.final_value - self.initial_cash

    def format(self) -> str:
        lines = [
            "初始资金 : %.2f" % self.initial_cash,
            "期末总资产: %.2f" % self.final_value,
            "净盈亏    : %.2f (%.2f%%)" % (
                self.pnl,
                100.0 * self.pnl / self.initial_cash if self.initial_cash else 0.0),
            "交易日数  : %d（其中上游没给清单 %d 天）"
            % (len(self.dates), len(self.missing_days)),
            "成交笔数  : %d（买 %d / 卖 %d）" % (
                len(self.trades),
                sum(1 for t in self.trades if t.side == "buy"),
                sum(1 for t in self.trades if t.side == "sell")),
            "被拦次数  : %d" % len(self.blocked),
            "爆仓标注  : %s" % (("%s（净值首次 < %.2f，此后曲线无策略信息量）"
                                 % (self.bankrupt_on, self.bankrupt_floor))
                                if self.bankrupt_on else "未触发"),
        ]
        if self.report is not None:
            lines.append("")
            lines.append(self.report.format())
        return "\n".join(lines)


# ══════════════════ 主循环（"总管"本体） ══════════════════

class CxqEngine(bt.Strategy):
    """每天喊一遍六个零件。

        参数
        ----
        picks        : {决策日: [(code, score), ...]} —— 上游清单（接口说明）
        sell_list    : {决策日: [code, ...]} —— 上游指定的无条件清仓票
        market_scores: {决策日: float} —— 上游的市场环境评分（配合 cfg.exposure_fn）
        cfg          : EngineConfig
        break_lookup : (code, date, raw_up_limit) -> (时刻, 开板价) | None
        clock_name   : 当日历用的数据（基准指数）
        """

    params = (
        ("picks", None),
        ("sell_list", None),
        ("market_scores", None),
        ("cfg", None),
        ("break_lookup", None),
        ("clock_name", "__benchmark__"),
    )

    # ══════════════ 初始化 ══════════════

    def __init__(self):
        self.cfg = self.p.cfg or EngineConfig()
        self.picks = {date_key(k): list(v) for k, v in (self.p.picks or {}).items()}
        self.sell_list = {date_key(k): list(v) for k, v in (self.p.sell_list or {}).items()}
        self.market_scores = ({date_key(k): float(v)
                               for k, v in (self.p.market_scores or {}).items()}
                              if self.p.market_scores else {})
        self.rules = RuleEngine()
        self.result = EngineResult(initial_cash=float(self.cfg.cash),
                                   bankrupt_floor=float(self.cfg.bankrupt_floor))
        self._pending = None          # (决策日, picks, sell_codes, market_score, has_list)
        self._last_close: Dict[str, float] = {}
        self._buy_times: Dict[str, dt.time] = {}
        # 🔴 名字不能叫 _orders —— backtrader 的 Strategy 内部就有 _orders(list)，
        #    撞名会在 self.clear() 里炸（踩过）
        self._order_meta = {}            # order.ref -> (code, side, reason)
        #: 一字板「排队成交模型」实际成交明细（字典里的每一笔都要能在报告里看到）
        self.sealed_fills: List[Tuple] = []
        # 硬止损判据：移动加权平均**开仓价**（含滑点的真实成交价），由 _apply_fill 维护。
        # 🔴 不用 broker.getposition().price —— 那是 backtrader 内部口径，
        #    与「我们记进 result.trades 的成交价」未必同源；自己记账才能手算对拍。
        self._shares: Dict[str, int] = {}        # code -> 当前股数（按成交累计）
        self._entry_price: Dict[str, float] = {}  # code -> 平均开仓价
        #: code -> **已持有交易日数**（开仓当天记 0，之后每个交易日 +1）。
        #: 只服务 `cfg.min_hold_days`（降换手），与硬止损/清仓判定无关。
        self._hold_days: Dict[str, int] = {}
        self._data_by_name = {d._name: d for d in self.datas}
        # 🔴 不能用 `or` 兜底：backtrader 的 data 对象 __bool__ 会返回 LineOwnOperation
        self._clock = self._data_by_name.get(self.p.clock_name)
        if self._clock is None:
            self._clock = self.datas[0]

    # ══════════════ 工具 ══════════════

    def _feed(self, code: str):
        return self._data_by_name.get(code)

    def _traded_today(self, d, today: dt.date) -> bool:
        """这只票今天有 bar 吗？（没有 = 停牌 / 未上市 / 缺数据）"""
        if d is None or len(d) == 0:
            return False
        try:
            return d.datetime.date(0) == today
        except (IndexError, ValueError):
            return False

    def _position_map(self) -> Dict[str, int]:
        out = {}
        for code, d in self._data_by_name.items():
            if code == self.p.clock_name:
                continue
            size = int(self.getposition(d).size)
            if size > 0:
                out[code] = size
        return out

    def _holdings(self) -> List[Holding]:
        return [Holding(c, n) for c, n in sorted(self._position_map().items())]

    def _date(self) -> dt.date:
        """今天（主时钟那一根 bar）的日期。

        🔴 为什么读的是**基准**（self._clock）的当前 bar：
           ``next_open`` 在 cerebro 里比 ``next`` 先跑（cerebro.py:1624 vs 1635），
           strategy 自己的 lines.datetime 那时还停在**上一根**（strategy.py:320-335 才更新它）。
           而每个 data 的 ``next()`` 在两个回调之前就已经推进过了，所以基准的
           ``datetime[0]`` 在两个回调里都等于「今天」。
        ``run_backtest`` 已把基准帧剪成与回测区间同宽（见 clock_frame），
        它同时是「日历」和 data0，所以它就是主时钟。
        """
        return self._clock.datetime.date(0)

    # ══════════════ backtrader 最小周期闸门（🔴 必须实现，否则净值曲线被腰斩） ══════════════

    def prenext(self):
        """在最小周期闸门期仍调用 next，保持已有交易日的记账。

        当某个数据源尚未出现首根bar时，Backtrader 会派发 prenext，默认实现
        为空。这里复用 next；读价前仍检查该数据源今天是否有bar，不能用前一根
        价格冒充今日行情。"""
        self.next()

    def prenext_open(self):
        """同上：闸门期也要执行清单，否则「开盘执行」在闸门期整段缺席。

        `next_open()` 内部每一步都自带「今天有没有这根 bar」的守卫
        （`_traded_today` + prices 字典），所以闸门期调用不会读到不存在的价格。
        """
        self.next_open()

    # ══════════════ T+1 日 09:30：执行昨天收盘定下的清单 ══════════════

    def _frozen_codes(self) -> set:
        """未满 `cfg.min_hold_days` 的持仓（降换手用；0 = 空集 = 现状）。"""
        n = int(getattr(self.cfg, "min_hold_days", 0) or 0)
        if n <= 0:
            return set()
        return {c for c, d in self._hold_days.items() if int(d) < n}

    def next_open(self):
        """09:30 锁名额和逐票预算；以后确认失败只留现金，不倒填替补。"""
        for code in list(self._shares):
            self._hold_days[code] = int(self._hold_days.get(code, 0)) + 1
        pend = self._pending
        if pend is None:
            return
        self._pending = None
        decide_date, picks, sell_codes, market_score, has_list = pend
        today = self._date()
        held = self._position_map()
        opening: Dict[str, float] = {}
        valuations: Dict[str, float] = {}
        can_sell: Dict[str, bool] = {}
        can_buy: Dict[str, bool] = {}
        for code in sorted({c for c, _ in picks} | set(sell_codes) | set(held)):
            d = self._feed(code)
            if self._traded_today(d, today):
                px = _f(d.open[0])
                if px is not None and px > 0:
                    opening[code] = px
                    valuations[code] = px
            if code in held and code not in valuations:
                # 只取前一交易日 next() 保存的有效收盘价，不读今日 close。
                last = self._last_close.get(code)
                if last is None:
                    raise ValueError("持仓 %s 没有可用估值价，不能把资产按零计算" % code)
                valuations[code] = last
            can_sell[code] = code in opening
            can_buy[code] = code in opening

        for code in sorted(held):
            if code not in opening:
                self.result.blocked.append(Blocked(today, code, "停牌/缺行情：不能卖，按上一有效收盘价估值"))
                continue
            d = self._feed(code)
            verdict = self.rules.check_sell(self._snapshot(code, d, today), today)
            if verdict and _eq(opening[code], _f(d.down_limit[0])):
                verdict = Verdict(False, "开盘封死跌停 -> 卖不出，顺延")
            lower = _f(d.down_limit[0])
            if verdict and lower is not None and self.cfg.costs.sell_price(opening[code]) < lower - 1e-9:
                verdict = Verdict(False, "滑点后卖价低于跌停价 -> 不预支卖出回款")
            can_sell[code] = bool(verdict)
            if not verdict:
                self.result.blocked.append(Blocked(today, code, str(verdict.reason)))

        forced_codes = set(sell_codes)
        sell_reasons: Dict[str, str] = {}
        for code in sorted(held):
            if not can_sell.get(code, False):
                continue
            why = stop_loss_reason(self._entry_price.get(code), opening[code], self.cfg.hard_stop_pct)
            if why:
                forced_codes.add(code)
                sell_reasons[code] = why
        if not has_list:
            for code in sorted(sell_reasons):
                self._submit_sell(code, int(held[code]), opening[code], sell_reasons[code], decide_date)
            return

        # 这里只过滤 09:30 已知的条件。一字板需要等待，仍占名额及预算。
        opening_picks: List[Tuple[str, float]] = []
        for code, score in picks:
            d = self._feed(code)
            why = ""
            if code not in opening:
                why = "今天没有开盘行情 -> 买不进"
            elif code in forced_codes:
                why = "今天需要清仓，不允许买回"
            elif int(d.is_st[0]) == 1:
                why = "ST 一律不买"
            elif self.cfg.skip_sealed_open and _eq(opening[code], _f(d.up_limit[0])):
                why = "开盘封涨停，主动不追（skip_sealed_open）"
            elif self.cfg.only_sealed_open and not _eq(opening[code], _f(d.up_limit[0])):
                why = "开盘未封涨停，only_sealed_open 不买"
            elif _SKIP_BOARDS_GTE and _BOARD_QUERY is not None:
                boards = _BOARD_QUERY(code, decide_date)
                if boards is not None and int(boards) >= _SKIP_BOARDS_GTE:
                    why = "%d 板 >= %d 板，按规则不买" % (int(boards), _SKIP_BOARDS_GTE)
            can_buy[code] = not bool(why)
            if why:
                self.result.blocked.append(Blocked(today, code, why))
            # 已持有但禁买的票仍可保留、减仓，不能从名次消失。
            if not why or (code in held and code in opening and code not in forced_codes):
                opening_picks.append((code, float(score)))

        plan = plan_day(
            holdings=self._holdings(), cash=float(self.broker.getcash()),
            prices=opening, valuation_prices=valuations, buy_prices=opening,
            sell_prices=opening, can_buy=can_buy, can_sell=can_sell,
            picks=opening_picks, sell_codes=forced_codes, sell_reasons=sell_reasons,
            frozen_codes=self._frozen_codes(), max_holdings=self.cfg.max_holdings,
            lot=self.cfg.lot, costs=self.cfg.costs,
            score_threshold=self.cfg.score_threshold, min_weight=self.cfg.min_weight,
            market_score=market_score, exposure_fn=self.cfg.exposure_fn)
        # 同一 feed 的 tick_open 属于全局字段。禁止同票双向单，杜绝互相覆盖。
        if {o.code for o in plan.sells} & {o.code for o in plan.buys}:
            raise RuntimeError("同一股票同轮双向下单，需独立订单价格绑定")
        for sale in plan.sells:
            self._submit_sell(sale.code, sale.shares, sale.ref_price, sale.reason, decide_date)

        confirmed_buys = []
        for buy in plan.buys:
            d = self._feed(buy.code)
            px, why = self._resolve_buy_price(buy.code, d, today)
            if px is None:
                self.result.blocked.append(Blocked(today, buy.code, why + "；保留现金，不回填09:30替补"))
                continue
            upper = _f(d.up_limit[0])
            experimental = any(day == today and code == buy.code for day, code, _ in self.sealed_fills)
            # 排队实验假设成交在涨停价，不能再加滑点穿越涨停；它不属于主结果。
            execution_price = px if experimental else self.cfg.costs.buy_price(px)
            if upper is not None and execution_price > upper + 1e-9:
                self.result.blocked.append(Blocked(today, buy.code, "滑点后买价超过涨停价"))
                continue
            # 预算在 09:30 已锁；价格变了也不增加股数或动用其他候选释放的现金。
            budget = buy.amount + buy.fee
            shares = int(buy.shares)
            while shares > 0:
                amount = shares * execution_price
                if amount + self.cfg.costs.buy_fee(amount) <= budget + 1e-9:
                    break
                shares -= self.cfg.lot
            if shares <= 0:
                self.result.blocked.append(Blocked(today, buy.code, "预留预算不足一手"))
                continue
            at = self._buy_times.get(buy.code, _ex.EXEC_TIME)
            confirmed_buys.append((at, buy.code, shares, px, execution_price, buy.reason, experimental))
        # 日线 broker 内按时间排序提交，晚确认的订单不得先占用早订单的资金。
        for at, code, shares, px, execution_price, why, experimental in sorted(confirmed_buys):
            d = self._feed(code)
            d.tick_open = float(execution_price)
            ref = self.buy(data=d, size=shares).ref
            self._order_meta[ref] = (code, "buy", why, at, px, decide_date,
                                     "排队概率实验（非观测成交）" if experimental else
                                     ("09:30开盘价" if at == _ex.EXEC_TIME else "分钟确认后的开盘近似价"))

    def _submit_sell(self, code, shares, price, reason, decide_date):
        """卖单始终使用自己的 09:30 参考价，不能被随后买入价改写。"""
        d = self._feed(code)
        lower = _f(d.down_limit[0])
        if lower is not None and self.cfg.costs.sell_price(price) < lower - 1e-9:
            self.result.blocked.append(Blocked(self._date(), code, "滑点后卖价低于跌停价"))
            return
        d.tick_open = float(self.cfg.costs.sell_price(price))
        ref = self.sell(data=d, size=int(shares)).ref
        self._order_meta[ref] = (code, "sell", reason, _ex.EXEC_TIME, price,
                                 decide_date, "09:30开盘价")

    # ══════════════ T 日收盘后：记净值 + 收下清单 ══════════════

    def next(self):
        today = self._date()
        # 清掉上一根 bar 注入的成交价，避免影响后面的 bar
        for d in self.datas:
            d.tick_open = None

        for d in self.datas:
            if self._traded_today(d, today):
                close = _f(d.close[0])
                if close is not None and close > 0:
                    self._last_close[d._name] = close
        self.result.dates.append(today)
        value = float(self.broker.getvalue())
        self.result.equity.append(value)
        self.result.cash.append(float(self.broker.getcash()))

        # 爆仓标注（接口约定）：净值跌破 floor -> 记下**第一个**这样的交易日。
        # 🔴 三件事刻意如此：
        #    (1) 只标注，**不停止运行 / 不跳过后面的 next** —— 净值点数照旧等于区间交易日数；
        #    (2) 只在 bankrupt_on is None 时写 -> 后来回到 floor 以上**不会**清掉已记的日期；
        #    (3) 判据是**当日收盘总资产**（与 equity 曲线同一个数），不是现金 —— 满仓下跌也算。
        if self.result.bankrupt_on is None and value < float(self.cfg.bankrupt_floor):
            self.result.bankrupt_on = today

        has_picks = today in self.picks
        has_sells = today in self.sell_list
        has_list = has_picks or has_sells
        if not has_list:
            self.result.missing_days.append(today)
        # 实验参数须与执行假设一起验证，不能从研究成绩推导默认可成交性。
        self._pending = (today,
                         self.picks.get(today, []),
                         self.sell_list.get(today, []),
                         self.market_scores.get(today),
                         has_list)

    # ══════════════ 成交回调 ══════════════

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        info = self._order_meta.pop(order.ref, None)
        code, side, reason = info[:3] if info else (None, None, "")
        if order.status == order.Completed:
            px = float(order.executed.price)
            sh = int(abs(order.executed.size))
            amount = px * sh
            is_buy = order.isbuy()
            self.result.trades.append(Trade(
                date=self._date(), code=code or "?",
                side="buy" if is_buy else "sell",
                shares=sh, price=px, amount=amount,
                fee=float(order.executed.comm or 0.0), reason=reason,
                at=info[3] if info else _ex.EXEC_TIME,
                reference_price=info[4] if info else None,
                decision_date=info[5] if info else None,
                price_source=info[6] if info else ""))
            if code:
                # 维护开仓价台账（硬止损的唯一判据）—— 必须在 record_sell 之前，
                # 因为 record_sell 会清掉 T+1 台账，两本账要在同一处成交点同步推进。
                self._apply_fill(code, "buy" if is_buy else "sell", sh, px)
            if is_buy:
                self.rules.record_buy(code, self._date())
            else:
                self.rules.record_sell(code, self._date())
        else:
            name = {order.Canceled: "已取消", order.Margin: "资金不足被拒",
                    order.Rejected: "被拒", order.Expired: "过期"}.get(order.status, str(order.status))
            self.result.blocked.append(Blocked(self._date(), code or "?", "订单%s" % name))

    def _apply_fill(self, code: str, side: str, shares: int, px: float) -> None:
        """按成交维护**移动加权平均开仓价** —— 第 3 层硬止损的唯一判据。

        口径（A股移动加权平均法）：
          · 买入：新成本 = (旧成本×旧股数 + 本次价×本次股数) / 新股数
          · 卖出：**不改变单位成本**（只减股数）
          · 清仓（股数归 0）：忘掉该票 -> 下次建仓重新起算，
            绝不能让上一轮的老成本去判新仓（否则一建仓就被误止损）

        ⚠️ 这里**刻意不用** broker.getposition().price：那是 backtrader 的内部口径，
           与本引擎记进 result.trades 的成交价未必同源，手算就对不上了。
        """
        old = self._shares.get(code, 0)
        if side == "buy":
            new = old + int(shares)
            if old <= 0 and new > 0:
                self._hold_days[code] = 0        # 建仓当天记 0（之后每个交易日 +1）
            if new > 0:
                prev_px = self._entry_price.get(code)
                if old > 0 and prev_px is not None:
                    self._entry_price[code] = (prev_px * old + px * int(shares)) / new
                else:
                    self._entry_price[code] = px
            self._shares[code] = new
        else:
            new = old - int(shares)
            if new <= 0:
                self._shares.pop(code, None)
                self._entry_price.pop(code, None)
                self._hold_days.pop(code, None)   # 清仓 -> 忘了这一轮
            else:
                self._shares[code] = new

    # ══════════════ 内部：一字板特例 ══════════════

    def _resolve_buy_price(self, code: str, d, today: dt.date):
        """今天这只票**能按什么价买到**？

                普通情况：09:30 开盘价（不碰分钟线，快）。
                涨停开盘：看 09:30~09:50 有没有开板 ——
                    开板了 -> 按开板那一刻的价格（接口说明 #4）
                    没开板 -> 买不进（一字板）
                """
        px = _f(d.open[0])
        up = _f(d.up_limit[0])
        self._buy_times[code] = _ex.EXEC_TIME
        if px is None or px <= 0:
            return None, "没有开盘价"
        if up is None or not _eq(px, up):
            if self.cfg.only_sealed_open:
                return None, "开盘没封在涨停价 -> only_sealed_open 只买一字板（名额顺延）"
            return px, ""                      # 常规：开盘价成交
        # 🔴 主动过滤（cfg.skip_sealed_open）：开盘即封涨停 -> 今天不碰它，
        #    不等开板、不追高。名额顺延给下一只（调用方 continue，不动 ok_picks）。
        if self.cfg.skip_sealed_open:
            return None, "开盘封涨停，主动不追（skip_sealed_open）"
        lookup = self.p.break_lookup
        if lookup is None:
            return None, "一字板缺分钟行情，未知状态禁止概率补成交"
        got = lookup(code, today, _f(d.raw_up_limit[0]))
        if got is None:
            return None, "一字板分钟查询未知，禁止概率补成交"
        if isinstance(got, _ex.BreakLookupResult):
            if got.status == "no_break":
                # 概率模型只是显式实验；仅完整有效窗口可以进入，按窗口结束时间审计。
                self._buy_times[code] = T_BOARD_WINDOW_END
                return self._sealed_fallback(code, today, up, "一字板，09:50 前未开板 -> 买不进")
            if got.status != "filled":
                return None, got.reason or "没有合法分钟成交"
            at, raw_px = got.at, got.price
            if got.confirmed_at is None or at is None or at < got.confirmed_at:
                return None, "分钟成交早于确认时间"
        else:
            # 兼容显式外部适配器：(可执行时刻, 该时刻开盘价)，时序由适配器负责。
            at, raw_px = got
        if at is None or at > T_BOARD_WINDOW_END:
            return None, "一字板，09:50 后才开板 -> 买不进"
        if at < _ex.EXEC_TIME:
            return None, "分钟成交时刻早于开盘"
        ratio = _f(d.adj_ratio[0]) or 1.0
        p = _f(raw_px)
        if p is None or p <= 0:
            return None, "开板价异常 -> 买不进"
        if _eq(p * ratio, up) or p * ratio > up:
            return None, "确认后的买入价已回封涨停"
        lower = _f(d.down_limit[0])
        if lower is not None and p * ratio < lower - 1e-9:
            return None, "分钟买入价低于跌停价，行情不合法"
        self._buy_times[code] = at
        return p * ratio, ""

    def _sealed_fallback(self, code: str, today, up, why: str):
        """一字板**买不进**的兜底出口 —— 只有开了「排队成交模型」才可能改成成交。

        `cfg.sealed_fill_prob` = 0（默认）-> 原样返回买不进，**行为与历史完全一致**。
        >0 -> 按确定性概率抽一次：抽中则按**涨停价**成交（一字板只能按涨停价成交），
              并把这一笔记进 `self.sealed_fills`（报告要单独列出来）。
        """
        prob = min(1.0, max(0.0, float(getattr(self.cfg, "sealed_fill_prob", 0.0) or 0.0)))
        fn = getattr(self.cfg, "sealed_fill_fn", None)
        if fn is not None:                     # 数据驱动模型优先（缺数据时它自己返回 0）
            try:
                prob = min(1.0, max(0.0, float(fn(code, today) or 0.0)))
            except Exception:                  # noqa: BLE001 - 模型出错绝不许影响主流程
                prob = 0.0
        if prob > 0 and sealed_draw(code, today, self.cfg.sealed_fill_seed) < prob:
            self.sealed_fills.append((today, code, prob))
            return up, ""
        return None, why

    def _snapshot(self, code: str, d, today: dt.date) -> BarSnapshot:
        """只读**开盘那一刻已知**的字段（见模块头的诚实说明）。"""
        if not self._traded_today(d, today):
            return BarSnapshot(code=code, is_suspended=True)
        return BarSnapshot(
            code=code,
            is_st=bool(int(d.is_st[0])) if _f(d.is_st[0]) is not None else False,
            is_suspended=False,
            open=_f(d.open[0]),
            up_limit=_f(d.up_limit[0]),
            down_limit=_f(d.down_limit[0]),
            limit_status=None,          # 🔴 刻意不用（那是收盘后才知道的）
        )


# ══════════════════ 跑一次回测 ══════════════════

def run_engine(*, frames: Mapping[str, pd.DataFrame],
               benchmark_frame: pd.DataFrame,
               picks: Mapping,
               cfg: Optional[EngineConfig] = None,
               sell_list: Optional[Mapping] = None,
               market_scores: Optional[Mapping] = None,
               break_lookup: Optional[Callable] = None,
               runonce: bool = False,
               bench_returns_frame: Optional[pd.DataFrame] = None) -> EngineResult:
    """用**已经准备好的帧**跑一遍（单元测试可直接提供人工构造帧）。

    benchmark_frame
        主时钟 + 基准数据。**它的 bar 数就是净值点数**：
        每一根 bar 都会推进一次策略（`prenext/nextstart` 也转发到 `next`，
        见 `CxqEngine.prenext`），所以调用方要给「回测区间」那一截
        （`run_backtest` 用 `clock_frame` 剪）。
    bench_returns_frame
        只用于算**基准日收益**的帧，默认与 benchmark_frame 同一个。
        它需要比主时钟多一段前值（`_benchmark_returns` 要拿 dates[0] 之前最近的收盘价）。
    """
    cfg = cfg or EngineConfig()
    if len(benchmark_frame) == 0:
        raise ValueError("基准帧为空 —— 基准同时当日历用，必须有")

    cer = bt.Cerebro(cheat_on_open=True)
    cer.broker.setcash(float(cfg.cash))
    cer.broker.addcommissioninfo(AShareCommission(costs=cfg.costs))
    # 每笔订单在自己的参考价上应用一次滑点。不能让日线 high/low 再裁剪成交价，
    # 否则今日尚不可见的区间会影响开盘成交。broker 只负责现金、持仓及费用。
    cer.broker.set_slippage_perc(0.0)
    if hasattr(cer.broker, "set_coo"):
        cer.broker.set_coo(True)

    # 开盘买单的资金检查应依据执行时价格，不能用当日收盘价决定开盘是否买得起。
    if hasattr(cer.broker, "set_checksubmit"):
        cer.broker.set_checksubmit(False)

    cer.adddata(ParquetFeed(dataname=benchmark_frame), name="__benchmark__")
    for code, fr in frames.items():
        if len(fr) == 0:
            continue
        cer.adddata(ParquetFeed(dataname=fr), name=code)

    cer.addstrategy(CxqEngine, picks=picks, sell_list=sell_list,
                    market_scores=market_scores, cfg=cfg,
                    break_lookup=break_lookup)
    cer.run(runonce=runonce)
    strat = cer.runstrats[0][0]
    result: EngineResult = strat.result
    result.sealed_fills = list(getattr(strat, "sealed_fills", []) or [])

    # ── 净值曲线 -> 第 6 层绩效 ──
    eq = [float(cfg.cash)] + list(result.equity)
    bench = _benchmark_returns(
        bench_returns_frame if bench_returns_frame is not None else benchmark_frame,
        result.dates)
    amounts = [t.amount for t in result.trades]
    result.benchmark_returns = bench
    result.report = evaluate(eq, benchmark_returns=(bench if bench else None),
                            trade_amounts=amounts,
                            benchmark_name=cfg.benchmark_name)
    return result


def _benchmark_returns(benchmark_frame: pd.DataFrame,
                       dates: Sequence[dt.date]) -> List[float]:
    """基准日收益，**长度与策略日收益严格一致**。

    策略收益是 N 个（净值曲线 N+1 个点），所以基准也要 N 个：
    第一个点用「dates[0] 的前一个交易日」做基准，不能少一个也不能多一个。
    """
    if len(dates) == 0 or len(benchmark_frame) == 0:
        return []
    close = benchmark_frame["close"].astype(float)
    if len(close) == 0:
        return []
    pos = {ts.date(): i for i, ts in enumerate(close.index)}
    prev = None
    for i, ts in enumerate(close.index):          # dates[0] 之前最近的一个收盘价
        if ts.date() < dates[0]:
            prev = float(close.iloc[i])
        else:
            break
    out = []
    for d in dates:
        i = pos.get(d)
        if i is None:                              # 基准当天没数据 -> 0 收益占位，不猜价格
            out.append(0.0)
            continue
        cur = float(close.iloc[i])
        if prev is None or prev <= 0:
            out.append(0.0)
        else:
            out.append(cur / prev - 1.0)
        prev = cur
    return out


# ══════════════════ 便捷入口：自己读 Parquet ══════════════════

#: 引擎真正需要的日线列（repo.daily 的默认列**不含** adj_factor / vol，必须显式要）
_ENGINE_COLS = ("trade_date", "open", "high", "low", "close", "vol",
                "adj_factor", "up_limit", "down_limit", "is_st")


def prepare_frame(repo: ParquetRepo, code: str, start, end, *, qfq: bool = True):
    """从主库取一只票的日线 -> 引擎帧（没有数据返回 None）。"""
    daily = repo.daily(code, start, end, cols=_ENGINE_COLS)
    if daily is None or len(daily) == 0:
        return None
    return qfq_frame(daily, qfq=qfq)


def prepare_frames(repo: ParquetRepo, codes: Iterable[str], start, end,
                   *, qfq: bool = True) -> Dict[str, pd.DataFrame]:
    """批量读盘后按代码切分行情帧。

    可用 daily_many 时一次过滤代码集合，减少重复解码；每只股票仍经过
    同一 qfq_frame 整理。无批量读取接口时退回逐票读取，保持兼容。"""
    codes = [str(c) for c in codes]
    if not codes:
        return {}
    fn = getattr(repo, "daily_many", None)
    bulk = None
    if fn is not None:
        try:
            bulk = fn(codes, start, end, cols=_ENGINE_COLS)
        except Exception:                          # noqa: BLE001 - 批量失败就退回逐票
            bulk = None
    out: Dict[str, pd.DataFrame] = {}
    if bulk is None or not len(bulk):
        for code in codes:
            fr = prepare_frame(repo, code, start, end, qfq=qfq)
            if fr is not None and len(fr):
                out[code] = fr
        return out
    for code, grp in bulk.groupby("ts_code", sort=False):
        if not len(grp):
            continue
        fr = qfq_frame(grp, qfq=qfq)
        if fr is not None and len(fr):
            out[str(code)] = fr
    return out


def run_backtest(*, picks: Mapping, start, end,
                 cfg: Optional[EngineConfig] = None,
                 repo: Optional[ParquetRepo] = None,
                 benchmark: Optional[str] = None,
                 codes: Optional[Iterable[str]] = None,
                 sell_list: Optional[Mapping] = None,
                 market_scores: Optional[Mapping] = None,
                 benchmark_start=None) -> EngineResult:
    """正式入口：自己读数据、自己拼帧、跑主循环。

    codes 不传就从 picks / sell_list 里自动收集（引擎只加载**用得到的票**）。
    """
    cfg = cfg or EngineConfig()
    repo = repo or ParquetRepo()
    bench_code = benchmark or "000300.SH"

    if codes is None:
        got = set()
        for v in (picks or {}).values():
            got |= {c for c, _ in v}
        for v in (sell_list or {}).values():
            got |= set(v)
        codes = sorted(got)

    # 基准要往前多取一段（算基准收益要有个前值）
    b_start = benchmark_start or (pd.Timestamp(start) - pd.Timedelta(days=400))
    bench_full = benchmark_frame(repo.index_daily(bench_code, b_start, end))
    # 基准前值不能作为策略主时钟，否则权益统计窗口会包含回测区间外日期。
    clock = clock_frame(bench_full, start, end)
    if clock is None or len(clock) == 0:
        raise ValueError("回测区间 [%s, %s] 内基准为空 —— 基准同时当日历用，必须有"
                         % (start, end))
    frames = prepare_frames(repo, codes, start, end, qfq=cfg.qfq)
    break_lookup = make_break_lookup(
        repo, preload=_break_queries(frames, clock, picks))
    return run_engine(frames=frames, benchmark_frame=clock, picks=picks,
                      cfg=cfg, sell_list=sell_list, market_scores=market_scores,
                      bench_returns_frame=bench_full,
                      break_lookup=break_lookup)
