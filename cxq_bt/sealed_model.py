# -*- coding: utf-8 -*-
"""额外实验：用竞价額和前一日封单额构造成交模型值。

x=竞价成交额/前一交易日收盘封单额，p=clamp(x/q,0,pmax)。
q 是调用者设置的队列假设，不是实测排队分位；p 不是经真实盘口
验证的成交概率。缺少输入时返回0并记录缺失，区分数据不足与模型值0。

本模块不选股，未接入公开日级默认入口。实际撮合、排队、撤单和反向
选择需要独立数据和核验，不能从这个比例推导真实可成交性或策略盈利。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

# 队列假设参数，未校准为真实账户排队分位。
DEFAULT_QUEUE_PCT = 0.35

#: 概率上限：**任何情况下都不假设 100% 一定成交**
PMAX = 0.95

#: 封得死/松 的分界（x = 竞价额/封单额）：低于它算「封得死」
TIGHT_X = 0.30


def fill_prob(auction_amount: Optional[float], seal_amount: Optional[float],
              queue_pct: float = DEFAULT_QUEUE_PCT, pmax: float = PMAX) -> float:
    """按假设公式返回模型值p，未校准为真实成交概率。

    人工手算：x=0.15、q=0.35得到0.4286；超上限截到pmax，缺输入返回0。"""
    if auction_amount is None or seal_amount is None:
        return 0.0
    try:
        a, s = float(auction_amount), float(seal_amount)
    except (TypeError, ValueError):
        return 0.0
    if not (a == a and s == s) or s <= 0 or a <= 0:
        return 0.0
    q = float(queue_pct)
    if q <= 0:
        return 0.0
    return min(float(pmax), a / s / q)


@dataclass
class SealedFillModel:
    """只读外部输入，按代码和日期返回实验模型值；不联网。

    使用前须核验输入在执行时已可知，不把模型值解释为实际成交概率。"""

    repo: object
    trade_days: List = field(default_factory=list)
    queue_pct: float = DEFAULT_QUEUE_PCT
    lookback_rows: int = 0
    #: 统计（报告要用）：命中/缺数据/封得死/封得松
    stats: Dict[str, int] = field(default_factory=lambda: {
        "asked": 0, "missing": 0, "zero": 0, "tight": 0, "loose": 0})

    _limit: Optional[pd.DataFrame] = None
    _auc: Optional[pd.DataFrame] = None
    _prev: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        days = list(self.trade_days or [])
        if not days:
            return
        d0, d1 = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
        self._prev = {str(days[i]): str(days[i - 1]) for i in range(1, len(days))}
        root = Path(getattr(self.repo, "root", "data"))
        p_ll = root / "股票数据" / "打板专题数据" / "limit_list_d.parquet"
        if p_ll.exists():
            t = pq.read_table(p_ll, columns=["ts_code", "trade_date", "limit",
                                             "fd_amount", "float_mv"],
                              filters=[("trade_date", ">=", d0 - pd.Timedelta(days=10)),
                                       ("trade_date", "<=", d1)])
            df = t.to_pandas(ignore_metadata=True).reset_index()
            df = df[df["limit"].astype(str).str.strip() == "U"]
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
            self._limit = df
            self.lookback_rows = len(df)
        p_auc = root / "股票数据" / "特色数据" / "stock_auction_o_daily.parquet"
        if p_auc.exists():
            t = pq.read_table(p_auc, columns=["ts_code", "trade_date", "amount"],
                              filters=[("trade_date", ">=", d0), ("trade_date", "<=", d1)])
            df = t.to_pandas(ignore_metadata=True).reset_index()
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
            self._auc = df

    def prob(self, code: str, buy_date) -> float:
        """**买入日**（= T+1）这只一字板的可成交概率。缺数据 -> 0（且记一笔）。"""
        self.stats["asked"] += 1
        d = str(buy_date)[:10]
        t = self._prev.get(d)
        if self._limit is None or self._auc is None or not t:
            self.stats["missing"] += 1
            return 0.0
        seal = self._limit[(self._limit["ts_code"] == code)
                           & (self._limit["trade_date"] == t)]
        auc = self._auc[(self._auc["ts_code"] == code) & (self._auc["trade_date"] == d)]
        if not len(seal) or not len(auc):
            self.stats["missing"] += 1
            return 0.0
        a = auc.iloc[0].get("amount")
        s = seal.iloc[0].get("fd_amount")
        p = fill_prob(a, s, self.queue_pct)
        if p <= 0:
            self.stats["zero"] += 1
            return 0.0
        try:
            x = float(a) / float(s)
        except (TypeError, ValueError, ZeroDivisionError):
            x = 0.0
        self.stats["tight" if x < TIGHT_X else "loose"] += 1
        return p
