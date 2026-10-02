# -*- coding: utf-8 -*-
"""面向约定 Parquet 目录布局的只读数据适配器。

公开日级入口使用 framework.load_market_data 的显式字段契约，见
 docs/data_contract.md。本模块属于额外适配组件，不分发数据。

通过 Arrow 投影和日期过滤，避免全部历史数据转入 pandas；日级读取
忽略索引元数据，分钟读取保留约定 MultiIndex。输入布局约定日线 vol
为手、amount 为千元，分钟与竞价表为股和元，调用者必须核验单位。
legacy 方法 is_suspended 把日线缺行视为停牌，只适用于已确认缺行
语义的数据，不能把任意数据缺口猜成停牌；公开入口要求显式停牌行。"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow.parquet as pq

from .rules import BarSnapshot
from .sizing import Holding

__all__ = [
    "ParquetRepo", "DAILY_VOL_UNIT", "DAILY_AMOUNT_UNIT",
    "MINUTE_VOL_UNIT", "hand_to_shares",
    "ENV_DATA_ROOT", "resolve_data_root",
]

#: 数据包根目录的环境变量名
ENV_DATA_ROOT = "CXQ_BT_DATA"


def resolve_data_root(root=None) -> Path:
    """数据目录优先级：显式 root > 环境变量 CXQ_BT_DATA > 包父目录/data。

    调用者选择自己有权限使用的目录，框架不提供数据。"""
    if root is not None:
        return Path(root)
    env = os.environ.get(ENV_DATA_ROOT)
    if env and str(env).strip():
        return Path(str(env).strip())
    return Path(__file__).resolve().parent.parent / "data"

# 约定输入单位，调用者须核验，不能推广到任意数据文件。
DAILY_VOL_UNIT = "手"        # stock_daily.vol
DAILY_AMOUNT_UNIT = "千元"   # stock_daily.amount
MINUTE_VOL_UNIT = "股"       # 分钟线 / 竞价表 vol
MINUTE_AMOUNT_UNIT = "元"

#: 第 3 层 BarSnapshot 需要的日线列
_DAILY_COLS = ("trade_date", "ts_code", "open", "high", "low", "close",
               "pre_close", "up_limit", "down_limit", "is_st", "suspend_type")


def hand_to_shares(vol_hand) -> float:
    """**手 -> 股**。stock_daily.vol 是手，分钟线/竞价表是股 —— 别混用。"""
    return float(vol_hand) * 100.0


class ParquetRepo:
    """主库数据访问层。**所有路径相对 data/，只读不写。**

    用法：
        repo = ParquetRepo()                       # 默认 <项目根>/data
        df   = repo.daily("000001.SZ", "2024-01-01", "2024-12-31")
        px   = repo.open_price("000001.SZ", "2024-01-02")
        bar  = repo.bar_snapshot("000001.SZ", "2024-01-02")
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = resolve_data_root(root)
        # 目录不存在立即报错，避免把路径错误误当成没有行情。
        if not self.root.is_dir():
            raise FileNotFoundError(
                '数据包根目录不存在：%s\n请通过环境变量 %s 选择自有数据目录，或显式传 ParquetRepo(root=...)。设置示例：set %s=<你的数据目录>'
                % (self.root, ENV_DATA_ROOT, ENV_DATA_ROOT))
        self._stock = self.root / "股票数据"
        self._index = self.root / "指数数据"
        self._dates_cache: Optional[list] = None

    # ══════════════ 路径 ══════════════

    @property
    def daily_path(self) -> Path:
        return self._stock / "行情数据" / "stock_daily.parquet"

    def minute_path(self, code: str, freq: str = "1min") -> Path:
        return self._stock / "行情数据" / ("stock_" + freq) / (code + ".parquet")

    @property
    def auction_open_path(self) -> Path:
        return self._stock / "特色数据" / "stock_auction_o_daily.parquet"

    @property
    def limit_ths_path(self) -> Path:
        return self._stock / "打板专题数据" / "limit_list_ths.parquet"

    # ══════════════ 日线 ══════════════

    def daily(self, code: str, start=None, end=None,
              cols: Iterable = _DAILY_COLS) -> pd.DataFrame:
        """取单只票的日线。带下推过滤，不用全量加载。"""
        f = [("ts_code", "==", code)]
        if start is not None:
            f.append(("trade_date", ">=", _ts(start)))
        if end is not None:
            f.append(("trade_date", "<=", _ts(end)))
        t = pq.read_table(self.daily_path, columns=list(cols), filters=f)
        df = t.to_pandas(ignore_metadata=True)     # 🔴 不加这个 trade_date 会变索引
        return df.sort_values("trade_date").reset_index(drop=True)

    def daily_many(self, codes: Iterable, start=None, end=None,
                   cols: Iterable = _DAILY_COLS) -> pd.DataFrame:
        """批量读取多个代码的日线长表。

        按代码集合与日期在 Arrow 层过滤，减少重复解码。投影包含 ts_code
        以便调用者分组；与 daily 的日期边界一致。"""
        codes = [str(c) for c in codes]
        if not codes:
            return pd.DataFrame(columns=list(cols))
        # 投影必须包含 ts_code 供调用方分组，只留业务列会丢失代码。
        proj = list(dict.fromkeys(["ts_code"] + [str(c) for c in cols]))
        f: list = [("ts_code", "in", codes)]
        if start is not None:
            f.append(("trade_date", ">=", _ts(start)))
        if end is not None:
            f.append(("trade_date", "<=", _ts(end)))
        t = pq.read_table(self.daily_path, columns=proj, filters=f)
        df = t.to_pandas(ignore_metadata=True)     # 🔴 不加这个 trade_date 会变索引
        if len(df):
            df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        return df

    # 框架与解释器会按名称隐式调用这些入口，不能只按直接调用次数判断是否删除。

    def trade_dates(self, start=None, end=None) -> list:
        """交易日列表（从 stock_daily 的日期唯一值取，不另建日历）。"""
        if self._dates_cache is None:
            t = pq.read_table(self.daily_path, columns=["trade_date"])
            s = t.to_pandas(ignore_metadata=True)["trade_date"]
            self._dates_cache = sorted(pd.to_datetime(s).dt.normalize().unique())
        out = self._dates_cache
        if start is not None:
            out = [d for d in out if d >= _ts(start)]
        if end is not None:
            out = [d for d in out if d <= _ts(end)]
        return [pd.Timestamp(d).date() for d in out]

    def is_suspended(self, code: str, date) -> bool:
        """约定布局的 legacy 判定：日线缺行返回 True。

        只用于已确认缺行代表停牌的数据；公开日级入口要求显式停牌行。"""
        return len(self.daily(code, date, date)) == 0

    # ══════════════ 分钟线 ══════════════

    def minute(self, code: str, date, freq: str = "1min") -> pd.DataFrame:
        """读取指定代码、日期和频率的分钟线。

        在 Arrow 层下推当天过滤，保留约定 trade_date/trade_time MultiIndex；
        索引布局不符时返回空表，不用日线补出分钟数据。"""
        p = self.minute_path(code, freq)
        if not p.exists():
            return pd.DataFrame()
        d = _ts(date)
        df = None
        try:
            _t = pq.read_table(p, filters=[("trade_date", "==", d)])
            if _t.num_rows:
                df = _t.to_pandas()               # 不用 ignore_metadata：要重建 MultiIndex
        except Exception:                         # noqa: BLE001 - 过滤读不了就退回原路径
            df = None
        if df is None or not len(df):
            df = pd.read_parquet(p)
        if isinstance(df.index, pd.MultiIndex) and "trade_date" in (df.index.names or []):
            try:
                return df.loc[(d, slice(None)), :]
            except KeyError:
                return df.iloc[0:0]
        if "trade_date" in df.columns:
            return df[pd.to_datetime(df["trade_date"]).dt.normalize() == d]
        return df.iloc[0:0]

    def first_break_time(self, code: str, date,
                         up_limit: Optional[float]) -> Optional[dt.time]:
        """从有效一分钟 low 寻找首次打开涨停的时刻；缺输入返回 None。"""
        if up_limit is None or up_limit <= 0:
            return None
        bars = self.minute(code, date)
        if len(bars) == 0 or "low" not in bars.columns:
            return None
        from .execution import first_break
        info = first_break(bars, float(up_limit))
        return info.at if info is not None else None

    # ══════════════ 开盘价 ══════════════

    def open_price(self, code: str, date,
                   prefer: str = "auction") -> Optional[float]:
        """读取约定开盘价来源：auction 表或分钟表的09:30 open。

        调用者须核对各来源的时间、单位与价格口径，不能假定任何数据都相等。"""
        if prefer == "auction":
            p = self.auction_open_path
            if p.exists():
                t = pq.read_table(p, columns=["trade_date", "ts_code", "open"],
                                  filters=[("ts_code", "==", code),
                                           ("trade_date", "==", _ts(date))])
                df = t.to_pandas(ignore_metadata=True)
                if len(df):
                    v = float(df.iloc[0]["open"])
                    if v == v and v > 0:          # 非 NaN 且为正
                        return v
        bars = self.minute(code, date)
        if len(bars):
            from .execution import open_price as _op
            return _op(bars)
        return None

    def open_prices(self, date, codes: Iterable, prefer: str = "auction") -> dict:
        """批量取开盘价 -> {code: price}。取不到的票**不进字典**（不猜价格）。"""
        out = {}
        for c in codes:
            px = self.open_price(c, date, prefer=prefer)
            if px is not None and px > 0:
                out[c] = px
        return out

    # ══════════════ 涨停榜 ══════════════

    def limit_status(self, date) -> dict:
        """读取非空的代码到涨停状态映射；缺失状态不自动当成可交易。"""
        p = self.limit_ths_path
        if not p.exists():
            return {}
        t = pq.read_table(p, columns=["trade_date", "ts_code", "limit_type", "status"],
                          filters=[("trade_date", "==", _ts(date))])
        df = t.to_pandas(ignore_metadata=True)
        if len(df) == 0 or "status" not in df.columns:
            return {}
        out = {}
        for _, r in df.iterrows():
            s = r["status"]
            if s is None or (isinstance(s, float) and s != s):
                continue                            # 空值不进字典
            out[str(r["ts_code"])] = str(s).strip()
        return out

    # ══════════════ 组装引擎要的入参 ══════════════

    def bar_snapshot(self, code: str, date) -> BarSnapshot:
        """组装第 3 层要的 BarSnapshot（含首次开板时刻）。"""
        d = self.daily(code, date, date)
        if len(d) == 0:
            # 没有日线 = 停牌（或当天不存在）
            return BarSnapshot(code=code, is_suspended=True)
        r = d.iloc[0]
        up = _f(r.get("up_limit"))
        return BarSnapshot(
            code=code,
            is_st=bool(int(r["is_st"])) if pd.notna(r.get("is_st")) else False,
            is_suspended=False,
            open=_f(r.get("open")), high=_f(r.get("high")),
            low=_f(r.get("low")), close=_f(r.get("close")),
            up_limit=up, down_limit=_f(r.get("down_limit")),
            limit_status=self.limit_status(date).get(code),
            first_break_time=self.first_break_time(code, date, up),
        )

    def holdings_from(self, shares_by_code: dict) -> list:
        """{code: shares} -> [Holding]，自动丢掉 0 股。"""
        return [Holding(c, int(s)) for c, s in shares_by_code.items() if s and s > 0]

    # ══════════════ 指数（第 6 层基准用）══════════════

    def index_daily(self, code: str, start=None, end=None) -> pd.DataFrame:
        """指数日线。基准用 000300.SH（沪深300）/ 000852.SH（中证1000）。"""
        p = self._index / "index_daily" / (code + ".parquet")
        if not p.exists():
            raise FileNotFoundError("指数文件不存在：%s" % p)
        f = []
        if start is not None:
            f.append(("trade_date", ">=", _ts(start)))
        if end is not None:
            f.append(("trade_date", "<=", _ts(end)))
        t = pq.read_table(p, filters=f) if f else pq.read_table(p)
        df = t.to_pandas(ignore_metadata=True)
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def benchmark_returns(self, start=None, end=None,
                          code: str = "000300.SH") -> pd.Series:
        """基准收益率序列（index = 日期），喂给第 6 层的 Information Ratio。"""
        df = self.index_daily(code, start, end)
        col = "close" if "close" in df.columns else "pct_chg"
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                      index=pd.to_datetime(df["trade_date"]))
        if col == "close":
            s = s.pct_change()
        else:
            s = s / 100.0
        return s.dropna()


# ══════════════════ 小工具 ══════════════════

def _ts(v) -> pd.Timestamp:
    """统一成 pd.Timestamp（parquet 里 trade_date 是 timestamp 类型）。"""
    return pd.Timestamp(v)


def _f(v) -> Optional[float]:
    """转 float，NaN/None -> None（规则层绝不猜值）。"""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x
