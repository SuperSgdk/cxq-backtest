"""公开入口：策略在收盘后给目标仓位，Backtrader 在下一交易日模拟成交。"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from . import compat as _compat
import backtrader as bt
import pandas as pd

from .costs import AShareCommission, AShareCosts
from .performance import evaluate

MARKET_COLUMNS = ("date", "code", "open", "high", "low", "close", "volume",
                  "up_limit", "down_limit", "is_st", "is_suspended")


def _date(value):
    return pd.Timestamp(value).date()


def _read(source):
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(source)
    return pd.read_csv(source, dtype={"date": "string", "code": "string"})


def load_market_data(source) -> pd.DataFrame:
    """检查同一价格口径的日行情；缺行、缺字段和未知状态直接报错。"""
    data = _read(source)
    missing = set(MARKET_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError("行情缺字段：" + ", ".join(sorted(missing)))
    data = data.loc[:, list(MARKET_COLUMNS)].copy()
    if data.empty:
        raise ValueError("行情为空")
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="raise").dt.normalize()
    if data["date"].isna().any() or data["code"].isna().any():
        raise ValueError("日期或代码为空")
    data["code"] = data["code"].astype(str).str.strip()
    if (data["code"] == "").any():
        raise ValueError("代码不能为空")
    if data.duplicated(["date", "code"]).any():
        raise ValueError("同一日期、同一代码有重复行情")
    for column in MARKET_COLUMNS[2:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].map(lambda v: pd.notna(v) and math.isfinite(float(v))).all():
            raise ValueError("行情存在缺失或非有限值：" + column)
    for column in ("is_st", "is_suspended"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(column + " 必须明确为 0 或 1")
    prices = data[["open", "high", "low", "close", "up_limit", "down_limit"]]
    if (prices <= 0).any().any() or (data["volume"] < 0).any():
        raise ValueError("价格必须大于 0，成交量不能为负")
    if ((data["high"] < data[["open", "close", "low"]].max(axis=1)) |
            (data["low"] > data[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("行情 OHLC 大小关系错误")
    if ((data["up_limit"] < data["down_limit"]) |
            (data["high"] > data["up_limit"] + 1e-8) |
            (data["low"] < data["down_limit"] - 1e-8)).any():
        raise ValueError("行情 OHLC 超出涨跌停范围或价格关系错误；请核对价格口径")
    calendar = set(data["date"])
    if len(calendar) < 2:
        raise ValueError("至少需要两个交易日：一天产生决定，下一天执行")
    for code, group in data.groupby("code"):
        if set(group["date"]) != calendar:
            raise ValueError(f"{code} 缺少交易日行情；停牌需要显式行，不能把缺数当停牌")
        group = group.sort_values("date")
        previous = None
        for row in group.itertuples(index=False):
            if row.is_suspended:
                if row.volume != 0 or any(abs(getattr(row, k) - row.close) > 1e-8
                                          for k in ("open", "high", "low")):
                    raise ValueError(f"{code} 停牌行需零成交量和明确的同一估值价")
                if previous is not None and abs(row.close - previous) > 1e-8:
                    raise ValueError(f"{code} 停牌行估值与前一有效收盘价不一致")
            else:
                previous = row.close
    return data.sort_values(["date", "code"]).reset_index(drop=True)


def _weights(values, maximum=None):
    if not isinstance(values, Mapping):
        raise TypeError("策略须返回代码到目标权重的映射，或 None（保持持仓）")
    out = {}
    for code, weight in values.items():
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0 or weight > 1:
            raise ValueError("目标权重必须是 0 到 1 的有限数")
        out[str(code)] = weight
    if sum(out.values()) > 1 + 1e-10:
        raise ValueError("目标权重之和不能超过 1；不支持杠杆")
    if maximum is not None and sum(w > 0 for w in out.values()) > maximum:
        raise ValueError("目标持仓数量超过 max_positions")
    return out


def load_target_weights(source) -> dict:
    """日期对应完整目标清单；无清单日保持持仓，显式零权重表示退出。"""
    data = _read(source)
    if not {"date", "code", "weight"}.issubset(data.columns):
        raise ValueError("策略清单需要 date、code、weight 三列")
    if data[["date", "code", "weight"]].isna().any().any():
        raise ValueError("策略清单包含空值")
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="raise").dt.date
    data["code"] = data["code"].astype(str).str.strip()
    if (data["code"] == "").any() or data.duplicated(["date", "code"]).any():
        raise ValueError("策略清单代码为空或日期代码重复")
    return {date: _weights(dict(zip(group.code, group.weight)))
            for date, group in data.groupby("date")}


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    costs: AShareCosts = field(default_factory=AShareCosts)
    lot: int = 100
    max_positions: int = 20
    exclude_st: bool = True
    hard_stop_pct: float | None = None

    def __post_init__(self):
        if not math.isfinite(float(self.initial_cash)) or self.initial_cash <= 0:
            raise ValueError("起始资金必须是大于 0 的有限数")
        for name in ("lot", "max_positions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " 必须是正整数")
        for name, value in asdict(self.costs).items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("费用必须是非负有限数：" + name)
            if name != "min_commission" and value >= 1:
                raise ValueError("费率必须小于 1：" + name)
        if self.hard_stop_pct is not None and not (-1 < self.hard_stop_pct < 0):
            raise ValueError("止损比例必须在 -1 与 0 之间，None 表示关闭")


@dataclass(frozen=True)
class StrategyContext:
    date: dt.date
    cash: float
    equity: float
    positions: Mapping
    _history: Mapping = field(repr=False)

    @property
    def codes(self):
        return tuple(self._history)

    def history(self, code, count=None):
        """只返回决策日及以前的数据副本，不暴露明天的行情。"""
        frame = self._history[str(code)]
        return (frame if count is None else frame.tail(count)).copy(deep=True)


class TargetWeightStrategy:
    def __init__(self, signals):
        self.signals = {_date(date): _weights(weights) for date, weights in signals.items()}

    def on_close(self, context):
        return self.signals.get(context.date)

    def parameters(self):
        encoded = {str(date): weights for date, weights in sorted(self.signals.items())}
        return {"signals_sha256": hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest()}


class MovingAverageStrategy:
    """通用策略接口样例：短均线高于长均线时等权持有，不宣称盈利。"""
    def __init__(self, short_window=5, long_window=20):
        if not (isinstance(short_window, int) and isinstance(long_window, int)
                and 0 < short_window < long_window):
            raise ValueError("均线窗口须为正整数，短窗口小于长窗口")
        self.short_window, self.long_window = short_window, long_window

    def on_close(self, context):
        selected = []
        for code in context.codes:
            history = context.history(code, self.long_window)
            if len(history) >= self.long_window and history.close.tail(self.short_window).mean() > history.close.mean():
                selected.append(code)
        return {code: 1 / len(selected) for code in selected}

    def parameters(self):
        return {"short_window": self.short_window, "long_window": self.long_window}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


@dataclass
class BacktestResult:
    settings: dict
    summary: dict = field(default_factory=dict)
    equity: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    orders: list = field(default_factory=list)

    def to_dict(self):
        return _json_safe({"schema_version": "1.0", "settings": self.settings,
                           "summary": self.summary, "equity": self.equity,
                           "trades": self.trades, "orders": self.orders})


class _DailyFeed(bt.feeds.PandasData):
    lines = ("is_st", "is_suspended", "up_limit", "down_limit")
    params = tuple((name, -1) for name in lines)


class _Portfolio(bt.Strategy):
    params = (("market", None), ("user_strategy", None), ("config", None), ("result", None))

    def __init__(self):
        self.cfg, self.output = self.p.config, self.p.result
        self.feeds = {data._name: data for data in self.datas}
        self.pending, self.decision_date = None, None
        self.bought_on, self.meta = {}, {}

    def _positions(self):
        return {code: int(self.getposition(data).size) for code, data in self.feeds.items()
                if self.getposition(data).size > 0}

    def _blocked(self, code, side, reason, today):
        self.output.orders.append(dict(decision_date=str(self.decision_date), date=str(today),
                                       execution_time="09:30", code=code, side=side,
                                       status="blocked", reason=reason))

    def _price(self, code, side, today):
        data = self.feeds[code]
        if data.is_suspended[0]:
            return None, "停牌，持仓继续估值但不能交易"
        if side == "buy" and self.cfg.exclude_st and data.is_st[0]:
            return None, "当前配置禁止买入 ST"
        opened, upper, lower = float(data.open[0]), float(data.up_limit[0]), float(data.down_limit[0])
        if side == "buy" and opened >= upper - 1e-8:
            return None, "开盘涨停，日级模型不推测排队或盘中开板成交"
        if side == "sell" and opened <= lower + 1e-8:
            return None, "开盘跌停，本次卖出未成交；后续执行需策略再次给出计划"
        if side == "sell" and self.bought_on.get(code) == today:
            return None, "T+1：当日买入不能卖出"
        price = self.cfg.costs.buy_price(opened) if side == "buy" else self.cfg.costs.sell_price(opened)
        if price > upper + 1e-8 or price < lower - 1e-8:
            return None, "滑点后的成交价越过涨跌停边界"
        return price, ""

    def _submit(self, code, side, size, price, today, reason):
        data = self.feeds[code]
        # 每只股票每轮仅有一个净方向，tick_open 不会被另一方向覆盖。
        data.tick_open = float(price)
        order = self.buy(data=data, size=size) if side == "buy" else self.sell(data=data, size=size)
        self.meta[order.ref] = dict(decision_date=str(self.decision_date), date=str(today),
                                   execution_time="09:30", code=code, side=side,
                                   shares=size, reference_price=float(data.open[0]),
                                   reason=reason)

    def next_open(self):
        today = self.datas[0].datetime.date(0)
        held = self._positions()
        forced = set()
        if self.cfg.hard_stop_pct is not None:
            for code in held:
                position = self.getposition(self.feeds[code])
                stop_price = position.price * (1 + self.cfg.hard_stop_pct)
                tolerance = max(1.0, abs(position.price)) * 1e-12
                if not self.feeds[code].is_suspended[0] and self.feeds[code].open[0] <= stop_price + tolerance:
                    forced.add(code)
        if self.pending is None and not forced:
            return
        # 估值价不受成交权限影响；本日开盘已知，未读取本日 high/low/close。
        equity = float(self.broker.getcash()) + sum(held[c] * float(self.feeds[c].open[0]) for c in held)
        desired = dict(held) if self.pending is None else {}
        if self.pending is not None:
            for code, weight in self.pending.items():
                reference = self.cfg.costs.buy_price(float(self.feeds[code].open[0]))
                desired[code] = int(equity * weight / reference / self.cfg.lot) * self.cfg.lot
        for code in forced:
            desired[code] = 0
        projected_cash = float(self.broker.getcash())
        projected_positions = dict(held)
        # 先卖后买：只能花实际可卖出部分预计产生的净回款。
        for code in sorted(held):
            size = held[code] - desired.get(code, 0)
            if size <= 0:
                continue
            price, why = self._price(code, "sell", today)
            if price is None:
                self._blocked(code, "sell", why, today)
                continue
            amount = size * price
            fee = self.cfg.costs.sell_fee(amount)
            if amount < fee:
                self._blocked(code, "sell", "卖出金额不足以支付费用", today)
                continue
            self._submit(code, "sell", size, price, today, "开盘止损" if code in forced else "策略目标仓位")
            projected_cash += amount - fee
            projected_positions[code] -= size
        for code in sorted(desired):
            size = desired[code] - held.get(code, 0)
            if size <= 0 or code in forced:
                continue
            price, why = self._price(code, "buy", today)
            if price is None:
                self._blocked(code, "buy", why, today)
                continue
            occupied = sum(shares > 0 for shares in projected_positions.values())
            if not projected_positions.get(code, 0) and occupied >= self.cfg.max_positions:
                self._blocked(code, "buy", "受限持仓仍占名额，达到持仓上限", today)
                continue
            # 二分整手数量，包含最低佣金，避免高费率时出现长循环。
            lo, hi = 0, size // self.cfg.lot
            while lo < hi:
                mid = (lo + hi + 1) // 2
                amount = mid * self.cfg.lot * price
                if amount + self.cfg.costs.buy_fee(amount) <= projected_cash + 1e-9:
                    lo = mid
                else:
                    hi = mid - 1
            affordable = lo * self.cfg.lot
            if affordable == 0:
                self._blocked(code, "buy", "资金不足以支付一整手及费用", today)
                continue
            self._submit(code, "buy", affordable, price, today, "策略目标仓位")
            projected_cash -= affordable * price + self.cfg.costs.buy_fee(affordable * price)
            projected_positions[code] = projected_positions.get(code, 0) + affordable
        self.pending = None

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        record = self.meta.pop(order.ref, None)
        if record is None:
            return
        if order.status == order.Completed:
            size, price = abs(int(order.executed.size)), float(order.executed.price)
            record.update(status="filled", shares=size, price=price,
                          amount=size * price, fee=float(order.executed.comm))
            self.output.trades.append(dict(record))
            if order.isbuy():
                self.bought_on[record["code"]] = _date(record["date"])
        else:
            record.update(status="rejected", reason=order.getstatusname())
        self.output.orders.append(record)

    def next(self):
        date = self.datas[0].datetime.date(0)
        equity, cash = float(self.broker.getvalue()), float(self.broker.getcash())
        if not math.isfinite(equity) or cash < -1e-6:
            raise RuntimeError("账户出现非有限资产或负现金，停止回测")
        positions = self._positions()
        self.output.equity.append(dict(date=str(date), equity=equity, cash=cash, positions=positions))
        visible = self.p.market[self.p.market.date.dt.date <= date]
        history = {code: group.copy(deep=True) for code, group in visible.groupby("code")}
        context = StrategyContext(date, cash, equity, MappingProxyType(positions), MappingProxyType(history))
        weights = self.p.user_strategy.on_close(context)
        if weights is not None:
            weights = _weights(weights, self.cfg.max_positions)
            unknown = set(weights) - set(self.feeds)
            if unknown:
                raise ValueError("策略引用没有行情的代码：" + ", ".join(sorted(unknown)))
        self.pending, self.decision_date = weights, date

    def prenext(self):
        self.next()

    def prenext_open(self):
        self.next_open()

    def stop(self):
        if self.pending is not None:
            for code in sorted(set(self.pending) | set(self._positions())):
                self.output.orders.append(dict(decision_date=str(self.decision_date), code=code,
                    side="target", target_weight=self.pending.get(code, 0),
                    status="unexecuted_end", reason="数据结束，没有下一交易日执行"))


def run_backtest(data, strategy, config=None) -> BacktestResult:
    """统一处理策略、交易规则与账户记账。策略须实现 on_close(context)。"""
    market = load_market_data(data)
    config = config or BacktestConfig()
    if not callable(getattr(strategy, "on_close", None)):
        raise TypeError("策略需要实现 on_close(context)")
    if isinstance(strategy, TargetWeightStrategy):
        dates = set(market.date.dt.date)
        if set(strategy.signals) - dates:
            raise ValueError("策略日期超出行情日历，请明确匹配回测窗口")
    settings = asdict(config)
    settings.update(input_sha256=hashlib.sha256(market.to_csv(index=False).encode()).hexdigest(),
        strategy=type(strategy).__module__ + ":" + type(strategy).__qualname__,
        strategy_parameters=strategy.parameters() if callable(getattr(strategy, "parameters", None)) else {},
        execution_model="next_open_daily_v1", annualization_periods=252,
        start_date=str(market.date.min().date()), end_date=str(market.date.max().date()),
        package_version="0.1.0")
    settings["code_sha256"] = hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
        for name in ("framework.py", "costs.py", "performance.py", "compat.py"))).hexdigest()
    result = BacktestResult(settings=settings)
    cerebro = bt.Cerebro(cheat_on_open=True, stdstats=False)
    cerebro.broker.setcash(float(config.initial_cash))
    cerebro.broker.set_coo(True)
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.addcommissioninfo(AShareCommission(costs=config.costs))
    # 滑点在每个净方向的 tick_open 中一次性应用；关闭 broker 二次滑点。
    cerebro.broker.set_slippage_perc(0.0)
    for code, group in market.groupby("code"):
        frame = group.set_index("date").drop(columns="code")
        cerebro.adddata(_DailyFeed(dataname=frame), name=code)
    cerebro.addstrategy(_Portfolio, market=market, user_strategy=strategy, config=config, result=result)
    cerebro.run(runonce=False)
    amounts = [trade["amount"] for trade in result.trades]
    curve = [config.initial_cash] + [row["equity"] for row in result.equity]
    result.summary = evaluate(curve, trade_amounts=amounts).to_dict()
    result.summary.update(final_equity=curve[-1], total_fees=sum(t["fee"] for t in result.trades),
                          trade_count=len(result.trades), initial_cash=config.initial_cash,
                          blocked_count=sum(row["status"] == "blocked" for row in result.orders))
    result.summary = _json_safe(result.summary)
    return result
