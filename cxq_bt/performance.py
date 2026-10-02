# -*- coding: utf-8 -*-
"""绩效指标、成交统计与持仓周期分析。

日收益 r_t = E_t/E_(t-1)-1。累计收益 prod(1+r)-1；年化系数为252；
波动率使用样本标准差 ddof=1。Sharpe 为算术平均超额日收益/样本
标准差*sqrt(252)，sharpe_geometric 使用几何年化收益和年化波动率，
二者不能混用。最大回撤为正跌幅 max(1-E_t/历史最高权益)。换手率由
实际成交额计算；Information Ratio 是日频 active return 均值/标准差。

人工净值 [1.0,1.1,1.05,1.2] 可用于独立手算和可选库交叉核对。
短样本年化可能很大，不代表真实长期收益；浮点缩放可能产生微小差异。
零波动等边界可能返回 nan/inf，导出时应明确处理。empyrical 和
quantstats 仅作可选核对库，不是框架运行的必需依赖。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "TRADING_DAYS_PER_YEAR", "DEFAULT_HORIZONS",
    "to_returns", "total_return", "annual_return", "annual_volatility",
    "sharpe_ratio", "sharpe_from_annual", "sortino_ratio",
    "max_drawdown", "max_drawdown_from_returns", "calmar_ratio",
    "Turnover", "turnover",
    "active_returns", "information_ratio", "tracking_error",
    "PerformanceReport", "evaluate",
    "CheckRow", "cross_check",
    "Trade", "RealizedStats", "DecayCurve", "decay_curve", "from_trades",
    "PerformanceAnalyzer",
]

#: A股一年交易日数（年化系数。empyrical / quantstats 默认也是 252）
TRADING_DAYS_PER_YEAR = 252

#: 持仓周期衰减曲线的默认观察点（接口约定 第 6 层）
DEFAULT_HORIZONS = (1, 2, 3, 5, 10)


# ══════════════════ 内部工具 ══════════════════

def _floats(seq, name):
    """转成 float 列表，顺便挡掉 nan / inf（脏数据不许悄悄混进指标）。"""
    out = []
    for x in seq:
        v = float(x)
        if not math.isfinite(v):
            raise ValueError("%s 里出现非有限值：%r" % (name, x))
        out.append(v)
    return out


def _std(xs, ddof=1):
    """标准差。ddof=1 是样本标准差（与 empyrical nanstd(..., ddof=1) 一致）。"""
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def _ratio_or_degenerate(numer, denom):
    """分母为 0 时按 numpy 的除法语义返回 +-inf / nan。

    empyrical 走的是 numpy 除法（np.errstate(divide="ignore")），
    所以 0 波动时它给的是 inf 而不是抛异常，这里保持一致。
    """
    if denom != 0.0:
        return numer / denom
    if numer > 0.0:
        return float("inf")
    if numer < 0.0:
        return float("-inf")
    return float("nan")


# ══════════════════ 净值曲线 -> 收益率 ══════════════════

def to_returns(equity):
    """净值曲线 -> 日收益序列。E_0 是期初，不产生收益，所以返回 n-1 个收益。

    手算示例：净值 [1.0, 1.1, 1.05, 1.2]
              -> [0.1, -0.045454545454545456, 0.1428571428571429]
    """
    eq = _floats(equity, "净值曲线")
    if len(eq) < 2:
        raise ValueError("净值曲线至少 2 个点，收到 %d 个" % len(eq))
    if any(v <= 0.0 for v in eq):
        raise ValueError("净值必须全为正数（含 0 或负数说明数据接错了）")
    return [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq))]


# ══════════════════ 收益类指标 ══════════════════

def total_return(returns):
    """累计收益 R = prod(1 + r_t) - 1。

    手算示例：日收益 [1/10, -1/22, 1/7] -> 1.1 * (21/22) * (8/7) = 1.2 -> R = 0.2
    """
    rs = _floats(returns, "日收益")
    if len(rs) < 1:
        raise ValueError("日收益序列不能为空")
    if any(r <= -1.0 for r in rs):
        raise ValueError("日收益不能 <= -100%（净值归零后无法复利）")
    ending = 1.0
    for r in rs:
        ending *= (1.0 + r)
    return ending - 1.0


def annual_return(returns, periods=TRADING_DAYS_PER_YEAR):
    """年化收益 AR = (1 + R) ** (periods / n) - 1。n = 日收益个数。

    等价写法（empyrical.stats.annual_return，stats.py:389）：
        num_years = len(returns) / 252
        return ending_value ** (1 / num_years) - 1

    手算示例（n = 252 时年化系数正好是 1，年化收益 = 累计收益）：
        252 天每天 +0.001 -> 1.001 ** 252 - 1 = 0.286434044376187746
    手算示例（n = 3，4 个点的小样本，数字大是正常的）：
        R = 0.2 -> (1.2) ** (252/3) - 1 = 1.2 ** 84 - 1 = 4479448.7388245724
    """
    rs = _floats(returns, "日收益")
    n = len(rs)
    if n < 1:
        raise ValueError("日收益序列不能为空")
    if periods <= 0:
        raise ValueError("年化系数必须为正数，收到 %r" % (periods,))
    ending = 1.0
    for r in rs:
        if r <= -1.0:
            raise ValueError("日收益不能 <= -100%（净值归零后无法复利）")
        ending *= (1.0 + r)
    return ending ** (float(periods) / n) - 1.0


def annual_volatility(returns, periods=TRADING_DAYS_PER_YEAR, ddof=1):
    """年化波动率 AV = std(r, ddof=1) * sqrt(periods)。

        🔴 ddof=1（样本标准差），使用样本标准差，与本模块公式约定一致。
        样本少于 2 个时返回 nan（empyrical stats.py:471 同样是 nan）。

        手算示例：[0.01, -0.01, 0.01, -0.01]
            mean = 0，平方偏差和 = 4 * 1e-4，ddof=1 -> 4e-4 / 3
            std = 0.01 * sqrt(4/3)，AV = 0.01 * sqrt(4/3) * sqrt(252)
                = 0.01 * sqrt(336) = 0.183303027798233600
        """
    rs = _floats(returns, "日收益")
    if len(rs) < 2:
        return float("nan")
    return _std(rs, ddof) * math.sqrt(periods)


def sharpe_ratio(returns, risk_free=0.0, periods=TRADING_DAYS_PER_YEAR, ddof=1):
    """Sharpe（算术平均版，与 empyrical / quantstats 同口径）。

        Sharpe = mean(r - rf) / std(r - rf, ddof=1) * sqrt(periods)

    🔴 risk_free 是**日频**常数，从每一天的收益里直接减掉
    （empyrical.stats.sharpe_ratio 的 risk_free 就是这个语义，见其文档串
    "Constant daily risk-free return throughout the period"）。
    如果手上是年化无风险利率 rf_a，先换算：rf = (1 + rf_a) ** (1/252) - 1。

    标准差为 0 时返回 +-inf / nan（跟 empyrical 的 numpy 除法一致）。

    手算示例：日收益 [1/10, -1/22, 1/7]，rf = 0
        mean  = 0.065800865800865801
        std   = 0.098704146689373719（ddof=1）
        Sharpe = 0.065800865800865801 / 0.098704146689373719 * sqrt(252)
               = 10.582699884516706
    手算示例（同一样本，rf = 0.001）：
        mean - rf = 0.064800865800865801 -> Sharpe = 10.421870695482274
    """
    rs = _floats(returns, "日收益")
    if len(rs) < 2:
        return float("nan")
    adj = [r - float(risk_free) for r in rs]
    m = sum(adj) / len(adj)
    return _ratio_or_degenerate(m, _std(adj, ddof)) * math.sqrt(periods)


def sharpe_from_annual(annual_ret, annual_vol, risk_free=0.0):
    """Sharpe（几何年化版，按年化收益与年化波动率计算）。

            Sharpe = (年化收益 - 无风险利率) / 年化波动率

        这里的 risk_free 是**年化**利率（和上一条的日频 risk_free 不是一回事，
        所以单独开一个函数，避免调用方传错）。

        手算示例（净值 [1.0, 1.1, 1.05, 1.2]，3 个收益）：
            年化收益 = 1.2 ** 84 - 1 = 4479448.7388245724
            年化波动 = 1.566879753065533112
            Sharpe   = 2858833.7618510437
        """
    return _ratio_or_degenerate(float(annual_ret) - float(risk_free),
                                float(annual_vol))


def sortino_ratio(returns, required_return=0.0, periods=TRADING_DAYS_PER_YEAR):
    """Sortino（与 empyrical / quantstats 同口径）。

        adj      = r - required_return            （日频门槛）
        downside = sqrt(mean(min(adj, 0) ** 2))   （分母是全样本数，ddof=0）
        Sortino  = mean(adj) * sqrt(periods) / downside

    等价于 empyrical（stats.py:737-746）：
        average_annual_return    = mean(adj) * 252
        annualized_downside_risk = sqrt(mean(min(adj,0)**2)) * sqrt(252)
        Sortino = 上面两个相除
    quantstats（stats.py:1029）用 sqrt(sum(负收益平方) / len(全部收益))，
    与这里的 mean(min(adj,0)**2) 是同一个东西（0 的平方不贡献）。

    ⚠️ 下行波动为 0（从来没亏过）时：empyrical 给 inf，quantstats 给 nan，
    本模块跟 empyrical。

    手算示例：日收益 [1/10, -1/22, 1/7]，required_return = 0
        只有 -1/22 是负的 -> mean(min(adj,0)**2) = (1/22)**2 / 3 = 1/1452
        downside = sqrt(1/1452) = 0.026243194054073898
        Sortino  = 0.065800865800865801 * sqrt(252) / 0.026243194054073898
                 = 39.802943179045010
    """
    rs = _floats(returns, "日收益")
    if len(rs) < 2:
        return float("nan")
    adj = [r - float(required_return) for r in rs]
    m = sum(adj) / len(adj)
    downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in adj) / len(adj))
    return _ratio_or_degenerate(m * math.sqrt(periods), downside)


# ══════════════════ 回撤 ══════════════════

def max_drawdown(equity):
    """最大回撤（净值曲线版，**返回正数 = 跌幅**）。

            MDD = max over t of (1 - E_t / max(E_0 .. E_t))

        🔴 **期初净值 E_0 也算一个历史最高点**（所以第一天就亏钱是能被算进去的）。
        这与 empyrical 一致：drawdown_series 从起始值 100 开始做 running max
        （stats.py:290-295），quantstats 也会补一个基准点（stats.py:2488-2496）。

        ⚠️ 符号：empyrical / quantstats 返回**负数**（-0.0454...），本模块返回**正数**
        （0.0454...），因为 接口说明 写的是 max(1 - 净值/历史最高净值)。
        对拍时取相反数即可。

        手算示例：净值 [1.0, 1.1, 1.05, 1.2]
            历史最高依次 = 1.0, 1.1, 1.1, 1.2
            回撤依次     = 0, 0, 1 - 1.05/1.1 = 1/22, 0
            MDD = 1/22 = 0.045454545454545455
        手算示例（第一天就亏）：净值 [1.0, 0.95, 0.969] -> MDD = 1 - 0.95/1.0 = 0.05

        实现细节：循环里写的是 (peak - v) / peak，与 接口说明 的 1 - v/peak
        **代数等价**，但避开了 1 - x（x 接近 1）的抵消误差。以 [1.0, 1.1, 1.05, 1.2]
        这一步为例（精确值 1/22 = 0.04545454545454545455）：
            1 - 1.05/1.1       = 0.045454545454545525   相对误差 1.53e-15
            (1.1 - 1.05) / 1.1 = 0.04545454545454549    相对误差 7.63e-16
        两种写法在双精度下都只能做到 ~1e-15，对拍时用相对误差比较，别用绝对误差卡。
        """
    eq = _floats(equity, "净值曲线")
    if len(eq) < 1:
        raise ValueError("净值曲线不能为空")
    peak = eq[0]
    mdd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def max_drawdown_from_returns(returns):
    """最大回撤（收益率序列版），返回值同样是正数。

    先把收益复利成净值（起点 1.0，且 1.0 参与历史最高点的比较），
    再走 max_drawdown。这样与 empyrical.max_drawdown(returns) / quantstats
    的输入形式对齐，三方可以直接对拍。
    """
    rs = _floats(returns, "日收益")
    eq = [1.0]
    for r in rs:
        eq.append(eq[-1] * (1.0 + r))
    return max_drawdown(eq)


def calmar_ratio(returns, periods=TRADING_DAYS_PER_YEAR):
    """Calmar = 年化收益 / 最大回撤（回撤取绝对值）。

    ⚠️ 没有回撤（MDD = 0）时：empyrical 返回 nan（stats.py:526-531），
    quantstats 返回 inf（stats.py:1677 直接除 0）。本模块跟 empyrical。

    手算示例：日收益 [0.01, -0.01, 0.01, -0.01]
        MDD  = 1 - 0.99980001/1.01 = 0.010099
        AR   = 0.99980001 ** 63 - 1 = -0.012521574501525
        Calmar = -0.012521574501525 / 0.010099 = -1.2398826123
    """
    mdd = max_drawdown_from_returns(returns)
    if mdd <= 0.0:
        return float("nan")
    return annual_return(returns, periods) / mdd


# ══════════════════ 换手率 ══════════════════

@dataclass(frozen=True)
class Turnover:
    """换手率结果。金额口径，买卖都算（双边）。"""
    total: float          #: 区间累计换手率 = sum(|成交金额|) / 平均权益
    annualized: float     #: 年化换手率 = total * 252 / n
    avg_equity: float     #: 平均权益 = 期初 + 每日末权益 的算术平均
    trade_amount: float   #: sum(|成交金额|)
    n_periods: int        #: 交易日数（= 净值点数 - 1）

    @property
    def one_way_annualized(self):
        """单边年化换手率 = 双边 / 2（很多研报用的是这个口径）。"""
        return self.annualized / 2.0


def turnover(equity, trade_amounts, periods=TRADING_DAYS_PER_YEAR):
    """换手率（自算，接口说明 要求）。

            累计换手率 = sum(|每笔成交金额|) / 平均权益
            年化换手率 = 累计换手率 * periods / n

        其中 平均权益 = (E_0 + E_1 + ... + E_n) / (n + 1)，即**把期初也算进去**的
        算术平均（用期初而不是只用期末，是因为资金进出会让期末权益失真）。

        trade_amounts 只吃金额、不吃方向：买入金额和卖出金额都按正数传进来
        （引擎的 BuyOrder.amount / SellOrder.amount 本身就是正数）。

        手算示例 1（只有买入，年化正好是整数）：
            权益 [100000, 110000]，买入 50000
            平均权益 = (100000 + 110000) / 2 = 105000
            累计 = 50000 / 105000 = 0.476190476190476190
            年化 = 0.476190476190476190 * 252 / 2 = 60.0
        手算示例 2（买 + 卖）：
            权益 [100000, 110000, 105000, 120000]，成交 50000 + 50000
            平均权益 = 108750
            累计 = 100000 / 108750 = 80/87 = 0.919540229885057471
            年化 = 80/87 * 252 / 4 = 1680/29 = 57.931034482758620690
        """
    eq = _floats(equity, "净值曲线")
    if len(eq) < 2:
        raise ValueError("净值曲线至少 2 个点，收到 %d 个" % len(eq))
    amounts = [abs(float(a)) for a in trade_amounts]
    n = len(eq) - 1
    avg_equity = sum(eq) / len(eq)
    if avg_equity <= 0.0:
        raise ValueError("平均权益必须为正数，收到 %r" % (avg_equity,))
    amount = sum(amounts)
    total = amount / avg_equity
    return Turnover(total=total, annualized=total * periods / n,
                    avg_equity=avg_equity, trade_amount=amount, n_periods=n)


# ══════════════════ 基准相关 ══════════════════

def active_returns(returns, benchmark_returns):
    """超额收益序列 r_p - r_b。

        🔴 长度必须**完全相等**且已经按同一天对齐：本函数不插值、不补零、不猜
        （接口说明 的口径：缺价格的票直接跳过，不猜）。长度不一致直接报错。
        """
    rp = _floats(returns, "策略日收益")
    rb = _floats(benchmark_returns, "基准日收益")
    if len(rp) != len(rb):
        raise ValueError("策略 %d 天、基准 %d 天，长度不一致：请先按同一交易日历对齐"
                         % (len(rp), len(rb)))
    return [a - b for a, b in zip(rp, rb)]


def information_ratio(returns, benchmark_returns):
    """Information Ratio（**日频，不年化**）：

        IR = mean(r_p - r_b) / std(r_p - r_b, ddof=1)

    口径来源（两种库叫法不同、数值相同）：
        empyrical.excess_sharpe(returns, factor_returns)   <- empyrical 没有 information_ratio
        quantstats.stats.information_ratio(returns, benchmark)
    两个库都**不做年化**；要年化自己乘 sqrt(252)。

    跟踪误差为 0 时返回 0.0（quantstats stats.py:2671-2673 就是这么写的）。

    手算示例：
        策略 [0.02, -0.01, 0.03, 0.00]，基准 [0.01, -0.02, 0.01, 0.01]
        超额 = [0.01, 0.01, 0.02, -0.01]
        mean = 0.03 / 4 = 0.0075
        方差 = (0.0025**2 * 2 + 0.0125**2 + 0.0175**2) / 3 = 4.75e-4 / 3
        std  = 0.012583057392117916
        IR   = 0.0075 / 0.012583057392117916 = 0.596039560679269713
    """
    act = active_returns(returns, benchmark_returns)
    if len(act) < 2:
        return float("nan")
    sd = _std(act, 1)
    if sd == 0.0:
        return 0.0
    return (sum(act) / len(act)) / sd


def tracking_error(returns, benchmark_returns, periods=TRADING_DAYS_PER_YEAR):
    """跟踪误差 = std(r_p - r_b, ddof=1) * sqrt(252)（年化）。

    手算示例（同上那 4 天）：0.012583057392117916 * sqrt(252) = 0.199749843554381789
    """
    act = active_returns(returns, benchmark_returns)
    if len(act) < 2:
        return float("nan")
    return _std(act, 1) * math.sqrt(periods)


# ══════════════════ 汇总报告 ══════════════════

@dataclass
class PerformanceReport:
    """一次回测的全部绩效指标（第 6 层的输出）。"""

    # ── 基本信息 ──
    n_periods: int = 0                 #: 交易日数（= 净值点数 - 1）
    periods_per_year: int = TRADING_DAYS_PER_YEAR
    risk_free: float = 0.0             #: 日频无风险收益
    equity_start: float = 0.0
    equity_end: float = 0.0

    # ── 收益与风险 ──
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    max_drawdown: float = 0.0          #: 正数 = 跌幅

    # ── 风险调整收益 ──
    sharpe: float = 0.0                #: 算术平均版（empyrical 口径）
    sharpe_geometric: float = 0.0      #: 几何年化版（接口约定 口径）
    sharpe_ddof0: float = 0.0          #: 只在 ddof 对照实验里用，正式结论别看这个
    sortino: float = 0.0
    calmar: float = 0.0

    # ── 换手 ──
    turnover: float = 0.0              #: 年化双边换手率
    turnover_total: float = 0.0        #: 区间累计双边换手率

    # ── 基准（可选）──
    benchmark_name: str = ""
    excess_return: float = float("nan")
    excess_return_annualized: float = float("nan")
    tracking_error: float = float("nan")
    information_ratio: float = float("nan")
    information_ratio_annualized: float = float("nan")

    def to_dict(self):
        """给外部（报告 / JSON / 上游策略）用的扁平字典。"""
        return {
            "n_periods": self.n_periods,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "sharpe_geometric": self.sharpe_geometric,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "turnover": self.turnover,
            "turnover_total": self.turnover_total,
            "benchmark_name": self.benchmark_name,
            "excess_return": self.excess_return,
            "excess_return_annualized": self.excess_return_annualized,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "information_ratio_annualized": self.information_ratio_annualized,
        }

    def format(self):
        """中文对齐文本，直接能贴进报告。"""
        rows = [
            ("交易日数", "%d" % self.n_periods),
            ("期初 / 期末权益", "%.2f -> %.2f" % (self.equity_start, self.equity_end)),
            ("累计收益", "%.4f%%" % (self.total_return * 100.0)),
            ("年化收益", "%.4f%%" % (self.annual_return * 100.0)),
            ("年化波动率", "%.4f%%" % (self.annual_volatility * 100.0)),
            ("最大回撤", "%.4f%%" % (self.max_drawdown * 100.0)),
            ("Sharpe（算术）", "%.4f" % self.sharpe),
            ("Sharpe（几何年化）", "%.4f" % self.sharpe_geometric),
            ("Sortino", "%.4f" % self.sortino),
            ("Calmar", "%.4f" % self.calmar),
            ("年化换手率（双边）", "%.4f" % self.turnover),
        ]
        if self.benchmark_name or not math.isnan(self.information_ratio):
            rows += [
                ("基准", self.benchmark_name or "（未命名）"),
                ("累计超额收益", "%.4f%%" % (self.excess_return * 100.0)),
                ("跟踪误差（年化）", "%.4f%%" % (self.tracking_error * 100.0)),
                ("Information Ratio（日频）", "%.4f" % self.information_ratio),
                ("Information Ratio（年化）", "%.4f" % self.information_ratio_annualized),
            ]
        width = max(len(k) for k, _ in rows)
        return "\n".join("%-*s : %s" % (width, k, v) for k, v in rows)


def evaluate(equity, benchmark_returns=None, trade_amounts=(),
             benchmark_name="", periods_per_year=TRADING_DAYS_PER_YEAR,
             risk_free=0.0):
    """一次算完第 6 层的全部指标。**这是给回测引擎调的主入口。**

        参数
        ----
        equity            : 净值曲线（期初 + 每日末权益），金额或归一化都行
        benchmark_returns : 基准日收益（沪深300 / 中证1000），长度必须与策略收益一致
        trade_amounts     : 每笔成交金额（买卖都传正数），用于换手率
        benchmark_name    : 基准名字，只用于报告显示
        periods_per_year  : 年化系数，默认 252
        risk_free         : **日频**无风险收益，默认 0（接口说明 口径）
        """
    rs = to_returns(equity)
    eq = _floats(equity, "净值曲线")

    report = PerformanceReport(
        n_periods=len(rs),
        periods_per_year=periods_per_year,
        risk_free=risk_free,
        equity_start=eq[0],
        equity_end=eq[-1],
        total_return=total_return(rs),
        annual_return=annual_return(rs, periods_per_year),
        annual_volatility=annual_volatility(rs, periods_per_year),
        max_drawdown=max_drawdown(eq),
        sharpe=sharpe_ratio(rs, risk_free, periods_per_year),
        sortino=sortino_ratio(rs, 0.0, periods_per_year),
        calmar=calmar_ratio(rs, periods_per_year),
        sharpe_ddof0=sharpe_ratio(rs, risk_free, periods_per_year, ddof=0),
    )
    report.sharpe_geometric = sharpe_from_annual(report.annual_return,
                                                 report.annual_volatility)

    tv = turnover(eq, trade_amounts, periods_per_year)
    report.turnover = tv.annualized
    report.turnover_total = tv.total

    if benchmark_returns is not None:
        rb = _floats(benchmark_returns, "基准日收益")
        report.benchmark_name = benchmark_name
        report.excess_return = report.total_return - total_return(rb)
        report.excess_return_annualized = (report.annual_return
                                           - annual_return(rb, periods_per_year))
        report.tracking_error = tracking_error(rs, rb, periods_per_year)
        report.information_ratio = information_ratio(rs, rb)
        report.information_ratio_annualized = (report.information_ratio
                                               * math.sqrt(periods_per_year))
    return report


# ══════════════════ 三方对拍 ══════════════════

@dataclass(frozen=True)
class CheckRow:
    """一行对拍结果。"""
    name: str
    self_value: float
    empyrical_value: float
    quantstats_value: float
    rel_diff: float          #: 三方两两之间的最大相对差
    note: str = ""


def _rel_diff(a, b):
    """相对差；两边都是 nan 记 0，一正一 nan 记 inf（不允许蒙混过关）。"""
    if a is None or b is None:
        return float("nan")
    a_nan, b_nan = math.isnan(a), math.isnan(b)
    if a_nan and b_nan:
        return 0.0
    if a_nan or b_nan:
        return float("inf")
    if a == b:
        return 0.0
    if math.isinf(a) or math.isinf(b):
        return float("inf")
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def cross_check(equity, benchmark_returns=None,
                periods_per_year=TRADING_DAYS_PER_YEAR, risk_free=0.0):
    """自算 vs empyrical vs quantstats 三方对拍。

    返回 list[CheckRow]。库缺失时直接报错（不假装算过）。
    注意 empyrical 没有 information_ratio，用它的别名 excess_sharpe 对拍。
    """
    try:
        import numpy as np
        import pandas as pd
        import empyrical
        import quantstats as qs
    except ImportError as exc:      # pragma: no cover - 环境缺库时如实报出来
        raise RuntimeError("三方对拍需要 numpy/pandas/empyrical/quantstats：%s" % exc)

    rs = to_returns(equity)
    idx = pd.date_range("2000-01-03", periods=len(rs), freq="B")
    s = pd.Series(rs, index=idx, dtype="float64")

    rows = []

    def add(name, mine, emp, qs_val, note=""):
        diffs = [d for d in (_rel_diff(mine, emp), _rel_diff(mine, qs_val),
                             _rel_diff(emp, qs_val)) if not math.isnan(d)]
        rows.append(CheckRow(name, mine, emp, qs_val,
                             max(diffs) if diffs else float("nan"), note))

    add("年化收益", annual_return(rs, periods_per_year),
        float(empyrical.annual_return(s, period="daily")),
        float(qs.stats.cagr(s, periods=periods_per_year)))
    add("年化波动率", annual_volatility(rs, periods_per_year),
        float(empyrical.annual_volatility(s, period="daily")),
        float(qs.stats.volatility(s, periods=periods_per_year)))
    add("Sharpe", sharpe_ratio(rs, risk_free, periods_per_year),
        float(empyrical.sharpe_ratio(s, risk_free=risk_free, period="daily")),
        float(qs.stats.sharpe(s, rf=0.0, periods=periods_per_year)),
        note="quantstats 的 rf 是年化口径，只有 rf=0 时两者才等价")
    add("Sortino", sortino_ratio(rs, 0.0, periods_per_year),
        float(empyrical.sortino_ratio(s, required_return=0, period="daily")),
        float(qs.stats.sortino(s, rf=0.0, periods=periods_per_year)))
    mdd = max_drawdown_from_returns(rs)
    add("Max Drawdown", mdd,
        -float(empyrical.max_drawdown(s)),
        -float(qs.stats.max_drawdown(s)),
        note="两个库返回负数，这里取相反数后比较")
    add("Calmar", calmar_ratio(rs, periods_per_year),
        float(empyrical.calmar_ratio(s, period="daily")),
        float(qs.stats.calmar(s, periods=periods_per_year)))

    if benchmark_returns is not None:
        rb = _floats(benchmark_returns, "基准日收益")
        sb = pd.Series(rb, index=idx, dtype="float64")
        add("Information Ratio（日频）", information_ratio(rs, rb),
            float(empyrical.excess_sharpe(s, sb)),
            float(qs.stats.information_ratio(s, sb)),
            note="empyrical 的 IR 叫 excess_sharpe，它没有 information_ratio")
    return rows


# ══════════════════ 持仓周期衰减曲线 ══════════════════

@dataclass(frozen=True)
class Trade:
    """一笔完整交易（买入 + 卖出）。字段与 接口说明 的交易口径一致。"""
    code: str
    buy_date: str
    buy_price: float
    sell_date: str
    sell_price: float

    @property
    def realized_return(self):
        """已实现收益 = 卖出价 / 买入价 - 1（价格本身应已含滑点）。"""
        return self.sell_price / self.buy_price - 1.0


@dataclass(frozen=True)
class RealizedStats:
    """已实现交易统计（只描述真发生过的交易，不掺推测值）。"""
    n: int
    mean_return: float
    win_rate: float
    mean_hold_days: float


@dataclass(frozen=True)
class DecayCurve:
    """持仓周期衰减曲线：T+1 / T+2 / T+3 / T+5 / T+10 的平均收益。"""
    horizons: tuple
    means: dict                  #: k -> 平均收益（没有样本的 k 不在字典里）
    counts: dict                 #: k -> 有效样本数
    n_entries: int               #: 一共喂进来多少个买入点
    realized: RealizedStats = None

    @property
    def best_horizon(self):
        """平均收益最高的持有期；没有任何样本时返回 None（不猜）。"""
        if not self.means:
            return None
        return max(self.means, key=lambda k: self.means[k])

    def to_dict(self):
        return {
            "n_entries": self.n_entries,
            "means": dict(self.means),
            "counts": dict(self.counts),
            "best_horizon": self.best_horizon,
        }

    def format(self):
        lines = ["持仓周期衰减曲线（样本 %d 笔）" % self.n_entries]
        for k in self.horizons:
            if k in self.means:
                lines.append("  T+%-2d 平均收益 %+8.4f%%   样本 %d 笔"
                             % (k, self.means[k] * 100.0, self.counts[k]))
            else:
                lines.append("  T+%-2d 无有效样本" % k)
        if self.realized is not None:
            r = self.realized
            lines.append("  已实现：%d 笔，平均收益 %+.4f%%，胜率 %.2f%%，平均持有 %.2f 个交易日"
                         % (r.n, r.mean_return * 100.0, r.win_rate * 100.0,
                            r.mean_hold_days))
        return "\n".join(lines)


def _coerce_trade(t):
    """把 Trade / dict / 5 元组 统一成 Trade。认不出来直接报错，不猜。"""
    if isinstance(t, Trade):
        return t
    if isinstance(t, Mapping):
        try:
            return Trade(str(t["code"]), str(t["buy_date"]), float(t["buy_price"]),
                         str(t["sell_date"]), float(t["sell_price"]))
        except KeyError as exc:
            raise ValueError("交易记录缺字段 %s（需要 code/buy_date/buy_price/"
                             "sell_date/sell_price）" % exc)
    if isinstance(t, (tuple, list)) and len(t) == 5:
        return Trade(str(t[0]), str(t[1]), float(t[2]), str(t[3]), float(t[4]))
    raise ValueError("认不出的交易记录：%r（支持 Trade / dict / 5 元组）" % (t,))


def _price_axis(bars):
    """把 {日期: 价格} 或 [(日期, 价格)] 统一成按日期升序的 [(日期, 价格)]。"""
    if isinstance(bars, Mapping):
        items = list(bars.items())
    else:
        items = [(d, p) for d, p in bars]
    items.sort(key=lambda kv: kv[0])
    return [(d, float(p)) for d, p in items]


def decay_curve(prices, entries, horizons=DEFAULT_HORIZONS):
    """按价格路径算 T+k 平均收益。

        参数
        ----
        prices  : {code: {日期: 价格}} 或 {code: [(日期, 价格), ...]}
                  日期用能排序、能比较的类型（推荐 "YYYY-MM-DD" 字符串）。
                  **必须传该票完整的日线序列**（有缺口的序列会把 T+k 算错）。
        entries : [(code, buy_date, buy_price), ...]
        horizons: 观察点，默认 (1, 2, 3, 5, 10)

        🔴 买入日当天不算（接口说明 记过 hold_days 买入日不计数导致"3 日到期永不触发"
        的真 bug）：T+k = 买入日在价格序列里的下标 + k，
        所以「持有 1 天」= 买入日之后第 1 个交易日卖出。

        ⚠️ 停牌口径（接口说明 #7「停牌 = 冻结，等复牌后继续」）：持有天数按**该票自己
        的行情序列**往后数，停牌日没有行情就不占格。所以传进来的 prices 应该是
        该票"有行情的交易日"序列。

        缺失处理：买入日不在价格序列里 -> 直接 ValueError（那说明入场点就是错的）；
        买入日之后不足 k 个交易日 -> 这笔在该 k 上不计入（样本还没到期，不是 0）。
        """
    hl = tuple(int(k) for k in horizons)
    if not hl or any(k <= 0 for k in hl):
        raise ValueError("观察点必须都是正整数，收到 %r" % (horizons,))

    axis_cache = {}
    means = {k: 0.0 for k in hl}
    counts = {k: 0 for k in hl}
    n_entries = 0

    for entry in entries:
        if isinstance(entry, Trade):
            code, buy_date, buy_price = entry.code, entry.buy_date, entry.buy_price
        else:
            code, buy_date, buy_price = entry[0], entry[1], float(entry[2])
        code = str(code)
        buy_price = float(buy_price)
        if buy_price <= 0.0:
            raise ValueError("%s 在 %s 的买入价必须为正数，收到 %r"
                             % (code, buy_date, buy_price))
        if code not in prices:
            raise ValueError("价格表里没有 %s（entries 里有这笔，prices 必须补齐）" % code)
        axis = axis_cache.get(code)
        if axis is None:
            axis = _price_axis(prices[code])
            axis_cache[code] = axis
        index = {d: i for i, (d, _) in enumerate(axis)}
        if buy_date not in index:
            raise ValueError("%s 的买入日 %s 不在价格序列里（价格区间 %s ~ %s）"
                             % (code, buy_date, axis[0][0], axis[-1][0]))
        i = index[buy_date]
        n_entries += 1
        for k in hl:
            if i + k >= len(axis):
                continue                    # 样本还没到期，不计入（也不记 0）
            r = axis[i + k][1] / buy_price - 1.0
            means[k] += r
            counts[k] += 1

    out_means = {k: means[k] / counts[k] for k in hl if counts[k] > 0}
    out_counts = {k: counts[k] for k in hl if counts[k] > 0}
    return DecayCurve(horizons=hl, means=out_means, counts=out_counts,
                      n_entries=n_entries)


def from_trades(trades, prices, horizons=DEFAULT_HORIZONS):
    """便捷入口：吃一批交易记录 -> 持仓周期衰减曲线 + 已实现统计。

    参数
    ----
    trades : 一批 Trade / dict / 5 元组，字段 = code, buy_date, buy_price,
             sell_date, sell_price
    prices : 价格表，同 decay_curve（**必须有**，理由见下）
    horizons : 观察点，默认 (1, 2, 3, 5, 10)

    🔴 为什么必须传 prices：只知道买入价和卖出价，是**算不出** T+1..T+10 的。
    拿"实际卖出价"冒充"买入后第 k 天价格"就是伪造数据（数据铁律），
    所以这里宁可多要一个参数、也不给一个看起来能跑但口径错的默认值。

    已实现统计里的持有天数同样**不含买入日**（A股 T+1，买入日当天不能卖）：
        持有交易日数 = 卖出日在价格序列里的下标 - 买入日的下标
    """
    tl = [_coerce_trade(t) for t in trades]
    curve = decay_curve(prices, tl, horizons)

    # ── 已实现统计（真发生过的交易）──
    rets = []
    holds = []
    for t in tl:
        if t.buy_price <= 0.0:
            raise ValueError("%s 在 %s 的买入价必须为正数" % (t.code, t.buy_date))
        axis = _price_axis(prices[t.code])
        index = {d: i for i, (d, _) in enumerate(axis)}
        if t.buy_date not in index or t.sell_date not in index:
            raise ValueError("%s 的买入/卖出日不在价格序列里：%s / %s"
                             % (t.code, t.buy_date, t.sell_date))
        holds.append(index[t.sell_date] - index[t.buy_date])
        rets.append(t.realized_return)

    realized = RealizedStats(
        n=len(tl),
        mean_return=sum(rets) / len(rets) if rets else float("nan"),
        win_rate=(sum(1 for r in rets if r > 0) / len(rets)) if rets else float("nan"),
        mean_hold_days=sum(holds) / len(holds) if holds else float("nan"),
    )
    return DecayCurve(horizons=curve.horizons, means=curve.means,
                      counts=curve.counts, n_entries=curve.n_entries,
                      realized=realized)


# ══════════════════ backtrader 适配器（第 6 层的挂载点） ══════════════════
from . import compat as _compat  # noqa: E402  必须早于 backtrader
import backtrader as bt  # noqa: E402


# 框架与解释器会按名称隐式调用这些入口，不能只按直接调用次数判断是否删除。
FRAMEWORK_CALLBACKS = ("start", "next", "stop", "get_analysis")


class PerformanceAnalyzer(bt.Analyzer):
    """将绩效计算接入 Backtrader Analyzer。

        用法：
            cerebro.addanalyzer(PerformanceAnalyzer, _name="perf")
            strat = cerebro.run()[0]
            report = strat.analyzers.perf.report       # PerformanceReport
            print(report.format())

        每根 bar 收盘记一次 broker.getvalue()，stop() 时交给 evaluate()。
        净值点数 = bar 数（第一根 bar 记的是期初权益），所以收益期数 = bar 数 - 1。
        """

    params = (
        ("periods_per_year", TRADING_DAYS_PER_YEAR),
        ("risk_free", 0.0),                 #: 日频
        ("benchmark_returns", None),
        ("benchmark_name", ""),
        ("trade_amounts", ()),
    )

    def start(self):
        self._equity = []
        self.report = None

    def next(self):
        self._equity.append(float(self.strategy.broker.getvalue()))

    def stop(self):
        if len(self._equity) < 2:
            # 一根 bar 都没有（或只有一根）-> 算不出收益，如实留空
            self.report = None
            return
        self.report = evaluate(
            self._equity,
            benchmark_returns=self.p.benchmark_returns,
            trade_amounts=self.p.trade_amounts,
            benchmark_name=self.p.benchmark_name,
            periods_per_year=self.p.periods_per_year,
            risk_free=self.p.risk_free,
        )

    def get_analysis(self):
        """backtrader 要求返回 dict。"""
        if self.report is None:
            return {}
        return self.report.to_dict()
