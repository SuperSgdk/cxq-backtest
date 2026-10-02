# -*- coding: utf-8 -*-
"""本地回测面板；只在用户点击运行后计算，结果保留当时的完整参数。"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any


def build_config(
    *,
    initial_cash: float,
    commission_percent: float,
    min_commission: float,
    stamp_duty_percent: float,
    transfer_fee_percent: float,
    slippage_bps: float,
    max_positions: int,
    exclude_st: bool,
    hard_stop_percent: float | None,
):
    """将面板里的百分比、基点转换成引擎费率；止损开关关闭时传 None。"""
    from cxq_bt.costs import AShareCosts
    from cxq_bt.framework import BacktestConfig

    return BacktestConfig(
        initial_cash=float(initial_cash),
        costs=AShareCosts(
            commission_rate=float(commission_percent) / 100.0,
            min_commission=float(min_commission),
            stamp_duty_rate=float(stamp_duty_percent) / 100.0,
            transfer_fee_rate=float(transfer_fee_percent) / 100.0,
            slippage_rate=float(slippage_bps) / 10_000.0,
        ),
        lot=100,
        max_positions=int(max_positions),
        exclude_st=bool(exclude_st),
        hard_stop_pct=(
            None if hard_stop_percent is None else -float(hard_stop_percent) / 100.0
        ),
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _snapshot(settings: dict[str, Any]) -> str:
    return _sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _read_csv(content: bytes):
    import pandas as pd

    # 股票代码按文字读入，避免 000001 变成 1。缺失字段交给引擎严格检查。
    frame = pd.read_csv(io.BytesIO(content), dtype={"date": "string", "code": "string"})
    if "date" not in frame.columns:
        raise ValueError("CSV 缺少 date 列，无法确定回测日期范围。")
    if frame.empty:
        raise ValueError("CSV 没有数据行。")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("CSV 中有空日期，请补齐 date 列。")
    frame["date"] = dates.dt.strftime("%Y-%m-%d")
    return frame


def _metric(value: Any, *, percent: bool = False, money: bool = False) -> str:
    import math

    if value is None:
        return "不可计算"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "不可计算"
    if percent:
        return f"{number:.2%}"
    if money:
        return f"¥{number:,.2f}"
    return f"{number:,.2f}"


def _show_result(st, result: dict[str, Any]) -> None:
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    presentation = result.get("presentation", {})
    is_handcalc = presentation.get("source_kind") == "handcalc"
    st.subheader("本次结果")
    if is_handcalc:
        st.warning("手算测试样例（非真实历史业绩）。这些数字只用于检查成交、费用和界面。")
    else:
        st.caption("以下结果来自本地上传文件；数据授权、复权口径及覆盖完整性需要由数据使用者核对。")
    summary = result["summary"]
    columns = st.columns(3)
    columns[0].metric("总收益", _metric(summary.get("total_return"), percent=True))
    columns[1].metric("最大回撤", _metric(summary.get("max_drawdown"), percent=True))
    columns[2].metric("期末权益", _metric(summary.get("final_equity"), money=True))
    columns = st.columns(3)
    trade_count = summary.get("trade_count")
    columns[0].metric("成交次数", "不可计算" if trade_count is None else f"{trade_count:,}")
    columns[1].metric("手续费合计", _metric(summary.get("total_fees"), money=True))
    columns[2].metric("回测区间", f"{presentation.get('start_date', '?')} → {presentation.get('end_date', '?')}")

    equity = pd.DataFrame(result["equity"])
    if not equity.empty:
        initial_cash = float(presentation["controls"]["initial_cash"])
        values = equity["equity"].astype(float)
        running_peak = values.cummax().clip(lower=initial_cash)
        drawdown = values / running_peak - 1.0
        chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                              subplot_titles=("净值（期初资金 = 1）", "回撤"))
        chart.add_trace(go.Scatter(x=equity["date"], y=values / initial_cash,
                                  name="净值", line={"color": "#13837f", "width": 2}), row=1, col=1)
        chart.add_trace(go.Scatter(x=equity["date"], y=drawdown, name="回撤", fill="tozeroy",
                                  line={"color": "#c35151", "width": 1.5}), row=2, col=1)
        chart.update_yaxes(tickformat=".1%", row=2, col=1)
        chart.update_xaxes(type="date")
        chart.update_layout(height=510, template="plotly_white", showlegend=False,
                            margin={"l": 20, "r": 20, "t": 55, "b": 25})
        st.plotly_chart(chart, width="stretch")

    trades_tab, orders_tab, equity_tab = st.tabs(["成交记录", "订单与拒单", "每日权益"])
    with trades_tab:
        if result["trades"]:
            st.dataframe(pd.DataFrame(result["trades"]), width="stretch", hide_index=True)
        else:
            st.info("本次没有成交。可在订单表中查看未成交的原因。")
    with orders_tab:
        if result["orders"]:
            st.dataframe(pd.DataFrame(result["orders"]), width="stretch", hide_index=True)
            st.caption("status 和 reason 保留引擎原始值，方便定位拒单及市场限制。")
        else:
            st.info("策略在所选区间内没有产生订单。")
    with equity_tab:
        st.dataframe(equity, width="stretch", hide_index=True)

    with st.expander("本次实际参数、数据指纹及全部指标", expanded=False):
        st.json({"settings": result["settings"], "presentation": presentation,
                 "summary": summary})
        st.caption("参数与文件指纹跟随结果保存。不可计算的指标保留空值，不用 0 代替。")
    exported = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    st.download_button("导出完整结果 JSON", data=exported.encode("utf-8"),
                       file_name="backtest_result.json", mime="application/json")


def main() -> None:
    import pandas as pd
    import streamlit as st

    from cxq_bt.framework import (
        MovingAverageStrategy,
        TargetWeightStrategy,
        load_market_data,
        load_target_weights,
        run_backtest,
    )

    st.set_page_config(page_title="A 股回测框架", page_icon="📈", layout="wide")
    st.title("A 股回测框架")
    st.caption("在本机读取数据，明确交易规则，再检查每笔成交与费用。设置变化后需要重新点击运行。")
    root = Path(__file__).resolve().parent

    with st.sidebar:
        st.header("回测设置")
        initial_cash = st.number_input("初始资金（元）", min_value=100.0, value=1_000_000.0,
                                       step=10_000.0, format="%.2f")
        max_positions = st.number_input("最多持仓数量", min_value=1, max_value=1000, value=5, step=1)
        st.subheader("费用与滑点")
        commission = st.number_input("双边佣金（%）", min_value=0.0, max_value=10.0,
                                    value=0.05, step=0.01, format="%.4f")
        minimum = st.number_input("最低佣金（元）", min_value=0.0, value=5.0, step=1.0)
        stamp = st.number_input("卖出印花税（%）", min_value=0.0, max_value=10.0,
                               value=0.05, step=0.01, format="%.4f")
        transfer = st.number_input("双边过户费（%）", min_value=0.0, max_value=10.0,
                                  value=0.001, step=0.001, format="%.4f")
        slippage = st.number_input("单边滑点（bps）", min_value=0.0, max_value=1000.0,
                                  value=5.0, step=1.0)
        st.caption("1 bps = 0.01%。买入成交价上浮，卖出成交价下浮。")
        st.subheader("可选策略规则")
        exclude_st = st.checkbox("不新买入 ST 股票", value=True)
        stop_enabled = st.checkbox("启用硬止损", value=False)
        hard_stop = st.number_input("止损幅度（%）", min_value=0.1, max_value=99.0,
                                   value=10.0, step=1.0, disabled=not stop_enabled)
        st.caption("这两项是可调策略政策。T+1、100 股整手、停牌和涨跌停交易限制由引擎执行。")

    controls = {
        "initial_cash": initial_cash, "commission_percent": commission,
        "min_commission": minimum, "stamp_duty_percent": stamp,
        "transfer_fee_percent": transfer, "slippage_bps": slippage,
        "max_positions": max_positions, "exclude_st": exclude_st,
        "hard_stop_percent": hard_stop if stop_enabled else None,
    }
    source = st.radio("数据来源", ["手算测试样例（非真实历史业绩）", "本地行情 CSV"], horizontal=True)
    is_handcalc = source.startswith("手算")
    strategy_name = st.selectbox("策略", ["目标仓位 CSV（接入你的策略）", "均线示例策略"])
    market_bytes = None
    signals_bytes = None
    if is_handcalc:
        st.warning("当前是手算测试样例（非真实历史业绩）。不代表策略收益或历史市场表现。")
        try:
            market_bytes = (root / "examples" / "market_handcalc.csv").read_bytes()
            if strategy_name.startswith("目标"):
                signals_bytes = (root / "examples" / "signals_handcalc.csv").read_bytes()
        except OSError:
            st.error("手算样例文件缺失或无法读取，请检查是否已下载完整项目。")
        st.caption("切换到「本地行情 CSV」后可使用你有权使用的数据；运行结果不会上传到外部服务。")
    else:
        uploaded = st.file_uploader("上传本地行情 CSV", type=["csv"], key="market_csv")
        st.caption("必需列：date, code, open, high, low, close, volume, up_limit, down_limit, is_st, is_suspended。")
        if uploaded is not None:
            market_bytes = uploaded.getvalue()
        if strategy_name.startswith("目标"):
            signals_upload = st.file_uploader("上传策略目标仓位 CSV", type=["csv"], key="signals_csv")
            st.caption("必需列：date, code, weight。weight 是 0 到 1 的资金权重；当日信号用于下一交易日开盘。")
            if signals_upload is not None:
                signals_bytes = signals_upload.getvalue()

    if strategy_name.startswith("目标"):
        st.caption("每个信号日期表示完整目标组合，未列出的已有持仓目标为 0；没有该日信号时保持持仓。"
                   "清仓可给该日的股票列出 weight=0。")

    short_window = None
    long_window = None
    if strategy_name.startswith("均线"):
        st.info("均线只是接口示例。你可以由自己的策略生成目标仓位 CSV，或通过 Python 接口接入自定义策略。")
        st.caption("均线从所选开始日期重新累计；窗口未满时不发买入信号。")
        short_col, long_col = st.columns(2)
        short_window = short_col.number_input("短均线天数", min_value=1, max_value=500, value=2, step=1)
        long_window = long_col.number_input("长均线天数", min_value=2, max_value=1000, value=3, step=1)

    start_date = None
    end_date = None
    market_frame = None
    input_error = None
    if market_bytes is not None:
        try:
            market_frame = _read_csv(market_bytes)
            actual_dates = pd.to_datetime(market_frame["date"]).dt.date
            first_date, last_date = actual_dates.min(), actual_dates.max()
            left, right = st.columns(2)
            date_key = _sha256(market_bytes)[:12]
            start_date = left.date_input("开始日期", value=first_date, min_value=first_date,
                                         max_value=last_date, key=f"start_{date_key}")
            end_date = right.date_input("结束日期", value=last_date, min_value=first_date,
                                        max_value=last_date, key=f"end_{date_key}")
            if start_date > end_date:
                raise ValueError("开始日期不能晚于结束日期。")
            mask = (actual_dates >= start_date) & (actual_dates <= end_date)
            market_frame = market_frame.loc[mask].copy()
            if market_frame.empty:
                raise ValueError("所选区间没有行情，请调整日期。")
            st.caption(f"所选区间：{market_frame['date'].nunique()} 个数据日期，"
                       f"{market_frame['code'].nunique() if 'code' in market_frame else '?'} 只股票，"
                       f"{len(market_frame):,} 行行情。")
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
            input_error = str(exc)
            st.error(f"行情文件读取失败：{input_error}")

    request_settings = {
        "controls": controls, "source_kind": "handcalc" if is_handcalc else "local_csv",
        "strategy": strategy_name, "short_window": short_window, "long_window": long_window,
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None,
        "market_file_sha256": _sha256(market_bytes) if market_bytes is not None else None,
        "signals_file_sha256": _sha256(signals_bytes) if signals_bytes is not None else None,
    }
    request_id = _snapshot(request_settings)
    ready = market_frame is not None and input_error is None
    ready = ready and (signals_bytes is not None or strategy_name.startswith("均线"))
    if not ready and input_error is None:
        st.info("请提供行情和策略文件，随后点击运行。")

    if st.button("运行回测", type="primary", disabled=not ready):
        try:
            config = build_config(**controls)
            market_data = load_market_data(market_frame)
            if strategy_name.startswith("目标"):
                signals_frame = _read_csv(signals_bytes)
                signals_frame = signals_frame.loc[
                    (signals_frame["date"] >= str(start_date))
                    & (signals_frame["date"] <= str(end_date))
                ].copy()
                strategy = TargetWeightStrategy(load_target_weights(signals_frame))
            else:
                if short_window >= long_window:
                    raise ValueError("短均线天数必须小于长均线天数。")
                strategy = MovingAverageStrategy(int(short_window), int(long_window))
            with st.spinner("正在计算成交、费用和每日权益……"):
                calculated = run_backtest(market_data, strategy, config).to_dict()
            calculated["presentation"] = request_settings
            # 先验证导出是严格 JSON，避免页面成功后才发现不能保存结果。
            json.dumps(calculated, ensure_ascii=False, allow_nan=False)
            st.session_state["last_result"] = calculated
            st.session_state["last_request_id"] = request_id
            st.success("回测完成。结果和导出文件均保存了本次参数。")
        except (ValueError, TypeError, KeyError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"回测未完成：{exc}")

    if "last_result" in st.session_state:
        if st.session_state.get("last_request_id") != request_id:
            st.warning("输入文件、日期、策略或参数已改变。下面仍是上次结果，请重新点击「运行回测」更新。")
        _show_result(st, st.session_state["last_result"])


if __name__ == "__main__":
    main()
