"""本地命令行：加载自有行情与自定义策略，保存含参数的回测结果。"""

import argparse
import importlib
import json
from pathlib import Path

from .costs import AShareCosts
from .framework import (BacktestConfig, MovingAverageStrategy, TargetWeightStrategy,
                        load_target_weights, run_backtest)


def main(argv=None):
    parser = argparse.ArgumentParser(description="日级 A 股回测：策略在收盘给计划，下一交易日开盘模拟执行")
    parser.add_argument("--data", required=True, help="自有日行情 CSV / Parquet 路径")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--signals", help="完整目标组合 CSV / Parquet：date,code,weight")
    group.add_argument("--strategy", choices=["sma"], help="均线策略接口样例")
    group.add_argument("--custom-strategy", help="本地 Python 模块:工厂，返回实现 on_close 的策略")
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--long-window", type=int, default=20)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--commission-rate", type=float, default=0.0005, help="小数费率，0.0005=0.05%%")
    parser.add_argument("--minimum-commission", type=float, default=5.0, help="每笔最低佣金，元")
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005, help="卖出印花税，小数费率")
    parser.add_argument("--transfer-rate", type=float, default=0.00001, help="双边过户费，小数费率")
    parser.add_argument("--slippage", type=float, default=0.0005, help="单边滑点，小数比例")
    parser.add_argument("--lot", type=int, default=100)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--allow-st", action="store_true", help="关闭禁止买入 ST 的可选政策")
    parser.add_argument("--hard-stop", type=float, default=None, help="可选开盘止损比例，例如 -0.05；默认关闭")
    parser.add_argument("--output", default="outputs/result.json")
    args = parser.parse_args(argv)
    try:
        if args.signals:
            strategy = TargetWeightStrategy(load_target_weights(args.signals))
        elif args.custom_strategy:
            name, sep, factory = args.custom_strategy.partition(":")
            if not sep:
                raise ValueError("自定义策略格式为 module:factory")
            strategy = getattr(importlib.import_module(name), factory)()
        else:
            strategy = MovingAverageStrategy(args.short_window, args.long_window)
        costs = AShareCosts(args.commission_rate, args.minimum_commission,
                            args.stamp_duty_rate, args.transfer_rate, args.slippage)
        config = BacktestConfig(args.initial_cash, costs, args.lot, args.max_positions,
                                not args.allow_st, args.hard_stop)
        result = run_backtest(args.data, strategy, config)
        example = Path(__file__).resolve().parent.parent / "examples" / "market_handcalc.csv"
        result.settings["data_kind"] = ("hand_calculated_test_fixture" if Path(args.data).resolve() == example
                                         else "user_supplied_data")
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, TypeError, OSError, ImportError, AttributeError) as error:
        parser.exit(2, "回测未完成：" + str(error) + "\n")
    label = "手算测试样例，非真实历史业绩" if result.settings["data_kind"] == "hand_calculated_test_fixture" else "使用者自有数据"
    print(label)
    print("起始资金 %.2f；期末资产 %.2f；成交 %d 笔；费用 %.4f" %
          (config.initial_cash, result.summary["final_equity"], result.summary["trade_count"], result.summary["total_fees"]))
    print("参数与数据指纹已记录：" + str(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
