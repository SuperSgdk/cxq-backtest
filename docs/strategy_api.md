# 策略接口

策略只回答：在本日收盘后，下一次开盘想把资产分配到哪些股票，各占多少。框架负责成交限制、费用和账户记账。

## 输入与输出

实现 `on_close(context)`，返回 `{股票代码: 目标权重}` 或 `None`。

`context.history(code)` 只能提供截至当前决策日的数据。它不会把之后的行情交给策略。一个策略也不应在其他地方读入未来文件，再把未来结果写进清单。

```python
class MyStrategy:
    def on_close(self, context):
        history = context.history("600000")
        if len(history) < 20:
            return None
        closes = history["close"]
        if closes.iloc[-1] > closes.tail(20).mean():
            return {"600000": 0.5}
        return {"600000": 0.0}
```

这里的代码只是接口示意，不是对该股票的投资建议，也没有提供策略盈利证据。返回权重后，框架在下一交易日开盘尝试执行；执行不了时记录原因。

### 清单的含义

- `None`：该日不主动调仓。
- Python 回调返回 `{}`：目标空仓，尝试退出全部已有持仓。
- 完整非空映射：按本次目标清单调仓，未列入的旧持仓尝试退出。
- 显式 `0` 权重：尝试退出该股票。

不要把缺失某日清单理解成清仓。CSV 没有“空映射”这一行的表达方式，需要清仓时明确列出已有股票的 0 权重。

每次目标计划只在下一交易日尝试一次。拒单后策略须再次给出计划才能重试，不能假定订单自动顺延；风控配置可另行产生退出决定。

目标权重不是每日推荐分数。框架不会把私有评分、排名或模型输出自动转换为实盘下单权限。

## 从命令行加载自定义策略

仓库的 `examples/my_strategy.py` 用人工样例中的 `TEST_A` 示范两日均线条件，提供 `make_strategy()` 工厂函数：

```powershell
.\.venv\Scripts\python.exe -m cxq_bt --data examples/market_handcalc.csv --custom-strategy examples.my_strategy:make_strategy --output outputs/custom.json
```

Linux / macOS 激活环境后，将虚拟环境 Python 的 Windows 路径换成 `python`。`module:factory` 表示导入本机 Python 模块并调用工厂函数；只加载自己信任的策略源码。该示例只验证接口行为，不是历史业绩。

策略可以实现 `parameters()` 返回参数字典，用于结果设置记录；没有该方法时记录空字典 `{}`。参数字典不能代替外部策略的源码版本，研究者应自行保存源码版本或提交编号。

## 内置策略

`TargetWeightStrategy(signals)` 读取日期到目标权重的映射，便于使用自己的外部研究输出。信号日期表示收盘后的决策日，而不是成交日。

`MovingAverageStrategy(short_window, long_window)` 用历史收盘价计算短、长均线。历史不足时不能用未来数据补齐窗口。均线示例用于演示如何通过 `context.history()` 写策略，不代表一个经过真实市场验证的策略。

```python
from cxq_bt.framework import (
    BacktestConfig,
    MovingAverageStrategy,
    load_market_data,
    run_backtest,
)

data = load_market_data("examples/market_handcalc.csv")
strategy = MovingAverageStrategy(short_window=2, long_window=3)
result = run_backtest(data, strategy, BacktestConfig(initial_cash=1_000_000))
print(result.summary)
```

## 回测设置

`BacktestConfig` 提供初始资金、费用、每手数量、持仓上限、ST 排除和硬止损等参数。策略产生的是目标，执行层仍需检查参数与交易限制。

例如，目标权重对应的金额不足一手时，可能没有买入；持仓停牌或开盘处于不可卖出的跌停状态时，即使目标是 0 也不能假定已经卖出。

## 策略核验

建议先构造可手算的场景，检查明确的预期：某日决策是否只在下一日开盘执行，卖出受限时是否继续持有，费用是否正确，缺信号是否保留持仓。

真实研究应另行检查数据可用日期、样本外窗口和成交假设。仅保证 `context.history()` 不含未来行，不足以证明整个外部研究流程没有未来信息。
