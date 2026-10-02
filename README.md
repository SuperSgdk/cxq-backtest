# CXQ Backtest：A 股日级回测框架

把自己的行情和策略放进来，按明确的成交时间、交易限制和费用规则，输出净值、回撤与交易记录。公开版只包含框架、接口和人工手算样例，不附带采购行情、原研究报告或私有评分策略。

框架基于 [Backtrader](https://www.backtrader.com/) 扩展。费用、整手交易、T+1、涨跌停、停牌和结果记录，是本项目重点验证的部分。它帮助检查策略在这些约束下会发生什么；样例测试通过不代表真实策略盈利。

## 运行方式

Python 要求 3.11 以上。本机使用 Python 3.13 核验；仓库配置了 Windows / Linux、Python 3.11 / 3.13 的 CI，运行结果见 GitHub Actions。建议在独立环境中安装：

```bash
git clone https://github.com/SuperSgdk/cxq-backtest.git
cd cxq-backtest
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui,validation]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install -e ".[ui,validation]"
```

只使用命令行或 Python 接口时，安装框架本体 `-e .` 即可。`ui` 提供本地可视化页面；`validation` 提供用于指标交叉核对的可选库。依赖版本范围见 `pyproject.toml`，本次核验的固定版本见 `requirements.txt`。Windows 直接调用虚拟环境的 Python，不需要改 PowerShell 执行策略；网络安装失败时可使用上面的清华镜像。

先用仓库里的人工构造场景跑通：

```powershell
.\.venv\Scripts\python.exe -m cxq_bt --data examples/market_handcalc.csv --signals examples/signals_handcalc.csv --initial-cash 1000000 --output outputs/handcalc.json
```

后续命令以 Windows 为例；Linux / macOS 激活环境后，将 `.\.venv\Scripts\python.exe` 换成 `python`。

**这些 CSV 是测试用的人工构造数据，不是真实历史行情。** 它们只用于演示输入格式和检查费用、交易限制，输出的收益不能当成历史业绩。

启动界面：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Windows 也可以双击 `启动面板.cmd`。页面仅监听本机，关闭 Streamlit 使用统计遥测。

使用 Docker 时，容器内部监听 `0.0.0.0`；主机端只向本机映射，例如 `docker run --rm -p 127.0.0.1:8501:8501 cxq-backtest`。Docker 构建方式为 `docker build -t cxq-backtest .`。

界面当前接受 CSV 文件。Python 接口支持 CSV / Parquet 路径或 DataFrame，CLI 支持 CSV / Parquet 文件。在浏览器里载入你有使用权限的行情，选择目标权重或均线策略，设置初始资金和费用，然后查看净值、回撤、交易与订单记录。上传文件由本机进程处理；发布部署到服务器时，应另外配置访问控制和数据保存规则。

面板可调整初始资金、最大持仓数、ST 排除、硬止损与全部费用。佣金、印花税、过户费显示为百分比，例如 `0.05` 表示 `0.05%`；最低佣金以元输入；滑点以 bps 输入，`1 bps = 0.01%`。硬止损输入正的跌幅，例如 `9%`，框架按 `-0.09` 判断。修改设置后重新运行，结果会保留本次参数快照。

也可以在命令行使用均线策略：

```powershell
.\.venv\Scripts\python.exe -m cxq_bt --data examples/market_handcalc.csv --strategy sma --initial-cash 1000000 --output outputs/sma.json
```

### 换成自己的数据

行情支持 CSV / Parquet，字段为：

```text
date,code,open,high,low,close,volume,up_limit,down_limit,is_st,is_suspended
```

目标权重文件支持 CSV / Parquet，字段为：

```text
date,code,weight
```

先阅读 [数据字段与时间约定](docs/data_contract.md)。缺字段、无效价格或重复日期代码组合会报错，框架不会用补零或猜测值把文件凑齐。行情与涨跌停价必须使用同一价格口径；v0.1 不处理分红、送转、拆股等公司行为。

## 一次回测怎样进行

1. 策略在 T 日收盘后读取截至 T 日的行情，给出目标权重。
2. 框架在下一个行情交易日的开盘尝试调仓，先检查 T+1、整手、停牌、ST 与涨跌停限制。
3. 根据设置计入滑点和费用，记录成交以及未成交的原因；受限持仓继续占用资产。
4. 按收盘价计算账户权益，输出绩效指标和逐笔记录。

没有某日信号，表示保持持仓；某日给出完整非空目标清单，未列入的持仓会尝试退出；显式 `0` 权重也表示尝试退出。被交易限制拦住的卖单不会变成可使用的现金。

每次目标计划只在下一交易日尝试一次。被拒绝的订单不会自动顺延，策略需要再次给出计划才能重新尝试；未成交持仓保留，已配置的风控规则另行处理。

公开入口承诺的是**日级收盘决策、下一交易日开盘成交模型**。日线无法证明真实挂单排队、盘口深度或一字板打开时能否成交。核心模块中保留的其他成交模型是额外实验代码，应单独验证后使用。

开盘成交判断不读取当日尚未完成的成交量或日内最高、最低价；这些完整日级数据只能在收盘后交给策略。模型没有按成交量限制订单规模，也不保证开盘价对应足够的真实流动性。

## Python 接口

```python
from cxq_bt.framework import (
    BacktestConfig,
    TargetWeightStrategy,
    load_market_data,
    load_target_weights,
    run_backtest,
)
from cxq_bt.costs import AShareCosts

data = load_market_data("examples/market_handcalc.csv")
signals = load_target_weights("examples/signals_handcalc.csv")
config = BacktestConfig(
    initial_cash=1_000_000,
    costs=AShareCosts(
        commission_rate=0.0005,
        min_commission=5.0,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    ),
)
result = run_backtest(data, TargetWeightStrategy(signals), config)
print(result.summary)
```

初始资金可以调整，必须大于 0。费率是模型参数，不是对所有账户和所有历史时期都适用的断言。命令行的费用参数包括 `--commission-rate`、`--minimum-commission`、`--stamp-duty-rate`、`--transfer-rate` 和 `--slippage`；完整选项见 `python -m cxq_bt --help`。

自定义策略只需实现 `on_close(context)`，返回权重映射或 `None`。详见 [策略接口](docs/strategy_api.md)。

## 结果与核验

`BacktestResult` 提供 `summary`、`equity`、`trades`、`orders` 和 `settings`，可以通过 `to_dict()` 导出。结果记录输入指纹、策略和模型设置，便于判断两次回测是否真的使用了相同输入。常用指标包括期末权益、总收益、最大回撤、年化收益、Sharpe、费用与成交笔数；不同收益口径和窗口的结果不能直接混用。

```powershell
.\.venv\Scripts\python.exe run_tests.py
.\.venv\Scripts\python.exe scripts/audit_public.py
```

单元测试重点使用可手算的人工场景，不依赖采购行情。可选交叉核对库缺失时，相关测试会跳过；需要完整核验时安装 `validation`。公开检查脚本只审查 Git 已跟踪的发布文件，报告不允许的文件、个人路径和疑似凭据位置，不打印疑似密钥值。

CSV 发布白名单只有 `examples/market_handcalc.csv` 与 `examples/signals_handcalc.csv`；新增样例须人工检查后显式扩展。用户行情放在已忽略的 `data/`，运行输出放在 `outputs/`。

## 项目结构

```text
cxq_bt/     回测、费用、规则、数据与绩效模块
tests/      人工场景与接口测试
examples/   明确标注的人工手算 CSV
docs/       架构、行情规范与策略接口
scripts/    发布文件检查
app.py      Streamlit 本地界面
```

更多设计见 [框架架构](docs/architecture.md)。界面是输入和结果的展示层，成交与费用规则由同一 Python 回测入口处理。

本项目采用 **GPL-3.0-or-later**，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。Backtrader 的作者和许可保持归原项目；本仓库的框架扩展由 SuperSgdk 维护。
