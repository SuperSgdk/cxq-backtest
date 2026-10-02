# 手算测试样例

本目录 CSV 完全由人工构造。TEST_A、TEST_B 是虚构测试标识，日期只用来演示交易日顺序。这些数值不是购买数据、真实行情或策略历史业绩。

输入给出：1 月 2 日收盘决定持有 TEST_A；1 月 4 日收盘决定切换到 TEST_B；1 月 8 日收盘决定退出。

运行（在项目根目录）：

```sh
python -m cxq_bt --data examples/market_handcalc.csv --signals examples/signals_handcalc.csv --initial-cash 100000 --output outputs/handcalc.json
```

手算检查：第一次成交在 1 月 3 日，第二次换仓在 1 月 5 日，最后退出在 1 月 9 日，不能提前到决定当天成交。另见 tests/test_bt_framework.py，分别检验费用、资产守恒、未知数据拦截和受限持仓估值。

用自己的策略时，以自己有权使用的真实行情替换 CSV；回测日期、行情字段与目标清单必须对应。每个信号日期是一份完整目标组合，该日未列出的旧持仓目标为零。没有信号的日期保持原持仓。
