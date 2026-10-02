"""自定义策略接口示意：使用人工 TEST_A 样例，不代表真实历史业绩。"""


class ExampleStrategy:
    def __init__(self, lookback=2, target_weight=0.5):
        self.lookback = lookback
        self.target_weight = target_weight

    def on_close(self, context):
        history = context.history("TEST_A")
        if len(history) < self.lookback:
            return None
        closes = history["close"]
        active = closes.iloc[-1] > closes.tail(self.lookback).mean()
        return {"TEST_A": self.target_weight if active else 0.0}

    def parameters(self):
        return {"lookback": self.lookback, "target_weight": self.target_weight}


def make_strategy():
    """命令行 module:factory 入口。"""
    return ExampleStrategy()
