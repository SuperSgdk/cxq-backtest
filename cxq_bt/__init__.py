"""A 股回测组件包。

公开日级入口见 cxq_bt.framework；设计见 docs/architecture.md。
费用、规则与绩效组件不依赖外部私有策略。"""

# Python 3.13 兼容垫片必须早于任何 Backtrader 导入。
from . import compat as compat  # noqa: F401

__version__ = "0.1.0"
