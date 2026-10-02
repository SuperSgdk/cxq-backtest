# -*- coding: utf-8 -*-
"""Backtrader 的现代 Python 兼容垫片。

在导入 Backtrader 之前，补回 collections 中已迁至 collections.abc 的
抽象类别名。只补缺失名称，使用相同对象；重复执行不会覆盖已有名称。
修补保存在项目代码中，不修改第三方安装文件。"""

from __future__ import annotations

import collections
import collections.abc as _abc

# Python 2 → 3.10 被移出 collections 的 ABC 名字
_SHIMS = (
    "Iterable", "Iterator", "Callable", "Mapping", "MutableMapping",
    "Sequence", "MutableSequence", "Set", "MutableSet", "Hashable",
    "Sized", "Container", "KeysView", "ItemsView", "ValuesView",
    "Generator", "Awaitable", "Coroutine", "AsyncIterable", "AsyncIterator",
)


def apply() -> list:
    """把缺失的 ABC 名字补回 collections。幂等，返回本次补了哪些名字。"""
    patched = []
    for name in _SHIMS:
        if not hasattr(collections, name) and hasattr(_abc, name):
            setattr(collections, name, getattr(_abc, name))
            patched.append(name)
    return patched


# import 本模块即生效
apply()
