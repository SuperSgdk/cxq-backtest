# -*- coding: utf-8 -*-
"""A 股代码规范化：裸六位代码与 SH/SZ/BJ 后缀之间转换。

调用者应在接入数据和目标清单时统一代码表示。无法识别的长度、前缀
或后缀直接报错，避免把代码不匹配错误地当成空清单。
前缀规则用于常见股票、基金、可转债和北交所代码，不推断未知品种。"""

from __future__ import annotations

from typing import Iterable, List

__all__ = ["to_ts_code", "to_ts_codes", "strip_suffix", "SUFFIXES"]

SUFFIXES = ("SH", "SZ", "BJ")

#: 显式前缀表（顺序有意义：北交所的 92 必须排在上海 90/9x 之前判断）
_BJ = ("43", "83", "87", "88", "92", "4", "8")
_SH = ("60", "68", "90", "5", "11", "13")
_SZ = ("00", "30", "20", "15", "16", "18", "12")


def to_ts_code(code) -> str:
    """裸码 / 带后缀 -> 标准 `ts_code`（`000001.SZ`）。**认不出来就抛 ValueError**。

    手算示例：
        '002084'      -> '002084.SZ'
        '605577'      -> '605577.SH'
        '920992'      -> '920992.BJ'
        '000001.SZ'   -> '000001.SZ'（幂等）
        '600000.sh'   -> '600000.SH'（后缀大写归一）
        'abc' / '' / '12345' / '600000.XX' -> ValueError
    """
    c = str(code).strip().upper()
    if not c:
        raise ValueError("代码为空，不能推断股票代码")
    if "." in c:
        base, _, suf = c.partition(".")
        if suf not in SUFFIXES:
            raise ValueError("认不出的后缀：%r（只接受 SH/SZ/BJ）" % (code,))
        if len(base) != 6 or not base.isdigit():
            raise ValueError("认不出的代码：%r（后缀前必须是 6 位数字）" % (code,))
        return "%s.%s" % (base, suf)
    if len(c) != 6 or not c.isdigit():
        raise ValueError("认不出的代码：%r（只接受 6 位数字，或带 .SH/.SZ/.BJ 后缀）" % (code,))
    if c.startswith(_BJ):
        return c + ".BJ"
    if c.startswith(_SH):
        return c + ".SH"
    if c.startswith(_SZ):
        return c + ".SZ"
    raise ValueError("认不出的代码前缀：%r，不能推断交易所" % (code,))


def to_ts_codes(codes: Iterable) -> List[str]:
    """批量转换（顺序保留；任一失败就抛错，不做部分成功）。"""
    return [to_ts_code(c) for c in codes]


def strip_suffix(code) -> str:
    """移除交易所后缀，返回点号前的代码字符串。"""
    c = str(code).strip().upper()
    return c.partition(".")[0] if "." in c else c
