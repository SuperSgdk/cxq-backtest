# -*- coding: utf-8 -*-
"""代码规范化测试。人工字符串场景检查转换幂等及未知代码报错。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cxq_bt.codes import strip_suffix, to_ts_code, to_ts_codes  # noqa: E402


class TestToTsCode(unittest.TestCase):

    def test_常见市场裸码转换(self):
        # 外部策略
        cases = [("601086", "601086.SH"), ("002909", "002909.SZ"),
                 ("605577", "605577.SH"), ("002059", "002059.SZ"),
                 ("002300", "002300.SZ"), ("605069", "605069.SH")]
        for raw, want in cases:
            self.assertEqual(to_ts_code(raw), want, raw)

    def test_各板块前缀手算(self):
        cases = {
            "600000": "600000.SH",   # 沪主板
            "688111": "688111.SH",   # 科创板
            "900901": "900901.SH",   # 沪 B 股
            "510300": "510300.SH",   # 沪 ETF
            "113050": "113050.SH",   # 沪可转债
            "000001": "000001.SZ",   # 深主板
            "300750": "300750.SZ",   # 创业板
            "200011": "200011.SZ",   # 深 B 股
            "159915": "159915.SZ",   # 深 ETF
            "128036": "128036.SZ",   # 深可转债
            "430047": "430047.BJ",   # 北交所（43）
            "830799": "830799.BJ",   # 北交所（83）
            "871981": "871981.BJ",   # 北交所（87）
            "920992": "920992.BJ",   # 北交所新号段（92）
        }
        for raw, want in cases.items():
            self.assertEqual(to_ts_code(raw), want, raw)

    def test_带后缀幂等且大小写归一(self):
        self.assertEqual(to_ts_code("000001.SZ"), "000001.SZ")
        self.assertEqual(to_ts_code("600000.sh"), "600000.SH")
        self.assertEqual(to_ts_code(" 002084 "), "002084.SZ")

    def test_认不出来必须抛错不猜(self):
        for bad in ("", "   ", "abc", "12345", "1234567", "777777",
                    "600000.XX", "60000.SH", "ABCDEF.SZ"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                to_ts_code(bad)

    def test_批量保序且遇错即停(self):
        self.assertEqual(to_ts_codes(["002084", "600000"]),
                         ["002084.SZ", "600000.SH"])
        with self.assertRaises(ValueError):
            to_ts_codes(["002084", "bad"])

    def test_strip_suffix(self):
        self.assertEqual(strip_suffix("000001.SZ"), "000001")
        self.assertEqual(strip_suffix("002084"), "002084")


if __name__ == "__main__":
    unittest.main(verbosity=2)
