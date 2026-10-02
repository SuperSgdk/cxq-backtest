# -*- coding: utf-8 -*-
"""检查面板单位到引擎费率的转换，防止手续费相差 100 倍。"""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app import build_config


class TestAppSettings(unittest.TestCase):
    def test_percent_and_basis_point_units(self):
        config = build_config(
            initial_cash=250_000,
            commission_percent=0.08,
            min_commission=8,
            stamp_duty_percent=0.06,
            transfer_fee_percent=0.002,
            slippage_bps=12,
            max_positions=7,
            exclude_st=True,
            hard_stop_percent=9,
        )
        self.assertAlmostEqual(config.costs.commission_rate, 0.0008)
        self.assertAlmostEqual(config.costs.stamp_duty_rate, 0.0006)
        self.assertAlmostEqual(config.costs.transfer_fee_rate, 0.00002)
        self.assertAlmostEqual(config.costs.slippage_rate, 0.0012)
        self.assertEqual(config.costs.min_commission, 8)
        self.assertAlmostEqual(config.hard_stop_pct, -0.09)
        self.assertEqual(config.initial_cash, 250_000)
        self.assertEqual(config.max_positions, 7)
        self.assertEqual(config.lot, 100)

    def test_optional_strategy_policies_can_be_disabled(self):
        config = build_config(
            initial_cash=1_000_000,
            commission_percent=0.05,
            min_commission=5,
            stamp_duty_percent=0.05,
            transfer_fee_percent=0.001,
            slippage_bps=5,
            max_positions=5,
            exclude_st=False,
            hard_stop_percent=None,
        )
        self.assertFalse(config.exclude_st)
        self.assertIsNone(config.hard_stop_pct)


class TestAppFlow(unittest.TestCase):
    """验证用户改参数、点击运行、上传缺字段文件时看到的真实界面状态。"""

    def app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                 default_timeout=30).run()

    def test_click_to_run_and_keep_parameter_snapshot(self):
        at = self.app()
        self.assertFalse(at.exception)
        self.assertNotIn("last_result", at.session_state)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        original = at.session_state["last_result"]
        self.assertEqual(original["presentation"]["source_kind"], "handcalc")
        self.assertEqual(original["settings"]["initial_cash"], 1_000_000)
        json.dumps(original, allow_nan=False)

        at.number_input[0].set_value(250_000.0).run()
        at.number_input[2].set_value(0.08).run()
        self.assertTrue(any("上次结果" in warning.value for warning in at.warning))
        self.assertEqual(at.session_state["last_result"]["settings"]["initial_cash"], 1_000_000)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        refreshed = at.session_state["last_result"]
        self.assertEqual(refreshed["settings"]["initial_cash"], 250_000)
        self.assertAlmostEqual(refreshed["settings"]["costs"]["commission_rate"], 0.0008)
        self.assertFalse(any("上次结果" in warning.value for warning in at.warning))
        self.assertEqual(refreshed["presentation"]["controls"]["commission_percent"], 0.08)

    def test_missing_market_columns_show_error(self):
        at = self.app()
        # 这里只模拟缺列的文件上传；不生成或报告所谓真实历史业绩。
        market = b"date,code,close\n2026-01-05,TEST_A,10\n2026-01-06,TEST_A,11\n"
        signals = b"date,code,weight\n2026-01-05,TEST_A,0.5\n"

        def upload(label, **kwargs):
            return io.BytesIO(market if "行情" in label else signals)

        with patch("streamlit.file_uploader", side_effect=upload):
            at.radio[0].set_value("本地行情 CSV").run()
            self.assertFalse(at.button[0].disabled)
            at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("行情缺字段" in error.value for error in at.error))
        self.assertNotIn("last_result", at.session_state)

    def test_moving_average_example_runs(self):
        at = self.app()
        at.selectbox[0].set_value("均线示例策略").run()
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertIn("last_result", at.session_state)


if __name__ == "__main__":
    unittest.main()
