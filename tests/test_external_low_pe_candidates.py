# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from src.services.external_low_pe_candidates import ExternalLowPeCandidateService


class TestExternalLowPeCandidateService(unittest.TestCase):
    def test_prefilter_excludes_watchlist_st_and_bse_stocks(self):
        rows = pd.DataFrame(
            [
                {
                    "代码": "600519",
                    "名称": "贵州茅台",
                    "最新价": 1500,
                    "涨跌幅": 1.0,
                    "成交额": 500000000,
                    "市盈率-动态": 18.0,
                    "市净率": 3.0,
                    "换手率": 1.0,
                    "量比": 1.1,
                    "60日涨跌幅": 8.0,
                },
                {
                    "代码": "600000",
                    "名称": "浦发银行",
                    "最新价": 10.2,
                    "涨跌幅": 1.0,
                    "成交额": 250000000,
                    "市盈率-动态": 6.8,
                    "市净率": 0.6,
                    "换手率": 1.2,
                    "量比": 1.1,
                    "60日涨跌幅": 8.5,
                },
                {
                    "代码": "600001",
                    "名称": "*ST示例",
                    "最新价": 6.0,
                    "涨跌幅": 1.0,
                    "成交额": 250000000,
                    "市盈率-动态": 6.0,
                    "市净率": 0.8,
                    "换手率": 1.0,
                    "量比": 1.0,
                    "60日涨跌幅": 5.0,
                },
                {
                    "代码": "920001",
                    "名称": "北交示例",
                    "最新价": 6.0,
                    "涨跌幅": 1.0,
                    "成交额": 250000000,
                    "市盈率-动态": 6.0,
                    "市净率": 0.8,
                    "换手率": 1.0,
                    "量比": 1.0,
                    "60日涨跌幅": 5.0,
                },
            ]
        )

        service = ExternalLowPeCandidateService(fetcher=object())
        filtered = service._prefilter(rows, service._normalize_excluded(["SH600519"]))

        self.assertEqual(filtered["代码"].tolist(), ["600000"])
        self.assertGreater(filtered.iloc[0]["_score"], 0)
