# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace
from datetime import date, timedelta

import pandas as pd

from src.services.external_low_pe_candidates import ExternalLowPeCandidateService
from src.services.external_low_pe_candidates import ExternalLowPeCandidate


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

    def test_screen_separates_sina_unavailable_watchlist_and_deduplicates_industry(self):
        rows = pd.DataFrame(
            [
                {
                    "代码": "600000", "名称": "浦发银行", "所属行业": "银行",
                    "最新价": 10.2, "涨跌幅": 1.0, "成交额": 250000000,
                    "市盈率-动态": 6.8, "市净率": 0.6, "换手率": 1.2,
                    "量比": 1.1, "60日涨跌幅": 8.5,
                },
                {
                    "代码": "601000", "名称": "示例银行", "所属行业": "银行",
                    "最新价": 9.8, "涨跌幅": 0.8, "成交额": 220000000,
                    "市盈率-动态": 7.0, "市净率": 0.7, "换手率": 1.0,
                    "量比": 1.0, "60日涨跌幅": 9.0,
                },
                {
                    "代码": "000001", "名称": "平安银行", "所属行业": "软件开发",
                    "最新价": 11.5, "涨跌幅": 1.2, "成交额": 180000000,
                    "市盈率-动态": 9.0, "市净率": 1.1, "换手率": 1.5,
                    "量比": 1.2, "60日涨跌幅": 6.0,
                },
            ]
        )

        class FakeFetcher:
            def get_a_share_spot_snapshot(self):
                return rows

            def get_realtime_quote(self, code, source):
                if code == "000001":
                    return None
                return SimpleNamespace(price=10.2, change_pct=1.0)

            def get_daily_data(self, code, days):
                return pd.DataFrame()

        trend = SimpleNamespace(
            trend_status=SimpleNamespace(value="多头排列"),
            trend_strength=72,
            buy_signal=SimpleNamespace(value="观望"),
            ma_alignment="MA5>MA10>MA20",
            signal_reasons=["均线向上"],
            risk_factors=["短线波动"],
            ma5=10.1,
            ma10=10.0,
            ma20=9.6,
            support_ma10=True,
        )
        trend_analyzer = SimpleNamespace(analyze=lambda df, code: trend)
        service = ExternalLowPeCandidateService(
            fetcher=FakeFetcher(),
            trend_analyzer=trend_analyzer,
        )

        result = service.screen_with_observations([], limit=3, watch_limit=3)

        self.assertEqual([candidate.code for candidate in result.featured], ["600000"])
        self.assertEqual([candidate.code for candidate in result.watchlist], ["000001"])
        self.assertEqual(result.watchlist[0].verification_status, "新浪暂不可用")
        self.assertIn("10.00", result.featured[0].entry_trigger)
        self.assertIn("9.60", result.featured[0].invalidation_condition)

    def test_screen_includes_yfinance_verified_us_candidate(self):
        us_rows = pd.DataFrame([
            {
                "代码": "105.INTC", "名称": "Intel", "最新价": 22.5,
                "涨跌幅": 0.8, "成交额": 50000000, "市盈率": 18.0,
                "总市值": 100000000000,
            }
        ])

        class FakeFetcher:
            def get_a_share_spot_snapshot(self):
                return pd.DataFrame()

            def get_us_stock_spot_snapshot(self):
                return us_rows

        class FakeUsFetcher:
            def get_realtime_quote(self, code):
                return SimpleNamespace(price=22.4, change_pct=0.7)

            def get_daily_data(self, code, days):
                return pd.DataFrame()

        trend = SimpleNamespace(
            trend_status=SimpleNamespace(value="多头排列"), trend_strength=70,
            buy_signal=SimpleNamespace(value="观望"), ma_alignment="MA5>MA10",
            signal_reasons=[], risk_factors=[], ma5=22.0, ma10=21.5,
            ma20=20.0, support_ma10=True,
        )
        service = ExternalLowPeCandidateService(
            fetcher=FakeFetcher(),
            us_fetcher=FakeUsFetcher(),
            trend_analyzer=SimpleNamespace(analyze=lambda df, code: trend),
        )

        result = service.screen_with_observations([], limit=3)

        self.assertEqual([candidate.code for candidate in result.featured], ["INTC"])
        self.assertEqual(result.featured[0].market, "us")
        self.assertEqual(result.featured[0].verification_status, "Yahoo Finance 已复核")

    def test_reduce_timer_requires_recent_news_sector_strength_and_existing_gain(self):
        self.assertTrue(ExternalLowPeCandidateService._is_heat_catalyst_news("政策支持带动订单预增"))
        self.assertFalse(ExternalLowPeCandidateService._is_heat_catalyst_news("公司发布日常公告"))
        candidate = ExternalLowPeCandidate(
            code="600000", name="浦发银行", market="cn", change_60d=15.0,
            sector_change_pct=2.0,
        )
        ExternalLowPeCandidateService._apply_cn_reduce_timer(
            candidate,
            [date.today() - timedelta(days=2)],
        )
        self.assertIn("减仓时钟", candidate.reduce_alert)
        self.assertIn("利好兑现", candidate.risk_alerts[0])

        no_gain = ExternalLowPeCandidate(
            code="600001", name="示例", market="cn", change_60d=5.0,
            sector_change_pct=2.0,
        )
        ExternalLowPeCandidateService._apply_cn_reduce_timer(no_gain, [date.today()])
        self.assertEqual(no_gain.reduce_alert, "")
