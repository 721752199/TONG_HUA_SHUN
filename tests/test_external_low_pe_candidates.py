# -*- coding: utf-8 -*-
import unittest
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

from src.services.external_low_pe_candidates import (
    ExternalLowPeCandidate,
    ExternalLowPeCandidateService,
)


class FakeAshareFetcher:
    def get_realtime_quote(self, code, source="sina"):
        if code == "000333":
            return SimpleNamespace(price=65.2, change_pct=1.1, name="美的集团")
        if code in {"002415", "601899", "002001", "000513", "603986"}:
            return SimpleNamespace(price=20.0, change_pct=0.5, name=code)
        return None

    def get_daily_data(self, code, days):
        return pd.DataFrame()


class FakeUsFetcher:
    def get_realtime_quote(self, code):
        if code == "AAPL":
            return SimpleNamespace(price=210.0, change_pct=0.8, amount=None)
        if code in {"GOOGL", "PDD", "BRK-B", "AXP", "OXY"}:
            return SimpleNamespace(price=100.0, change_pct=0.2, amount=None)
        return None

    def get_daily_data(self, code, days):
        return pd.DataFrame()


class FakeSearchService:
    is_available = True

    def __init__(self, query_results):
        self.query_results = query_results

    def search_stock_news(self, stock_code, stock_name, max_results=5, focus_keywords=None):
        results = self.query_results.get(stock_code, [])
        return SimpleNamespace(success=True, results=results)


class TestExternalMasterCandidateService(unittest.TestCase):
    def test_screen_uses_master_holdings_and_excludes_self_stocks(self):
        trend = SimpleNamespace(
            trend_status=SimpleNamespace(value="多头排列"),
            trend_strength=72,
            buy_signal=SimpleNamespace(value="观望"),
            ma_alignment="MA5>MA10>MA20",
            signal_reasons=["均线向上"],
            risk_factors=["短线波动"],
            ma5=64.0,
            ma10=63.5,
            ma20=61.0,
            support_ma10=True,
        )
        service = ExternalLowPeCandidateService(
            fetcher=FakeAshareFetcher(),
            us_fetcher=FakeUsFetcher(),
            trend_analyzer=SimpleNamespace(analyze=lambda df, code: trend),
        )

        result = service.screen_with_observations(["000333", "AAPL"], limit=3)

        self.assertNotIn("000333", [item.code for item in result.featured])
        self.assertNotIn("AAPL", [item.code for item in result.featured])
        self.assertLessEqual(len([item for item in result.featured if item.market == "cn"]), 3)
        self.assertLessEqual(len([item for item in result.featured if item.market == "us"]), 3)
        self.assertIn("cn", result.market_status)
        self.assertIn("us", result.market_status)

    def test_recent_add_news_boosts_candidate_and_sets_evidence(self):
        news = SimpleNamespace(
            title="高毅资产新进美的集团十大流通股东",
            snippet="机构持仓显示高毅资产加仓美的集团",
            url="https://example.com/midea",
            published_date=date.today().isoformat(),
        )
        trend = SimpleNamespace(
            trend_status=SimpleNamespace(value="震荡上行"),
            trend_strength=80,
            buy_signal=SimpleNamespace(value="观望"),
            ma_alignment="MA5>MA10",
            signal_reasons=[],
            risk_factors=[],
            ma5=64.0,
            ma10=63.5,
            ma20=61.0,
            support_ma10=True,
        )
        service = ExternalLowPeCandidateService(
            fetcher=FakeAshareFetcher(),
            us_fetcher=FakeUsFetcher(),
            trend_analyzer=SimpleNamespace(analyze=lambda df, code: trend),
            search_service=FakeSearchService({"000333": [news]}),
        )

        result = service.screen_with_observations([], limit=3)
        midea = next(item for item in result.featured if item.code == "000333")

        self.assertIn("大师新增/加仓", midea.catalyst_signals)
        self.assertEqual(midea.holding_confidence, "新增/加仓优先")
        self.assertIn("高毅资产新进美的集团", midea.source_titles[0])
        self.assertGreater(midea.score + midea.catalyst_score, 90)

    def test_reduce_news_sets_risk_timer(self):
        candidate = ExternalLowPeCandidate(
            code="AAPL",
            name="Apple",
            market="us",
            source_date=date.today() - timedelta(days=2),
        )

        ExternalLowPeCandidateService._apply_reduce_alert(candidate)

        self.assertIn("出现减仓/清仓线索", candidate.reduce_alert)
        self.assertIn((date.today() + timedelta(days=5)).isoformat(), candidate.reduce_alert)

    def test_limit_per_market_uses_total_score(self):
        candidates = [
            ExternalLowPeCandidate(code=f"60000{index}", name=str(index), score=50, catalyst_score=index, market="cn")
            for index in range(4)
        ] + [
            ExternalLowPeCandidate(code=f"US{index}", name=str(index), score=40 + index, market="us")
            for index in range(4)
        ]

        selected = ExternalLowPeCandidateService._limit_per_market(candidates, 3)

        self.assertEqual([item.code for item in selected[:3]], ["600003", "600002", "600001"])
        self.assertEqual([item.code for item in selected[3:]], ["US3", "US2", "US1"])


if __name__ == "__main__":
    unittest.main()
