# -*- coding: utf-8 -*-
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from src.services.external_low_pe_candidates import (
    ExternalLowPeCandidate,
    ExternalLowPeScreeningResult,
)
from src.services.investment_radar import InvestmentRadarService
from src.services.investor_holdings_database import MasterInvestorHoldingsDatabase
from src.services.recommendation_lifecycle import (
    LifecycleRecommendation,
    RecommendationLifecycleService,
)


def _candidate(*, action="核心持仓观察", price=100.0, title="", source_date=None):
    return ExternalLowPeCandidate(
        code="AAPL",
        name="Apple",
        market="us",
        price=price,
        score=82.0,
        investors=["巴菲特/伯克希尔"],
        investor_actions=[action],
        action_summary=action,
        source_titles=[title] if title else [],
        source_urls=["https://example.com/source"] if title else [],
        source_date=source_date,
        holding_confidence="公开持仓观察",
        reasons=["高质量现金流"],
    )


class TestMasterInvestorHoldingsDatabase(unittest.TestCase):
    def test_records_public_holding_changes_without_inferring_exit_from_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            database = MasterInvestorHoldingsDatabase(directory)
            first = database.update_from_candidates(
                [_candidate()], observed_on=date(2026, 7, 1)
            )
            self.assertEqual(first.active_count, 1)
            self.assertEqual(first.holdings[0]["investor"], "巴菲特/伯克希尔")
            self.assertEqual(first.holdings[0]["status"], "active")

            increased = database.update_from_candidates(
                [_candidate(action="公开新闻显示加仓", title="巴菲特加仓 Apple", source_date=date(2026, 7, 2))],
                observed_on=date(2026, 7, 2),
            )
            self.assertEqual(increased.changes[0]["change_type"], "increase")

            reduced = database.update_from_candidates(
                [_candidate(action="公开新闻显示减仓", title="伯克希尔减仓 Apple", source_date=date(2026, 7, 3))],
                observed_on=date(2026, 7, 3),
            )
            self.assertEqual(reduced.changes[0]["change_type"], "decrease")
            self.assertEqual(reduced.holdings[0]["status"], "active")

            unchanged = database.update_from_candidates([], observed_on=date(2026, 7, 4))
            self.assertEqual(unchanged.holdings[0]["status"], "active")

            exited = database.update_from_candidates(
                [_candidate(action="公开新闻显示清仓", title="伯克希尔清仓 Apple", source_date=date(2026, 7, 5))],
                observed_on=date(2026, 7, 5),
            )
            self.assertEqual(exited.changes[0]["change_type"], "exit")
            self.assertEqual(exited.holdings[0]["status"], "exited")


class TestRecommendationLifecycleService(unittest.TestCase):
    def test_tracks_performance_and_generates_risk_exit_reminders(self):
        with tempfile.TemporaryDirectory() as directory:
            service = RecommendationLifecycleService(directory)
            first = LifecycleRecommendation(
                code="AAPL",
                name="Apple",
                market="us",
                source="external_master",
                source_label="外部大师雷达",
                price=100.0,
            )
            service.update([first], observed_on=date(2026, 7, 1))

            stopped = service.update(
                [
                    LifecycleRecommendation(
                        code="AAPL",
                        name="Apple",
                        market="us",
                        source="external_master",
                        source_label="外部大师雷达",
                        price=91.0,
                    )
                ],
                observed_on=date(2026, 7, 2),
            )
            record = stopped.records[0]
            self.assertEqual(record["status"], "stop_loss")
            self.assertAlmostEqual(record["return_pct"], -9.0)
            self.assertIn("止损", record["alert"])

            reduce = service.update(
                [
                    LifecycleRecommendation(
                        code="MSFT",
                        name="Microsoft",
                        market="us",
                        source="external_master",
                        source_label="外部大师雷达",
                        price=100.0,
                        reduce_signal="公开持仓出现减仓信号，建议复查",
                    )
                ],
                observed_on=date(2026, 7, 3),
            )
            msft = next(item for item in reduce.records if item["code"] == "MSFT")
            self.assertEqual(msft["status"], "reduce")

    def test_marks_unseen_recommendation_for_review_before_expiring(self):
        with tempfile.TemporaryDirectory() as directory:
            service = RecommendationLifecycleService(directory)
            recommendation = LifecycleRecommendation(
                code="600519",
                name="贵州茅台",
                market="cn",
                source="self_stock_signal",
                source_label="自选股信号",
                price=1500.0,
            )
            service.update([recommendation], observed_on=date(2026, 7, 1))
            update = service.update([], observed_on=date(2026, 7, 16))
            self.assertEqual(update.records[0]["status"], "watch")
            self.assertIn("未再次推荐", update.records[0]["alert"])


class _FakeCandidateService:
    def __init__(self, candidate):
        self.candidate = candidate

    def screen_with_observations(self, stock_list, *, limit, watch_limit):
        return ExternalLowPeScreeningResult(
            featured=[self.candidate],
            watchlist=[],
            observed=[self.candidate],
            prefiltered_count=1,
            market_status={"cn": "无可用跟踪标的", "us": "复核通过 1 个"},
        )


class TestInvestmentRadarService(unittest.TestCase):
    def test_keeps_unselected_master_exit_signal_in_lifecycle_feed(self):
        featured = _candidate(action="核心持仓观察")
        exited = ExternalLowPeCandidate(
            code="MSFT",
            name="Microsoft",
            market="us",
            price=100.0,
            investors=["巴菲特/伯克希尔"],
            investor_actions=["公开新闻显示清仓"],
            action_summary="公开新闻显示清仓",
        )

        candidates = InvestmentRadarService._lifecycle_candidates(
            [featured],
            [featured, exited],
        )

        self.assertEqual([item.code for item in candidates], ["AAPL", "MSFT"])

    def test_writes_artifacts_for_holdings_and_recommendation_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = _candidate(action="公开新闻显示加仓", title="巴菲特加仓 Apple", source_date=date(2026, 7, 1))
            service = InvestmentRadarService(
                candidate_service=_FakeCandidateService(candidate),
                holdings_database=MasterInvestorHoldingsDatabase(root / "data"),
                lifecycle_service=RecommendationLifecycleService(root / "data"),
            )
            self_result = SimpleNamespace(
                success=True,
                code="600519",
                name="贵州茅台",
                action="buy",
                operation_advice="买入",
                current_price=1500.0,
                sentiment_score=76,
                buy_reason="盈利质量稳定",
                analysis_summary="",
            )

            snapshot = service.run([], [self_result], observed_on=date(2026, 7, 1))
            paths = snapshot.write_artifacts(root / "reports")

            self.assertEqual(snapshot.holdings.active_count, 1)
            self.assertEqual(snapshot.lifecycle.tracked_count, 2)
            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["json"].exists())
            report = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("推荐生命周期", report)
            self.assertIn("大佬持仓数据库", report)


if __name__ == "__main__":
    unittest.main()
