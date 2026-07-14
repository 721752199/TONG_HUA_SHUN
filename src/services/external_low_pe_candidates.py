# -*- coding: utf-8 -*-
"""Screen external low-PE A-share candidates for the PushPlus appendix."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Set

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import is_bse_code, is_st_stock, normalize_stock_code
from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ExternalLowPeCandidate:
    code: str
    name: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    amount: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    volume_ratio: Optional[float] = None
    turnover_rate: Optional[float] = None
    change_60d: Optional[float] = None
    sina_price: Optional[float] = None
    sina_change_pct: Optional[float] = None
    score: float = 0.0
    industry: str = ""
    opportunity_type: str = "低估值关注"
    verification_status: str = "新浪待复核"
    data_status: str = "东方财富全市场快照"
    reasons: List[str] = field(default_factory=list)
    technical_summary: str = ""
    entry_trigger: str = ""
    invalidation_condition: str = ""
    positive_catalysts: List[str] = field(default_factory=list)
    risk_alerts: List[str] = field(default_factory=list)


@dataclass
class ExternalLowPeScreeningResult:
    """Separate verified recommendations from unverified observation candidates."""

    featured: List[ExternalLowPeCandidate] = field(default_factory=list)
    watchlist: List[ExternalLowPeCandidate] = field(default_factory=list)
    prefiltered_count: int = 0
    sina_unavailable_count: int = 0


class ExternalLowPeCandidateService:
    """Find low-PE candidates outside the user's STOCK_LIST."""

    def __init__(
        self,
        fetcher: Optional[AkshareFetcher] = None,
        trend_analyzer: Optional[StockTrendAnalyzer] = None,
        search_service: Optional[Any] = None,
    ) -> None:
        self.fetcher = fetcher or AkshareFetcher(sleep_min=0.2, sleep_max=0.6)
        self.trend_analyzer = trend_analyzer or StockTrendAnalyzer()
        self.search_service = search_service

    def screen(
        self,
        stock_list: Sequence[str],
        *,
        limit: int = 3,
        prefilter_limit: int = 40,
    ) -> List[ExternalLowPeCandidate]:
        """Return only Sina-verified candidates for backward-compatible callers."""
        return self.screen_with_observations(
            stock_list,
            limit=limit,
            prefilter_limit=prefilter_limit,
        ).featured

    def screen_with_observations(
        self,
        stock_list: Sequence[str],
        *,
        limit: int = 3,
        watch_limit: int = 3,
        prefilter_limit: int = 40,
    ) -> ExternalLowPeScreeningResult:
        """Screen broader candidates while keeping unverified names out of recommendations."""
        excluded = self._normalize_excluded(stock_list)
        try:
            snapshot = self.fetcher.get_a_share_spot_snapshot()
        except Exception as exc:
            logger.warning("外部低 PE 候选：东方财富全市场快照获取失败: %s", exc)
            return ExternalLowPeScreeningResult()

        rows = self._prefilter(snapshot, excluded)
        result = ExternalLowPeScreeningResult(prefiltered_count=len(rows))
        featured_industries: Set[str] = set()
        watch_industries: Set[str] = set()
        for _, row in rows.head(prefilter_limit).iterrows():
            candidate = self._row_to_candidate(row)
            if candidate is None:
                continue
            verification_status = self._confirm_with_sina(candidate)
            candidate.verification_status = verification_status
            if verification_status == "新浪已复核":
                candidate.data_status = "东方财富快照 + 新浪行情复核"
                if self._is_duplicate_industry(candidate.industry, featured_industries):
                    continue
                featured_industries.add(candidate.industry)
                self._attach_technical_context(candidate)
                self._attach_news_context(candidate)
                result.featured.append(candidate)
                if len(result.featured) >= limit:
                    break
                continue

            if verification_status == "新浪暂不可用":
                result.sina_unavailable_count += 1
                if len(result.watchlist) >= watch_limit:
                    continue
                if self._is_duplicate_industry(
                    candidate.industry,
                    featured_industries | watch_industries,
                ):
                    continue
                watch_industries.add(candidate.industry)
                candidate.data_status = "东方财富快照；新浪行情暂不可用"
                candidate.entry_trigger = "待新浪行情复核及技术面确认后再评估，不构成交易建议"
                candidate.invalidation_condition = "若后续复核价格与东财偏差超过 5%，停止跟踪"
                result.watchlist.append(candidate)

        return result

    def _prefilter(self, df: pd.DataFrame, excluded: Set[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        required = ["代码", "名称", "最新价", "成交额", "市盈率-动态", "换手率", "量比", "60日涨跌幅"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning("外部低 PE 候选：东财快照缺少字段: %s", ",".join(missing))
            return pd.DataFrame()

        work = df.copy()
        work["代码"] = work["代码"].map(lambda value: normalize_stock_code(str(value).strip()))
        work["名称"] = work["名称"].map(lambda value: str(value or "").strip())
        for col in ["最新价", "涨跌幅", "成交额", "市盈率-动态", "市净率", "换手率", "量比", "60日涨跌幅"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        mask = (
            work["代码"].map(lambda code: code.isdigit() and len(code) == 6)
            & ~work["代码"].isin(excluded)
            & ~work["名称"].map(is_st_stock)
            & ~work["代码"].map(is_bse_code)
            & work["最新价"].between(2, 300, inclusive="both")
            & work["成交额"].ge(50000000)
            & work["市盈率-动态"].between(3, 30, inclusive="both")
            & work["换手率"].between(0.3, 15, inclusive="both")
            & work["量比"].between(0.7, 5, inclusive="both")
            & work["60日涨跌幅"].ge(-8)
            & work["涨跌幅"].between(-5, 5, inclusive="both")
        )
        if "市净率" in work.columns:
            mask = mask & work["市净率"].between(0.2, 5, inclusive="both")

        filtered = work.loc[mask].copy()
        if filtered.empty:
            return filtered

        filtered["_score"] = filtered.apply(self._score_row, axis=1)
        return filtered.sort_values("_score", ascending=False)

    def _row_to_candidate(self, row: pd.Series) -> Optional[ExternalLowPeCandidate]:
        code = normalize_stock_code(str(row.get("代码", "")).strip())
        name = str(row.get("名称", "") or "").strip()
        if not code or not name:
            return None
        pe = self._float(row.get("市盈率-动态"))
        turnover = self._float(row.get("换手率"))
        volume_ratio = self._float(row.get("量比"))
        change_60d = self._float(row.get("60日涨跌幅"))
        pb = self._float(row.get("市净率"))
        industry = str(row.get("所属行业", "") or "").strip()
        reasons = [
            f"动态 PE {pe:.1f}" if pe is not None else "动态 PE 低位",
            f"成交额 {self._amount_yi(row.get('成交额'))} 亿" if self._float(row.get("成交额")) else "流动性达标",
            f"60日涨幅 {change_60d:.1f}%" if change_60d is not None else "中期趋势非负",
        ]
        if turnover is not None and volume_ratio is not None:
            reasons.append(f"换手 {turnover:.2f}%、量比 {volume_ratio:.2f}")
        return ExternalLowPeCandidate(
            code=code,
            name=name,
            price=self._float(row.get("最新价")),
            change_pct=self._float(row.get("涨跌幅")),
            amount=self._float(row.get("成交额")),
            pe_ratio=pe,
            pb_ratio=self._float(row.get("市净率")),
            volume_ratio=volume_ratio,
            turnover_rate=turnover,
            change_60d=change_60d,
            score=self._float(row.get("_score")) or 0.0,
            industry=industry,
            opportunity_type=self._classify_opportunity(pe, pb, change_60d),
            reasons=reasons,
        )

    def _confirm_with_sina(self, candidate: ExternalLowPeCandidate) -> str:
        try:
            quote = self.fetcher.get_realtime_quote(candidate.code, source="sina")
        except Exception as exc:
            logger.info("外部低 PE 候选：新浪复核失败 %s: %s", candidate.code, exc)
            return "新浪暂不可用"
        if quote is None or not quote.price:
            return "新浪暂不可用"
        candidate.sina_price = quote.price
        candidate.sina_change_pct = quote.change_pct
        if candidate.price and quote.price:
            price_gap = abs(candidate.price - quote.price) / max(candidate.price, 0.01)
            if price_gap > 0.05:
                logger.info(
                    "外部低 PE 候选：新浪/东财价格偏差过大，跳过 %s gap=%.2f%%",
                    candidate.code,
                    price_gap * 100,
                )
                return "新浪价格偏差过大"
        return "新浪已复核"

    def _attach_technical_context(self, candidate: ExternalLowPeCandidate) -> None:
        try:
            df = self.fetcher.get_daily_data(candidate.code, days=90)
            trend = self.trend_analyzer.analyze(df, candidate.code)
        except Exception as exc:
            logger.info("外部低 PE 候选：技术分析失败 %s: %s", candidate.code, exc)
            candidate.entry_trigger = "等待技术面数据确认后再评估"
            candidate.invalidation_condition = "技术面未确认或后续行情复核异常时停止跟踪"
            return
        candidate.technical_summary = self._format_technical_summary(trend)
        candidate.entry_trigger, candidate.invalidation_condition = self._build_execution_plan(trend)
        candidate.risk_alerts.extend(self._strip_prefix(item) for item in trend.risk_factors[:2])

    def _attach_news_context(self, candidate: ExternalLowPeCandidate) -> None:
        service = self.search_service
        if service is None or not getattr(service, "is_available", False):
            return
        try:
            response = service.search_stock_news(
                candidate.code,
                candidate.name,
                max_results=2,
                focus_keywords=[candidate.name, candidate.code, "业绩", "订单", "回购", "低估值"],
            )
        except Exception as exc:
            logger.info("外部低 PE 候选：新闻搜索失败 %s: %s", candidate.code, exc)
            return
        if not getattr(response, "success", False):
            return
        for result in getattr(response, "results", [])[:2]:
            title = str(getattr(result, "title", "") or "").strip()
            snippet = str(getattr(result, "snippet", "") or "").strip()
            text = title or snippet
            if text:
                candidate.positive_catalysts.append(text[:80])

    @staticmethod
    def _format_technical_summary(trend: TrendAnalysisResult) -> str:
        parts = [
            f"{trend.trend_status.value}",
            f"趋势强度 {trend.trend_strength:.0f}/100",
            f"信号 {trend.buy_signal.value}",
        ]
        if trend.ma_alignment:
            parts.append(trend.ma_alignment)
        if trend.signal_reasons:
            parts.append("；".join(trend.signal_reasons[:2]))
        return "，".join(part for part in parts if part)

    @staticmethod
    def _score_row(row: pd.Series) -> float:
        pe = ExternalLowPeCandidateService._float(row.get("市盈率-动态")) or 30
        amount = ExternalLowPeCandidateService._float(row.get("成交额")) or 0
        turnover = ExternalLowPeCandidateService._float(row.get("换手率")) or 0
        volume_ratio = ExternalLowPeCandidateService._float(row.get("量比")) or 0
        change_60d = ExternalLowPeCandidateService._float(row.get("60日涨跌幅")) or 0
        pb = ExternalLowPeCandidateService._float(row.get("市净率")) or 5
        return (
            max(0, 30 - pe) * 2.4
            + min(amount / 100000000, 20) * 1.2
            + min(turnover, 8) * 2
            + min(volume_ratio, 3) * 4
            + min(change_60d, 40) * 0.8
            + max(0, 5 - pb) * 1.6
        )

    @staticmethod
    def _classify_opportunity(
        pe_ratio: Optional[float],
        pb_ratio: Optional[float],
        change_60d: Optional[float],
    ) -> str:
        if pe_ratio is not None and pe_ratio <= 12 and pb_ratio is not None and pb_ratio <= 1:
            return "低 PB 价值"
        if change_60d is not None and change_60d <= 10:
            return "低估值修复"
        return "价值趋势"

    @staticmethod
    def _is_duplicate_industry(industry: str, selected_industries: Set[str]) -> bool:
        normalized = str(industry or "").strip()
        if not normalized:
            return False
        return normalized in selected_industries

    @staticmethod
    def _build_execution_plan(trend: TrendAnalysisResult) -> tuple[str, str]:
        ma5 = float(getattr(trend, "ma5", 0) or 0)
        ma10 = float(getattr(trend, "ma10", 0) or 0)
        ma20 = float(getattr(trend, "ma20", 0) or 0)
        support = ma10 if getattr(trend, "support_ma10", False) and ma10 > 0 else ma5
        if support > 0:
            entry = f"回踩 {support:.2f} 附近企稳且量能不恶化后再评估"
        elif ma5 > 0:
            entry = f"站稳 MA5 {ma5:.2f} 并保持量价配合后再评估"
        else:
            entry = "等待技术面数据确认后再评估"

        if ma20 > 0:
            invalidation = f"收盘跌破 MA20 {ma20:.2f} 且下一交易日未收回时停止跟踪"
        else:
            invalidation = "趋势转弱或出现新的基本面风险时停止跟踪"
        return entry, invalidation

    @staticmethod
    def _normalize_excluded(stock_list: Iterable[str]) -> Set[str]:
        return {
            normalize_stock_code(str(code).strip())
            for code in stock_list or []
            if str(code).strip()
        }

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _amount_yi(value: Any) -> str:
        amount = ExternalLowPeCandidateService._float(value)
        if amount is None:
            return "N/A"
        return f"{amount / 100000000:.1f}"

    @staticmethod
    def _strip_prefix(value: Any) -> str:
        text = str(value or "").strip()
        return text.lstrip("⚠️ ").strip()
