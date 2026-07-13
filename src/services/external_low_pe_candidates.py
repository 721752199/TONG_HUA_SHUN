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
    reasons: List[str] = field(default_factory=list)
    technical_summary: str = ""
    positive_catalysts: List[str] = field(default_factory=list)
    risk_alerts: List[str] = field(default_factory=list)


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
        prefilter_limit: int = 12,
    ) -> List[ExternalLowPeCandidate]:
        excluded = self._normalize_excluded(stock_list)
        try:
            snapshot = self.fetcher.get_a_share_spot_snapshot()
        except Exception as exc:
            logger.warning("外部低 PE 候选：东方财富全市场快照获取失败: %s", exc)
            return []

        rows = self._prefilter(snapshot, excluded)
        candidates: List[ExternalLowPeCandidate] = []
        for _, row in rows.head(prefilter_limit).iterrows():
            candidate = self._row_to_candidate(row)
            if candidate is None:
                continue
            if not self._confirm_with_sina(candidate):
                continue
            self._attach_technical_context(candidate)
            self._attach_news_context(candidate)
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates

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
            & work["最新价"].between(2, 200, inclusive="both")
            & work["成交额"].ge(100000000)
            & work["市盈率-动态"].between(3, 20, inclusive="both")
            & work["换手率"].between(0.5, 12, inclusive="both")
            & work["量比"].between(0.8, 4, inclusive="both")
            & work["60日涨跌幅"].ge(0)
            & work["涨跌幅"].between(-3, 8, inclusive="both")
        )
        if "市净率" in work.columns:
            mask = mask & work["市净率"].between(0.3, 4, inclusive="both")

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
            reasons=reasons,
        )

    def _confirm_with_sina(self, candidate: ExternalLowPeCandidate) -> bool:
        try:
            quote = self.fetcher.get_realtime_quote(candidate.code, source="sina")
        except Exception as exc:
            logger.info("外部低 PE 候选：新浪复核失败 %s: %s", candidate.code, exc)
            return False
        if quote is None or not quote.price:
            return False
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
                return False
        return True

    def _attach_technical_context(self, candidate: ExternalLowPeCandidate) -> None:
        try:
            df = self.fetcher.get_daily_data(candidate.code, days=90)
            trend = self.trend_analyzer.analyze(df, candidate.code)
        except Exception as exc:
            logger.info("外部低 PE 候选：技术分析失败 %s: %s", candidate.code, exc)
            return
        candidate.technical_summary = self._format_technical_summary(trend)
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
        pe = ExternalLowPeCandidateService._float(row.get("市盈率-动态")) or 20
        amount = ExternalLowPeCandidateService._float(row.get("成交额")) or 0
        turnover = ExternalLowPeCandidateService._float(row.get("换手率")) or 0
        volume_ratio = ExternalLowPeCandidateService._float(row.get("量比")) or 0
        change_60d = ExternalLowPeCandidateService._float(row.get("60日涨跌幅")) or 0
        pb = ExternalLowPeCandidateService._float(row.get("市净率")) or 4
        return (
            max(0, 20 - pe) * 3
            + min(amount / 100000000, 20) * 1.2
            + min(turnover, 8) * 2
            + min(volume_ratio, 3) * 4
            + min(change_60d, 40) * 0.8
            + max(0, 4 - pb) * 2
        )

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
