# -*- coding: utf-8 -*-
"""Screen external low-PE A-share candidates for the PushPlus appendix."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, List, Optional, Sequence, Set

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import is_bse_code, is_st_stock, normalize_stock_code
from data_provider.yfinance_fetcher import YfinanceFetcher
from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ExternalLowPeCandidate:
    code: str
    name: str
    market: str = "cn"
    currency: str = "CNY"
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
    sector_change_pct: Optional[float] = None
    sector_heat_summary: str = ""
    reduce_alert: str = ""
    catalyst_signals: List[str] = field(default_factory=list)
    catalyst_score: float = 0.0
    positive_catalysts: List[str] = field(default_factory=list)
    risk_alerts: List[str] = field(default_factory=list)


@dataclass
class ExternalLowPeScreeningResult:
    """Separate verified recommendations from unverified observation candidates."""

    featured: List[ExternalLowPeCandidate] = field(default_factory=list)
    watchlist: List[ExternalLowPeCandidate] = field(default_factory=list)
    prefiltered_count: int = 0
    sina_unavailable_count: int = 0
    market_status: dict[str, str] = field(default_factory=dict)


class ExternalLowPeCandidateService:
    """Find low-PE candidates outside the user's STOCK_LIST."""

    def __init__(
        self,
        fetcher: Optional[AkshareFetcher] = None,
        us_fetcher: Optional[YfinanceFetcher] = None,
        trend_analyzer: Optional[StockTrendAnalyzer] = None,
        search_service: Optional[Any] = None,
    ) -> None:
        self.fetcher = fetcher or AkshareFetcher(sleep_min=0.2, sleep_max=0.6)
        self.us_fetcher = us_fetcher or YfinanceFetcher()
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
        result = ExternalLowPeScreeningResult()
        self._screen_a_shares(result, excluded, limit, watch_limit, prefilter_limit)
        self._screen_us_stocks(result, excluded, limit, watch_limit, prefilter_limit)
        result.featured = self._limit_per_market(result.featured, limit)
        return result

    def _screen_a_shares(
        self,
        result: ExternalLowPeScreeningResult,
        excluded: Set[str],
        limit: int,
        watch_limit: int,
        prefilter_limit: int,
    ) -> None:
        try:
            snapshot = self.fetcher.get_a_share_spot_snapshot()
        except Exception as exc:
            result.market_status["cn"] = f"A 股快照不可用: {type(exc).__name__}"
            return
        rows = self._prefilter(snapshot, excluded)
        result.prefiltered_count += len(rows)
        result.market_status["cn"] = f"A 股初筛 {len(rows)} 只"
        sector_changes = self._get_sector_changes()
        self._screen_rows(result, rows, "cn", limit, watch_limit, prefilter_limit, sector_changes)

    def _screen_us_stocks(
        self,
        result: ExternalLowPeScreeningResult,
        excluded: Set[str],
        limit: int,
        watch_limit: int,
        prefilter_limit: int,
    ) -> None:
        try:
            snapshot = self.fetcher.get_us_stock_spot_snapshot()
        except Exception as exc:
            result.market_status["us"] = str(exc)
            return
        rows = self._prefilter_us(snapshot, excluded)
        result.prefiltered_count += len(rows)
        result.market_status["us"] = f"美股初筛 {len(rows)} 只"
        self._screen_rows(result, rows, "us", limit, watch_limit, prefilter_limit, {})

    def _screen_rows(
        self,
        result: ExternalLowPeScreeningResult,
        rows: pd.DataFrame,
        market: str,
        limit: int,
        watch_limit: int,
        prefilter_limit: int,
        sector_changes: dict[str, float],
    ) -> None:
        featured_industries: Set[str] = set()
        watch_industries: Set[str] = set()
        featured = [candidate for candidate in result.featured if candidate.market == market]
        watches = [candidate for candidate in result.watchlist if candidate.market == market]
        for _, row in rows.head(prefilter_limit).iterrows():
            candidate = self._row_to_candidate(row, market=market)
            if candidate is None:
                continue
            if market == "cn":
                candidate.sector_change_pct = sector_changes.get(candidate.industry)
                if candidate.sector_change_pct is not None:
                    candidate.sector_heat_summary = f"所属板块当日 {candidate.sector_change_pct:+.2f}%"
                    if candidate.sector_change_pct >= 1.5:
                        candidate.catalyst_signals.append("市场情绪：板块强势")
                        candidate.catalyst_score += 3
            verification_status = self._confirm_with_sina(candidate) if market == "cn" else self._confirm_with_yfinance(candidate)
            candidate.verification_status = verification_status
            verified = verification_status in {"新浪已复核", "Yahoo Finance 已复核"}
            if verified:
                candidate.data_status = "东方财富快照 + 行情复核"
                if self._is_duplicate_industry(candidate.industry, featured_industries):
                    continue
                featured_industries.add(candidate.industry)
                self._attach_technical_context(candidate, market=market)
                self._attach_news_context(candidate)
                result.featured.append(candidate)
                featured.append(candidate)
                candidate_limit = limit * 3 if market == "cn" else limit
                if len(featured) >= candidate_limit:
                    break
                continue

            if verification_status in {"新浪暂不可用", "Yahoo Finance 暂不可用"}:
                result.sina_unavailable_count += 1
                if len(watches) >= watch_limit:
                    continue
                if self._is_duplicate_industry(
                    candidate.industry,
                    featured_industries | watch_industries,
                ):
                    continue
                watch_industries.add(candidate.industry)
                candidate.data_status = "全市场快照；单股行情暂不可用"
                candidate.entry_trigger = "待行情复核及技术面确认后再评估，不构成交易建议"
                candidate.invalidation_condition = "若后续复核价格与东财偏差超过 5%，停止跟踪"
                result.watchlist.append(candidate)
                watches.append(candidate)

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

    def _prefilter_us(self, df: pd.DataFrame, excluded: Set[str]) -> pd.DataFrame:
        required = ["代码", "名称", "最新价", "成交额", "市盈率"]
        if df is None or df.empty or any(col not in df.columns for col in required):
            return pd.DataFrame()
        work = df.copy()
        work["代码"] = work["代码"].map(self._us_ticker)
        work["名称"] = work["名称"].fillna("").astype(str).str.strip()
        for column in ["最新价", "涨跌幅", "成交额", "市盈率", "总市值"]:
            if column in work.columns:
                work[column] = pd.to_numeric(work[column], errors="coerce")
        mask = (
            work["代码"].map(lambda code: bool(code) and code.replace("-", "").isalnum())
            & ~work["代码"].isin({str(code).upper() for code in excluded})
            & work["最新价"].between(5, 1000, inclusive="both")
            & work["成交额"].ge(10000000)
            & work["市盈率"].between(3, 45, inclusive="both")
            & work["涨跌幅"].between(-7, 7, inclusive="both")
        )
        filtered = work.loc[mask].copy()
        if filtered.empty:
            return filtered
        filtered["_score"] = filtered.apply(self._score_us_row, axis=1)
        return filtered.sort_values("_score", ascending=False)

    @staticmethod
    def _us_ticker(value: Any) -> str:
        text = str(value or "").strip().upper()
        return text.split(".", 1)[-1] if "." in text else text

    def _row_to_candidate(self, row: pd.Series, *, market: str = "cn") -> Optional[ExternalLowPeCandidate]:
        code = normalize_stock_code(str(row.get("代码", "")).strip())
        if market == "us":
            code = self._us_ticker(row.get("代码", ""))
        name = str(row.get("名称", "") or "").strip()
        if not code or not name:
            return None
        pe = self._float(row.get("市盈率-动态" if market == "cn" else "市盈率"))
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
        if market == "us":
            reasons = [
                f"市盈率 {pe:.1f}" if pe is not None else "市盈率处于可筛选范围",
                f"成交额 {self._amount_yi(row.get('成交额'))} 亿" if self._float(row.get("成交额")) else "流动性达标",
                "待日线趋势确认",
            ]
        return ExternalLowPeCandidate(
            code=code,
            name=name,
            market=market,
            currency="USD" if market == "us" else "CNY",
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

    def _confirm_with_yfinance(self, candidate: ExternalLowPeCandidate) -> str:
        try:
            quote = self.us_fetcher.get_realtime_quote(candidate.code)
        except Exception as exc:
            logger.info("美股候选：Yahoo Finance 复核失败 %s: %s", candidate.code, exc)
            return "Yahoo Finance 暂不可用"
        if quote is None or not quote.price:
            return "Yahoo Finance 暂不可用"
        candidate.sina_price = quote.price
        candidate.sina_change_pct = quote.change_pct
        if candidate.price and abs(candidate.price - quote.price) / max(candidate.price, 0.01) > 0.08:
            return "Yahoo Finance 价格偏差过大"
        return "Yahoo Finance 已复核"

    def _attach_technical_context(self, candidate: ExternalLowPeCandidate, *, market: str = "cn") -> None:
        try:
            fetcher = self.us_fetcher if market == "us" else self.fetcher
            df = fetcher.get_daily_data(candidate.code, days=120)
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
        recent_news_dates: List[date] = []
        for result in getattr(response, "results", [])[:2]:
            title = str(getattr(result, "title", "") or "").strip()
            snippet = str(getattr(result, "snippet", "") or "").strip()
            text = title or snippet
            if text:
                candidate.positive_catalysts.append(text[:80])
            if candidate.market == "cn":
                for category in self._categorize_a_share_catalysts(f"{title} {snippet}"):
                    signal = f"{category}催化"
                    if signal not in candidate.catalyst_signals:
                        candidate.catalyst_signals.append(signal)
                        candidate.catalyst_score += 6 if category in {"政策", "未来盈利"} else 4
            published = self._parse_news_date(getattr(result, "published_date", None))
            if (
                self._is_heat_catalyst_news(text)
                and published is not None
                and 0 <= (date.today() - published).days <= 7
            ):
                recent_news_dates.append(published)
        self._apply_cn_reduce_timer(candidate, recent_news_dates)

    @staticmethod
    def _limit_per_market(
        candidates: List[ExternalLowPeCandidate],
        limit: int,
    ) -> List[ExternalLowPeCandidate]:
        selected: List[ExternalLowPeCandidate] = []
        for market in ("cn", "us"):
            market_candidates = [item for item in candidates if item.market == market]
            market_candidates.sort(key=lambda item: item.score + item.catalyst_score, reverse=True)
            selected.extend(market_candidates[:limit])
        return selected

    def _get_sector_changes(self) -> dict[str, float]:
        try:
            rankings = self.fetcher.get_sector_rankings(n=100)
        except Exception as exc:
            logger.info("外部候选：板块排行获取失败: %s", exc)
            return {}
        if not rankings:
            return {}
        changes: dict[str, float] = {}
        for group in rankings:
            for item in group or []:
                name = str(item.get("name", "") or "").strip()
                change = self._float(item.get("change_pct"))
                if name and change is not None:
                    changes[name] = change
        return changes

    @staticmethod
    def _parse_news_date(value: Any) -> Optional[date]:
        if not value:
            return None
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            return None if pd.isna(parsed) else parsed.date()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_heat_catalyst_news(text: str) -> bool:
        keywords = (
            "政策", "规划", "补贴", "资金", "主力", "北向", "融资",
            "情绪", "人气", "题材", "业绩", "盈利", "订单", "预增",
        )
        return any(keyword in str(text or "") for keyword in keywords)

    @staticmethod
    def _categorize_a_share_catalysts(text: str) -> List[str]:
        content = str(text or "")
        categories = {
            "政策": ("政策", "规划", "补贴", "监管", "试点"),
            "资金": ("资金", "主力", "北向", "融资", "增持"),
            "市场情绪": ("情绪", "人气", "题材", "热度", "涨停"),
            "未来盈利": ("业绩", "盈利", "订单", "预增", "利润", "营收"),
        }
        return [label for label, keywords in categories.items() if any(word in content for word in keywords)]

    @staticmethod
    def _apply_cn_reduce_timer(candidate: ExternalLowPeCandidate, recent_dates: List[date]) -> None:
        if candidate.market != "cn" or not recent_dates:
            return
        if (candidate.change_60d or 0) < 12 or (candidate.sector_change_pct or 0) < 1.5:
            return
        event_date = max(recent_dates)
        deadline = event_date + timedelta(days=7)
        days_left = (deadline - date.today()).days
        candidate.reduce_alert = (
            f"题材新闻后减仓时钟：{event_date.isoformat()} 触发，"
            f"建议在 {deadline.isoformat()} 前分批降低仓位（剩余 {max(days_left, 0)} 天）"
        )
        candidate.risk_alerts.insert(0, "板块强势且个股 60 日涨幅较大，警惕利好兑现")

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
    def _score_us_row(row: pd.Series) -> float:
        pe = ExternalLowPeCandidateService._float(row.get("市盈率")) or 45
        amount = ExternalLowPeCandidateService._float(row.get("成交额")) or 0
        market_cap = ExternalLowPeCandidateService._float(row.get("总市值")) or 0
        change = abs(ExternalLowPeCandidateService._float(row.get("涨跌幅")) or 0)
        return (
            max(0, 45 - pe) * 1.5
            + min(amount / 10000000, 20) * 1.5
            + min(market_cap / 1000000000, 50) * 0.3
            + max(0, 7 - change) * 1.2
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
