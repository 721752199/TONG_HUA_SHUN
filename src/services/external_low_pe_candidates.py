# -*- coding: utf-8 -*-
"""External stock ideas driven by respected-investor holding changes.

The public API intentionally keeps the historical class names so the PushPlus
pipeline can evolve without a broad refactor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, List, Optional, Sequence, Set

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import normalize_stock_code
from data_provider.yfinance_fetcher import YfinanceFetcher
from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MasterTrackingIdea:
    code: str
    name: str
    market: str
    investors: tuple[str, ...]
    thesis: str
    base_action: str = "核心持仓观察"
    sector: str = ""


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
    opportunity_type: str = "大师持仓跟踪"
    verification_status: str = "待复核"
    data_status: str = "公开持仓/新闻线索 + 行情复核"
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
    investors: List[str] = field(default_factory=list)
    investor_actions: List[str] = field(default_factory=list)
    action_summary: str = ""
    source_titles: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    source_date: Optional[date] = None
    holding_confidence: str = "观察"


@dataclass
class ExternalLowPeScreeningResult:
    """Separate verified recommendations from unverified observation candidates."""

    featured: List[ExternalLowPeCandidate] = field(default_factory=list)
    watchlist: List[ExternalLowPeCandidate] = field(default_factory=list)
    prefiltered_count: int = 0
    sina_unavailable_count: int = 0
    market_status: dict[str, str] = field(default_factory=dict)


class ExternalLowPeCandidateService:
    """Find external candidates from investor-master holding and change signals."""

    _BUY_WORDS = (
        "新进", "新买入", "买入", "首次建仓", "建仓", "加仓", "增持", "大幅增持",
        "added", "new stake", "initiated", "increased", "boosted", "bought",
    )
    _SELL_WORDS = (
        "减仓", "减持", "卖出", "清仓", "退出", "降低持仓",
        "trimmed", "reduced", "sold", "cut stake", "exited",
    )
    _FILING_WORDS = ("13F", "持仓", "一季报", "半年报", "三季报", "年报", "portfolio", "holdings")

    # The static universe is deliberately small: it is a resilience layer when
    # live search is unavailable, not a replacement for current holding-change news.
    _TRACKED_IDEAS: tuple[MasterTrackingIdea, ...] = (
        MasterTrackingIdea("AAPL", "Apple", "us", ("巴菲特/伯克希尔", "段永平"), "消费电子生态和高质量现金流", "核心持仓观察", "科技消费"),
        MasterTrackingIdea("GOOGL", "Alphabet", "us", ("段永平", "景林资产", "高瓴/HHLR"), "AI 基础设施与搜索广告现金流", "名人重仓观察", "互联网"),
        MasterTrackingIdea("PDD", "PDD Holdings", "us", ("段永平", "景林资产"), "跨境电商和高经营效率", "名人重仓观察", "互联网电商"),
        MasterTrackingIdea("BRK-B", "Berkshire Hathaway", "us", ("段永平",), "复利型控股平台，跟随优秀资本配置", "长期核心观察", "金融控股"),
        MasterTrackingIdea("AXP", "American Express", "us", ("巴菲特/伯克希尔",), "品牌、支付网络和高端消费韧性", "核心持仓观察", "金融"),
        MasterTrackingIdea("OXY", "Occidental Petroleum", "us", ("巴菲特/伯克希尔",), "能源资产和回购/分红潜力", "加仓线索观察", "能源"),
        MasterTrackingIdea("000333", "美的集团", "cn", ("高毅资产/邓晓峰", "长期价值派"), "家电龙头、全球化和现金流质量", "A股价值龙头观察", "家电"),
        MasterTrackingIdea("002415", "海康威视", "cn", ("高毅资产/冯柳",), "安防和 AIoT 复苏弹性", "A股机构重仓观察", "计算机设备"),
        MasterTrackingIdea("601899", "紫金矿业", "cn", ("高毅资产/邓晓峰",), "铜金资源和全球矿业扩张", "资源龙头观察", "有色金属"),
        MasterTrackingIdea("002001", "新和成", "cn", ("重阳投资", "长期价值派"), "精细化工、维生素周期和盈利修复", "A股价值修复观察", "化工"),
        MasterTrackingIdea("000513", "丽珠集团", "cn", ("高瓴/HHLR", "长期价值派"), "创新药和稳定现金流组合", "医药价值观察", "医药"),
        MasterTrackingIdea("603986", "兆易创新", "cn", ("高毅资产", "长期成长派"), "国产半导体周期修复和存储弹性", "科技成长观察", "半导体"),
    )

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
        excluded = self._normalize_excluded(stock_list)
        ideas = [idea for idea in self._TRACKED_IDEAS if self._idea_code(idea) not in excluded]
        result = ExternalLowPeScreeningResult(prefiltered_count=len(ideas))
        self._screen_market(result, ideas, "cn", limit, watch_limit, prefilter_limit)
        self._screen_market(result, ideas, "us", limit, watch_limit, prefilter_limit)
        result.featured = self._limit_per_market(result.featured, limit)
        result.watchlist = self._limit_per_market(result.watchlist, watch_limit)
        return result

    def _screen_market(
        self,
        result: ExternalLowPeScreeningResult,
        ideas: Sequence[MasterTrackingIdea],
        market: str,
        limit: int,
        watch_limit: int,
        prefilter_limit: int,
    ) -> None:
        market_ideas = [idea for idea in ideas if idea.market == market][:prefilter_limit]
        if not market_ideas:
            result.market_status[market] = "无可用跟踪标的"
            return
        result.market_status[market] = f"跟踪 {len(market_ideas)} 个大师持仓标的"

        verified: List[ExternalLowPeCandidate] = []
        watches: List[ExternalLowPeCandidate] = []
        for idea in market_ideas:
            candidate = self._idea_to_candidate(idea)
            self._attach_investor_news(candidate)
            candidate.verification_status = (
                self._confirm_with_sina(candidate)
                if market == "cn"
                else self._confirm_with_yfinance(candidate)
            )
            is_verified = candidate.verification_status in {"新浪已复核", "Yahoo Finance 已复核"}
            if is_verified:
                candidate.data_status = "大师持仓/调仓线索 + 行情复核"
                self._attach_technical_context(candidate, market=market)
                candidate.score += self._action_score(candidate)
                verified.append(candidate)
            else:
                candidate.data_status = "大师持仓线索；单股行情暂不可用"
                candidate.entry_trigger = "等待行情复核后再纳入精选"
                candidate.invalidation_condition = "后续若确认大师已减仓/清仓，停止跟踪"
                watches.append(candidate)
                result.sina_unavailable_count += 1

        verified.sort(key=lambda item: item.score + item.catalyst_score, reverse=True)
        watches.sort(key=lambda item: item.score + item.catalyst_score, reverse=True)
        result.featured.extend(verified[: max(limit * 2, limit)])
        result.watchlist.extend(watches[:watch_limit])
        selected_count = len([item for item in verified if item.market == market])
        if selected_count:
            result.market_status[market] = f"大师持仓跟踪 {len(market_ideas)} 个，复核通过 {selected_count} 个"
        else:
            result.market_status[market] = f"大师持仓跟踪 {len(market_ideas)} 个，行情复核暂未通过"

    def _idea_to_candidate(self, idea: MasterTrackingIdea) -> ExternalLowPeCandidate:
        investors = list(idea.investors)
        candidate = ExternalLowPeCandidate(
            code=self._idea_code(idea),
            name=idea.name,
            market=idea.market,
            currency="USD" if idea.market == "us" else "CNY",
            industry=idea.sector,
            opportunity_type=idea.base_action,
            investors=investors,
            investor_actions=[idea.base_action],
            action_summary=f"{'、'.join(investors)}：{idea.base_action}",
            holding_confidence="公开持仓观察",
            score=58.0 + min(len(investors), 3) * 6,
            reasons=[
                f"跟踪对象：{'、'.join(investors)}",
                idea.thesis,
                "优先关注新增/加仓；减仓新闻触发风险提示",
            ],
        )
        if idea.market == "cn":
            candidate.catalyst_signals.append("A股机构持仓线索")
        else:
            candidate.catalyst_signals.append("13F/公开持仓线索")
        candidate.catalyst_score += min(len(investors), 3) * 3
        return candidate

    def _attach_investor_news(self, candidate: ExternalLowPeCandidate) -> None:
        service = self.search_service
        if service is None or not getattr(service, "is_available", False):
            candidate.risk_alerts.append("新闻搜索未启用，本次使用公开持仓观察池")
            return

        keywords = self._news_keywords(candidate)
        try:
            response = service.search_stock_news(
                candidate.code,
                candidate.name,
                max_results=4,
                focus_keywords=keywords,
            )
        except Exception as exc:
            logger.info("大师持仓新闻搜索失败 %s: %s", candidate.code, exc)
            candidate.risk_alerts.append("大师调仓新闻搜索失败，先按公开持仓观察")
            return
        if not getattr(response, "success", False):
            candidate.risk_alerts.append("大师调仓新闻暂无结果，先按公开持仓观察")
            return

        buy_hits = 0
        sell_hits = 0
        dates: List[date] = []
        for item in getattr(response, "results", [])[:4]:
            title = str(getattr(item, "title", "") or "").strip()
            snippet = str(getattr(item, "snippet", "") or "").strip()
            text = f"{title} {snippet}"
            if not text.strip():
                continue
            url = str(getattr(item, "url", "") or "").strip()
            published = self._parse_news_date(getattr(item, "published_date", None))
            if published:
                dates.append(published)
            candidate.source_titles.append((title or snippet)[:90])
            if url:
                candidate.source_urls.append(url)
            lower_text = text.lower()
            if any(word.lower() in lower_text for word in self._BUY_WORDS):
                buy_hits += 1
            if any(word.lower() in lower_text for word in self._SELL_WORDS):
                sell_hits += 1
            if any(word.lower() in lower_text for word in self._FILING_WORDS):
                candidate.catalyst_score += 2

        if dates:
            candidate.source_date = max(dates)
        if buy_hits:
            candidate.investor_actions.append("新闻出现新增/加仓信号")
            candidate.catalyst_signals.append("大师新增/加仓")
            candidate.positive_catalysts.append("近期新闻检索到新增/加仓相关线索")
            candidate.catalyst_score += buy_hits * 9
            candidate.score += buy_hits * 6
            candidate.holding_confidence = "新增/加仓优先"
        if sell_hits:
            candidate.investor_actions.append("新闻出现减仓/清仓信号")
            candidate.risk_alerts.insert(0, "近期新闻检索到减仓/清仓线索，谨慎追高")
            candidate.catalyst_score -= sell_hits * 8
            candidate.score -= sell_hits * 10
            candidate.holding_confidence = "减仓风险观察"
            self._apply_reduce_alert(candidate)
        if candidate.source_titles:
            candidate.positive_catalysts.extend(candidate.source_titles[:2])
        candidate.action_summary = "；".join(candidate.investor_actions[:3])

    def _news_keywords(self, candidate: ExternalLowPeCandidate) -> List[str]:
        investors = list(getattr(candidate, "investors", []) or [])
        if candidate.market == "us":
            return investors + [
                candidate.name,
                candidate.code,
                "13F",
                "portfolio",
                "new stake",
                "increased",
                "trimmed",
            ]
        return investors + [
            candidate.name,
            candidate.code,
            "持仓",
            "新进",
            "加仓",
            "减仓",
            "十大流通股东",
        ]

    def _confirm_with_sina(self, candidate: ExternalLowPeCandidate) -> str:
        try:
            quote = self.fetcher.get_realtime_quote(candidate.code, source="sina")
        except Exception as exc:
            logger.info("A股大师候选新浪复核失败 %s: %s", candidate.code, exc)
            return "新浪暂不可用"
        if quote is None or not quote.price:
            return "新浪暂不可用"
        candidate.sina_price = quote.price
        candidate.sina_change_pct = quote.change_pct
        candidate.price = quote.price
        candidate.change_pct = quote.change_pct
        candidate.name = candidate.name or getattr(quote, "name", candidate.code)
        return "新浪已复核"

    def _confirm_with_yfinance(self, candidate: ExternalLowPeCandidate) -> str:
        try:
            quote = self.us_fetcher.get_realtime_quote(candidate.code)
        except Exception as exc:
            logger.info("美股大师候选 Yahoo Finance 复核失败 %s: %s", candidate.code, exc)
            return "Yahoo Finance 暂不可用"
        if quote is None or not quote.price:
            return "Yahoo Finance 暂不可用"
        candidate.sina_price = quote.price
        candidate.sina_change_pct = quote.change_pct
        candidate.price = quote.price
        candidate.change_pct = quote.change_pct
        candidate.amount = getattr(quote, "amount", None)
        return "Yahoo Finance 已复核"

    def _attach_technical_context(self, candidate: ExternalLowPeCandidate, *, market: str) -> None:
        try:
            fetcher = self.us_fetcher if market == "us" else self.fetcher
            df = fetcher.get_daily_data(candidate.code, days=120)
            trend = self.trend_analyzer.analyze(df, candidate.code)
        except Exception as exc:
            logger.info("大师候选技术面分析失败 %s: %s", candidate.code, exc)
            candidate.entry_trigger = "等待技术面数据确认后分批观察"
            candidate.invalidation_condition = "出现明确调仓减持或跌破中期趋势时停止跟踪"
            return
        candidate.technical_summary = self._format_technical_summary(trend)
        candidate.entry_trigger, candidate.invalidation_condition = self._build_execution_plan(trend)
        for risk in getattr(trend, "risk_factors", [])[:2]:
            text = self._strip_prefix(risk)
            if text:
                candidate.risk_alerts.append(text)
        trend_strength = self._float(getattr(trend, "trend_strength", None))
        if trend_strength is not None:
            candidate.score += max(0, min(trend_strength, 100) - 50) * 0.35

    @staticmethod
    def _format_technical_summary(trend: TrendAnalysisResult) -> str:
        parts = [
            str(getattr(getattr(trend, "trend_status", None), "value", "") or ""),
            f"趋势强度 {float(getattr(trend, 'trend_strength', 0) or 0):.0f}/100",
            str(getattr(getattr(trend, "buy_signal", None), "value", "") or ""),
        ]
        ma_alignment = str(getattr(trend, "ma_alignment", "") or "")
        if ma_alignment:
            parts.append(ma_alignment)
        for reason in list(getattr(trend, "signal_reasons", []) or [])[:2]:
            if reason:
                parts.append(str(reason))
        return "；".join(part for part in parts if part)

    @staticmethod
    def _build_execution_plan(trend: TrendAnalysisResult) -> tuple[str, str]:
        ma5 = float(getattr(trend, "ma5", 0) or 0)
        ma10 = float(getattr(trend, "ma10", 0) or 0)
        ma20 = float(getattr(trend, "ma20", 0) or 0)
        support = ma10 if getattr(trend, "support_ma10", False) and ma10 > 0 else ma5
        if support > 0:
            entry = f"回踩 {support:.2f} 附近企稳，且没有新增减仓消息时再分批观察"
        elif ma5 > 0:
            entry = f"站稳 MA5 {ma5:.2f} 且调仓消息偏正面后再观察"
        else:
            entry = "等待技术面和调仓新闻进一步确认"

        if ma20 > 0:
            invalidation = f"收盘跌破 MA20 {ma20:.2f}，或出现大师减仓/清仓消息"
        else:
            invalidation = "趋势转弱或出现大师减仓/清仓消息"
        return entry, invalidation

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

    @staticmethod
    def _action_score(candidate: ExternalLowPeCandidate) -> float:
        score = 0.0
        actions = " ".join(candidate.investor_actions)
        if any(word in actions for word in ("新增", "加仓", "增持", "new", "increased")):
            score += 18
        if any(word in actions for word in ("减仓", "清仓", "减持", "trimmed", "reduced")):
            score -= 20
        if candidate.source_date and (date.today() - candidate.source_date).days <= 14:
            score += 8
        return score

    @staticmethod
    def _apply_reduce_alert(candidate: ExternalLowPeCandidate) -> None:
        trigger = candidate.source_date or date.today()
        deadline = trigger + timedelta(days=7)
        days_left = max((deadline - date.today()).days, 0)
        candidate.reduce_alert = (
            f"{trigger.isoformat()} 出现减仓/清仓线索，建议 {deadline.isoformat()} 前复查仓位，"
            f"剩余 {days_left} 天"
        )

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
    def _idea_code(idea: MasterTrackingIdea) -> str:
        return idea.code.upper() if idea.market == "us" else normalize_stock_code(idea.code)

    @staticmethod
    def _normalize_excluded(stock_list: Iterable[str]) -> Set[str]:
        excluded: Set[str] = set()
        for code in stock_list or []:
            text = str(code or "").strip()
            if not text:
                continue
            excluded.add(normalize_stock_code(text))
            excluded.add(text.upper())
        return excluded

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _strip_prefix(value: Any) -> str:
        return str(value or "").strip().lstrip("⚠️ ").strip()
