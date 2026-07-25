# -*- coding: utf-8 -*-
"""Orchestrate the persistent master-holding and recommendation radar."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from src.services.external_low_pe_candidates import (
    ExternalLowPeCandidate,
    ExternalLowPeCandidateService,
    ExternalLowPeScreeningResult,
)
from src.services.investor_holdings_database import HoldingsDatabaseUpdate, MasterInvestorHoldingsDatabase
from src.services.recommendation_lifecycle import (
    RecommendationLifecycleService,
    RecommendationLifecycleUpdate,
    build_lifecycle_recommendations,
)

logger = logging.getLogger(__name__)


_CHANGE_LABELS = {
    "new": "新增",
    "first_seen": "首次入库",
    "increase": "加仓/增持",
    "decrease": "减仓/减持",
    "exit": "清仓/退出",
    "reentered": "重新进入",
}
_LIFECYCLE_LABELS = {
    "active": "跟踪中",
    "profit_protection": "保护利润",
    "reduce": "减仓提醒",
    "exit": "退出提醒",
    "stop_loss": "止损复查",
    "take_profit": "止盈提醒",
    "watch": "待复查",
    "expired": "本轮结束",
}


@dataclass
class InvestmentRadarSnapshot:
    screening: ExternalLowPeScreeningResult
    holdings: HoldingsDatabaseUpdate
    lifecycle: RecommendationLifecycleUpdate
    observed_on: date

    @property
    def featured(self) -> list[ExternalLowPeCandidate]:
        return list(self.screening.featured)

    @property
    def watchlist(self) -> list[ExternalLowPeCandidate]:
        return list(self.screening.watchlist)

    @property
    def market_status(self) -> dict[str, str]:
        return dict(self.screening.market_status)

    def to_markdown(self) -> str:
        lines = [
            f"# 持续投研雷达 · {self.observed_on.isoformat()}",
            "",
            "## 推荐生命周期",
            "",
            (
                f"> 累计跟踪 {self.lifecycle.tracked_count} 条推荐；"
                f"当前跟踪 {len(self.lifecycle.active_records)} 条；"
                f"需处理 {len(self.lifecycle.alert_records)} 条。"
            ),
            "",
        ]
        self._append_lifecycle_lines(lines)
        lines.extend([
            "## 大佬持仓数据库",
            "",
            (
                f"> 跟踪 {self.holdings.investor_count} 位投资人、"
                f"{self.holdings.active_count} 条公开持仓观察；"
                "无新增公开证据时不会把缺失记录判定为退出。"
            ),
            "",
        ])
        self._append_holding_change_lines(lines)
        self._append_holding_snapshot_lines(lines)
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_on": self.observed_on.isoformat(),
            "screening_status": self.market_status,
            "featured": [_candidate_payload(item) for item in self.featured],
            "watchlist": [_candidate_payload(item) for item in self.watchlist],
            "holdings": self.holdings.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
        }

    def write_artifacts(self, report_dir: Optional[Path | str] = None) -> dict[str, Path]:
        target_dir = Path(report_dir) if report_dir else Path(__file__).resolve().parents[2] / "reports"
        target_dir.mkdir(parents=True, exist_ok=True)
        date_text = self.observed_on.strftime("%Y%m%d")
        markdown_path = target_dir / f"investment_radar_{date_text}.md"
        json_path = target_dir / "investment_radar_latest.json"
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {"markdown": markdown_path, "json": json_path}

    def _append_lifecycle_lines(self, lines: list[str]) -> None:
        alerts = self.lifecycle.alert_records[:8]
        lines.extend(["### 减仓/退出提醒", ""])
        if alerts:
            for item in alerts:
                status = _LIFECYCLE_LABELS.get(str(item.get("status") or ""), "复查")
                lines.append(
                    f"- **{item.get('name') or item.get('code')} · {item.get('code')}**：{status}；"
                    f"累计 {float(item.get('return_pct') or 0):+.1f}%；{item.get('alert') or '建议复查'}"
                )
        else:
            lines.append("> 暂无减仓、退出、止损或长期未复核提醒。")
        lines.extend(["", "### 持续跟踪", ""])
        active = self.lifecycle.active_records[:10]
        if active:
            for item in active:
                status = _LIFECYCLE_LABELS.get(str(item.get("status") or ""), "跟踪中")
                lines.append(
                    f"- **{item.get('name') or item.get('code')} · {item.get('code')}**：{status}；"
                    f"累计 {float(item.get('return_pct') or 0):+.1f}%；"
                    f"第 {len(item.get('observations') or [])} 次观察。"
                )
        else:
            lines.append("> 暂无可持续跟踪的已复核推荐。")
        lines.append("")

    def _append_holding_change_lines(self, lines: list[str]) -> None:
        lines.extend(["### 本次持仓变化", ""])
        changes = self.holdings.actionable_changes[:12]
        if not changes:
            lines.append("> 本次未发现有新公开证据支持的新增、加仓、减仓或退出变化。")
            lines.append("")
            return
        for item in changes:
            label = _CHANGE_LABELS.get(str(item.get("change_type") or ""), "更新")
            source = str(item.get("source_title") or "公开持仓观察")
            lines.append(
                f"- **{item.get('investor')} · {item.get('name')} · {item.get('code')}**：{label}；{source}"
            )
        lines.append("")

    def _append_holding_snapshot_lines(self, lines: list[str]) -> None:
        lines.extend(["### 最新持仓快照", ""])
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in self.holdings.holdings:
            if item.get("status") != "active":
                continue
            groups.setdefault(str(item.get("investor") or "其他"), []).append(item)
        if not groups:
            lines.append("> 暂无可用持仓快照。")
            lines.append("")
            return
        for investor, holdings in sorted(groups.items()):
            lines.extend([f"#### {investor}", ""])
            for item in holdings[:8]:
                source_date = str(item.get("last_source_date") or item.get("last_seen_on") or "")
                action = _CHANGE_LABELS.get(str(item.get("last_action") or ""), "持仓观察")
                lines.append(
                    f"- {item.get('name') or item.get('code')} · {item.get('code')}（{item.get('market')}）："
                    f"{action}，最近证据 {source_date or '未标注日期'}"
                )
            lines.append("")


class InvestmentRadarService:
    """Build a persistent investment radar from the existing external screen."""

    def __init__(
        self,
        *,
        search_service: Optional[Any] = None,
        candidate_service: Optional[ExternalLowPeCandidateService] = None,
        holdings_database: Optional[MasterInvestorHoldingsDatabase] = None,
        lifecycle_service: Optional[RecommendationLifecycleService] = None,
    ) -> None:
        self.candidate_service = candidate_service or ExternalLowPeCandidateService(search_service=search_service)
        self.holdings_database = holdings_database or MasterInvestorHoldingsDatabase()
        self.lifecycle_service = lifecycle_service or RecommendationLifecycleService()

    def run(
        self,
        stock_list: Sequence[str],
        analysis_results: Iterable[Any],
        *,
        observed_on: Optional[date] = None,
        limit: int = 3,
        watch_limit: int = 3,
    ) -> InvestmentRadarSnapshot:
        observed_on = observed_on or date.today()
        result_list = list(analysis_results or [])
        screening = self.candidate_service.screen_with_observations(
            stock_list,
            limit=limit,
            watch_limit=watch_limit,
        )
        observed_candidates = list(getattr(screening, "observed", None) or [])
        if not observed_candidates:
            observed_candidates = list(screening.featured) + list(screening.watchlist)
        holdings = self.holdings_database.update_from_candidates(
            observed_candidates,
            observed_on=observed_on,
        )
        lifecycle_candidates = self._lifecycle_candidates(
            screening.featured,
            observed_candidates,
        )
        lifecycle = self.lifecycle_service.update(
            build_lifecycle_recommendations(lifecycle_candidates, result_list),
            observed_on=observed_on,
        )
        snapshot = InvestmentRadarSnapshot(
            screening=screening,
            holdings=holdings,
            lifecycle=lifecycle,
            observed_on=observed_on,
        )
        logger.info(
            "持续投研雷达完成: 精选 %s，观察 %s，持仓 %s，生命周期提醒 %s",
            len(snapshot.featured),
            len(snapshot.watchlist),
            snapshot.holdings.active_count,
            len(snapshot.lifecycle.alert_records),
        )
        return snapshot

    @staticmethod
    def _lifecycle_candidates(
        featured: Iterable[Any],
        observed: Iterable[Any],
    ) -> list[Any]:
        """Keep an explicit master sell signal visible after it leaves the top picks.

        A holding can fall out of the limited recommendation list precisely because a
        trim or exit was reported. Keeping that observation in the lifecycle feed
        lets a previously recommended name emit its reduction/exit reminder.
        """
        selected = list(featured or [])
        selected_keys = {
            (
                str(getattr(item, "market", "") or "").lower(),
                str(getattr(item, "code", "") or "").upper(),
            )
            for item in selected
        }
        signal_words = (
            "减仓",
            "减持",
            "清仓",
            "退出",
            "卖出",
            "trimmed",
            "reduced",
            "exited",
            "sold",
        )
        for candidate in observed or []:
            key = (
                str(getattr(candidate, "market", "") or "").lower(),
                str(getattr(candidate, "code", "") or "").upper(),
            )
            if key in selected_keys:
                continue
            text = " ".join(
                [
                    *[str(value) for value in (getattr(candidate, "investor_actions", None) or [])],
                    str(getattr(candidate, "action_summary", "") or ""),
                    str(getattr(candidate, "reduce_alert", "") or ""),
                ]
            ).lower()
            if any(word.lower() in text for word in signal_words):
                selected.append(candidate)
                selected_keys.add(key)
        return selected


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    source_date = getattr(candidate, "source_date", None)
    return {
        "code": getattr(candidate, "code", ""),
        "name": getattr(candidate, "name", ""),
        "market": getattr(candidate, "market", ""),
        "price": getattr(candidate, "price", None),
        "change_pct": getattr(candidate, "change_pct", None),
        "score": getattr(candidate, "score", 0.0),
        "investors": list(getattr(candidate, "investors", None) or []),
        "investor_actions": list(getattr(candidate, "investor_actions", None) or []),
        "action_summary": getattr(candidate, "action_summary", ""),
        "source_date": source_date.isoformat() if isinstance(source_date, date) else str(source_date or ""),
        "source_titles": list(getattr(candidate, "source_titles", None) or []),
        "source_urls": list(getattr(candidate, "source_urls", None) or []),
        "data_status": getattr(candidate, "data_status", ""),
        "reasons": list(getattr(candidate, "reasons", None) or []),
        "entry_trigger": getattr(candidate, "entry_trigger", ""),
        "reduce_alert": getattr(candidate, "reduce_alert", ""),
        "catalyst_signals": list(getattr(candidate, "catalyst_signals", None) or []),
        "risk_alerts": list(getattr(candidate, "risk_alerts", None) or []),
    }
