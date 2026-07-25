# -*- coding: utf-8 -*-
"""Persistent follow-up tracking for stock recommendations."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from src.services.investor_holdings_database import DEFAULT_RADAR_STORAGE_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LifecycleRecommendation:
    code: str
    name: str
    market: str
    source: str
    source_label: str
    price: float
    score: float = 0.0
    investors: tuple[str, ...] = ()
    reason: str = ""
    reduce_signal: str = ""
    exit_signal: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}|{self.market}|{self.code}"


@dataclass
class RecommendationLifecycleUpdate:
    records: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = ""

    @property
    def tracked_count(self) -> int:
        return len(self.records)

    @property
    def active_records(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.records
            if item.get("status") in {"active", "profit_protection"}
        ]

    @property
    def alert_records(self) -> list[dict[str, Any]]:
        alert_states = {"reduce", "exit", "stop_loss", "take_profit", "watch", "expired"}
        records = [item for item in self.records if item.get("status") in alert_states]
        priority = {"exit": 0, "stop_loss": 1, "reduce": 2, "take_profit": 3, "watch": 4, "expired": 5}
        return sorted(records, key=lambda item: (priority.get(item.get("status"), 9), item.get("code", "")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_at": self.updated_at,
            "tracked_count": self.tracked_count,
            "active_count": len(self.active_records),
            "alert_count": len(self.alert_records),
            "records": self.records,
            "events": self.events,
        }


class RecommendationLifecycleService:
    """Track recommendation performance without treating missing data as an exit."""

    _SCHEMA_VERSION = 1
    _EVENT_LIMIT = 1800
    _OBSERVATION_LIMIT = 120
    _STOP_LOSS_PCT = -8.0
    _PROFIT_PROTECTION_PCT = 25.0
    _TAKE_PROFIT_DRAWDOWN_PCT = -7.0
    _WATCH_AFTER_DAYS = 14
    _EXPIRE_AFTER_DAYS = 30

    def __init__(self, storage_dir: Optional[Path | str] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_RADAR_STORAGE_DIR
        self.path = self.storage_dir / "recommendation_lifecycle.json"

    def update(
        self,
        recommendations: Iterable[LifecycleRecommendation],
        *,
        observed_on: Optional[date] = None,
    ) -> RecommendationLifecycleUpdate:
        observed_on = observed_on or date.today()
        observed_text = observed_on.isoformat()
        payload = self._load()
        records_by_key = {
            str(item.get("key") or ""): dict(item)
            for item in payload.get("recommendations", [])
            if isinstance(item, dict) and item.get("key")
        }
        known_event_ids = {
            str(item.get("event_id") or "")
            for item in payload.get("events", [])
            if isinstance(item, dict)
        }
        events: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for recommendation in recommendations or []:
            if not self._valid_price(recommendation.price):
                continue
            key = recommendation.key
            seen_keys.add(key)
            previous = records_by_key.get(key)
            record, event = self._update_record(previous, recommendation, observed_on)
            records_by_key[key] = record
            if event and event["event_id"] not in known_event_ids:
                payload["events"].append(event)
                known_event_ids.add(event["event_id"])
                events.append(event)

        for key, record in records_by_key.items():
            if key in seen_keys:
                continue
            event = self._mark_stale(record, observed_on)
            if event and event["event_id"] not in known_event_ids:
                payload["events"].append(event)
                known_event_ids.add(event["event_id"])
                events.append(event)

        records = sorted(
            records_by_key.values(),
            key=lambda item: (
                self._status_priority(str(item.get("status") or "")),
                str(item.get("source") or ""),
                str(item.get("market") or ""),
                str(item.get("code") or ""),
            ),
        )
        payload["recommendations"] = records
        payload["events"] = payload["events"][-self._EVENT_LIMIT :]
        payload["updated_at"] = self._timestamp()
        self._save(payload)
        return RecommendationLifecycleUpdate(records=records, events=events, updated_at=payload["updated_at"])

    def _update_record(
        self,
        previous: Optional[dict[str, Any]],
        recommendation: LifecycleRecommendation,
        observed_on: date,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        observed_text = observed_on.isoformat()
        price = round(float(recommendation.price), 4)
        if previous is None:
            record = {
                "key": recommendation.key,
                "source": recommendation.source,
                "source_label": recommendation.source_label,
                "market": recommendation.market,
                "code": recommendation.code,
                "name": recommendation.name,
                "first_recommended_on": observed_text,
                "last_seen_on": observed_text,
                "entry_price": price,
                "latest_price": price,
                "peak_price": price,
                "trough_price": price,
                "return_pct": 0.0,
                "drawdown_from_peak_pct": 0.0,
                "score": round(float(recommendation.score or 0.0), 2),
                "investors": list(recommendation.investors),
                "reason": recommendation.reason,
                "status": "active",
                "alert": "",
                "observations": [],
            }
            self._append_observation(record, observed_text, price)
            self._apply_state(record, recommendation)
            return record, self._build_event(record, "recommended", observed_text, "首次纳入跟踪")

        record = dict(previous)
        record.update(
            {
                "name": recommendation.name or record.get("name", recommendation.code),
                "last_seen_on": observed_text,
                "latest_price": price,
                "peak_price": max(float(record.get("peak_price") or price), price),
                "trough_price": min(float(record.get("trough_price") or price), price),
                "score": round(float(recommendation.score or 0.0), 2),
                "investors": list(recommendation.investors) or list(record.get("investors") or []),
                "reason": recommendation.reason or record.get("reason", ""),
            }
        )
        self._append_observation(record, observed_text, price)
        previous_status = str(record.get("status") or "active")
        previous_alert = str(record.get("alert") or "")
        self._apply_state(record, recommendation)
        if record["status"] != previous_status or record["alert"] != previous_alert:
            return record, self._build_event(record, str(record["status"]), observed_text, str(record["alert"]))
        return record, None

    def _apply_state(self, record: dict[str, Any], recommendation: LifecycleRecommendation) -> None:
        entry_price = float(record.get("entry_price") or 0)
        latest_price = float(record.get("latest_price") or 0)
        peak_price = float(record.get("peak_price") or latest_price)
        return_pct = self._percent_change(latest_price, entry_price)
        drawdown = self._percent_change(latest_price, peak_price)
        record["return_pct"] = round(return_pct, 2)
        record["drawdown_from_peak_pct"] = round(drawdown, 2)

        if recommendation.exit_signal:
            record["status"] = "exit"
            record["alert"] = recommendation.exit_signal
        elif recommendation.reduce_signal:
            record["status"] = "reduce"
            record["alert"] = recommendation.reduce_signal
        elif return_pct <= self._STOP_LOSS_PCT:
            record["status"] = "stop_loss"
            record["alert"] = f"相对首次推荐已下跌 {abs(return_pct):.1f}%，触发止损复查"
        elif return_pct >= self._PROFIT_PROTECTION_PCT and drawdown <= self._TAKE_PROFIT_DRAWDOWN_PCT:
            record["status"] = "take_profit"
            record["alert"] = (
                f"累计上涨 {return_pct:.1f}%，较高点回撤 {abs(drawdown):.1f}%，建议分批止盈"
            )
        elif return_pct >= self._PROFIT_PROTECTION_PCT:
            record["status"] = "profit_protection"
            record["alert"] = f"累计上涨 {return_pct:.1f}%，建议上移保护线"
        else:
            record["status"] = "active"
            record["alert"] = ""

    def _mark_stale(self, record: dict[str, Any], observed_on: date) -> Optional[dict[str, Any]]:
        status = str(record.get("status") or "")
        if status in {"exit", "stop_loss", "take_profit", "expired"}:
            return None
        last_seen = self._parse_date(record.get("last_seen_on"))
        if last_seen is None:
            return None
        absent_days = (observed_on - last_seen).days
        previous_status = status
        if absent_days >= self._EXPIRE_AFTER_DAYS:
            record["status"] = "expired"
            record["alert"] = f"已连续 {absent_days} 日未再次推荐或复核，结束本轮跟踪"
        elif absent_days >= self._WATCH_AFTER_DAYS:
            record["status"] = "watch"
            record["alert"] = f"已连续 {absent_days} 日未再次推荐或复核，建议人工复查"
        else:
            return None
        if record["status"] == previous_status:
            return None
        return self._build_event(record, str(record["status"]), observed_on.isoformat(), str(record["alert"]))

    def _append_observation(self, record: dict[str, Any], observed_on: str, price: float) -> None:
        observations = list(record.get("observations") or [])
        observation = {
            "observed_on": observed_on,
            "price": round(float(price), 4),
            "return_pct": round(self._percent_change(price, float(record.get("entry_price") or price)), 2),
        }
        if observations and observations[-1].get("observed_on") == observed_on:
            observations[-1] = observation
        else:
            observations.append(observation)
        record["observations"] = observations[-self._OBSERVATION_LIMIT :]

    @staticmethod
    def _build_event(record: dict[str, Any], event_type: str, observed_on: str, detail: str) -> dict[str, Any]:
        material = "|".join([str(record.get("key") or ""), event_type, observed_on, detail])
        return {
            "event_id": hashlib.sha1(material.encode("utf-8")).hexdigest(),
            "observed_on": observed_on,
            "event_type": event_type,
            "source": record.get("source", ""),
            "market": record.get("market", ""),
            "code": record.get("code", ""),
            "name": record.get("name", ""),
            "return_pct": record.get("return_pct", 0.0),
            "detail": detail,
        }

    def _load(self) -> dict[str, Any]:
        default = {"schema_version": self._SCHEMA_VERSION, "updated_at": "", "recommendations": [], "events": []}
        if not self.path.exists():
            return default
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("读取推荐生命周期账本失败，将使用空账本: %s", exc)
            return default
        if not isinstance(payload, dict):
            return default
        payload.setdefault("schema_version", self._SCHEMA_VERSION)
        payload.setdefault("updated_at", "")
        payload.setdefault("recommendations", [])
        payload.setdefault("events", [])
        if not isinstance(payload["recommendations"], list):
            payload["recommendations"] = []
        if not isinstance(payload["events"], list):
            payload["events"] = []
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    @staticmethod
    def _valid_price(value: Any) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _percent_change(value: float, baseline: float) -> float:
        if baseline <= 0:
            return 0.0
        return (float(value) - baseline) / baseline * 100

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        try:
            return date.fromisoformat(str(value)[:10])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _status_priority(status: str) -> int:
        return {"exit": 0, "stop_loss": 1, "reduce": 2, "take_profit": 3, "watch": 4, "expired": 5}.get(status, 9)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_lifecycle_recommendations(
    external_candidates: Iterable[Any],
    analysis_results: Iterable[Any],
) -> list[LifecycleRecommendation]:
    """Build lifecycle inputs for featured external ideas and buy/add self-stock signals."""

    recommendations: list[LifecycleRecommendation] = []
    for candidate in external_candidates or []:
        price = _as_price(getattr(candidate, "price", None))
        code = _normalized_code(getattr(candidate, "code", ""))
        if price is None or not code:
            continue
        action_text = " ".join(str(item) for item in (getattr(candidate, "investor_actions", None) or []))
        action_text += " " + str(getattr(candidate, "action_summary", "") or "")
        exit_signal = ""
        reduce_signal = ""
        if any(word in action_text for word in ("清仓", "退出", "卖出")):
            exit_signal = "大师持仓出现清仓/退出信号，建议退出跟踪"
        elif any(word in action_text for word in ("减仓", "减持", "降低持仓")):
            reduce_signal = str(getattr(candidate, "reduce_alert", "") or "大师持仓出现减仓信号，建议减仓复查")
        elif getattr(candidate, "reduce_alert", ""):
            reduce_signal = str(getattr(candidate, "reduce_alert"))
        reasons = list(getattr(candidate, "reasons", None) or [])
        recommendations.append(
            LifecycleRecommendation(
                code=code,
                name=str(getattr(candidate, "name", "") or code),
                market=str(getattr(candidate, "market", "cn") or "cn"),
                source="external_master",
                source_label="外部大师雷达",
                price=price,
                score=float(getattr(candidate, "score", 0.0) or 0.0) + float(getattr(candidate, "catalyst_score", 0.0) or 0.0),
                investors=tuple(str(item) for item in (getattr(candidate, "investors", None) or []) if str(item).strip()),
                reason=str(reasons[0] if reasons else getattr(candidate, "action_summary", "") or ""),
                reduce_signal=reduce_signal,
                exit_signal=exit_signal,
            )
        )

    for result in analysis_results or []:
        if not getattr(result, "success", True):
            continue
        action = str(getattr(result, "action", "") or "").strip().lower()
        advice = str(getattr(result, "operation_advice", "") or "")
        if action not in {"buy", "add"} and not any(word in advice for word in ("买入", "加仓")):
            continue
        price = _as_price(getattr(result, "current_price", None))
        code = _normalized_code(getattr(result, "code", ""))
        if price is None or not code:
            continue
        recommendations.append(
            LifecycleRecommendation(
                code=code,
                name=str(getattr(result, "name", "") or code),
                market=_market_from_code(code),
                source="self_stock_signal",
                source_label="自选股信号",
                price=price,
                score=float(getattr(result, "sentiment_score", 0.0) or 0.0),
                reason=str(getattr(result, "buy_reason", "") or getattr(result, "analysis_summary", "") or advice),
            )
        )
    return recommendations


def _as_price(value: Any) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _normalized_code(value: Any) -> str:
    text = str(value or "").strip()
    return text.upper() if any(char.isalpha() for char in text) else text


def _market_from_code(code: str) -> str:
    upper = code.upper()
    if upper.startswith("HK"):
        return "hk"
    return "us" if any(char.isalpha() for char in upper) else "cn"
