# -*- coding: utf-8 -*-
"""Persistent public-observation ledger for master-investor holdings."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


DEFAULT_RADAR_STORAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "investment_radar"


@dataclass
class HoldingsDatabaseUpdate:
    """The current master-holding snapshot and the changes observed this run."""

    holdings: list[dict[str, Any]] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = ""

    @property
    def active_count(self) -> int:
        return sum(1 for item in self.holdings if item.get("status") == "active")

    @property
    def investor_count(self) -> int:
        return len({str(item.get("investor") or "") for item in self.holdings if item.get("investor")})

    @property
    def actionable_changes(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.changes
            if item.get("change_type") in {"new", "increase", "decrease", "exit", "reentered"}
        ]

    def change_count(self, change_type: str) -> int:
        return sum(1 for item in self.changes if item.get("change_type") == change_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_at": self.updated_at,
            "active_count": self.active_count,
            "investor_count": self.investor_count,
            "holdings": self.holdings,
            "changes": self.changes,
        }


class MasterInvestorHoldingsDatabase:
    """Keep an evidence-backed ledger of public master-investor observations.

    A missing observation never becomes an exit. Exits are recorded only when a
    public source explicitly signals selling, trimming, or clearing a position.
    """

    _SCHEMA_VERSION = 1
    _EVENT_LIMIT = 1200
    _ACTION_WORDS = {
        "exit": ("清仓", "退出", "卖出", "exited", "sold", "closed position"),
        "decrease": ("减仓", "减持", "降低持仓", "trimmed", "reduced", "cut stake"),
        "new": ("新进", "新买入", "首次建仓", "new stake", "initiated"),
        "increase": ("加仓", "增持", "买入", "increased", "added", "boosted", "bought"),
    }
    _INVESTOR_ALIASES = (
        ("巴菲特/伯克希尔", ("巴菲特", "伯克希尔", "buffett", "berkshire")),
        ("段永平", ("段永平", "duan yongping")),
        ("高毅资产", ("高毅", "邓晓峰", "冯柳", "gaoyi")),
        ("景林资产", ("景林", "jinglin")),
        ("高瓴/HHLR", ("高瓴", "高领", "hhlr", "hillhouse")),
    )

    def __init__(self, storage_dir: Optional[Path | str] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_RADAR_STORAGE_DIR
        self.path = self.storage_dir / "master_investor_holdings.json"

    def update_from_candidates(
        self,
        candidates: Iterable[Any],
        *,
        observed_on: Optional[date] = None,
    ) -> HoldingsDatabaseUpdate:
        observed_on = observed_on or date.today()
        payload = self._load()
        records_by_key = {
            str(item.get("key") or ""): dict(item)
            for item in payload.get("holdings", [])
            if isinstance(item, dict) and item.get("key")
        }
        known_event_ids = {
            str(item.get("event_id") or "")
            for item in payload.get("events", [])
            if isinstance(item, dict)
        }
        changes: list[dict[str, Any]] = []

        for candidate in candidates or []:
            code = self._normalized_code(getattr(candidate, "code", ""))
            market = str(getattr(candidate, "market", "cn") or "cn").lower()
            if not code:
                continue
            name = str(getattr(candidate, "name", "") or code).strip()
            investors = list(getattr(candidate, "investors", None) or [])
            if not investors:
                continue
            action = self._infer_action(candidate)
            evidence = self._build_evidence(candidate, action)
            for investor_raw in investors:
                investor = self._canonical_investor(investor_raw)
                if not investor:
                    continue
                key = f"{investor}|{market}|{code}"
                previous = records_by_key.get(key)
                record, change = self._merge_observation(
                    previous,
                    key=key,
                    investor=investor,
                    market=market,
                    code=code,
                    name=name,
                    action=action,
                    evidence=evidence,
                    observed_on=observed_on,
                )
                records_by_key[key] = record
                if change and change["event_id"] not in known_event_ids:
                    payload["events"].append(change)
                    known_event_ids.add(change["event_id"])
                    changes.append(change)

        holdings = sorted(
            records_by_key.values(),
            key=lambda item: (
                item.get("investor", ""),
                item.get("status") != "active",
                item.get("market", ""),
                item.get("code", ""),
            ),
        )
        payload["holdings"] = holdings
        payload["events"] = payload["events"][-self._EVENT_LIMIT :]
        payload["updated_at"] = self._timestamp()
        self._save(payload)
        return HoldingsDatabaseUpdate(
            holdings=holdings,
            changes=changes,
            updated_at=payload["updated_at"],
        )

    def _merge_observation(
        self,
        previous: Optional[dict[str, Any]],
        *,
        key: str,
        investor: str,
        market: str,
        code: str,
        name: str,
        action: str,
        evidence: dict[str, str],
        observed_on: date,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        observed_text = observed_on.isoformat()
        source_fingerprint = evidence["fingerprint"]
        if previous is None:
            status = "exited" if action == "exit" else "active"
            record = {
                "key": key,
                "investor": investor,
                "market": market,
                "code": code,
                "name": name,
                "status": status,
                "first_seen_on": observed_text,
                "last_seen_on": observed_text,
                "last_change_on": observed_text,
                "last_action": action,
                "last_source_date": evidence["source_date"],
                "source_title": evidence["source_title"],
                "source_url": evidence["source_url"],
                "evidence_fingerprint": source_fingerprint,
                "confidence": evidence["confidence"],
            }
            change_type = "new" if action == "new" else "first_seen"
            if action in {"increase", "decrease", "exit"}:
                change_type = action
            return record, self._build_event(record, change_type, action, evidence, observed_text)

        record = dict(previous)
        previous_status = str(record.get("status") or "active")
        previous_action = str(record.get("last_action") or "hold")
        previous_fingerprint = str(record.get("evidence_fingerprint") or "")
        record.update(
            {
                "name": name or record.get("name", code),
                "last_seen_on": observed_text,
                "last_source_date": evidence["source_date"] or record.get("last_source_date", ""),
                "source_title": evidence["source_title"] or record.get("source_title", ""),
                "source_url": evidence["source_url"] or record.get("source_url", ""),
                "confidence": evidence["confidence"] or record.get("confidence", ""),
            }
        )

        change_type: Optional[str] = None
        if action == "exit":
            record["status"] = "exited"
            change_type = "exit" if previous_status != "exited" or previous_fingerprint != source_fingerprint else None
        elif previous_status == "exited" and action in {"new", "increase"}:
            record["status"] = "active"
            change_type = "reentered"
        elif previous_status == "exited":
            # A static watchlist entry is not evidence that a reported exit was reversed.
            record["status"] = "exited"
        else:
            record["status"] = "active"
            if action != "hold" and (
                action != previous_action or previous_fingerprint != source_fingerprint
            ):
                change_type = action

        if action != "hold":
            record["last_action"] = action
        record["evidence_fingerprint"] = source_fingerprint or previous_fingerprint
        if change_type:
            record["last_change_on"] = observed_text
            return record, self._build_event(record, change_type, action, evidence, observed_text)
        return record, None

    def _build_event(
        self,
        record: dict[str, Any],
        change_type: str,
        action: str,
        evidence: dict[str, str],
        observed_on: str,
    ) -> dict[str, Any]:
        event_material = "|".join(
            [record["key"], change_type, action, observed_on, evidence["fingerprint"]]
        )
        event_id = hashlib.sha1(event_material.encode("utf-8")).hexdigest()
        return {
            "event_id": event_id,
            "observed_on": observed_on,
            "source_date": evidence["source_date"],
            "change_type": change_type,
            "action": action,
            "investor": record["investor"],
            "market": record["market"],
            "code": record["code"],
            "name": record["name"],
            "source_title": evidence["source_title"],
            "source_url": evidence["source_url"],
            "confidence": evidence["confidence"],
        }

    def _build_evidence(self, candidate: Any, action: str) -> dict[str, str]:
        titles = [str(item).strip() for item in (getattr(candidate, "source_titles", None) or []) if str(item).strip()]
        urls = [str(item).strip() for item in (getattr(candidate, "source_urls", None) or []) if str(item).strip()]
        source_date = self._date_text(getattr(candidate, "source_date", None))
        confidence = str(getattr(candidate, "holding_confidence", "") or "公开持仓观察").strip()
        source_title = titles[0] if titles else ""
        source_url = urls[0] if urls else ""
        material = "|".join([action, source_date, source_title, source_url, confidence])
        return {
            "source_date": source_date,
            "source_title": source_title,
            "source_url": source_url,
            "confidence": confidence,
            "fingerprint": hashlib.sha1(material.encode("utf-8")).hexdigest(),
        }

    def _infer_action(self, candidate: Any) -> str:
        parts = [
            *[str(item) for item in (getattr(candidate, "investor_actions", None) or [])],
            str(getattr(candidate, "action_summary", "") or ""),
            *[str(item) for item in (getattr(candidate, "source_titles", None) or [])],
        ]
        text = " ".join(parts).lower()
        for action in ("exit", "decrease", "new", "increase"):
            if any(word.lower() in text for word in self._ACTION_WORDS[action]):
                return action
        return "hold"

    def _load(self) -> dict[str, Any]:
        default = {"schema_version": self._SCHEMA_VERSION, "updated_at": "", "holdings": [], "events": []}
        if not self.path.exists():
            return default
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("读取大佬持仓数据库失败，将使用空账本: %s", exc)
            return default
        if not isinstance(loaded, dict):
            return default
        loaded.setdefault("schema_version", self._SCHEMA_VERSION)
        loaded.setdefault("updated_at", "")
        loaded.setdefault("holdings", [])
        loaded.setdefault("events", [])
        if not isinstance(loaded["holdings"], list):
            loaded["holdings"] = []
        if not isinstance(loaded["events"], list):
            loaded["events"] = []
        return loaded

    def _save(self, payload: dict[str, Any]) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    @classmethod
    def _canonical_investor(cls, value: Any) -> str:
        raw = str(value or "").strip()
        lowered = raw.lower()
        for canonical, aliases in cls._INVESTOR_ALIASES:
            if any(alias.lower() in lowered for alias in aliases):
                return canonical
        return raw

    @staticmethod
    def _normalized_code(value: Any) -> str:
        text = str(value or "").strip()
        return text.upper() if any(char.isalpha() for char in text) else text

    @staticmethod
    def _date_text(value: Any) -> str:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        text = str(value or "").strip()
        return text[:10] if len(text) >= 10 else text

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
