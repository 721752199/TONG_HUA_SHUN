# -*- coding: utf-8 -*-
"""Helpers for parsing the user-facing STOCK_LIST value."""

from __future__ import annotations

import re
import logging
from typing import List

_STOCK_LIST_SEPARATOR_RE = re.compile(r"[\s,;\uFF0C\u3001\uFF1B]+")
logger = logging.getLogger(__name__)


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= ch <= "\u9fff" for ch in text or "")


def split_stock_list(value: str) -> List[str]:
    """Split STOCK_LIST values on common copy/paste separators."""
    return [
        item.strip()
        for item in _STOCK_LIST_SEPARATOR_RE.split(value or "")
        if item.strip()
    ]


def serialize_stock_list(value: str) -> str:
    """Return STOCK_LIST in the canonical comma-separated storage form."""
    return ",".join(parse_stock_list(value))


def parse_stock_list(value: str) -> List[str]:
    """Parse STOCK_LIST into normalized codes, resolving Chinese names when possible."""
    # Keep ordinary ticker/code input on the local fast path. The name resolver
    # can use AkShare for unknown Chinese names, so only call it when needed.
    from src.services.name_to_code_resolver import resolve_name_to_code
    from src.services.stock_code_utils import normalize_code

    parsed: List[str] = []
    seen = set()
    for item in split_stock_list(value):
        resolved = normalize_code(item) or resolve_name_to_code(item)
        if not resolved:
            if _contains_cjk(item):
                logger.warning("跳过无法识别的自选股中文名称: %s", item)
                continue
            resolved = item.strip().upper()
        code = resolved.strip().upper()
        if code and code not in seen:
            parsed.append(code)
            seen.add(code)
    return parsed
