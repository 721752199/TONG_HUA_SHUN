# -*- coding: utf-8 -*-
"""Tests for STOCK_LIST separator handling."""

from unittest.mock import patch

from src.services.stock_list_parser import parse_stock_list, serialize_stock_list, split_stock_list


def test_split_stock_list_accepts_common_copy_paste_separators() -> None:
    value = "600519，300750  hk00700;AAPL、7203.T\n005930.KS；002594"

    assert split_stock_list(value) == [
        "600519",
        "300750",
        "hk00700",
        "AAPL",
        "7203.T",
        "005930.KS",
        "002594",
    ]


def test_serialize_stock_list_uses_canonical_commas() -> None:
    assert serialize_stock_list("600519，300750\nAAPL") == "600519,300750,AAPL"


def test_parse_stock_list_resolves_chinese_names_and_deduplicates() -> None:
    assert parse_stock_list("贵州茅台，宁德时代,600519,AAPL") == [
        "600519",
        "300750",
        "AAPL",
    ]


@patch("src.services.name_to_code_resolver.resolve_name_to_code", return_value=None)
def test_parse_stock_list_skips_unresolved_chinese_names(mock_resolver) -> None:
    assert parse_stock_list("不存在股票名,600519") == ["600519"]
    mock_resolver.assert_called_once_with("不存在股票名")
