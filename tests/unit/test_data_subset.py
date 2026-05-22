"""Tests for data/subset.py — schema validation, resolve_subset_indices, SubsetDataset."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from custom_sam_peft.config.schema import LimitConfig

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("train", None),
        ("val", None),
        ("train", 1),
        ("train", 64),
        ("val", 100),
        ("train", 0.5),
        ("train", 1.0),
        ("val", 0.01),
    ],
)
def test_limit_config_valid(field: str, value: object) -> None:
    cfg = LimitConfig(**{field: value})
    assert getattr(cfg, field) == value


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("train", True, "bool"),
        ("train", False, "bool"),
        ("val", True, "bool"),
        ("train", 0, "int"),
        ("train", -1, "int"),
        ("val", 0, "int"),
        ("train", 0.0, "float"),
        ("train", -0.1, "float"),
        ("train", 1.1, "float"),
        ("val", 1.5, "float"),
    ],
)
def test_limit_config_invalid(field: str, value: object, match: str) -> None:
    with pytest.raises(ValidationError):
        LimitConfig(**{field: value})


def test_limit_config_defaults() -> None:
    cfg = LimitConfig()
    assert cfg.train is None
    assert cfg.val is None
    assert cfg.seed == 42
    assert cfg.strategy == "random"


def test_limit_config_strategy_valid() -> None:
    for s in ("random", "stratified", "first_n"):
        cfg = LimitConfig(strategy=s)  # type: ignore[arg-type]
        assert cfg.strategy == s
