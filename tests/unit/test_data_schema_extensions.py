"""Tests for the new data-loading config schema additions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import TextPromptConfig


def test_text_prompt_config_defaults() -> None:
    cfg = TextPromptConfig()
    assert cfg.mode == "present"
    assert cfg.negatives_per_image == 0
    assert cfg.k == 16


def test_text_prompt_config_k_bounded() -> None:
    with pytest.raises(ValidationError):
        TextPromptConfig(k=17)
    with pytest.raises(ValidationError):
        TextPromptConfig(k=0)


from esam3.config.schema import NormalizeConfig


def test_normalize_config_defaults() -> None:
    cfg = NormalizeConfig()
    assert cfg.mean == [0.485, 0.456, 0.406]
    assert cfg.std == [0.229, 0.224, 0.225]
    assert len(cfg.mean) == 3 and len(cfg.std) == 3


def test_normalize_config_validation_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[0.1, 0.2], std=[0.1, 0.1, 0.1])


def test_normalize_config_validation_rejects_nonpositive_std() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[0.1, 0.1, 0.1], std=[0.0, 0.1, 0.1])


def test_normalize_config_validation_rejects_mean_out_of_range() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[1.5, 0.1, 0.1], std=[0.1, 0.1, 0.1])
