"""Tests for data/transforms.py."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from esam3.config.schema import NormalizeConfig
from esam3.data.transforms import build_eval_transforms, resolve_normalization


@contextmanager
def _patch_proc_to_imagenet() -> Iterator[None]:
    """Patch transformers.AutoImageProcessor so resolve_normalization falls back to ImageNet defaults."""
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")
    with patch("transformers.AutoImageProcessor", mock_aip):
        yield


def test_resolve_normalization_uses_image_processor_when_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_proc = SimpleNamespace(image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = fake_proc

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="esam3.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    mock_aip.from_pretrained.assert_called_once_with(
        "facebook/sam3.1", local_files_only=True
    )
    assert mean == [0.1, 0.2, 0.3]
    assert std == [0.4, 0.5, 0.6]
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )


def test_resolve_normalization_falls_back_on_oserror(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_aip = MagicMock()
    mock_aip.from_pretrained.side_effect = OSError("no cache")

    with patch("transformers.AutoImageProcessor", mock_aip):
        caplog.set_level(logging.INFO, logger="esam3.data.transforms")
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
    assert any(
        re.search(r"AutoImageProcessor cache miss", rec.message) for rec in caplog.records
    )


def test_resolve_normalization_falls_back_on_attribute_error() -> None:
    mock_aip = MagicMock()
    mock_aip.from_pretrained.return_value = SimpleNamespace()  # missing image_mean/image_std

    with patch("transformers.AutoImageProcessor", mock_aip):
        mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())

    assert mean == [0.485, 0.456, 0.406]


def test_eval_transforms_resizes_to_square() -> None:
    with _patch_proc_to_imagenet():
        compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    masks = [np.ones((40, 80), dtype=np.uint8)]
    out = compose(image=img, bboxes=[[0.0, 0.0, 80.0, 40.0]], masks=masks, class_labels=[0])
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 64, 64)
    assert out["image"].dtype == torch.float32
    bx = out["bboxes"][0]
    assert 0 <= bx[0] <= 1 and 0 <= bx[1] <= 1
    assert 60 <= bx[2] <= 64 and 28 <= bx[3] <= 36
    assert out["masks"][0].shape == (64, 64)


def test_eval_transforms_pad_position_top_left() -> None:
    """The right/bottom region should be zero-padded (top-left preserves original)."""
    with _patch_proc_to_imagenet():
        compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
    img = np.full((32, 64, 3), 255, dtype=np.uint8)
    out = compose(image=img, bboxes=[], masks=[], class_labels=[])
    top_row = out["image"][0, 0, :]
    bottom_row = out["image"][0, 60, :]
    assert top_row.mean().item() > 0
    assert bottom_row.mean().item() < 0
