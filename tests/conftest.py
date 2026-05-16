"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from esam3.data.coco import COCODataset
from esam3.tracking.noop import NoopTracker
from tests.fixtures.tiny_sam3_stub import TinySam3Stub

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tiny_coco_dir() -> Path:
    return FIXTURES / "tiny_coco"


@pytest.fixture
def tiny_coco_dataset(tiny_coco_dir: Path) -> COCODataset:
    """A COCODataset pointing at the tiny_coco fixture (bbox prompt mode)."""
    from esam3.config.schema import NormalizeConfig, TextPromptConfig
    from esam3.data.transforms import build_eval_transforms

    transforms = build_eval_transforms(
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def stub_model() -> TinySam3Stub:
    return TinySam3Stub()


@pytest.fixture
def noop_tracker() -> NoopTracker:
    return NoopTracker()
