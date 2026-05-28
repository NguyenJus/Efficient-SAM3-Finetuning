"""Tests for src/custom_sam_peft/cli/_config_rewrite.py — in-place line-surgery helper."""

from __future__ import annotations

from pathlib import Path

import yaml

from custom_sam_peft.cli._config_rewrite import _rewrite_sizing_block
from custom_sam_peft.config.loader import load_config


def _write_config_with_comments(path: Path) -> None:
    """Write a config YAML that includes unrelated comments in multiple sections."""
    content = """\
# This is the top-level comment — must survive rewrite
run:
  name: test-run
  output_dir: ./runs
  seed: 42

# model section comment
model:
  name: facebook/sam3.1
  local_dir: models/sam3.1
  checkpoint_file: sam3.1_multiplex.pt
  dtype: bfloat16  # original dtype comment

data:
  format: coco
  train:
    annotations: data/train.json
    images: data/train/

peft:
  # peft section comment
  method: lora  # original method comment
  r: 16  # original r comment
  alpha: 32
  dropout: 0.05

train:
  epochs: 10
  batch_size: 1  # original batch_size comment
  grad_accum_steps: 8  # original grad_accum_steps comment
  optimizer: auto
  learning_rate: 1.0e-4
  multiplex:
    classes_per_forward: 16

tracking:
  backend: none
"""
    path.write_text(content)


def test_rewrite_sizing_block_annotation_present(tmp_path: Path) -> None:
    """The annotation comment appears in the rewritten file."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    body = cfg_path.read_text()
    assert "# calibrated 2026-05-28" in body


def test_rewrite_sizing_block_values_changed(tmp_path: Path) -> None:
    """The sized fields are updated to the new values."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    parsed = yaml.safe_load(cfg_path.read_text())
    assert parsed["peft"]["method"] == "qlora"
    assert parsed["peft"]["r"] == 8
    assert parsed["train"]["batch_size"] == 2
    assert parsed["train"]["grad_accum_steps"] == 4
    assert parsed["model"]["dtype"] == "float16"


def test_rewrite_sizing_block_unrelated_lines_survive(tmp_path: Path) -> None:
    """Comments and lines unrelated to the sized fields are preserved."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    body = cfg_path.read_text()
    # Top-level comment must survive
    assert "# This is the top-level comment" in body
    # model section comment must survive
    assert "# model section comment" in body
    # peft section comment must survive
    assert "# peft section comment" in body
    # Unrelated fields must survive
    assert "alpha: 32" in body
    assert "dropout: 0.05" in body
    assert "learning_rate: 1.0e-4" in body
    assert "epochs: 10" in body


def test_rewrite_sizing_block_still_parses_via_load_config(tmp_path: Path) -> None:
    """After rewrite, the file is still a valid TrainConfig."""
    cfg_path = tmp_path / "config.yaml"
    _write_config_with_comments(cfg_path)

    _rewrite_sizing_block(
        cfg_path,
        method="qlora",
        r=8,
        batch_size=2,
        grad_accum_steps=4,
        dtype="float16",
        annotation="# calibrated 2026-05-28",
    )

    cfg = load_config(cfg_path)
    assert cfg is not None
    assert cfg.peft.method == "qlora"
    assert cfg.peft.r == 8
    assert cfg.train.batch_size == 2
    assert cfg.train.grad_accum_steps == 4
    assert cfg.model.dtype == "float16"
