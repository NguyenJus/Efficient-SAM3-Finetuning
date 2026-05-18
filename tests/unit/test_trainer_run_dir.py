"""Trainer.fit must use the caller-provided run_dir, not compute one internally."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

import esam3.train.trainer as trainer_mod
from esam3.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from esam3.data.base import Example, Instance, TextPrompts
from esam3.eval.metrics import MetricsReport
from esam3.peft_adapters.lora import apply_lora
from esam3.tracking.noop import NoopTracker
from esam3.train.trainer import Trainer
from tests.fixtures.tiny_sam3_lora_stub import FIXTURE_SCOPE_PATTERNS, make_stub_wrapper


def test_fit_uses_caller_provided_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a minimal Trainer that won't actually train; we just want to see
    # which dir gets the config.yaml write.
    cfg = MagicMock()
    cfg.run.output_dir = str(tmp_path / "ignored")  # Trainer must NOT use this.
    cfg.run.name = "irrelevant"
    cfg.run.seed = 0
    cfg.data.prompt_mode = "text"
    cfg.train.num_workers = 0
    cfg.train.batch_size = 1
    cfg.train.epochs = 0  # Skip the train loop entirely.
    cfg.train.warmup_steps = 0
    cfg.train.lr_schedule = "constant"
    cfg.train.lr = 1e-4
    cfg.train.optimizer = "adamw"
    cfg.train.box_hint.p_start = 0.0
    cfg.train.box_hint.p_end = 0.0
    cfg.train.box_hint.decay_steps = 1
    cfg.peft.method = "lora"
    cfg.export.merge = False
    cfg.model_dump.return_value = {"run": {"name": "irrelevant"}}

    # Stub model with at least one trainable parameter on CPU.
    # Use side_effect so each parameters() call gets a fresh iterator.
    _param = torch.nn.Parameter(torch.zeros(1))
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([_param])

    # Length=1 satisfies DataLoader's RandomSampler; epochs=0 means it's never iterated.
    train_ds = MagicMock(__len__=lambda self: 1, class_names=[])
    val_ds = MagicMock(__len__=lambda self: 0, class_names=[])
    tracker = MagicMock()

    # Patch Evaluator and save_adapter where trainer.py imports them (module-level names).
    monkeypatch.setattr(
        trainer_mod,
        "Evaluator",
        lambda _cfg: MagicMock(
            evaluate=MagicMock(
                return_value=MagicMock(overall={}, per_class={}, n_images=0, n_predictions=0)
            )
        ),
    )
    monkeypatch.setattr(
        trainer_mod, "save_adapter", lambda model, path: path.mkdir(parents=True, exist_ok=True)
    )

    chosen = tmp_path / "explicit-run"
    trainer = Trainer(model, train_ds, val_ds, tracker, cfg)
    result = trainer.fit(run_dir=chosen)

    assert result.run_dir == chosen
    assert (chosen / "config.yaml").exists()
    assert not (tmp_path / "ignored").exists()


class _TinyTextDataset:
    """Two-example dataset with text prompts, suitable for the stub wrapper."""

    def __init__(self) -> None:
        self._examples = [
            Example(
                image=torch.zeros(3, 8, 8),
                image_id=f"img{i}",
                prompts=TextPrompts(classes=["A"]),
                instances=[
                    Instance(
                        mask=torch.zeros(8, 8, dtype=torch.bool),
                        class_id=0,
                        box=torch.tensor([1.0, 1.0, 5.0, 5.0]),
                    )
                ],
            )
            for i in range(2)
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        return self._examples[i]

    @property
    def class_names(self) -> list[str]:
        return ["A"]


def test_fit_creates_expected_layout(tmp_path: Path) -> None:
    ds = _TinyTextDataset()
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = TrainConfig(
        run=RunConfig(name="layout-test", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
            prompt_mode="text",
        ),
        peft=PEFTConfig(method="lora", target_modules=FIXTURE_SCOPE_PATTERNS["vision"]),
        train=TrainHyperparams(
            epochs=1,
            grad_accum_steps=1,
            save_every=2,
            log_every=1,
            warmup_steps=0,
            num_workers=0,
        ),
    )
    apply_lora(wrapper, cfg.peft)
    trainer = Trainer(wrapper, ds, ds, NoopTracker(), cfg)
    run_dir = tmp_path / "layout-test-run"
    result = trainer.fit(run_dir=run_dir)
    rd = result.run_dir
    assert rd == run_dir
    assert rd.exists()
    assert (rd / "config.yaml").exists()
    assert (rd / "adapter" / "adapter_config.json").exists()
    assert (rd / "metrics.json").exists()
    assert (rd / "checkpoints").exists()
    assert isinstance(result.final_metrics, MetricsReport)
    assert result.merged_path is None
    payload = json.loads((rd / "metrics.json").read_text())
    assert "global_step" in payload
    assert "overall" in payload
