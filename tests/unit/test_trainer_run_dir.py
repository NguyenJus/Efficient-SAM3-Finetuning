"""End-to-end Trainer.fit() on the stub: verify run-dir layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from esam3.config.schema import (
    DataConfig,
    DataSplit,
    PEFTConfig,
    RunConfig,
    TrainConfig,
    TrainHyperparams,
)
from esam3.data.base import Example, Instance, TextPrompts
from esam3.models.sam3 import Sam3Wrapper
from esam3.peft_adapters.lora import apply_lora
from esam3.tracking.noop import NoopTracker
from esam3.train.trainer import Trainer


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


class _AttnBlock(nn.Module):
    """SAM 3.1-style attention block with fused qkv + proj (LoRA targets)."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)  # type: ignore[assignment]
        self.attn.proj = nn.Linear(dim, dim)  # type: ignore[assignment]


class _TinyWorkingBase(nn.Module):
    """Inner base: SAM 3.1-style module names + a working forward.

    forward() routes through ``vision_encoder.block0.attn.qkv`` so that LoRA
    A/B matrices (which are inserted there by apply_lora) participate in the
    computation graph and produce a grad_fn on the outputs.
    """

    def __init__(self, dim: int = 8, num_queries: int = 4, mask_size: int = 8) -> None:
        super().__init__()
        self.vision_encoder = nn.Module()
        self.vision_encoder.block0 = _AttnBlock(dim)  # type: ignore[assignment]
        self.num_queries = num_queries
        self.mask_size = mask_size
        self.dim = dim

    def forward(
        self, images: torch.Tensor, prompts: Any, box_hints: Any = None
    ) -> dict[str, torch.Tensor]:
        b = images.shape[0]
        q, m = self.num_queries, self.mask_size
        # Route through the LoRA-targeted qkv linear so the computation graph
        # includes LoRA A/B matrices after apply_lora().
        flat = images.reshape(b, 3, -1).mean(dim=-1)  # (B, 3)
        feat = torch.nn.functional.pad(flat, (0, self.dim - 3))  # (B, dim)
        feat = self.vision_encoder.block0.attn.qkv(feat)  # (B, dim*3)
        scalar = feat.mean()
        return {
            "pred_logits": torch.zeros(b, q, 1) + scalar,
            "pred_boxes": torch.zeros(b, q, 4) + scalar,
            "pred_masks": torch.zeros(b, q, m, m) + scalar,
            "presence_logit_dec": torch.zeros(b, 1) + scalar,
        }


class _WorkingAdapter(nn.Module):
    """Adapter that holds the base and delegates forward — so apply_lora can reach model.model."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.model = base

    def forward(
        self, images: torch.Tensor, prompts: Any, box_hints: Any = None
    ) -> dict[str, torch.Tensor]:
        return self.model(images, prompts, box_hints=box_hints)  # type: ignore[return-value]


def _make_lora_wrapper(dim: int = 8) -> Sam3Wrapper:
    """Build a Sam3Wrapper with SAM 3.1-style LoRA targets and a working forward."""
    base = _TinyWorkingBase(dim=dim)
    adapter = _WorkingAdapter(base)
    return Sam3Wrapper(adapter, image_size=8, mask_size=8)


def test_fit_creates_expected_layout(tmp_path: Path) -> None:
    ds = _TinyTextDataset()
    wrapper = _make_lora_wrapper(dim=8)
    cfg = TrainConfig(
        run=RunConfig(name="layout-test", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(annotations="a.json", images="i"),
            val=DataSplit(annotations="a.json", images="i"),
            prompt_mode="text",
        ),
        peft=PEFTConfig(method="lora", scope="vision"),
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
    result = trainer.fit()
    rd = result.run_dir
    assert rd.exists()
    assert (rd / "config.yaml").exists()
    assert (rd / "adapter" / "adapter_config.json").exists()
    assert (rd / "metrics.json").exists()
    assert (rd / "checkpoints").exists()
    assert result.final_metrics is None
    assert result.merged_path is None
    payload = json.loads((rd / "metrics.json").read_text())
    assert "global_step" in payload
