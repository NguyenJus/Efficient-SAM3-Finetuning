"""Unit tests for _patch_enable_vit_act_checkpoint — CPU-only, synthetic modules.

The helper iterates an ``nn.Module`` tree and sets ``use_act_checkpoint=True``
on every submodule exposing that attribute (sam3's ViT-Det blocks). The
contract is attribute-level and sam3-agnostic, so the tests use synthetic
stand-ins rather than instantiating a full sam3 model (which would load ~3 GB
of weights and is unnecessary for verifying attribute-flip behavior).
"""

from __future__ import annotations

import logging

import pytest
import torch.nn as nn

from esam3.models.sam3 import _patch_enable_vit_act_checkpoint


class _FakeViTDetBlock(nn.Module):
    """Stand-in for a sam3 ViT-Det block exposing the use_act_checkpoint flag."""

    def __init__(self) -> None:
        super().__init__()
        self.use_act_checkpoint = False


class _FakeNonCheckpointable(nn.Module):
    """Stand-in for a module that doesn't expose the checkpoint flag."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)


class _FakeModel(nn.Module):
    def __init__(self, n_blocks: int = 4, with_non: bool = True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeViTDetBlock() for _ in range(n_blocks)])
        if with_non:
            self.other = _FakeNonCheckpointable()


def test_flips_use_act_checkpoint_to_true_on_every_exposing_module() -> None:
    model = _FakeModel(n_blocks=3)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is False
    _patch_enable_vit_act_checkpoint(model)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is True


def test_skips_modules_without_the_attribute() -> None:
    """Modules that don't expose ``use_act_checkpoint`` must be left untouched.

    The patch must NOT inject the attribute onto unrelated modules — only flip
    it where sam3 already declared it.
    """
    model = _FakeModel(n_blocks=2)
    _patch_enable_vit_act_checkpoint(model)
    assert not hasattr(model.other, "use_act_checkpoint")


def test_idempotency_double_apply_is_no_op() -> None:
    """Calling the helper twice leaves state correct and per-module sentinels set."""
    model = _FakeModel(n_blocks=2)
    _patch_enable_vit_act_checkpoint(model)
    _patch_enable_vit_act_checkpoint(model)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is True
        assert getattr(blk, "_esam3_act_checkpoint_patched", False)


def test_logs_positive_count(caplog: pytest.LogCaptureFixture) -> None:
    """Helper emits an INFO log with the number of patched modules (replaces
    the prior ``set_grad_checkpointing`` no-op warning)."""
    model = _FakeModel(n_blocks=5)
    with caplog.at_level(logging.INFO, logger="esam3.models.sam3"):
        _patch_enable_vit_act_checkpoint(model)
    messages = [rec.message for rec in caplog.records]
    assert any("5" in msg and "checkpoint" in msg.lower() for msg in messages), (
        f"expected INFO log mentioning count=5 and 'checkpoint', got: {messages}"
    )
