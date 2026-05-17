"""Tests for esam3.peft_adapters.lora.apply_lora and helpers."""

from __future__ import annotations

import inspect

import pytest
from torch import nn

from esam3.config.schema import PEFTConfig
from esam3.peft_adapters.lora import (
    SCOPE_TARGETS,
    apply_lora,
)
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper


def _trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _total(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _lora_param_names(m: nn.Module) -> list[str]:
    return [n for n, _ in m.named_parameters() if "lora_" in n]


def test_apply_lora_default_scope_freezes_base() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora"))
    # Every non-LoRA param is frozen.
    for n, p in w.model.model.named_parameters():
        if "lora_" in n:
            assert p.requires_grad, f"LoRA param {n} should be trainable"
        else:
            assert not p.requires_grad, f"Base param {n} should be frozen"


def test_apply_lora_vision_scope_matches_only_vision() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora", scope="vision"))
    lora_names = _lora_param_names(w.model.model)
    assert lora_names, "expected LoRA params under vision scope"
    assert all("vision_encoder" in n for n in lora_names), lora_names
    assert not any("mask_decoder" in n for n in lora_names), lora_names


def test_apply_lora_vision_decoder_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora", scope="vision_decoder"))
    lora_names = _lora_param_names(w.model.model)
    assert any("vision_encoder" in n for n in lora_names), lora_names
    assert any("mask_decoder" in n for n in lora_names), lora_names
    # Negative-control Linears must not be adapted.
    assert not any("neg_control" in n for n in lora_names), lora_names


def test_apply_lora_all_scope_includes_negative_controls() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora", scope="all"))
    lora_names = _lora_param_names(w.model.model)
    assert any("neg_control" in n for n in lora_names), lora_names


def test_target_modules_overrides_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(
        w,
        PEFTConfig(
            method="lora",
            scope="all",  # would normally adapt everything
            target_modules=["vision_encoder.block0.attn.qkv"],
        ),
    )
    lora_names = _lora_param_names(w.model.model)
    # Exactly one Linear adapted → two LoRA params (lora_A, lora_B).
    qkv_lora = [n for n in lora_names if "vision_encoder.block0.attn.qkv" in n]
    assert len(qkv_lora) >= 2, qkv_lora
    other = [n for n in lora_names if "vision_encoder.block0.attn.qkv" not in n]
    assert not other, f"target_modules override should ignore scope; got {other}"


def test_apply_lora_no_match_raises() -> None:
    w = make_stub_wrapper()
    with pytest.raises(ValueError) as exc:
        apply_lora(w, PEFTConfig(method="lora", target_modules=["nonexistent.module"]))
    msg = str(exc.value)
    assert "nonexistent.module" in msg
    # Error should also surface at least one real Linear path to help debugging.
    assert "vision_encoder" in msg or "neg_control" in msg, msg


def test_apply_lora_idempotent_guard() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora"))
    with pytest.raises(RuntimeError, match="already applied"):
        apply_lora(w, PEFTConfig(method="lora"))


def test_apply_lora_trainable_ratio_under_default_scope() -> None:
    w = make_stub_wrapper()
    apply_lora(w, PEFTConfig(method="lora"))
    ratio = _trainable(w.model.model) / _total(w.model.model)
    assert ratio < 0.20, f"trainable ratio {ratio:.2%} unexpectedly high on tiny stub"


def test_apply_lora_preserves_forward_signature() -> None:
    w = make_stub_wrapper()
    sig_before = inspect.signature(w.forward)
    apply_lora(w, PEFTConfig(method="lora"))
    sig_after = inspect.signature(w.forward)
    assert sig_before == sig_after
    assert list(sig_after.parameters) == ["images", "prompts"]


def test_apply_lora_sets_peft_model_handle() -> None:
    w = make_stub_wrapper()
    assert w.peft_model is None
    apply_lora(w, PEFTConfig(method="lora"))
    assert w.peft_model is not None
    # The handle is the same object that replaced wrapper.model.model.
    assert w.peft_model is w.model.model


def test_scope_targets_keys_match_lora_scope_literal() -> None:
    # Cheap guard: SCOPE_TARGETS must cover every literal value of LoraScope.
    assert set(SCOPE_TARGETS) == {"vision", "vision_decoder", "all"}
