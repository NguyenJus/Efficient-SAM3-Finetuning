"""CPU unit tests for tests/helpers/lora_predicates.py::has_plain_nn_linear.

These tests must NOT import sam3, qlora, bitsandbytes, or any GPU-only
dependency — they run in the lightweight CPU unit environment.
"""

from __future__ import annotations

from torch import nn

from tests.helpers.lora_predicates import has_plain_nn_linear

# ---------------------------------------------------------------------------
# Basic predicate behaviour: plain Linear outside MHA is flagged.
# ---------------------------------------------------------------------------


def test_plain_linear_outside_mha_returns_true() -> None:
    """A bare nn.Linear with no MHA ancestor must cause the predicate to return True."""
    model = nn.Sequential(nn.Linear(4, 4))
    assert has_plain_nn_linear(model), (
        "predicate must return True when a plain nn.Linear exists outside any MHA"
    )


def test_empty_module_returns_false() -> None:
    """A module with no nn.Linear children at all must return False."""

    class _NoLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(4)

    assert not has_plain_nn_linear(_NoLinear())


# ---------------------------------------------------------------------------
# MHA exclusion: internal Linears of nn.MultiheadAttention are not flagged.
# ---------------------------------------------------------------------------


def test_mha_only_module_returns_false() -> None:
    """A module whose only plain nn.Linear is inside nn.MultiheadAttention must
    return False — those children legitimately remain after apply_qlora.

    ``nn.MultiheadAttention`` holds an ``out_proj`` (nn.Linear) that the
    predicate must skip.
    """

    class _MHAOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2)

    assert not has_plain_nn_linear(_MHAOnly()), (
        "predicate must not flag nn.Linear children that live inside nn.MultiheadAttention"
    )


def test_mha_plus_plain_linear_returns_true() -> None:
    """An nn.MultiheadAttention alongside a bare nn.Linear must return True
    (the bare Linear is outside MHA and is the base leak we care about).
    """

    class _MixedBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2)
            self.fc = nn.Linear(8, 8)  # bare Linear — should be flagged

    assert has_plain_nn_linear(_MixedBlock()), (
        "predicate must return True when a plain nn.Linear sits outside any MHA"
    )


def test_nested_mha_children_excluded() -> None:
    """MHA nested inside an nn.Sequential must still exclude its internal Linears."""
    model = nn.Sequential(nn.MultiheadAttention(embed_dim=8, num_heads=2))
    assert not has_plain_nn_linear(model), (
        "predicate must exclude MHA-internal Linears even when MHA is nested inside a container"
    )


# ---------------------------------------------------------------------------
# Subclass of nn.Linear (mimicking bnb.nn.Linear4bit) is NOT flagged.
# ---------------------------------------------------------------------------


def test_linear_subclass_not_flagged() -> None:
    """``type(m) is nn.Linear`` must exclude subclasses (e.g. Linear4bit)."""

    class _Linear4bitSentinel(nn.Linear):
        pass

    model = nn.Sequential(_Linear4bitSentinel(4, 4))
    assert not has_plain_nn_linear(model), (
        "predicate must not flag subclasses of nn.Linear (type check is strict)"
    )


# ---------------------------------------------------------------------------
# LoRA adapter path tokens are still excluded (regression guard).
# ---------------------------------------------------------------------------


def test_lora_adapter_linears_not_flagged() -> None:
    """nn.Linear modules whose qualified name contains a LoRA adapter token
    (lora_A, lora_B, …) must not be flagged.
    """

    class _LoRAWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})

    assert not has_plain_nn_linear(_LoRAWrapper()), (
        "predicate must not flag lora_A / lora_B adapter Linears"
    )


def test_lora_adapter_plus_plain_linear_returns_true() -> None:
    """LoRA adapter Linears are ignored but a bare base Linear is still flagged."""

    class _LoRAWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})

    model = nn.Sequential(_LoRAWrapper(), nn.Linear(4, 4))
    assert has_plain_nn_linear(model), (
        "predicate must return True when a plain base Linear exists alongside LoRA adapter Linears"
    )
