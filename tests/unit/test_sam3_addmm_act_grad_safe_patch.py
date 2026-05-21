"""Unit tests for _patch_addmm_act_grad_safe — CPU-only.

Covers:
  1. grad-enabled  → grad-enabled branch fires (unchanged behavior).
  2. grad-disabled, regular nn.Linear → delegates to _orig.
  3. grad-disabled, fake Linear4bit → routes through __call__ + activation;
     does NOT call _orig.
  4. grad-disabled, fake Linear4bit, unsupported activation → raises ValueError.
"""

from __future__ import annotations

import pytest
import sam3.perflib.fused as _pf
import torch
import torch.nn.functional as F
from torch import nn

from custom_sam_peft.models.sam3 import (
    _apply_activation,
    _is_linear4bit,
    _patch_addmm_act_grad_safe,
)

# ---------------------------------------------------------------------------
# Helpers / fake types
# ---------------------------------------------------------------------------


class _FakeLinear4bit(nn.Linear):
    """Minimal stand-in for bitsandbytes.nn.Linear4bit.

    Registered in the bitsandbytes namespace (lazily created if absent) so
    that ``_is_linear4bit`` recognises instances without needing a real GPU or
    the full bnb install.
    """


def _install_fake_bnb(monkeypatch) -> type:  # type: ignore[no-untyped-def]
    """Register _FakeLinear4bit as bitsandbytes.nn.Linear4bit for this test."""
    import sys
    import types

    if "bitsandbytes" not in sys.modules:
        bnb = types.ModuleType("bitsandbytes")
        bnb.nn = types.SimpleNamespace()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    else:
        bnb = sys.modules["bitsandbytes"]

    monkeypatch.setattr(bnb.nn, "Linear4bit", _FakeLinear4bit, raising=False)
    return _FakeLinear4bit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_patch():
    """Restore sam3.perflib.fused.addmm_act + sentinel after each test."""
    original_fn = _pf.addmm_act
    original_sentinel = getattr(_pf, "_custom_sam_peft_addmm_act_grad_safe_patched", False)
    yield
    _pf.addmm_act = original_fn
    _pf._custom_sam_peft_addmm_act_grad_safe_patched = original_sentinel  # type: ignore[attr-defined]
    # Also restore vitdet binding if it was patched.
    try:
        import sam3.model.vitdet as _vd

        _vd.addmm_act = original_fn
    except ImportError:
        # sam3.model.vitdet not importable (CPU-only unit env); nothing to restore.
        pass


# ---------------------------------------------------------------------------
# _is_linear4bit
# ---------------------------------------------------------------------------


def test_is_linear4bit_returns_false_for_regular_linear(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _install_fake_bnb(monkeypatch)
    assert not _is_linear4bit(nn.Linear(4, 8))


def test_is_linear4bit_returns_true_for_fake_linear4bit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _install_fake_bnb(monkeypatch)
    module = _FakeLinear4bit(4, 8)
    assert _is_linear4bit(module)


def test_is_linear4bit_returns_false_when_bnb_absent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sys

    monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # type: ignore[arg-type]
    # Should not raise; returns False gracefully.
    assert not _is_linear4bit(nn.Linear(4, 8))


# ---------------------------------------------------------------------------
# _apply_activation
# ---------------------------------------------------------------------------


def test_apply_activation_relu() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = _apply_activation(F.relu, x)
    assert torch.allclose(out, F.relu(x))


def test_apply_activation_relu_class() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = _apply_activation(nn.ReLU, x)
    assert torch.allclose(out, F.relu(x))


def test_apply_activation_gelu() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = _apply_activation(F.gelu, x)
    assert torch.allclose(out, F.gelu(x))


def test_apply_activation_gelu_class() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = _apply_activation(nn.GELU, x)
    assert torch.allclose(out, F.gelu(x))


def test_apply_activation_unsupported_raises() -> None:
    x = torch.tensor([1.0])
    with pytest.raises(ValueError, match="unsupported activation"):
        _apply_activation(nn.Sigmoid, x)


# ---------------------------------------------------------------------------
# Patched addmm_act wrapper — four core scenarios
# ---------------------------------------------------------------------------


def test_grad_enabled_branch_fires(monkeypatch) -> None:
    """Scenario 1: grad enabled → grad-enabled branch (linear + activation).

    The wrapper closes over _orig at patch time, so we cannot spy on _orig
    directly. Instead we verify behavior: a grad_fn must propagate through
    the call (proving an autograd-tracked nn.Linear forward ran, not the
    no-grad sam3 fused kernel).
    """
    _install_fake_bnb(monkeypatch)
    _patch_addmm_act_grad_safe()

    linear = nn.Linear(4, 8, bias=True)
    mat1 = torch.randn(2, 4, requires_grad=True)

    assert torch.is_grad_enabled()
    with torch.enable_grad():
        out = _pf.addmm_act(nn.ReLU, linear, mat1)

    assert out.shape == (2, 8)
    assert out.requires_grad


def test_no_grad_regular_linear_delegates_to_orig(monkeypatch) -> None:
    """Scenario 2: grad disabled, nn.Linear → delegates to _orig (not Linear4bit branch)."""
    _install_fake_bnb(monkeypatch)

    orig_call_count: list[int] = [0]
    fake_result = torch.tensor([42.0])

    # The wrapper closes over _orig at patch time.  To spy on _orig we reset
    # the sentinel, swap _pf.addmm_act to a spy, then re-patch so the new
    # wrapper closes over the spy.
    _pf._custom_sam_peft_addmm_act_grad_safe_patched = False  # type: ignore[attr-defined]

    def _spy_orig(activation, linear, mat1):  # type: ignore[no-untyped-def]
        orig_call_count[0] += 1
        return fake_result

    _pf.addmm_act = _spy_orig
    _patch_addmm_act_grad_safe()

    linear = nn.Linear(4, 8, bias=True)
    mat1 = torch.randn(2, 4)

    with torch.no_grad():
        out = _pf.addmm_act(nn.ReLU, linear, mat1)

    assert orig_call_count[0] == 1
    assert out is fake_result


def test_no_grad_linear4bit_skips_orig_and_calls_forward(monkeypatch) -> None:
    """Scenario 3: grad disabled, Linear4bit → routes through __call__, skips _orig."""
    _install_fake_bnb(monkeypatch)

    orig_call_count: list[int] = [0]
    forward_call_count: list[int] = [0]

    _pf._custom_sam_peft_addmm_act_grad_safe_patched = False  # type: ignore[attr-defined]

    def _spy_orig(activation, linear, mat1):  # type: ignore[no-untyped-def]
        orig_call_count[0] += 1
        raise AssertionError("_orig must NOT be called for Linear4bit")

    _pf.addmm_act = _spy_orig
    _patch_addmm_act_grad_safe()

    # Build a _FakeLinear4bit and wrap its forward to count calls.
    linear = _FakeLinear4bit(4, 8, bias=True)
    original_forward = linear.forward

    def _counting_forward(x):  # type: ignore[no-untyped-def]
        forward_call_count[0] += 1
        return original_forward(x)

    monkeypatch.setattr(linear, "forward", _counting_forward)

    mat1 = torch.randn(2, 4)
    with torch.no_grad():
        out = _pf.addmm_act(F.gelu, linear, mat1)

    assert orig_call_count[0] == 0, "_orig must not be called for Linear4bit"
    assert forward_call_count[0] == 1, "linear.forward must be called exactly once"
    assert out.shape == (2, 8)


def test_no_grad_linear4bit_unsupported_activation_raises(monkeypatch) -> None:
    """Scenario 4: grad disabled, Linear4bit, unsupported activation → ValueError."""
    _install_fake_bnb(monkeypatch)

    _pf._custom_sam_peft_addmm_act_grad_safe_patched = False  # type: ignore[attr-defined]

    def _spy_orig(activation, linear, mat1):  # type: ignore[no-untyped-def]
        raise AssertionError("should not reach _orig")

    _pf.addmm_act = _spy_orig
    _patch_addmm_act_grad_safe()

    linear = _FakeLinear4bit(4, 8, bias=True)
    mat1 = torch.randn(2, 4)

    with torch.no_grad(), pytest.raises(ValueError, match="unsupported activation"):
        _pf.addmm_act(nn.Sigmoid, linear, mat1)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotency(monkeypatch) -> None:
    """Re-applying the patch does not double-wrap."""
    _install_fake_bnb(monkeypatch)
    _patch_addmm_act_grad_safe()
    first_fn = _pf.addmm_act
    _patch_addmm_act_grad_safe()
    second_fn = _pf.addmm_act
    assert first_fn is second_fn
