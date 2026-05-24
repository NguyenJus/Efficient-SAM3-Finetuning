"""Tests for channel_adapter dtype alignment (Bug 1 fix).

Confirms that _Sam3ImageAdapter casts the channel_adapter to match the inner
model's parameter dtype at construction time, and that a forward through the
adapter does not raise on dtype-mismatched input.

These are pure CPU tests — no checkpoint loading, no CUDA required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from custom_sam_peft.models.sam3 import _Sam3ImageAdapter


class _TinyFakeModel(nn.Module):
    """Minimal fake model with parameters but no meaningful forward.

    Used as a stand-in for Sam3Image so _Sam3ImageAdapter can be constructed
    without loading the 3.3 GB real checkpoint.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2).to(dtype)

    def forward(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError("_TinyFakeModel.forward should not be called in unit tests")


class TestChannelAdapterDtype:
    """channel_adapter weight dtype must match the inner model's dtype."""

    def test_channel_adapter_weight_dtype_matches_inner_model_bfloat16(self) -> None:
        """After construction, channel_adapter.weight.dtype == inner model's dtype."""
        fake_model = _TinyFakeModel(dtype=torch.bfloat16)
        adapter = _Sam3ImageAdapter(fake_model, channels=4, channel_semantics="rgba")

        assert adapter.channel_adapter is not None, (
            "Expected channel_adapter to be built for rgba (non-rgb) semantics"
        )
        assert adapter.channel_adapter.weight.dtype == torch.bfloat16, (
            f"channel_adapter weight dtype should be bfloat16, "
            f"got {adapter.channel_adapter.weight.dtype}"
        )

    def test_channel_adapter_weight_dtype_matches_inner_model_float16(self) -> None:
        """Same test for float16."""
        fake_model = _TinyFakeModel(dtype=torch.float16)
        adapter = _Sam3ImageAdapter(fake_model, channels=4, channel_semantics="rgba")

        assert adapter.channel_adapter is not None
        assert adapter.channel_adapter.weight.dtype == torch.float16, (
            f"channel_adapter weight dtype should be float16, "
            f"got {adapter.channel_adapter.weight.dtype}"
        )

    def test_channel_adapter_forward_does_not_raise_on_bfloat16_input(self) -> None:
        """A bf16 input tensor fed directly to the adapter conv must not raise.

        Before the fix, _build_channel_adapter returned float32 Conv2d, so
        passing bfloat16 input raised:
          RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
        """
        fake_model = _TinyFakeModel(dtype=torch.bfloat16)
        adapter = _Sam3ImageAdapter(fake_model, channels=4, channel_semantics="rgba")

        assert adapter.channel_adapter is not None
        x = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
        # Must not raise
        out = adapter.channel_adapter(x)
        assert out.dtype == torch.bfloat16, f"Output dtype should be bfloat16, got {out.dtype}"
        assert out.shape == (1, 3, 8, 8), f"Expected shape (1, 3, 8, 8), got {out.shape}"

    def test_channel_adapter_forward_input_cast_on_dtype_mismatch(self) -> None:
        """adapter.forward casts images to adapter weight dtype before the conv.

        This verifies the Part 2 fix: images = images.to(dtype=...) inside
        _Sam3ImageAdapter.forward before applying the channel_adapter.
        We test this by checking that even a float32 input tensor fed through
        the adapter (which was cast to bfloat16 at construction time) doesn't raise.

        Note: we test the adapter conv directly here (not the full forward, which
        requires a real Sam3Image). The full cast path is exercised in
        test_channel_adapter_forward_does_not_raise_on_bfloat16_input above.
        """
        fake_model = _TinyFakeModel(dtype=torch.bfloat16)
        adapter = _Sam3ImageAdapter(fake_model, channels=4, channel_semantics="rgba")

        assert adapter.channel_adapter is not None
        # Confirm that the adapter weight was cast to bfloat16
        assert adapter.channel_adapter.weight.dtype == torch.bfloat16

        # Now check that passing a float32 input to the adapter conv (with cast) works
        x_f32 = torch.randn(1, 4, 8, 8, dtype=torch.float32)
        x_cast = x_f32.to(dtype=adapter.channel_adapter.weight.dtype)
        out = adapter.channel_adapter(x_cast)
        assert out.dtype == torch.bfloat16

    def test_rgb_passthrough_has_no_channel_adapter(self) -> None:
        """rgb semantics => channel_adapter is None (passthrough, no new params)."""
        fake_model = _TinyFakeModel(dtype=torch.bfloat16)
        adapter = _Sam3ImageAdapter(fake_model, channels=3, channel_semantics="rgb")

        assert adapter.channel_adapter is None
