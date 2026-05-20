"""Unit tests for _patch_roi_align_dtype — CPU-only, no GPU required."""

import pytest
import torch
import torchvision.ops

from custom_sam_peft.models.sam3 import _patch_roi_align_dtype


@pytest.fixture(autouse=True)
def _restore_roi_align():
    """Restore torchvision.ops.roi_align and the sentinel after each test."""
    original_fn = torchvision.ops.roi_align
    original_sentinel = getattr(torchvision.ops, "_custom_sam_peft_roi_align_dtype_patched", False)
    yield
    torchvision.ops.roi_align = original_fn
    torchvision.ops._custom_sam_peft_roi_align_dtype_patched = original_sentinel


def test_list_rois_dtype_mismatch_real_kernel() -> None:
    """fp16 rois (list form) are cast to fp32 input dtype before the kernel call."""
    _patch_roi_align_dtype()
    input_fp32 = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float16)]
    out = torchvision.ops.roi_align(input_fp32, boxes, output_size=2)
    assert out.dtype == torch.float32


def test_tensor_rois_dtype_mismatch_real_kernel() -> None:
    """fp16 rois (tensor form, shape (N,5)) are cast to fp32 input dtype."""
    _patch_roi_align_dtype()
    input_fp32 = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    # (N, 5) form: [batch_idx, x1, y1, x2, y2]
    boxes_fp16 = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0]], dtype=torch.float16)
    out = torchvision.ops.roi_align(input_fp32, boxes_fp16, output_size=2)
    assert out.dtype == torch.float32


def test_same_dtype_passthrough() -> None:
    """Patched roi_align produces identical output to the unpatched version when dtypes match."""
    input_fp32 = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)]

    # Capture unpatched result BEFORE installing the patch
    unpatched_out = torchvision.ops.roi_align(input_fp32, boxes, output_size=2)

    _patch_roi_align_dtype()
    patched_out = torchvision.ops.roi_align(input_fp32, boxes, output_size=2)

    assert torch.allclose(unpatched_out, patched_out)


def test_idempotency() -> None:
    """Calling _patch_roi_align_dtype twice does not double-wrap roi_align."""
    _patch_roi_align_dtype()
    after_first = torchvision.ops.roi_align

    _patch_roi_align_dtype()
    after_second = torchvision.ops.roi_align

    assert after_first is after_second
