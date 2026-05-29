"""CPU tests for device-aware bf16->fp16 coercion on CC<8.0 hardware."""

from __future__ import annotations

import logging

import torch

import custom_sam_peft.runtime._runtime as _rt


def test_bf16_coerced_to_fp16_below_cc80() -> None:
    assert _rt.coerce_dtype_for_capability(torch.bfloat16, capability=(6, 1)) is torch.float16
    assert _rt.coerce_dtype_for_capability(torch.bfloat16, capability=(7, 5)) is torch.float16


def test_bf16_preserved_at_cc80_and_above() -> None:
    assert _rt.coerce_dtype_for_capability(torch.bfloat16, capability=(8, 0)) is torch.bfloat16
    assert _rt.coerce_dtype_for_capability(torch.bfloat16, capability=(9, 0)) is torch.bfloat16


def test_non_bf16_never_coerced() -> None:
    assert _rt.coerce_dtype_for_capability(torch.float16, capability=(6, 1)) is torch.float16
    assert _rt.coerce_dtype_for_capability(torch.float32, capability=(6, 1)) is torch.float32


def test_warns_at_most_once_per_process(caplog) -> None:
    _rt._dtype_coercion_warned = False
    with caplog.at_level(logging.WARNING, logger="custom_sam_peft.runtime._runtime"):
        _rt.coerce_dtype_for_capability(torch.bfloat16, capability=(6, 1))
        _rt.coerce_dtype_for_capability(torch.bfloat16, capability=(6, 1))
    warnings = [r for r in caplog.records if "bfloat16" in r.message.lower()]
    assert len(warnings) == 1, [r.message for r in caplog.records]
