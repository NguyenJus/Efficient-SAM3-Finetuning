"""decide_eval_batch_size: forward-only VRAM math; calibrated/analytic/CPU paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

_GB = 1024**3


def _stub_gpu(monkeypatch: pytest.MonkeyPatch, total_bytes: int, name: str = "StubGPU") -> None:
    props = MagicMock(total_memory=total_bytes)
    props.name = name
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx: name)


def _write_cache(path: Path, **fields: object) -> None:
    base = {
        "schema_version": 2,
        "calibrated_at": "2026-05-22T00:00:00+00:00",
        "gpu_name": "StubGPU",
        "gpu_total_memory_bytes": int(40 * _GB),
        "sam3_checkpoint_sha": "deadbeef",
        "torch_version": "2.4.0",
        "custom_sam_peft_version": "0.0.0",
        "activation_bytes_per_example": int(0.5 * _GB),
        "peak_memory_bytes_at_probe": int(38 * _GB),
    }
    base.update(fields)
    path.write_text(json.dumps(base))


def test_decide_eval_batch_size_cpu_fallback(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from custom_sam_peft.presets import decide_eval_batch_size

    bs, predicted_bytes, provenance = decide_eval_batch_size()
    assert bs == 1
    assert predicted_bytes == 0
    assert provenance == "analytic"
    msgs = [r.message for r in caplog.records if "eval.batch_size=auto on CPU" in r.message]
    assert len(msgs) == 1


def test_decide_eval_batch_size_analytic_no_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a calibration cache, the analytic estimate runs at BASE_ACTIVATION_AT_1024."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _stub_gpu(monkeypatch, int(40 * _GB))
    # Ensure no calibration cache is found by setting cwd to a location without one.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    from custom_sam_peft.presets import decide_eval_batch_size

    bs, _predicted_bytes, provenance = decide_eval_batch_size()
    assert provenance == "analytic"
    assert bs >= 1


def test_decide_eval_batch_size_caps_search_at_64(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Search space is B in [1, 64]; never returns B > 64 even on huge GPUs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Use an enormous GPU (1 TiB) — should still cap at 64.
    _stub_gpu(monkeypatch, int(1024 * _GB))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    from custom_sam_peft.presets import decide_eval_batch_size

    bs, _predicted_bytes, _provenance = decide_eval_batch_size()
    assert bs <= 64


def test_decide_eval_batch_size_uses_calibrated_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a matching calibration cache, provenance='calibrated' and the cached
    activation_bytes_per_example is multiplied by forward_only_factor=0.25."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _stub_gpu(monkeypatch, int(40 * _GB), name="StubGPU")
    cache = tmp_path / ".custom_sam_peft_calibration.json"
    _write_cache(cache)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    from custom_sam_peft.presets import decide_eval_batch_size

    bs, _predicted_bytes, provenance = decide_eval_batch_size()
    assert provenance == "calibrated"
    assert bs >= 1


def test_predicted_bytes_eval_mode_excludes_optimizer_and_adapter(monkeypatch) -> None:
    """In mode='eval', _predicted_bytes skips _optimizer_bytes and _adapter_bytes,
    and scales activations by forward_only_factor."""
    from custom_sam_peft.presets import _predicted_bytes

    train_bytes = _predicted_bytes("lora", r=4, batch=1, image_size=1024, cache=None, mode="train")
    eval_bytes = _predicted_bytes("lora", r=4, batch=1, image_size=1024, cache=None, mode="eval")
    # eval drops optimizer state + adapter weights; activations scaled by 0.25.
    assert eval_bytes < train_bytes


def test_decide_eval_batch_size_sdpa_attention_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On a 23.46 GiB GPU the auto-batch must NOT return a batch size whose
    SDPA attention score matrix (B*H*N*N*dtype_bytes) would exceed the eval budget.

    SAM 3.1 vision backbone: patch_size=14 -> N=(1008/14)^2=5184, H=16.
    At bf16 (2 bytes), attn_per_example = 16*5184*5184*2 ~860 MiB.
    Without the cap the analytic model picks bs~35 -> 35*860 MiB ~29 GiB > 23 GiB budget.
    With the cap, bs must satisfy B*attn_per_example <= budget.  Issue #162.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # 23.46 GiB -- the exact card reported in issue #162.
    gpu_total = int(23.46 * _GB)
    _stub_gpu(monkeypatch, gpu_total, name="NVIDIA A10")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "custom_sam_peft.presets._current_sam3_checkpoint_sha",
        lambda: "deadbeef",
    )
    from custom_sam_peft.presets import decide_eval_batch_size

    bs, _predicted, _prov = decide_eval_batch_size()

    # SAM 3.1 vision backbone constants (patch_size=14, num_heads=16, bf16).
    _N = (1008 // 14) ** 2  # 5184 tokens
    _H = 16
    _dtype_bytes = 2  # bf16
    attn_bytes_for_bs = bs * _H * _N * _N * _dtype_bytes

    budget = gpu_total - _GB  # 1 GiB headroom
    assert attn_bytes_for_bs <= budget, (
        f"bs={bs} -> SDPA attn matrix = {attn_bytes_for_bs / _GB:.2f} GiB "
        f"exceeds eval budget {budget / _GB:.2f} GiB (issue #162)"
    )
