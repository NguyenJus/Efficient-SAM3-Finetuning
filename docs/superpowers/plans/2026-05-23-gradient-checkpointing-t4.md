# Gradient Checkpointing on T4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable `gradient_checkpointing` end-to-end on a Colab T4 across both entry points (static `load_sam31`, dynamic OOM ladder), fixing the three coupled breaks (recompute-metadata mismatch, dead static branch, dead dynamic rung) in one PR, with a measurable peak-VRAM reduction and correctness parity.

**Architecture:** A new config-gated patch module `models/_patches/vit_act_checkpoint.py` exposing `apply(model, runtime)` flips sam3's per-block `use_act_checkpoint=True` (idempotent, sentinel-guarded) AND wraps each exposing block's `forward` in a deterministic `torch.autocast` so non-reentrant checkpoint recompute reconstructs the same dispatch context (Fix A). It is invoked conditionally — NOT registered in `_ALL_PATCHES` — from `_construct_raw_model` (static) and from the OOM ladder (dynamic). The fix tier (A default; escalate to B/C) is selected by a Phase-0 Colab T4 diagnostic trace, which is the authority over the spec's hypothesis. Defaults and shipped YAMLs flip to `true`. CPU tests cover the flag-flip/wiring mechanism; GPU tests on T4 verify no `CheckpointError`, loss parity, and VRAM reduction.

**Tech Stack:** Python, PyTorch (`torch.utils.checkpoint`, `torch.autocast`, `torch.cuda.max_memory_allocated`), Meta `sam3` (external, present only on Colab T4; never edited), `pytest` (`-m gpu` release tier), `ruff`, `mypy`, Pydantic config (`TrainConfig`), `uv`.

**Reference spec:** `docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md`

---

## Pre-flight checks

Run once before Task 1. All paths are absolute against the worktree.

```bash
# 1. Confirm you are in the worktree.
git -C /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 rev-parse --show-toplevel
# Expected: /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89

# 2. Confirm a clean tree.
git -C /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 status --porcelain
# Expected: no output

# 3. Confirm sam3 is NOT importable on the dev box (the fix is monkeypatch-only; no sam3 source edits).
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run python -c "import sam3" 2>&1 | tail -1
# Expected: ModuleNotFoundError (or ImportError). Confirms CPU tests must use synthetic stand-ins.

# 4. Confirm the current CPU baseline is green.
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit -q 2>&1 | tail -3
# Expected: all passed (record the count).

# 5. Confirm the cited live-code anchors still match the spec §11.
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run python - <<'PY'
import pathlib
sam3 = pathlib.Path("src/custom_sam_peft/models/sam3.py").read_text().splitlines()
assert "set_grad_checkpointing" in sam3[597], sam3[596:611]
loop = pathlib.Path("src/custom_sam_peft/train/loop.py").read_text()
assert 'action="grad_ckpt_enabled"' in loop
schema = pathlib.Path("src/custom_sam_peft/config/schema.py").read_text()
assert "TODO(#60)" in schema
print("anchors OK")
PY
# Expected: anchors OK
```

If any check fails, STOP and re-derive line numbers before proceeding.

---

## File map (what gets touched)

| File | Action | Owning task | Phase |
| --- | --- | --- | --- |
| `src/custom_sam_peft/models/_patches/vit_act_checkpoint.py` | Create — `apply(model, runtime)`: flag-flip (Task 1) + deterministic-autocast wrap (Task 6, after Phase 0) | 1, 6 | 1 |
| `src/custom_sam_peft/models/sam3.py` | Modify — add `_patch_enable_vit_act_checkpoint` shim + `_VIT_ACT_CHECKPOINT_ATTR`; replace no-op `else` branch (597-610) | 2, 3 | 1, 2 |
| `tests/unit/test_sam3_act_checkpoint_patch.py` | Create — CPU mechanism tests (flag-flip, skip, idempotency, count) | 1 | 1 |
| `tests/unit/test_construct_raw_model_grad_ckpt.py` | Create — CPU test: static entry point invokes the shim | 3 | 2 |
| `src/custom_sam_peft/train/loop.py` | Modify — OOM ladder rung (126-140) applies the patch to the live model | 4 | 2 |
| `tests/unit/test_trainer_oom_retry.py` | Modify — add CPU case: `grad_ckpt_enabled` rung applies the patch to live model | 4 | 2 |
| `src/custom_sam_peft/config/schema.py` | Modify — `ModelConfig.gradient_checkpointing` default `False`→`True` (118-120); drop `# TODO(#60)` | 5 | 2 |
| `src/custom_sam_peft/cli/templates/coco_text_lora.yaml` | Modify — line 21 `false`→`true`; drop `#60` comment | 5 | 2 |
| `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml` | Modify — line 21 `false`→`true`; drop `#60` comment | 5 | 2 |
| `configs/examples/coco_text_lora.yaml` | Modify — line 18 `false`→`true`; drop `#60` comment | 5 | 2 |
| `configs/examples/coco_text_qlora.yaml` | Modify — line 18 `false`→`true`; drop `#60` comment | 5 | 2 |
| `configs/examples/gpu_smoke_lora.yaml` | Modify — line 10 `false`→`true`; drop `#60` comment | 5 | 2 |
| `configs/examples/gpu_smoke_qlora.yaml` | Modify — line 10 `false`→`true`; drop `#60` comment | 5 | 2 |
| `configs/examples/coco_text_auto_split.yaml` | Modify — line 16 `false`→`true` (no comment) | 5 | 2 |
| `configs/examples/coco_text_no_val.yaml` | Modify — line 18 `false`→`true` | 5 | 2 |
| `configs/examples/coco_text_lora_subset.yaml` | Modify — line 12 `false`→`true` | 5 | 2 |
| `src/custom_sam_peft/models/_patches/README.md` | Modify — add a "When SAM-3 bumps" row for `vit_act_checkpoint.py` | 1 | 1 |
| `tests/gpu/test_grad_checkpointing.py` | Create — GPU tests: no `CheckpointError`, loss parity, VRAM-lower, OOM-ladder rung | 8 | 3 |
| `notebooks/colab_gpu_tests.ipynb` | Modify — add Phase-0 diagnostic cell + Phase-3 verification cell (RUN ON COLAB T4) | 7, 9 | 0, 3 |
| `docs/superpowers/specs/2026-05-18-smoke-test-design.md` | Modify — dated reconciliation note | 10 | 4 |
| `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md` | Modify — dated note that the lever is now functional | 10 | 4 |
| `docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md` | Modify — fold Phase-0 trace result into §3 if it diverges | 6 | 1 |

**Dependency / GPU-gating arc:**
- **CPU-runnable now (no T4 needed):** Tasks 1–5 (patch flag-flip half + its CPU tests, both entry-point wirings + CPU tests, schema/YAML flips). The flag-flip mechanism is fix-tier-independent.
- **GPU-gated GATE:** Task 7 (Phase 0 diagnostic on Colab T4) — **classifies the divergence and SELECTS the Fix tier.** Do NOT write Task 6 until Task 7 returns a classification.
- **Tier-selected:** Task 6 (Phase 1 fix body — Fix A default, B/C conditional on the Task-7 trace).
- **GPU-gated verification:** Tasks 8–9 (Phase 3 on Colab T4).
- **Bookkeeping:** Task 10 (Phase 4).

Lint gate for every landing task that produces shippable code (run from the worktree root):

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
```

Per `superpowers:verification-before-completion`: a step is "done" only when you have SHOWN the command output, not asserted it.

---

## Task 1: New patch module — flag-flip half + CPU mechanism tests

**Phase:** 1. **CPU-runnable now.** This is the fix-tier-independent half of the patch (flipping `use_act_checkpoint=True` + idempotency). The deterministic-autocast wrap (Fix A item b) is added in Task 6 AFTER the Phase-0 trace; it is GPU-only behavior and is NOT testable here (honesty note below).

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Files:**
- Create: `src/custom_sam_peft/models/_patches/vit_act_checkpoint.py`
- Test: `tests/unit/test_sam3_act_checkpoint_patch.py`
- Modify: `src/custom_sam_peft/models/_patches/README.md`

**Pattern references (read, do not modify):**
- `src/custom_sam_peft/models/_patches/module_input_dtype.py:20-50` — `apply(model, runtime)` signature + per-module sentinel idempotency (`_custom_sam_peft_module_input_dtype_patched`).
- Deleted test from `889bd2c` (`git show 889bd2c:tests/unit/test_sam3_act_checkpoint_patch.py`) — synthetic `_FakeViTDetBlock` / `_FakeNonCheckpointable` / `_FakeModel` stand-ins. Re-created below, modernized to `custom_sam_peft` and the new sentinel name.

**Honesty note (mirror the `_patch_*_dtype` family precedent):** these CPU tests cover the flag-flip, the skip-non-exposing rule, and idempotency only. The deterministic-autocast wrapping behavior (Task 6 Fix A item b) and the recompute-determinism guarantee are GPU-only — there is no CPU assertion that the autocast context reconstructs identically across recompute. Do NOT over-claim coverage.

- [ ] **Step 1: Write the failing CPU mechanism tests.**

Create `tests/unit/test_sam3_act_checkpoint_patch.py`:

```python
"""Unit tests for the vit_act_checkpoint patch — CPU-only, synthetic modules.

The patch iterates an ``nn.Module`` tree and sets ``use_act_checkpoint=True``
on every submodule already exposing that attribute (sam3's ViT-Det blocks).
The contract is attribute-level and sam3-agnostic, so the tests use synthetic
stand-ins rather than instantiating a full sam3 model.

GPU-only behavior NOT covered here: the deterministic-autocast wrap added in
the Phase-1 fix (Fix A item b) and the recompute-determinism guarantee. Those
are verified in tests/gpu/test_grad_checkpointing.py on a real T4.
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from custom_sam_peft.models._patches import vit_act_checkpoint
from custom_sam_peft.runtime._runtime import Runtime

_CPU_RUNTIME = Runtime(device=torch.device("cpu"), dtype=torch.float32)


class _FakeViTDetBlock(nn.Module):
    """Stand-in for a sam3 ViT-Det block exposing the use_act_checkpoint flag."""

    def __init__(self) -> None:
        super().__init__()
        self.use_act_checkpoint = False
        self.lin = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _FakeNonCheckpointable(nn.Module):
    """Stand-in for a module that does NOT expose the checkpoint flag."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)


class _FakeModel(nn.Module):
    def __init__(self, n_blocks: int = 4, with_non: bool = True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeViTDetBlock() for _ in range(n_blocks)])
        if with_non:
            self.other = _FakeNonCheckpointable()


def test_flips_use_act_checkpoint_on_every_exposing_block() -> None:
    model = _FakeModel(n_blocks=3)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is False
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is True


def test_skips_modules_without_the_attribute() -> None:
    model = _FakeModel(n_blocks=2)
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    assert not hasattr(model.other, "use_act_checkpoint")


def test_idempotent_double_apply() -> None:
    model = _FakeModel(n_blocks=2)
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    for blk in model.blocks:
        assert blk.use_act_checkpoint is True
        assert getattr(blk, "_custom_sam_peft_act_checkpoint_patched", False) is True


def test_logs_positive_count(caplog: pytest.LogCaptureFixture) -> None:
    model = _FakeModel(n_blocks=5)
    with caplog.at_level(logging.INFO, logger="custom_sam_peft.models._patches.vit_act_checkpoint"):
        vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    messages = [rec.message for rec in caplog.records]
    assert any("5" in m and "checkpoint" in m.lower() for m in messages), messages
```

- [ ] **Step 2: Run the tests to verify they fail.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_sam3_act_checkpoint_patch.py -v
```

Expected: collection/import error — `ModuleNotFoundError: No module named 'custom_sam_peft.models._patches.vit_act_checkpoint'`.

- [ ] **Step 3: Write the minimal patch module (flag-flip half only).**

Create `src/custom_sam_peft/models/_patches/vit_act_checkpoint.py`. **Do NOT add the autocast wrap yet** — that is Task 6, gated on the Phase-0 trace.

```python
"""Patch: enable sam3 ViT-Det per-block activation checkpointing.

Config-gated (cfg.gradient_checkpointing) — NOT registered in _ALL_PATCHES.
Invoked conditionally from models/sam3.py::_construct_raw_model (static entry
point) and from train/loop.py's OOM ladder (dynamic entry point).

This module currently implements the FLAG-FLIP half only: it sets
``use_act_checkpoint=True`` on every submodule that already exposes that
attribute (sam3's ViT-Det blocks, vitdet.py:982). sam3 self-checkpoints inside
the block forward via ``checkpoint.checkpoint(blk, x, use_reentrant=False)``,
which raised a recompute-metadata CheckpointError on T4 (issue #60 / #89). The
deterministic-autocast wrap that resolves that mismatch (Fix A) is added by the
Phase-1 task after the Phase-0 Colab T4 diagnostic classifies the divergence.

See models/_patches/README.md "When SAM-3 bumps" and the spec
docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_sam_peft.runtime._runtime import Runtime

from torch import nn

logger = logging.getLogger(__name__)

# sam3's per-ViT-Det-block flag (vitdet.py:982). If a sam3 bump renames or
# removes this, the patch flips zero blocks and logs a loud warning.
_ACT_CHECKPOINT_ATTR = "use_act_checkpoint"
_SENTINEL_ATTR = "_custom_sam_peft_act_checkpoint_patched"


def apply(model: nn.Module, runtime: Runtime) -> None:
    """Enable activation checkpointing on every exposing ViT-Det block.

    Idempotent via a per-module sentinel (mirrors module_input_dtype.py:46-49).
    Only flips the flag where sam3 already declared it; never injects the
    attribute onto unrelated modules.

    The ``runtime`` argument is unused by the flag-flip half but is part of the
    patch contract and is consumed by the deterministic-autocast wrap added in
    the Phase-1 fix task.
    """
    patched_count = 0
    for submodule in model.modules():
        if not hasattr(submodule, _ACT_CHECKPOINT_ATTR):
            continue
        if getattr(submodule, _SENTINEL_ATTR, False):
            continue
        submodule.use_act_checkpoint = True  # type: ignore[assignment]
        setattr(submodule, _SENTINEL_ATTR, True)
        patched_count += 1

    if patched_count == 0:
        logger.warning(
            "vit_act_checkpoint: found ZERO modules exposing %r. Either the "
            "model has no ViT-Det blocks (wrong model) or sam3 renamed the "
            "attribute (see vitdet.py:982 and _patches/README.md 'When SAM-3 "
            "bumps').",
            _ACT_CHECKPOINT_ATTR,
        )
    else:
        logger.info(
            "Enabled activation checkpointing on %d ViT-Det block(s).",
            patched_count,
        )
```

- [ ] **Step 4: Run the tests to verify they pass.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_sam3_act_checkpoint_patch.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Add the SAM-3-bump checklist row.**

In `src/custom_sam_peft/models/_patches/README.md`, add a row to the "Patch index" table after the `text_pool_dtype.py` row:

```markdown
| `vit_act_checkpoint.py` | Config-gated (NOT in `_ALL_PATCHES`). Enables per-block activation checkpointing (`use_act_checkpoint=True`, `vitdet.py:982`) + deterministic-autocast wrap so non-reentrant checkpoint recompute is metadata-consistent. |
```

And add a note under the "When SAM-3 bumps" list (after item 5):

```markdown
6. `vit_act_checkpoint.py` targets the per-block `use_act_checkpoint` flag at
   `vitdet.py:982`. If that attribute is renamed/removed, the patch flips zero
   blocks and logs a loud warning; update `_ACT_CHECKPOINT_ATTR` or open a
   `sam3-bump` issue.
```

- [ ] **Step 6: Lint gate + commit.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
git add src/custom_sam_peft/models/_patches/vit_act_checkpoint.py tests/unit/test_sam3_act_checkpoint_patch.py src/custom_sam_peft/models/_patches/README.md
git commit -m "feat(sam3): add vit_act_checkpoint patch (flag-flip half) + CPU tests"
```

**Completion criteria:** `vit_act_checkpoint.apply` flips every exposing block, skips non-exposing modules, is idempotent, logs the count; lint gate green.

---

## Task 2: Delegation shim in `sam3.py`

**Phase:** 1. **CPU-runnable now.** Adds the `_patch_enable_vit_act_checkpoint(model)` shim matching the `_patch_*` family at `sam3.py:402-520`, so the static and dynamic entry points (Tasks 3, 4) and CPU tests can call a stable name.

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py` (add shim near the `_patch_*` family, ~lines 402-512)

**Pattern reference:** `sam3.py:506-511` (`_patch_module_input_dtype` shim) — same shape: lazy-import the patch module + `Runtime`, call `apply`.

- [ ] **Step 1: Add the delegation shim.**

In `src/custom_sam_peft/models/sam3.py`, immediately after `_patch_module_input_dtype` (ends at line 511), add:

```python
def _patch_enable_vit_act_checkpoint(model: nn.Module) -> None:
    """Delegate to models/_patches/vit_act_checkpoint.apply.

    Config-gated activation-checkpointing patch (NOT in _ALL_PATCHES). Called
    conditionally from _construct_raw_model and the OOM ladder. See that module
    for the recompute-mismatch rationale (issue #89).
    """
    from custom_sam_peft.models._patches import vit_act_checkpoint as _m
    from custom_sam_peft.runtime._runtime import Runtime

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32
    _m.apply(model, Runtime(device=device, dtype=dtype))
```

> Note: unlike the dtype-patch shims (which hardcode a CPU/fp32 `Runtime` because they were only ever called from `_apply_patches`), this shim derives `device`/`dtype` from the live model's parameters, because Fix A's autocast wrap (Task 6) needs the real device type (`cuda` on T4) and dtype to be correct. On the dev box the model is the synthetic stand-in (no params) → defaults to CPU/fp32, which is fine.

- [ ] **Step 2: Verify the shim imports and is callable on a synthetic model.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run python - <<'PY'
import torch
import torch.nn as nn
from custom_sam_peft.models.sam3 import _patch_enable_vit_act_checkpoint

class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_act_checkpoint = False
        self.lin = nn.Linear(2, 2)

m = nn.ModuleList([Blk(), Blk()])
_patch_enable_vit_act_checkpoint(m)
assert all(b.use_act_checkpoint for b in m), "shim did not flip flags"
print("shim OK")
PY
```

Expected: `shim OK`.

- [ ] **Step 3: Lint gate + commit.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
git add src/custom_sam_peft/models/sam3.py
git commit -m "feat(sam3): add _patch_enable_vit_act_checkpoint delegation shim"
```

**Completion criteria:** shim exists, derives device/dtype from the model, delegates to the patch; lint gate green.

---

## Task 3: Static entry point — replace the no-op warning branch + CPU test

**Phase:** 2. **CPU-runnable now (via spy).** Replaces the dead `else` branch in `_construct_raw_model` (`sam3.py:597-610`) with a call to the shim. The real sam3 model is unavailable on the dev box, so the CPU test monkeypatches the shim and asserts it is invoked when `cfg.gradient_checkpointing=True`.

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Implementer-confirmation seam (spec §6 / §8):** the patch must run on the raw model so its blocks are flipped + (post-Task-6) wrapped. Confirm placement composes with `_apply_patches`: `_construct_raw_model` runs BEFORE `_apply_patches` in `load_sam31` (`sam3.py:678-680`). The deterministic-autocast wrap (Task 6) is OUTER to the block forward; the dtype hooks installed by `_apply_patches` are forward-pre-hooks on inner Linear/LayerNorm/Conv. They compose, but the wrap must be applied to the block forward, and the dtype hooks are added afterward by `_apply_patches` — confirm this ordering does not double-wrap (idempotency sentinel guards it). **The static `set_grad_checkpointing` branch (line 598-599) is PRESERVED** as the preferred path for a future sam3 revision.

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py:597-610`
- Test: `tests/unit/test_construct_raw_model_grad_ckpt.py` (create)

- [ ] **Step 1: Write the failing CPU test (spy on the shim).**

Create `tests/unit/test_construct_raw_model_grad_ckpt.py`:

```python
"""CPU test: the static entry point invokes the activation-checkpoint patch.

The real sam3 model is unavailable on the dev box, so we monkeypatch
sam3.build_sam3_image_model to return a synthetic ViT-Det stand-in and spy on
_patch_enable_vit_act_checkpoint to confirm _construct_raw_model calls it when
cfg.gradient_checkpointing is True (and does NOT call it when False).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import custom_sam_peft.models.sam3 as sam3_mod
from custom_sam_peft.config.schema import ModelConfig


class _FakeBlk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_act_checkpoint = False
        self.lin = nn.Linear(2, 2)


class _FakeRawModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlk(), _FakeBlk()])
        # No set_grad_checkpointing — forces the patch branch.


def _install_fake_build(monkeypatch, model: nn.Module) -> None:
    monkeypatch.setattr(sam3_mod, "_locate_weights", lambda cfg: Path("/tmp/fake.pt"))
    monkeypatch.setattr(
        sam3_mod.sam3, "build_sam3_image_model", lambda **kw: model
    )


def _cfg(grad_ckpt: bool) -> ModelConfig:
    return ModelConfig(gradient_checkpointing=grad_ckpt, device="cpu")


def test_construct_raw_model_invokes_patch_when_enabled(monkeypatch) -> None:
    model = _FakeRawModel()
    _install_fake_build(monkeypatch, model)
    calls: list[nn.Module] = []
    monkeypatch.setattr(
        sam3_mod, "_patch_enable_vit_act_checkpoint", lambda m: calls.append(m)
    )
    out = sam3_mod._construct_raw_model(_cfg(True))
    assert calls == [out], "patch not invoked exactly once on the raw model"


def test_construct_raw_model_skips_patch_when_disabled(monkeypatch) -> None:
    model = _FakeRawModel()
    _install_fake_build(monkeypatch, model)
    calls: list[nn.Module] = []
    monkeypatch.setattr(
        sam3_mod, "_patch_enable_vit_act_checkpoint", lambda m: calls.append(m)
    )
    sam3_mod._construct_raw_model(_cfg(False))
    assert calls == [], "patch invoked when gradient_checkpointing=False"
```

> If `_construct_raw_model`'s stdout-capture / missing-keys logic chokes on the synthetic model, the implementer adapts the fake build to return cleanly (the synthetic model has no checkpoint-load print, so `_captured_stdout` is empty and the loop is skipped — confirm against `sam3.py:535-595`).

- [ ] **Step 2: Run the test to verify it fails.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_construct_raw_model_grad_ckpt.py -v
```

Expected: `test_construct_raw_model_invokes_patch_when_enabled` FAILS (`assert [] == [model]`) — current code hits the `logger.warning` no-op branch, never calls the shim.

- [ ] **Step 3: Replace the no-op branch.**

In `src/custom_sam_peft/models/sam3.py`, replace the `else` branch (lines 600-610):

```python
    if cfg.gradient_checkpointing:
        if hasattr(raw_model, "set_grad_checkpointing"):
            raw_model.set_grad_checkpointing(True)
        else:
            # sam3 has no set_grad_checkpointing on this revision; enable
            # activation checkpointing on its per-ViT-Det-block flags via the
            # config-gated vit_act_checkpoint patch (flips use_act_checkpoint +
            # deterministic-autocast wrap so non-reentrant recompute is
            # metadata-consistent). Fixes the dead static entry point (#89).
            _patch_enable_vit_act_checkpoint(raw_model)
```

- [ ] **Step 4: Run the test to verify it passes.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_construct_raw_model_grad_ckpt.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Lint gate + commit.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
git add src/custom_sam_peft/models/sam3.py tests/unit/test_construct_raw_model_grad_ckpt.py
git commit -m "fix(sam3): wire static gradient_checkpointing branch to vit_act_checkpoint patch (#89)"
```

**Completion criteria:** static branch calls the patch when enabled, skips when disabled, preserves the `set_grad_checkpointing` preferred path; lint gate green.

---

## Task 4: Dynamic entry point — OOM ladder applies the patch to the live model + CPU test

**Phase:** 2. **CPU-runnable now.** In `_train_step_with_oom_ladder` (`loop.py:126-140`), after `state.gradient_checkpointing = True`, actually apply the patch to the live `model` before the retry `continue`. The patch's idempotency makes a repeated apply harmless.

**Difficulty:** Medium. **Subagent:** implementer (sonnet/high).

**Implementer-confirmation seam (spec §8):** the ladder receives the live `Sam3Wrapper` as `model` (call site `loop.py:317-318`). The descent path to the raw ViT-Det blocks is: `Sam3Wrapper.model` (the `_Sam3ImageAdapter`) → `_Sam3ImageAdapter.model` (the raw sam3 model) → its ViT-Det blocks. **However**, `vit_act_checkpoint.apply` uses `model.modules()`, which recurses the ENTIRE tree — so calling the shim on the top-level `model` (whatever object it is) reaches the blocks regardless of nesting. Confirm: do NOT hand-descend; just call `_patch_enable_vit_act_checkpoint(model)` on whatever the ladder holds, relying on `.modules()` recursion. The `OomEvent` "grad_ckpt_enabled" record (`train/types.py:30-33`) is unchanged.

> **Caution — CPU test isolation:** the ladder helper currently catches `torch.cuda.OutOfMemoryError`. The existing tests inject that via a stub model. Do NOT import `sam3.py`'s shim at module top of the test if it would pull in heavy deps; import lazily or monkeypatch the shim symbol in `loop`'s namespace.

**Files:**
- Modify: `src/custom_sam_peft/train/loop.py:126-140`
- Test: `tests/unit/test_trainer_oom_retry.py` (add a case)

- [ ] **Step 1: Write the failing CPU test (add to the existing file).**

Append to `tests/unit/test_trainer_oom_retry.py`:

```python
def test_grad_ckpt_rung_applies_patch_to_live_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the ladder flips gradient_checkpointing on, it must apply the
    activation-checkpoint patch to the live model (Break 3 / #89), not just
    set the state flag."""
    import custom_sam_peft.train.loop as loop_mod

    applied: list[object] = []
    monkeypatch.setattr(
        loop_mod, "_patch_enable_vit_act_checkpoint", lambda m: applied.append(m), raising=False
    )

    state = _State(micro_batch_size=1, gradient_checkpointing=False)
    model = _OomThenOk(n_oom=1)  # mb already 1 → first OOM flips ckpt, retry succeeds
    _train_step_with_oom_ladder(model, _make_batch(1), state, forward_call=_fake_forward_call)

    assert state.gradient_checkpointing is True
    assert applied == [model], "patch not applied exactly once to the live model on the ckpt rung"
```

- [ ] **Step 2: Run the test to verify it fails.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_trainer_oom_retry.py::test_grad_ckpt_rung_applies_patch_to_live_model -v
```

Expected: FAILS — either `AttributeError` (no `_patch_enable_vit_act_checkpoint` symbol in `loop`) or `assert [] == [model]`.

- [ ] **Step 3: Import the shim into `loop.py` and apply it in the rung.**

In `src/custom_sam_peft/train/loop.py`, add a lazy import inside the rung (keeps module-import cost low and matches the existing lazy-import style elsewhere). Modify the `grad_ckpt_enabled` rung (currently lines 126-140):

```python
            if not state.gradient_checkpointing:
                state.gradient_checkpointing = True
                # Break 3 fix (#89): actually enable checkpointing on the LIVE
                # model before retry — setting the flag alone never reached the
                # model, so the retry ran unchanged and OOMed identically. The
                # patch is idempotent, so a repeated apply across steps is safe.
                from custom_sam_peft.models.sam3 import _patch_enable_vit_act_checkpoint

                _patch_enable_vit_act_checkpoint(model)
                state.pending_oom_events.append(
                    OomEvent(
                        step=state.step,
                        action="grad_ckpt_enabled",
                        new_micro_batch_size=state.micro_batch_size,
                        new_gradient_checkpointing=True,
                    )
                )
                _LOG.warning(
                    "OOM at step %d — enabling gradient_checkpointing on the live model",
                    state.step,
                )
                continue
```

> For the test's `monkeypatch.setattr(loop_mod, "_patch_enable_vit_act_checkpoint", ...)` to intercept the call, the symbol must be resolvable in `loop`'s namespace. The lazy `from ... import` inside the function binds a LOCAL name that monkeypatching the module attribute will NOT intercept. **Fix:** instead, add a module-level lazy import at the top of `loop.py` is undesirable (heavy). The clean pattern: import the module and call through it, OR expose a module-level reference. Use this form so the spy works — at module top of `loop.py` add:
>
> ```python
> if TYPE_CHECKING:
>     pass  # (existing imports unchanged)
> ```
>
> and in the rung call via the module so the monkeypatch target matches:
>
> ```python
>                 from custom_sam_peft.models import sam3 as _sam3_mod
>                 _sam3_mod._patch_enable_vit_act_checkpoint(model)
> ```
>
> Then change the test's monkeypatch target to `custom_sam_peft.models.sam3._patch_enable_vit_act_checkpoint`. **Implementer: pick ONE consistent approach** — either (a) module-call form above with the test patching `sam3._patch_enable_vit_act_checkpoint`, or (b) bind a module-level alias `_patch_enable_vit_act_checkpoint = None` lazily set. Approach (a) is simplest and is the recommended one; update Step 1's test to:
>
> ```python
>     import custom_sam_peft.models.sam3 as sam3_mod
>     applied: list[object] = []
>     monkeypatch.setattr(
>         sam3_mod, "_patch_enable_vit_act_checkpoint", lambda m: applied.append(m)
>     )
> ```

- [ ] **Step 4: Run the full OOM-ladder test file to verify the new case passes and the existing 11 still pass.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_trainer_oom_retry.py -v
```

Expected: all passed (the prior cases + the new one). In particular `test_oom_ckpt_toggle_is_once` still passes (the patch is only applied on the flip, and idempotency means a redundant apply is harmless).

- [ ] **Step 5: Lint gate + commit.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
git add src/custom_sam_peft/train/loop.py tests/unit/test_trainer_oom_retry.py
git commit -m "fix(train): OOM ladder applies vit_act_checkpoint patch to live model (#89)"
```

**Completion criteria:** the `grad_ckpt_enabled` rung applies the patch to the live model exactly once on flip; existing ladder tests unchanged-green; lint gate green.

---

## Task 5: Flip the default + all shipped YAMLs

**Phase:** 2. **CPU-runnable now.** Flip `ModelConfig.gradient_checkpointing` default to `True`, drop the `#60` TODO, and flip all 9 shipped YAMLs. All must still validate against `TrainConfig` (existing parametrized loader `tests/unit/test_config_examples.py`).

**Difficulty:** Easy (config/YAML edits, no logic). **Subagent:** implementer (sonnet/high — touches schema + 9 files; keep one agent for consistency).

**Files:** `src/custom_sam_peft/config/schema.py` + the 9 YAMLs in the File map (lines as cited).

- [ ] **Step 1: Flip the schema default.**

In `src/custom_sam_peft/config/schema.py` (lines 118-120), replace:

```python
    gradient_checkpointing: bool = (
        False  # TODO(#60): re-enable when sam3 activation-checkpointing recompute mismatch is fixed
    )
```

with:

```python
    gradient_checkpointing: bool = True
```

- [ ] **Step 2: Flip the 6 YAMLs carrying the `#60` comment.**

In each, change `gradient_checkpointing: false` → `gradient_checkpointing: true` and remove the trailing `# see issue #60 …` comment (or re-point to `#89` if a comment is desired — prefer removing it):
- `src/custom_sam_peft/cli/templates/coco_text_lora.yaml:21`
- `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml:21`
- `configs/examples/coco_text_lora.yaml:18`
- `configs/examples/coco_text_qlora.yaml:18`
- `configs/examples/gpu_smoke_lora.yaml:10`
- `configs/examples/gpu_smoke_qlora.yaml:10`

> Use `Read` on each file first to capture the exact line (the comment text varies); then `Edit` the exact string. Do NOT blind-`sed`.

- [ ] **Step 3: Flip the 3 YAMLs with no comment.**

`gradient_checkpointing: false` → `true`:
- `configs/examples/coco_text_auto_split.yaml:16`
- `configs/examples/coco_text_no_val.yaml:18`
- `configs/examples/coco_text_lora_subset.yaml:12`

> Spec §5 Phase 2 item 3: if any of these is a deliberate "checkpointing-off illustrative" config, document the exception inline and leave it `false`. Absent that intent (none is documented), flip it. Use judgment; default is flip.

- [ ] **Step 4: Verify all examples still validate against `TrainConfig`.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_config_examples.py -v
```

Expected: all parametrized cases pass (the loader validates every shipped YAML). Also confirm no test asserted the old `False` default:

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run grep -rn "gradient_checkpointing" tests/ | grep -i "false\|is False\|== False" || echo "no stale False assertion"
```

Expected: `no stale False assertion`, OR a hit the implementer updates to match the new `True` default.

- [ ] **Step 5: Lint gate + commit.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
git add src/custom_sam_peft/config/schema.py src/custom_sam_peft/cli/templates/*.yaml configs/examples/*.yaml
git commit -m "feat(config): default gradient_checkpointing=true; flip shipped YAMLs (#89)"
```

**Completion criteria:** default is `True`, `#60` TODO gone, all 9 YAMLs flipped and still validate; lint gate green.

---

## Task 6 (GATE — blocked on Task 7): Phase 0 diagnostic on Colab T4

**RUN ON COLAB T4. This task GATES the Phase-1 fix (Task 8 below).** Its output is a recorded diagnosis in the PR description (and a §3 spec correction if the trace diverges from the hypothesis), NOT shipped code. The scratch re-introduction is reverted before any fix lands.

> **Ordering note:** this is labeled "Task 6" by spec phase (Phase 0) but is dispatched AFTER Tasks 1–5 (which are CPU-runnable and tier-independent) and BEFORE Task 8 (the fix body). The orchestrator pauses here for a real T4 run.

**Difficulty:** Hard (diagnostic; requires T4). **Subagent:** none on the dev box — this is a manual T4 run by the user/orchestrator. The orchestrator surfaces the protocol and waits for the trace.

**Files:** none shipped. Scratch only. Optionally add a throwaway cell to `notebooks/colab_gpu_tests.ipynb` (Task 7) — but the diagnostic re-introduction is reverted, not committed.

- [ ] **Step 1 (RUN ON COLAB T4): Enable the checkpoint debugger and reproduce on the QLoRA fast-smoke.**

Copy-paste cell (the QLoRA path is the one that originally failed — it disables outer autocast, spec §3.2):

```python
# Colab T4 cell — Phase 0 diagnostic. THROWAWAY; do not commit the re-introduction.
import torch.utils.checkpoint as ckpt
ckpt.set_checkpoint_debug_enabled(True)   # per-op forward/recompute metadata table

# Ensure the flag-flip patch is wired (Tasks 1-3 already land it; confirm
# cfg.gradient_checkpointing=True routes through _patch_enable_vit_act_checkpoint).
# Run the QLoRA fast-smoke and CAPTURE the full CheckpointError trace:
!cd /content/custom-sam-peft && pytest -m gpu tests/gpu/test_real_train_qlora.py::test_qlora_smoke_fast -v 2>&1 | tee /content/phase0_qlora_trace.txt
```

Portable equivalent:

```bash
pytest -m gpu tests/gpu/test_real_train_qlora.py::test_qlora_smoke_fast -v
```

- [ ] **Step 2 (RUN ON COLAB T4): Identify the divergent op and classify.**

From the per-op metadata table, find the FIRST slot where forward and recompute disagree (the spec's "shift-by-one"). Classify into exactly one of:
- **autocast-only** → **Fix A** (deterministic-autocast wrap).
- **needs RNG/full-context control** → **Fix B** (`context_fn` pinning autocast + RNG).
- **benign non-differentiable divergence** (divergent tensors provably do not flow into any gradient) → **Fix C** (`determinism_check="none"` + a GPU gradient-parity gate).

- [ ] **Step 3 (RUN ON COLAB T4): Also run the LoRA smoke (outer autocast ON) to confirm whether it reproduces.**

```bash
pytest -m gpu tests/gpu/test_real_train_overfits.py::test_lora_smoke_fast -v
```

(Adjust the test id to the actual fast-smoke name in `test_real_train_overfits.py`.) The chosen fix must satisfy both PEFT paths.

- [ ] **Step 4: Record the diagnosis in the PR description.**

Record: the divergent op, its forward-vs-recompute metadata, the classification, and the chosen fix tier. If the trace contradicts spec §3, the implementer (Task 8) folds a correction into `docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md` §3.

- [ ] **Step 5: Revert the scratch re-introduction.** Phase 0 ships nothing.

**Completion criteria:** a recorded classification (A / B / C) with the divergent-op evidence. **The orchestrator does not dispatch Task 8 until this classification exists.**

---

## Task 7: Add the Phase-0 + Phase-3 Colab cells to the notebook

**Phase:** 0 + 3 scaffolding. **CPU-editable now** (notebook JSON edits; the cells RUN ON COLAB T4). Do this alongside Task 6 so the diagnostic protocol is reproducible and transfers to auto-provisioned GPUs (#124/#125).

**Difficulty:** Easy (notebook cell authoring). **Subagent:** implementer (haiku acceptable per Orchestrator routing for non-code, but sonnet/high is safer given the notebook JSON; use sonnet/high).

**Files:** Modify `notebooks/colab_gpu_tests.ipynb`.

- [ ] **Step 1: Read the notebook to find the existing GPU-test cell pattern and the install cell.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run jupyter nbconvert --to script --stdout notebooks/colab_gpu_tests.ipynb 2>/dev/null | head -80
```

- [ ] **Step 2: Add a markdown + code cell pair "## Phase 0 — Gradient-checkpointing diagnostic (RUN ON COLAB T4)"** containing the Task-6 Step-1/Step-3 cells (debugger enable + QLoRA-then-LoRA smoke with trace capture).

- [ ] **Step 3: Add a markdown + code cell pair "## Phase 3 — Gradient-checkpointing verification (RUN ON COLAB T4)"** containing the Task-9 verification command (`pytest -m gpu tests/gpu/test_grad_checkpointing.py -v`).

> Use the `NotebookEdit` tool (load its schema via ToolSearch) for cell insertion rather than hand-editing JSON.

- [ ] **Step 4: Confirm the notebook still parses.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run python -c "import json,nbformat; nbformat.read('notebooks/colab_gpu_tests.ipynb', as_version=4); print('notebook OK')"
```

Expected: `notebook OK`.

- [ ] **Step 5: Commit (no lint gate needed for notebook-only; still run ruff/format to be safe).**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff format --check . ; git add notebooks/colab_gpu_tests.ipynb && git commit -m "docs(notebook): add Phase-0 diagnostic + Phase-3 verification cells for grad checkpointing"
```

**Completion criteria:** notebook has the two RUN ON COLAB T4 cell-pairs and parses.

---

## Task 8 (BLOCKED on Task 6's classification): Phase 1 fix body in `vit_act_checkpoint.py`

**Phase:** 1. **The fix body is GPU-verified, not CPU-TDD'd** — the autocast-determinism / recompute correctness is GPU-only (verified in Task 9). Implement the LOWEST tier the Task-6 trace justifies. Escalate only on evidence.

**Difficulty:** Hard. **Subagent:** implementer (sonnet/high; opus/xhigh if the trace points to Fix B/C, which involve `context_fn`/`determinism_check` subtleties).

**Files:**
- Modify: `src/custom_sam_peft/models/_patches/vit_act_checkpoint.py`
- (Fix C only) Modify: `tests/gpu/test_grad_checkpointing.py` (add gradient-parity gate — see Task 9)
- (If trace diverges from §3) Modify: `docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md` §3

**Entry criteria:** Task 6 returned a classification. Do NOT start otherwise.

### Tier A (DEFAULT — entry: classification is "autocast-only", or no classification yet rules it out)

- [ ] **Step A1: Add the deterministic-autocast wrap to `apply`.**

Extend `vit_act_checkpoint.apply` so that, for each exposing block, in addition to flipping the flag, it wraps the block's `forward` so the block runs under an explicit deterministic autocast. Append after the flag-flip inside the loop (guarded by the same sentinel so it is idempotent — wrap exactly once):

```python
import functools

import torch


def _wrap_forward_with_autocast(module: nn.Module, runtime: Runtime) -> None:
    """Wrap module.forward so it always runs under the same explicit autocast.

    Non-reentrant torch.utils.checkpoint re-runs the SAME wrapped callable on
    backward recompute, so reconstructing the autocast context identically on
    forward and recompute pins SDPA backend selection / casts -> no metadata
    divergence (the CheckpointError shift-by-one). The wrap is per-ViT-Det-block
    (trunk, upstream of the decoder), so it does NOT envelop
    decoder.forward_ffn's autocast(enabled=False) fp32 region (spec §3.4 landmine).
    """
    orig_forward = module.forward
    enabled = runtime.device.type == "cuda"

    @functools.wraps(orig_forward)
    def _wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        ):
            return orig_forward(*args, **kwargs)

    module.forward = _wrapped  # type: ignore[method-assign]
```

And in the `apply` loop, after `setattr(submodule, _SENTINEL_ATTR, True)`:

```python
        _wrap_forward_with_autocast(submodule, runtime)
```

> **Keep PyTorch's recompute metadata check ON** (default `determinism_check="default"`). Fix A's correctness rests on the wrapped callable being deterministic, not on disabling the check. Do NOT touch `determinism_check` in Tier A.

- [ ] **Step A2: Confirm the CPU mechanism tests still pass** (the wrap must not break the flag-flip; on the synthetic CPU model `enabled=False`, so the wrap is a transparent passthrough).

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest tests/unit/test_sam3_act_checkpoint_patch.py -v
```

Expected: still 4 passed. Add one CPU test asserting the wrap is applied (forward still returns the same result, and `module.forward` differs from the unwrapped block's bound method) — a transparency check, NOT an autocast-determinism check (that is GPU-only):

```python
def test_wrap_is_transparent_on_cpu() -> None:
    model = _FakeModel(n_blocks=1)
    blk = model.blocks[0]
    x = torch.randn(3, 2)
    before = blk(x)
    vit_act_checkpoint.apply(model, _CPU_RUNTIME)
    after = blk(x)
    assert torch.allclose(before, after), "wrap changed CPU forward output"
```

- [ ] **Step A3: Lint gate + commit.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
git add src/custom_sam_peft/models/_patches/vit_act_checkpoint.py tests/unit/test_sam3_act_checkpoint_patch.py
git commit -m "feat(sam3): Fix A deterministic-autocast wrap for vit_act_checkpoint (#89)"
```

**Tier-A completion criteria:** wrap added, CPU transparency test green, metadata check left ON. Proceed to Task 9 (T4 verify). Escalate ONLY if Task 9 still raises `CheckpointError` or fails loss parity.

### Tier B (ESCALATE — entry: Task 6 classified "needs RNG/full-context control", OR Task 9 with Tier A still raises `CheckpointError`/fails parity)

- [ ] **Step B1: Patch the block forward to invoke `torch.utils.checkpoint.checkpoint` ourselves with a `context_fn`.**

Instead of relying on sam3's internal `checkpoint.checkpoint(blk, x, use_reentrant=False)`, wrap so OUR checkpoint call controls the context. Add to the patch:

```python
import torch.utils.checkpoint as torch_ckpt


def _deterministic_context_fn(runtime: Runtime):  # type: ignore[no-untyped-def]
    """Return a (forward_ctx, recompute_ctx) pair entering the SAME autocast
    (and, if the trace implicates RNG, the same fork_rng / preserved RNG state)
    so forward and recompute are bit-reproducible."""
    def _mk():  # type: ignore[no-untyped-def]
        return torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.device.type == "cuda",
        )
    return _mk(), _mk()
```

and wrap the block forward to call `torch_ckpt.checkpoint(orig_forward, *args, use_reentrant=False, context_fn=lambda: _deterministic_context_fn(runtime), **kwargs)`. **Implementer:** if the Task-6 trace implicates RNG (dropout / stochastic depth), extend `_mk()` to also enter a `torch.random.fork_rng`-preserved context per the PyTorch `context_fn` contract; the exact RNG-pinning form depends on the trace. Tier B subsumes Tier A (it also pins autocast).

> **Seam to confirm:** when our wrap calls `torch_ckpt.checkpoint` AND sam3's block also self-checkpoints internally (`vitdet.py:982` reads `use_act_checkpoint`), you would double-checkpoint. To avoid that, when entering Tier B, set `use_act_checkpoint=False` on the block (so sam3 does NOT self-checkpoint) and let OUR wrap own the checkpoint call. The Task-6 trace + a T4 run confirms which composition is correct. Document the choice inline.

- [ ] **Step B2: CPU tests still green + lint gate + commit.** Then re-run Task 9 on T4.

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest tests/unit -q
git add src/custom_sam_peft/models/_patches/vit_act_checkpoint.py
git commit -m "feat(sam3): escalate to Fix B (context_fn-pinned checkpoint) for vit_act_checkpoint (#89)"
```

### Tier C (LAST RESORT — entry: Task 6 OR a Tier-B trace PROVES the divergent tensors are non-differentiable AND no autocast/RNG pinning resolves the mismatch)

- [ ] **Step C1: Set `determinism_check="none"` on the affected checkpoint call** AND add the GPU gradient-parity test (Task 9 Step 5) as a MERGE GATE. Fix C is not allowed without that evidence.

```python
        torch_ckpt.checkpoint(
            orig_forward, *args,
            use_reentrant=False,
            context_fn=lambda: _deterministic_context_fn(runtime),
            determinism_check="none",
            **kwargs,
        )
```

- [ ] **Step C2: Commit; the gradient-parity GPU test (Task 9) gates the merge.**

**Escalation discipline:** default A; A→B on the explicit triggers above; B→C only with proof of non-differentiability + the parity gate. "At all costs" is bounded by this tree, not open-ended thrash.

---

## Task 9 (BLOCKED on Task 8): Phase 3 GPU verification on Colab T4

**Phase:** 3. **RUN ON COLAB T4.** GPU tests are release-tier (`-m gpu`); they collect-and-skip on the dev box via `requires_compatible_gpu` / `requires_checkpoint` (`tests/conftest.py:55-57`). Write the test file on the dev box (it imports cleanly and is collectable), then RUN ON COLAB T4.

**Difficulty:** Medium (test authoring) + Hard (T4 interpretation). **Subagent:** implementer (sonnet/high) writes the GPU test file on the dev box; the T4 run is manual/orchestrator.

**Files:** Create `tests/gpu/test_grad_checkpointing.py`.

**Pattern reference:** `tests/gpu/test_real_train_qlora.py:1-74` — `pytestmark` with `gpu`/`requires_compatible_gpu`/`requires_checkpoint`, `_RecordingTracker`, `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()`.

- [ ] **Step 1: Write the GPU test file (collectable on the dev box, runs on T4).**

Create `tests/gpu/test_grad_checkpointing.py`:

```python
"""GPU verification for gradient checkpointing on T4 (#89).

Release-tier (-m gpu); collect-and-skip on CPU. Run on Colab T4:
    pytest -m gpu tests/gpu/test_grad_checkpointing.py -v

Verifies (spec §4 acceptance): no CheckpointError on LoRA AND QLoRA smokes
(forward+backward complete); step-0 loss parity vs checkpointing-off (recompute
is numerically exact); peak VRAM measurably LOWER with checkpointing ON than OFF.
The absolute 14/10 GB ceilings live in the existing smoke tests and are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from custom_sam_peft.config.loader import load_config
from custom_sam_peft.train.runner import run_training
from tests.gpu.conftest import _RecordingTracker

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]

_CFG_DIR = Path(__file__).resolve().parents[2] / "configs" / "examples"
_LORA = _CFG_DIR / "gpu_smoke_lora.yaml"
_QLORA = _CFG_DIR / "gpu_smoke_qlora.yaml"
VRAM_MARGIN_GB = 0.3  # checkpointing-on must be lower by at least this margin


def _run(cfg_path: Path, tmp_path: Path, tiny_coco_dir: Path, monkeypatch, grad_ckpt: bool):
    cfg = load_config(
        cfg_path,
        overrides=[
            f"data.train.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.train.images={tiny_coco_dir / 'images'}",
            f"data.val.annotations={tiny_coco_dir / 'annotations.json'}",
            f"data.val.images={tiny_coco_dir / 'images'}",
            f"run.output_dir={tmp_path}",
            f"model.gradient_checkpointing={'true' if grad_ckpt else 'false'}",
        ],
    )
    tracker = _RecordingTracker()
    monkeypatch.setattr("custom_sam_peft.train.runner.build_tracker", lambda *_a, **_kw: tracker)
    torch.cuda.reset_peak_memory_stats()
    run_training(cfg)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    losses = [s["loss/total"] for _, s in tracker.scalars if s["loss/total"] > 0]
    return losses[0], peak_gb


def test_lora_no_checkpoint_error_and_vram_lower(tmp_path, tiny_coco_dir, monkeypatch) -> None:
    off_loss0, off_peak = _run(_LORA, tmp_path / "off", tiny_coco_dir, monkeypatch, grad_ckpt=False)
    on_loss0, on_peak = _run(_LORA, tmp_path / "on", tiny_coco_dir, monkeypatch, grad_ckpt=True)
    # No CheckpointError == run_training returned. Loss parity:
    assert abs(on_loss0 - off_loss0) <= 1e-2 * max(1.0, abs(off_loss0)), (
        f"step-0 loss parity failed: on={on_loss0} off={off_loss0}"
    )
    # VRAM lever is live:
    assert on_peak <= off_peak - VRAM_MARGIN_GB, (
        f"checkpointing did not lower peak VRAM: on={on_peak:.2f}GB off={off_peak:.2f}GB"
    )


@pytest.mark.requires_bnb
def test_qlora_no_checkpoint_error_and_vram_lower(tmp_path, tiny_coco_dir, monkeypatch) -> None:
    from tests.gpu.conftest import _bnb_available

    if not _bnb_available():
        pytest.skip("bitsandbytes not installed")
    off_loss0, off_peak = _run(_QLORA, tmp_path / "off", tiny_coco_dir, monkeypatch, grad_ckpt=False)
    on_loss0, on_peak = _run(_QLORA, tmp_path / "on", tiny_coco_dir, monkeypatch, grad_ckpt=True)
    assert abs(on_loss0 - off_loss0) <= 1e-2 * max(1.0, abs(off_loss0)), (
        f"QLoRA step-0 loss parity failed: on={on_loss0} off={off_loss0}"
    )
    assert on_peak <= off_peak - VRAM_MARGIN_GB, (
        f"QLoRA checkpointing did not lower peak VRAM: on={on_peak:.2f}GB off={off_peak:.2f}GB"
    )
```

> **Implementer:** confirm `_RecordingTracker`, `tiny_coco_dir`, and `_bnb_available` are importable from `tests/gpu/conftest.py` (they are used by `test_real_train_qlora.py`). Confirm `load_config` accepts a `model.gradient_checkpointing=` override string — if the loader coerces booleans differently, pass a Python bool via the override mechanism or set it on `cfg.model` after load.

- [ ] **Step 2: Add the OOM-ladder rung GPU assertion.**

Append a test that forces the `micro_batch_size==1` rung and asserts the live model's ViT-Det blocks have `use_act_checkpoint=True` after the rung fires and the retry proceeds. Use the spec §7.2 "OOM-ladder rung exercised" intent. If forcing a real OOM is impractical on T4, simulate by invoking `_train_step_with_oom_ladder` with `state.micro_batch_size=1` and a forward that raises `torch.cuda.OutOfMemoryError` once, then assert the wrapper's blocks (`wrapper.model.model.modules()` reaching every block) have `use_act_checkpoint is True`. Mark it `@pytest.mark.gpu` only if it needs a real model; if it can use a synthetic wrapper it belongs in CPU tests (Task 4 already covers the apply-call; this GPU test confirms the descent reaches REAL blocks).

- [ ] **Step 3: Confirm the file collects-and-skips on the dev box.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest -m gpu tests/gpu/test_grad_checkpointing.py -v
```

Expected: all tests SKIPPED (no compatible GPU / no checkpoint on the dev box), zero collection errors.

- [ ] **Step 4 (RUN ON COLAB T4): Run the full verification.**

```bash
pytest -m gpu tests/gpu/test_grad_checkpointing.py -v
```

Plus re-run the release suite under the UNCHANGED ceilings to prove no regression:

```bash
pytest -m gpu tests/gpu/test_real_train_overfits.py tests/gpu/test_real_train_qlora.py -v
```

Expected: no `CheckpointError`; loss-parity assertions pass; `on_peak < off_peak - 0.3GB`; existing 14/10 GB ceiling assertions still green. Capture the on/off loss and VRAM numbers for the PR description. Also confirm NO re-introduction of the `decoder.forward_ffn` bf16-vs-fp32 collision (spec §3.4/§6 step 6) — the per-block wrap is upstream of the decoder; the run proves it.

- [ ] **Step 5 (Fix-C only): gradient-parity gate.** If Task 8 selected Tier C, add and run a GPU test asserting gradients with checkpointing-on match checkpointing-off within a tight tolerance. This gates the merge.

- [ ] **Step 6: Commit the GPU test file (the T4 run results go in the PR description, not the repo).**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -m gpu tests/gpu/test_grad_checkpointing.py -v
git add tests/gpu/test_grad_checkpointing.py
git commit -m "test(gpu): verify grad checkpointing — no CheckpointError, loss parity, VRAM lower (#89)"
```

**Completion criteria:** on a real T4: no `CheckpointError` on both PEFT paths, loss parity, VRAM measurably lower, release ceilings unchanged-green. If Tier A fails any of these → escalate to Task 8 Tier B; re-run. The fix file collects-and-skips cleanly on the dev box.

---

## Task 10: Phase 4 bookkeeping

**Phase:** 4. **CPU/dev-box.** Re-scope #89, comment on the accidentally-closed #60, reconcile the two stale specs. Mechanical but real work.

**Difficulty:** Easy. **Subagent:** implementer (sonnet/high for the spec edits; the `gh` calls are orchestrator-run).

**Files:**
- Modify: `docs/superpowers/specs/2026-05-18-smoke-test-design.md`
- Modify: `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md`
- GitHub: #89 (re-scope), #60 (comment).

- [ ] **Step 1: Re-scope #89.**

```bash
gh issue edit 89 --add-assignee @me
gh issue view 89 --json body -q .body  # capture current body first
gh issue edit 89 --body "<updated body: 'fix activation-checkpointing recompute + re-wire both entry points + flip defaults'; link docs/superpowers/specs/2026-05-23-gradient-checkpointing-t4-design.md>"
```

> The PR's `Closes #89` lands the merge.

- [ ] **Step 2: Comment on #60 (accidental-closure correction).**

```bash
gh issue comment 60 --body "The deeper activation-checkpointing investigation #60 was meant to track (per 2dc4883's 'stays open' note) is resolved by the PR closing #89. #60 was closed COMPLETED on 2026-05-21 accidentally — a stray 'Closes #60.' rode along in the reverted feat 889bd2c and fired when PR #58 merged. Recording the correction here so the audit trail is intact; do not rely on #60's closed state as evidence the work was done."
```

- [ ] **Step 3: Reconcile `2026-05-18-smoke-test-design.md`.**

Add a dated reconciliation note near §5.1 (the `gpu_smoke_lora.yaml` listing showing `gradient_checkpointing: true` at ~line 259) and the §5.4 ceiling rationale (~lines 211-212):

```markdown
> **Reconciliation (2026-05-23, #89):** `gpu_smoke_lora.yaml` now ships
> `gradient_checkpointing: true` (the flag was flipped to `false` under #60 and
> is re-enabled by #89's working activation-checkpointing fix), so this listing
> is accurate again. The VRAM-ceiling rationale can now cite a *working*
> checkpointing lever (verified: peak VRAM lower with checkpointing on, on T4).
```

- [ ] **Step 4: Reconcile `2026-05-19-gpu-test-policy-design.md` §5.4.**

```markdown
> **Note (2026-05-23, #89):** the "gradient checkpointing knobs" named here as a
> T4-ceiling remediation are now FUNCTIONAL (they were a dead reference before
> #89's fix). The "ceilings must not be raised" policy is unchanged and
> reaffirmed — the fix reduces usage under the existing 14/10 GB ceilings.
```

- [ ] **Step 5: Lint gate (docs only; run format check) + commit.**

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff format --check .
git add docs/superpowers/specs/2026-05-18-smoke-test-design.md docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md
git commit -m "docs: reconcile smoke-test + gpu-test-policy specs; grad checkpointing lever now live (#89)"
```

**Completion criteria:** #89 re-scoped + assigned, #60 comment posted, both specs carry dated reconciliation notes.

---

## Final verification (before PR)

- [ ] Full CPU gate green:

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

- [ ] GPU tests collect-and-skip cleanly on the dev box:

```bash
cd /home/justin/projects/custom-sam-peft/.worktrees/gradient-checkpointing-89 && uv run pytest -m gpu -q 2>&1 | tail -3
```

Expected: all skipped, zero errors.

- [ ] The Phase-0 trace classification AND the Phase-3 T4 numbers (on/off loss, on/off peak VRAM) are recorded in the PR description (per `superpowers:verification-before-completion` — evidence before assertions). The PR body links the spec and this plan and carries `Closes #89`.

---

## Self-review notes (spec coverage map)

- **Phase 0** → Task 6 (diagnostic GATE) + Task 7 (notebook cell).
- **Phase 1** → Task 1 (flag-flip half + tests) + Task 2 (shim) + Task 8 (fix body, tier A/B/C).
- **Phase 2** → Task 3 (static), Task 4 (dynamic OOM ladder), Task 5 (default + 9 YAMLs).
- **Phase 3** → Task 9 (GPU verify) + Task 7 (notebook cell).
- **Phase 4** → Task 10 (bookkeeping).
- **§6 inventory:** every file covered — new patch (T1/T8), `sam3.py` shim+branch (T2/T3), `loop.py` (T4), `schema.py`+9 YAMLs (T5), CPU tests (T1/T3/T4), GPU test (T9), notebook (T7), README (T1), two stale specs + §3 fold-back (T10/T8). `_patches/__init__.py` is intentionally NOT modified (config-gated patch is NOT in `_ALL_PATCHES`) — confirmed in T1/T3 commentary.
