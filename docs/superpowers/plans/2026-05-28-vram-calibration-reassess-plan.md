# Reassess VRAM calibration: tightened formula + config-aware probe + B-then-K OOM ladder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the under-predicting VRAM estimator with a tightened analytic formula (multiplex-K + SDPA attention), an opt-in config-aware calibration probe, an `init`/`run`/`calibrate` wiring layer that bakes concrete sizing into `config.yaml`, and a B-then-K OOM ladder that re-chunks (never drops) classes — shipped as one PR (spec §1–§9).

**Architecture:** Three layers tied together by `config.yaml` as the frozen source of truth. (a) The **formula** (`presets.py`) is the always-available floor; it gains a `K_eff` activation term and a shared SDPA-attention helper, and `decide_preset` threads `k`. (b) The **probe** (`calibrate_cmd.py`) is the opt-in precision upgrade; it reads the config, probes at representative `(image_size, k, method, r, batch)`, rewrites the config in place, and writes the cache. (c) The **OOM ladder** (`train/loop.py`) is the run-time safety net; it halves micro-batch B to 1, then halves `effective_K` to 1 (re-chunking ALL classes into more, smaller groups and replaying the whole step), then hard-fails. Wiring (`init`/`run`/`calibrate` skip-init guards + wizard opt-in) closes the UX gap so the chosen preset actually reaches a run. Provenance (`runs/bundle.py`) renders formula-vs-calibrated and the new `multiplex_halved` events.

**Tech Stack:** Python 3, PyTorch (CUDA memory model + `torch.cuda.OutOfMemoryError`), Pydantic v2 (`TrainConfig` schema/validation), Typer + rich (CLI/prompts), `string.Template` (config template), manual targeted line-rewrite (pyyaml only — no new dep) for in-place annotation, pytest + pytest-cov (TDD, 80% gate on the FULL suite), pytest markers `gpu_local`/`gpu_t4`/`gpu_xl`/`requires_checkpoint`/`requires_compatible_gpu` for the GPU tier.

---

## Ground-truth facts verified against source (read before starting)

The spec was written against a slightly different mental model than the current `main`. These were confirmed by reading the actual files; the tasks below target the **real** symbols:

1. `decide_preset()` takes **no arguments** today — `image_size` comes from `SAM3_IMAGE_SIZE` (a module constant in `models/sam3.py`, value `1008`) via a local import (`presets.py:280-282`). Component 1's "thread `k`" means adding a `k` parameter to `decide_preset`, **not** an `image_size` parameter.
2. `PresetDecision` has **no `image_size` field** (it was removed in an earlier PR). Do not reintroduce one. Fields are: `method, r, batch_size, grad_accum_steps, dtype, headroom_bytes, predicted_bytes, budget_bytes, gpu_name, provenance, cache_path, calibrated_at`.
3. `_load_cache(gpu_name)` takes a single argument (`presets.py:207`), not `(image_size, gpu_name)`.
4. `decide_eval_batch_size(classes_per_forward=16)` already **accepts** `classes_per_forward` but ignores it; the SDPA-attention model is inline at `presets.py:395-405` (`_SAM3_PATCH=14`, `_SAM3_HEADS=16`, `_n_tokens=(image_size//14)**2`, `_attn_per_example=H*N*N*4`). Component 1 extracts this into a shared helper used by both the train branch and this eval path.
5. `_run_probe()` takes **no arguments** (`calibrate_cmd.py:64`); it loads SAM 3.1, attaches LoRA at fixed `r=4`, builds a single `TextPrompts(classes=["thing"])` at batch=1 zeros-image, calls `wrapper(images, prompts, support=None)` (forward kwarg is `support=`, **not** `box_hints=`), and reads `max_memory_allocated`. `calibrate(output, force)` no longer takes `--image-size`; `_cache_is_fresh(path, gpu_name)`.
6. `run`/`_orchestrate` (`run_cmd.py:69-82`) **already** calls `run_training(cfg, resume_from=resume)` — the train sizing tuple is **already consumed verbatim**. The `PresetDecision` from `_load_preset_or_fallback` only feeds `BundleContext.preset` (the label). So Component 3's only new work on `run` is the **skip-init guard** (§6.2); the "consumes literals verbatim" property already holds and is asserted, not implemented.
7. In `train_step` (`loop.py:171-358`), `effective_K = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)` is a **local** (line 212); groups come from `_chunked(classes_in_batch, effective_K)`; the `for group in groups` loop accumulates `backward()` per group; `optimizer.zero_grad(set_to_none=True)` fires **inside** `train_step` at the grad-accum boundary (lines ~350-352). The inner B-halving lives entirely in `_train_step_with_oom_ladder` (`loop.py:69-126`), which **hard-fails at micro_batch=1**. The NaN-driven group-skip uses `finite_group_count`/`is_finite` and a `ValueError` catch from the Hungarian matcher (lines ~315-325) — leave it untouched.
8. `OomState` (`loop.py:54-66`) has `step`, `micro_batch_size`, `pending_oom_events`. It is constructed only at `trainer.py:489`: `OomState(micro_batch_size=cfg.train.batch_size)`.
9. `OomEvent` (`train/types.py:15-29`) is `step: int`, `action: Literal["microbatch_halved"]`, `new_micro_batch_size: int`.
10. `_oom_edge_note` (`bundle.py:333-338`) reads `events[-1].new_micro_batch_size`; `_preset_block` (`bundle.py:313-330`) renders the `- Source: calibrated …`/`- Source: analytic estimate` line.
11. `init` (`init_cmd.py`) has an `--interactive/-i` branch that calls `setup_wizard.generate_config(...)`, and a flag branch that calls `run_init(...)` which renders `config_full.yaml` via `string.Template`. `init` does **not** currently call `decide_preset` — only the wizard's `_ask_peft_sizing` does (`setup_wizard.py:432-446`, applies `decide_preset().config_patch`). The wizard's calibrate opt-in offer goes in `generate_config` (`setup_wizard.py:571+`), after emit.
12. `scripts/` exists (`bench_multiplex_throughput.py`, `run_gpu_tests.sh`); `scripts/_derive_preset_constants.py` is **new**.
13. `PEFTConfig` (`schema.py:487`) has `method`, `r` (default 16), `alpha`, `dropout`, … `MultiplexConfig.classes_per_forward` (`schema.py:551`) is `Field(default=16, ge=1, le=16)`; `MULTIPLEX_CAP = 16` and `SAM3_IMAGE_SIZE = 1008` are in `models/sam3.py`.

### Hard invariants to preserve (enforced as explicit test assertions in the noted tasks)

- (a) **K-halving re-chunks ALL classes** — every class still trained, none bypassed (Task 4).
- (b) **zero_grad + whole-step replay** on a K-rung — no mid-step resume, no double-count (Task 4).
- (c) **NaN-driven group-skip path is untouched** (`finite_group_count`/`is_finite`/`ValueError`) (Task 4 asserts it still skips on non-finite cost; the OOM ladder never marks a group non-finite).
- (d) **`eval.batch_size: auto` stays run-time** via `decide_eval_batch_size` — never baked (Task 11).
- (e) **No `auto` sentinel** added to the train tuple in the schema; `init` always bakes concrete numbers (Task 9, Task 11).
- (f) **The formula must not under-predict** the real peak on the 16 GB card (GPU Task 14, the headline 10-vs-22 GiB validation).

> **Task numbering:** tasks are numbered 1–6 and 9–14 (there is no Task 7 or Task 8 — an artifact of a mid-draft renumber). All tasks are present and uniquely titled; the gap is cosmetic. References to the OOM-ladder invariants point to Task 4.

---

## File structure

**Component 1 — formula (`presets.py`, new script):**

- `src/custom_sam_peft/presets.py` — add `_attention_bytes_per_example(image_size)` shared helper; add a `K_eff` term to the train branch of `_activation_bytes`/`_predicted_bytes`; add the attention term to the train branch; thread `k` into `decide_preset`; refactor `decide_eval_batch_size` to call the shared helper. Re-seed `MODEL_PARAMS`/`BASE_ACTIVATION_AT_1024`/the new per-`K_eff` coefficient comment.
- `scripts/_derive_preset_constants.py` — **new**; loads SAM 3.1, runs a representative probe, prints re-derived seed values.

**Component 4 — OOM ladder (`train/types.py`, `train/loop.py`):**

- `src/custom_sam_peft/train/types.py` — widen `OomEvent.action` to `Literal["microbatch_halved", "multiplex_halved"]`; add optional `effective_K: int | None = None`.
- `src/custom_sam_peft/train/loop.py` — add `effective_K` to `OomState`; raise a distinguishable B-exhausted signal from `_train_step_with_oom_ladder`; wrap the `for group in groups` block in `train_step` with a K-replay loop (zero_grad → halve `effective_K` → re-chunk → replay from group 0); hard-fail when both B and K are 1.
- `src/custom_sam_peft/train/trainer.py` — initialise `OomState(..., effective_K=...)`.

**Component 2 — probe (`cli/calibrate_cmd.py`):**

- `src/custom_sam_peft/cli/calibrate_cmd.py` — config-aware `_run_probe(...)` + `calibrate(...)`; in-place config rewrite + `# calibrated` annotation; auto-init when no config; skip-init guard.

**Component 3 — wiring (`cli/init_cmd.py`, `cli/run_cmd.py`, `cli/setup_wizard.py`):**

- `src/custom_sam_peft/cli/init_cmd.py` — `init` bakes concrete formula-derived sizing + `# formula-derived` annotation (CPU-only safe-defaults fallback + warning).
- `src/custom_sam_peft/cli/run_cmd.py` — skip-init guard.
- `src/custom_sam_peft/cli/setup_wizard.py` — post-emit opt-in `csp calibrate` offer.

**Component 8 — provenance (`runs/bundle.py`):**

- `src/custom_sam_peft/runs/bundle.py` — render `multiplex_halved` events in `_oom_edge_note` (final `effective_K` alongside final `micro_batch`).

**Tests:**

- Modify: `tests/unit/test_presets.py`, `tests/unit/test_decide_eval_batch_size.py`, `tests/unit/test_trainer_oom_retry.py`, `tests/unit/test_calibrate_cmd.py`, `tests/unit/test_cli_init.py`, `tests/unit/cli/test_setup_wizard.py`, `tests/unit/runs/test_bundle.py`, `tests/integration/test_cli_run.py`, `tests/unit/test_train_loop_multiplex.py` (or a new focused loop test).
- Modify GPU: `tests/gpu/test_calibrate_real.py` (real config-aware probe + real-model peak); **new** GPU test for the formula-accuracy validation (Task 14).

---

## Sequencing rationale (read before starting)

1. **Component 1 formula first.** The shared `_attention_bytes_per_example` helper and the `K_eff` term are imported by the probe (Component 2 validates the calibrated activation against the same overhead model) and referenced by the wiring (`init` calls `decide_preset(k=…)`). The formula must land before the probe and before any task that calls `decide_preset(k=…)`. (Phase 1, Task 1; new script Task 2.)
2. **Component 4 OOM ladder is file-disjoint from Component 1** (`train/types.py`+`train/loop.py`+`trainer.py` vs `presets.py`). It can run **in parallel** with Component 1. Within Component 4, `train/types.py` (Task 3) must land before `train/loop.py` (Task 4) because the loop constructs the widened `OomEvent`.
3. **Component 2 probe after Component 1.** `calibrate` recomputes the overhead with the same `_model_bytes`/`_adapter_bytes`/`_optimizer_bytes`/`WORKSPACE_BYTES` model and (per §3.2) the attention term; it reads `cfg.peft.r`/`method`/`k`. So Task 5 follows Task 1. (Phase 3, Tasks 5–6.)
4. **Component 3 wiring after Components 1 and 2.** `init` (Task 9) calls `decide_preset(k=…)` (needs Task 1). The `run` skip-init guard (Task 10) and `calibrate` auto-init/skip-init (Task 6) both invoke `init`. The wizard calibrate offer (Task 12) needs the reworked `calibrate` (Task 6). (Phase 4.)
5. **Component 8 provenance after Component 4** — `_oom_edge_note` renders the `multiplex_halved` events that Task 4 produces. (Phase 5, Task 13.) It is file-disjoint from everything else and could run any time after Task 3 widens `OomEvent`.
6. **GPU tasks last** (Phase 6, Task 14) — run one at a time on the 16 GB card; they validate Components 1, 2 against reality.

**Parallelizable:** Phase 1 (Component 1: Tasks 1–2) and Phase 2 (Component 4: Tasks 3–4) are file-disjoint and may run concurrently. Within Phase 4, Task 9 (`init_cmd.py`) and Task 10 (`run_cmd.py`) touch disjoint files but both depend on Task 1; serialize Task 6→Task 12 (wizard offer needs reworked calibrate). Task 13 (`bundle.py`) is independent after Task 3.

---

## Phase 1 — Component 1: tightened analytic formula (`presets.py`)

> Single file (`presets.py`) for Task 1; all sub-edits serialized into one task, then its tests. Task 2 (new script) is file-disjoint and may follow or run in parallel.

### Task 1: `K_eff` activation term + shared SDPA-attention helper + `k`-threaded `decide_preset`

**Files:**

- Modify: `src/custom_sam_peft/presets.py`
- Test: `tests/unit/test_presets.py`, `tests/unit/test_decide_eval_batch_size.py`

- [ ] **Step 1: Write the failing formula tests**

In `tests/unit/test_presets.py`, append these tests (they import the new helper + the K-aware signatures):

```python
def test_attention_bytes_helper_matches_sdpa_model() -> None:
    """The shared helper reproduces the inline SDPA model: H * N^2 * 4 bytes."""
    from custom_sam_peft.presets import _attention_bytes_per_example

    image_size = 1008
    n_tokens = (image_size // 14) ** 2  # patch=14
    expected = 16 * n_tokens * n_tokens * 4  # heads=16, fp32
    assert _attention_bytes_per_example(image_size) == expected


def test_predicted_bytes_train_grows_with_k_eff() -> None:
    """Train-mode prediction is monotone in K_eff (more classes/group -> more activation)."""
    from custom_sam_peft.presets import _predicted_bytes

    small_k = _predicted_bytes("lora", r=8, batch=1, image_size=1008, cache=None, k_eff=1)
    big_k = _predicted_bytes("lora", r=8, batch=1, image_size=1008, cache=None, k_eff=16)
    assert big_k > small_k


def test_predicted_bytes_train_includes_attention_term() -> None:
    """Train-mode prediction includes the (dominant) SDPA attention term."""
    from custom_sam_peft.presets import (
        _attention_bytes_per_example,
        _model_bytes,
        _predicted_bytes,
    )

    pb = _predicted_bytes("lora", r=8, batch=1, image_size=1008, cache=None, k_eff=1)
    # The attention term alone must be a large fraction of the prediction at 1008px.
    assert _attention_bytes_per_example(1008) > 0
    assert pb > _model_bytes("lora") + _attention_bytes_per_example(1008)


def test_decide_preset_threads_k_into_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    """decide_preset(k=...) feeds K_eff into the train formula; larger k -> larger
    predicted_bytes for the chosen preset (monotone), all else equal."""
    _patch_cuda(monkeypatch, total=int(80 * _GB))  # large card so a preset always fits
    d_small = decide_preset(k=1)
    d_big = decide_preset(k=16)
    assert d_big.predicted_bytes >= d_small.predicted_bytes


def test_decide_preset_defaults_k_to_cap_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No k supplied -> conservative worst case == MULTIPLEX_CAP."""
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

    _patch_cuda(monkeypatch, total=int(80 * _GB))
    assert decide_preset().predicted_bytes == decide_preset(k=MULTIPLEX_CAP).predicted_bytes
```

If `tests/unit/test_presets.py` lacks a `_patch_cuda` helper or a `pytest` import, reuse the file's existing CUDA-patching fixture/helper (the existing `test_decide_preset_*` tests already patch `torch.cuda`); adapt the new tests to that helper's name rather than introducing a new one. Read the top of the file first and match its convention.

In `tests/unit/test_decide_eval_batch_size.py`, add an assertion that the eval path uses the same helper:

```python
def test_eval_path_uses_shared_attention_helper() -> None:
    """The eval SDPA cap and the train attention term cite ONE definition."""
    from custom_sam_peft.presets import _attention_bytes_per_example

    # H=16, N=(1008//14)^2=5184, fp32=4 bytes.
    assert _attention_bytes_per_example(1008) == 16 * 5184 * 5184 * 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py -k "attention or k_eff or threads_k or defaults_k or shared_attention" -v`
Expected: FAIL — `_attention_bytes_per_example` does not exist (ImportError); `_predicted_bytes(...)` rejects `k_eff=` (TypeError: unexpected keyword); `decide_preset(k=...)` rejects `k=` (TypeError).

- [ ] **Step 3: Add the shared SDPA-attention helper**

In `src/custom_sam_peft/presets.py`, add constants near the existing `BASE_ACTIVATION_AT_1024` block and a helper just above `_activation_per_example`:

```python
# SAM 3.1 vision backbone (hiera-large), from sam3/model_builder.py. Shared by
# the train-branch formula and decide_eval_batch_size's SDPA ceiling so both
# cite one definition (spec §3.2).
_SAM3_PATCH = 14  # vision backbone patch size
_SAM3_HEADS = 16  # vision backbone attention heads


def _attention_bytes_per_example(image_size: int) -> int:
    """Per-example SDPA score-matrix bytes: H * N^2 * 4 (fp32 math upcast).

    At SAM 3.1's image_size=1008, patch=14 -> N=5184 tokens, so this term is the
    dominant activation contributor and is exactly what the train formula omitted
    (the 10-vs-22 GiB miss). Spec §3.2.
    """
    n_tokens = (image_size // _SAM3_PATCH) ** 2
    return _SAM3_HEADS * n_tokens * n_tokens * 4
```

- [ ] **Step 4: Add the `K_eff` term to the train activation + thread it through `_activation_bytes`/`_predicted_bytes`**

Rewrite `_activation_bytes` and `_predicted_bytes` so the train branch scales the per-example activation by `k_eff` and adds the attention term; the eval branch is unchanged in behavior (K-agnostic, attention handled by `decide_eval_batch_size`'s own cap):

```python
def _activation_bytes(
    image_size: int, batch: int, cache: dict[str, Any] | None, k_eff: int = 1
) -> int:
    # The SAM 3.1 multiplex forward materializes per-class mask/box decoder
    # activations within a group, so per-example activation scales with k_eff
    # (the per-group class count). Spec §3.1.
    per = _activation_per_example(image_size, cache)
    return int(per * batch * k_eff)


def _predicted_bytes(
    method: str,
    r: int,
    batch: int,
    image_size: int,
    cache: dict[str, Any] | None,
    mode: Literal["train", "eval"] = "train",
    k_eff: int = 1,
) -> int:
    if mode == "train":
        return (
            _model_bytes(method)
            + _adapter_bytes(r)
            + _optimizer_bytes(r)
            # Per-group activation scales with k_eff; the SDPA attention term is
            # forward+backward (NOT scaled by forward_only_factor). Spec §3.1/§3.2.
            + _activation_bytes(image_size, batch, cache, k_eff=k_eff)
            + _attention_bytes_per_example(image_size) * batch
            + WORKSPACE_BYTES
        )
    # mode == "eval": no optimizer, no adapter bytes; activations x forward_only_factor.
    # K and attention are handled by decide_eval_batch_size's own cap, not here.
    activations = int(_activation_bytes(image_size, batch, cache) * forward_only_factor)
    return _model_bytes(method) + activations + WORKSPACE_BYTES
```

- [ ] **Step 5: Thread `k` through `decide_preset`**

In `decide_preset`, add a `k` parameter that defaults to the conservative cap, and pass `k_eff` into both `_predicted_bytes` calls. The function currently starts `def decide_preset() -> PresetDecision:` and imports `SAM3_IMAGE_SIZE`. Rewrite the signature + the K resolution + the two `_predicted_bytes` calls:

```python
def decide_preset(k: int | None = None) -> PresetDecision:
    """Pick the largest configuration that fits within the VRAM budget.

    Args:
      k: representative classes-per-forward for the train activation term. When
         None, uses the conservative worst case MULTIPLEX_CAP. Callers with a
         config in scope pass cfg.train.multiplex.classes_per_forward. Spec §3.1.

    Raises:
      RuntimeError: CUDA unavailable, env-var malformed, or no candidate fits.

    Spec: design §3 + §7.
    """
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE

    image_size = SAM3_IMAGE_SIZE
    k_eff = MULTIPLEX_CAP if k is None else min(k, MULTIPLEX_CAP)
    if k_eff < 1:
        raise ValueError(f"k must be >= 1 when provided; got {k}")
    if not torch.cuda.is_available():
        raise RuntimeError(_CUDA_HINT)
    # ... (unchanged: props/total/gpu_name/cc/dtype/headroom/budget/cache) ...
```

Then in the feasible loop and the "nothing fits" branch, pass `k_eff`:

```python
    feasible = []
    for method, r, batch in _candidates():
        pb = _predicted_bytes(method, r, batch, image_size, cache, k_eff=k_eff)
        if pb <= budget:
            feasible.append((method, r, batch, pb))

    if not feasible:
        budget_gib = budget / _GB
        headroom_gib = headroom / _GB
        min_needed = _predicted_bytes("qlora", 4, 1, image_size, cache, k_eff=k_eff)
        raise RuntimeError(
            f"pick_preset(): GPU has {budget_gib:.1f} GiB after {headroom_gib:.1f} GiB "
            f"headroom — SAM 3.1 needs ≈{min_needed / _GB:.1f} GiB even at QLoRA r=4 "
            f"batch=1. Use a larger GPU."
        )
```

Leave the sort/unpack/`PresetDecision(...)` construction unchanged (no `image_size`/`k` field on the dataclass).

- [ ] **Step 6: Refactor `decide_eval_batch_size` to call the shared helper**

In `decide_eval_batch_size`, replace the inline attention block (`presets.py:395-399`) with the helper, keeping the `attn_budget` subtraction and the cap logic identical otherwise:

```python
    # Attention-memory ceiling via the shared helper so the train term and this
    # eval cap cite one definition (spec §3.2 / issue #162).
    _attn_per_example = _attention_bytes_per_example(image_size)
    # Model weights and forward activations are ALREADY resident when SDPA runs;
    # subtract them from the budget before solving for the attention-bound bs.
    attn_budget = budget - _model_bytes("lora") - WORKSPACE_BYTES
    _act_per_example = int(_activation_per_example(image_size, cache) * forward_only_factor)
    _per_example = _attn_per_example + _act_per_example
    attn_cap = max(1, attn_budget // _per_example) if attn_budget > 0 else 1
```

Delete the now-unused `_SAM3_PATCH`/`_SAM3_HEADS`/`_n_tokens` locals inside the function (they moved to module scope). Keep the warning + recompute below it unchanged.

- [ ] **Step 7: Re-seed / re-annotate the constants**

Update the comment block at `MODEL_PARAMS`/`BASE_ACTIVATION_AT_1024` to note the new per-`K_eff` coefficient and that re-derivation runs via `scripts/_derive_preset_constants.py` (Task 2). Do **not** invent new numeric seeds here — the actual re-seed is validated on the 16 GB card in Task 14; for now keep `MODEL_PARAMS`/`BASE_ACTIVATION_AT_1024` as-is and add a one-line comment that `_attention_bytes_per_example` is the dominant term and `k_eff` scales `BASE_ACTIVATION_AT_1024`. (If Task 14's GPU validation later shows the formula still under-predicts, the seed bump lands as a follow-up commit within this PR — see Plan-time findings.)

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py -v`
Expected: PASS — helper exists and matches the SDPA model; train prediction grows with `k_eff` and includes the attention term; `decide_preset(k=...)` is monotone and defaults to the cap; existing `decide_preset()`/`_predicted_bytes(image_size=...)` callers still work (the new params are keyword-defaulted).

- [ ] **Step 9: Commit**

```bash
git add src/custom_sam_peft/presets.py tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py
git commit -m "feat(presets): add K_eff activation + shared SDPA attention term; thread k into decide_preset"
```

### Task 2: `scripts/_derive_preset_constants.py` (new)

**Files:**

- Create: `scripts/_derive_preset_constants.py`
- Test: none (a maintainer-only re-derivation script, run manually on the 16 GB card; not part of the pytest suite — it imports CUDA-heavy modules and prints, it does not assert). Coverage is not required for `scripts/` (it is outside `src/`).

- [ ] **Step 1: Create the derivation script**

Create `scripts/_derive_preset_constants.py`. It loads SAM 3.1, runs the config-aware probe at representative values, and prints re-derived seeds for `MODEL_PARAMS`, `BASE_ACTIVATION_AT_1024`, and the per-`K_eff` activation coefficient. It reuses the probe's overhead model so the printed `BASE_ACTIVATION_AT_1024` is `(peak - overhead - attention) / (batch * k_eff)` at `image_size=1024`-equivalent scaling:

```python
"""Re-derive presets.py seed constants from a measured probe on the local GPU.

Maintainer-only. Run on the 16 GB card:

    uv run python scripts/_derive_preset_constants.py --r 16 --k 16 --batch 1

Prints values to paste into presets.py (MODEL_PARAMS, BASE_ACTIVATION_AT_1024,
and the per-K_eff activation coefficient). Validates against the new
config-aware probe (Component 2). Not imported by the package or the test suite.

Spec: design §3.3.
"""

from __future__ import annotations

import argparse

import torch

from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.presets import (
    WORKSPACE_BYTES,
    _adapter_bytes,
    _attention_bytes_per_example,
    _model_bytes,
    _optimizer_bytes,
)

_GB = 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--method", choices=["lora", "qlora"], default="lora")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE, load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora

    image_size = SAM3_IMAGE_SIZE
    k_eff = min(args.k, MULTIPLEX_CAP)

    wrapper = load_sam31(ModelConfig(), channels=3, channel_semantics="rgb")
    apply_lora(wrapper, PEFTConfig(method=args.method, r=args.r))
    device = next(wrapper.parameters()).device
    images = torch.zeros(
        args.batch, 3, image_size, image_size, dtype=torch.bfloat16, device=device
    )
    prompts = [TextPrompts(classes=[f"class_{j}" for j in range(k_eff)]) for _ in range(args.batch)]

    torch.cuda.reset_peak_memory_stats()
    out = wrapper(images, prompts, support=None)
    loss = torch.zeros((), device=device, dtype=torch.float32)
    for t in out.values():
        if isinstance(t, torch.Tensor):
            loss = loss + t.float().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    peak = int(torch.cuda.max_memory_allocated())

    overhead = (
        _model_bytes(args.method)
        + _adapter_bytes(args.r)
        + _optimizer_bytes(args.r)
        + WORKSPACE_BYTES
        + _attention_bytes_per_example(image_size) * args.batch
    )
    activation_total = max(0, peak - overhead)
    per_k_per_example = activation_total / max(1, args.batch * k_eff)

    print(f"measured peak:            {peak / _GB:.2f} GiB")
    print(f"modeled overhead:         {overhead / _GB:.2f} GiB")
    print(f"residual activation:      {activation_total / _GB:.2f} GiB")
    print(f"-> per-K_eff activation:  {int(per_k_per_example)} bytes "
          f"({per_k_per_example / _GB:.3f} GiB)")
    print(f"-> BASE_ACTIVATION_AT_1024 candidate (scale to 1024px): "
          f"{int(per_k_per_example * (1024 / image_size) ** 2)} bytes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check it imports (no GPU needed for the import test)**

Run: `uv run python -c "import ast; ast.parse(open('scripts/_derive_preset_constants.py').read())"`
Expected: no output (parses clean). Do not run `main()` on CPU.

- [ ] **Step 3: Commit**

```bash
git add scripts/_derive_preset_constants.py
git commit -m "feat(scripts): add _derive_preset_constants probe for re-seeding presets constants"
```

---

## Phase 2 — Component 4: B-then-K OOM ladder (`train/types.py`, `train/loop.py`, `trainer.py`)

> File-disjoint from Phase 1; may run in parallel. Within this phase, Task 3 (`types.py`) lands before Task 4 (`loop.py`+`trainer.py`).

### Task 3: Widen `OomEvent` for the `multiplex_halved` action

**Files:**

- Modify: `src/custom_sam_peft/train/types.py`
- Test: `tests/unit/test_train_types.py` (if present) or add to `tests/unit/test_trainer_oom_retry.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_trainer_oom_retry.py` (it already imports `OomEvent`):

```python
def test_oom_event_supports_multiplex_halved_action() -> None:
    ev = OomEvent(step=5, action="multiplex_halved", new_micro_batch_size=1, effective_K=8)
    assert ev.action == "multiplex_halved"
    assert ev.effective_K == 8


def test_oom_event_microbatch_action_defaults_effective_k_none() -> None:
    ev = OomEvent(step=1, action="microbatch_halved", new_micro_batch_size=4)
    assert ev.effective_K is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_trainer_oom_retry.py -k "multiplex_halved_action or effective_k_none" -v`
Expected: FAIL — `action="multiplex_halved"` violates the `Literal["microbatch_halved"]` only at type-check time but at runtime the dataclass has no `effective_K` field (TypeError: unexpected keyword argument `effective_K`).

- [ ] **Step 3: Edit `train/types.py`**

Rewrite `OomEvent`:

```python
@dataclass(frozen=True)
class OomEvent:
    """One step where the trainer caught OOM and adapted before retrying.

    `action` records the adaptive rung:
      - "microbatch_halved": `state.micro_batch_size //= 2`, retry same step.
      - "multiplex_halved": inner B-ladder exhausted at micro_batch=1; the
        trainer zero_grad'd, halved `effective_K`, re-chunked ALL classes into
        more/smaller groups, and replayed the whole step. No class is dropped.
        Carries the new `effective_K`. Spec §4.

    The fields capture *post*-adaptation state so downstream rendering can
    reconstruct the run's safety-net history without re-traversing mutable state.
    """

    step: int
    action: Literal["microbatch_halved", "multiplex_halved"]
    new_micro_batch_size: int
    effective_K: int | None = None  # set only for "multiplex_halved" events
```

(If `tests/unit/test_train_types.py` exists and asserts the exact field tuple/order, update it to include `effective_K` as the trailing optional field.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_trainer_oom_retry.py -k "multiplex_halved_action or effective_k_none" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/types.py tests/unit/test_trainer_oom_retry.py
git commit -m "feat(train)!: widen OomEvent.action with multiplex_halved + effective_K"
```

### Task 4: B-then-K ladder in `train/loop.py` + `OomState.effective_K` + trainer init

**Files:**

- Modify: `src/custom_sam_peft/train/loop.py` (`OomState` ~54-66; `_train_step_with_oom_ladder` ~69-126; `train_step` ~171-358)
- Modify: `src/custom_sam_peft/train/trainer.py` (`OomState(...)` construction ~489)
- Test: `tests/unit/test_trainer_oom_retry.py` (helper-level rungs); `tests/unit/test_train_loop_multiplex.py` (whole-step K-replay + no-class-dropped invariant)

- [ ] **Step 1: Write the failing helper-level tests (B-exhaust signal)**

The K-rung is driven by `train_step`, but the **signal** that triggers it must come from `_train_step_with_oom_ladder` when B is exhausted. Replace `test_oom_after_microbatch_1_raises` in `tests/unit/test_trainer_oom_retry.py` with a test that the helper raises a **distinguishable** `_MicrobatchExhausted` (caught by `train_step`) rather than a generic `RuntimeError`:

```python
def test_oom_after_microbatch_1_signals_b_exhausted() -> None:
    """At micro_batch=1, the inner ladder no longer hard-fails: it raises the
    B-exhausted signal so train_step can try the K-rung. Spec §4.1."""
    from custom_sam_peft.train.loop import _MicrobatchExhausted

    state = _State(micro_batch_size=8)
    model = _OomThenOk(n_oom=4)  # 3 halvings -> mb=1, 4th OOM signals exhaustion
    with pytest.raises(_MicrobatchExhausted):
        _train_step_with_oom_ladder(model, _make_batch(8), state, forward_call=_fake_forward_call)
    assert state.micro_batch_size == 1
```

Keep `test_oom_first_attempt_halves_microbatch`, `test_oom_multiple_halvings_until_one`, `test_oom_microbatch_shrink_is_sticky`, `test_oom_optimizer_zero_grad_called_once_per_step`, `test_oom_gradient_magnitude_preserved_across_ladder` unchanged (the inner B-ladder behavior is unchanged below micro_batch=1).

- [ ] **Step 2: Write the failing whole-step K-replay tests**

These exercise `train_step`'s new outer K-replay loop with a fake wrapper that OOMs the first time it sees a group larger than 1 class, then succeeds. Add to `tests/unit/test_train_loop_multiplex.py` (read the file first to reuse its existing batch/wrapper/cfg fixtures and `class_names` conventions; adapt names to match). Sketch of the four assertions to add (use the file's existing `_make_batch`/`_make_cfg`/fake-wrapper builders):

```python
def test_oom_k_rung_rechunks_all_classes_none_dropped(...) -> None:
    """When the inner B-ladder exhausts at micro_batch=1, train_step halves
    effective_K, re-chunks ALL classes into more groups, replays the whole step,
    and trains every class. Spec §4.2 (hard invariant)."""
    # classes_in_batch has 4 classes; cfg.train.multiplex.classes_per_forward=4.
    # Fake wrapper OOMs whenever len(group) > 2, succeeds when len(group) <= 2.
    # Expect: K halves 4 -> 2, G goes 1 -> 2, and the set of classes the wrapper
    # was successfully forwarded over == the full classes_in_batch (none dropped).
    ...
    assert oom_state.effective_K == 2
    assert trained_classes == set(classes_in_batch)  # invariant (a)


def test_oom_k_rung_zero_grad_and_whole_step_replay(...) -> None:
    """The K-rung calls optimizer.zero_grad() (discarding the larger-K grads) and
    replays from group 0 — not mid-step. Spec §4.3 (invariant b)."""
    # optimizer is a MagicMock; assert zero_grad called on the K-rung BEFORE the
    # replay, and that the first group of the replay is group 0 (not a resume).
    ...


def test_oom_effective_k_sticky_across_steps(...) -> None:
    """effective_K shrinks once and stays shrunk for the next step. Spec §4.3."""
    # Step 1 forces K 4->2; step 2 with a never-OOM wrapper still uses K=2.
    ...
    assert oom_state.effective_K == 2


def test_oom_records_multiplex_halved_event(...) -> None:
    """A multiplex_halved OomEvent carrying the new effective_K is recorded
    alongside microbatch_halved events. Spec §4.4."""
    actions = [e.action for e in oom_state.pending_oom_events]
    assert "multiplex_halved" in actions
    ev = next(e for e in oom_state.pending_oom_events if e.action == "multiplex_halved")
    assert ev.effective_K == 2


def test_oom_final_hard_fail_only_when_b_and_k_both_one(...) -> None:
    """When micro_batch=1 AND effective_K=1 and OOM still fires, raise with the
    new message naming classes_per_forward=1. Spec §4.5."""
    with pytest.raises(RuntimeError, match=r"classes_per_forward=1"):
        ...


def test_nan_group_skip_path_untouched(...) -> None:
    """The Hungarian non-finite-cost group-skip still skips a group (does NOT
    re-chunk) and is independent of the OOM K-rung. Spec §4.2 (invariant c)."""
    # Wrapper/matcher raises ValueError (non-finite) for one group; assert that
    # group is skipped (finite_group_count reflects it), effective_K is unchanged,
    # and no multiplex_halved event is recorded.
    ...
    assert oom_state.effective_K == cfg.train.multiplex.classes_per_forward
    assert all(e.action != "multiplex_halved" for e in oom_state.pending_oom_events)
```

The implementer must flesh these out against the real `train_step` signature (`model, batch, optimizer, scheduler, cfg, class_names, global_step, nan_streak, peft_method, runtime, oom_state`). Use a `MagicMock`/stub wrapper whose `forward` raises `torch.cuda.OutOfMemoryError` conditioned on group size, and a `MagicMock` optimizer. Build `batch` as `{"images": tensor, "prompts": [...], "instances": [...]}` matching the file's existing fixtures.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/unit/test_trainer_oom_retry.py tests/unit/test_train_loop_multiplex.py -k "exhausted or k_rung or sticky or multiplex_halved or hard_fail or nan_group_skip" -v`
Expected: FAIL — `_MicrobatchExhausted` does not exist; `OomState` has no `effective_K`; `train_step` has no K-replay loop.

- [ ] **Step 4: Edit `train/loop.py` — add the B-exhausted signal + `OomState.effective_K`**

Add a private exception near the top of `loop.py` (after the imports):

```python
class _MicrobatchExhausted(Exception):
    """Internal signal: the inner B-ladder hit micro_batch=1 and OOMed again.

    Caught by train_step to trigger the outer K-rung (halve effective_K and
    replay the whole step). Never escapes train_step. Spec §4.
    """
```

Add `effective_K` to `OomState`:

```python
@dataclass
class OomState:
    """Mutable state the OOM ladder reads/writes across steps.

    Two sticky ladder dimensions, degraded in order (spec §4):
      - micro_batch_size: inner B-rung (halved within each group's forward).
      - effective_K: outer multiplex-K rung (re-chunks ALL classes; replays the
        whole step). Both are sticky: once shrunk they stay shrunk for later steps.
    """

    step: int = 0
    micro_batch_size: int = 1
    effective_K: int = 1
    pending_oom_events: list[OomEvent] = field(default_factory=list)
```

In `_train_step_with_oom_ladder`, replace the final hard-raise (currently `raise RuntimeError("OOM at step ... after micro_batch=1. ...")`) with the signal:

```python
            raise _MicrobatchExhausted(
                f"micro_batch exhausted at step {state.step}"
            ) from oom_err
```

(Everything else in the helper — the B-halving rung, the sticky `micro_batch_size`, the `(loss / n_micro).backward()` — is unchanged.)

- [ ] **Step 5: Edit `train/loop.py` — outer K-replay loop in `train_step`**

In `train_step`, replace the `effective_K` local + group build + the `for group in groups` block with a replay loop. The current code is:

```python
    effective_K = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    groups = _chunked(classes_in_batch, effective_K)
    G = len(groups)
    ... (auto-chunk log) ...
    accum = {...}; finite_group_count = 0; n_hint_applied = 0
    if oom_state is not None:
        oom_state.step = global_step
    for group in groups:
        ... (per-group forward+backward, B-ladder inside) ...
```

Rewrite so the whole group pass is replayable. When `oom_state is not None`, K comes from `oom_state.effective_K` (sticky); initialise it from the config the first time (handled at trainer init in Step 6, so here just read it). On `_MicrobatchExhausted`, zero_grad, halve K (or hard-fail if K==1), re-chunk, and replay from group 0:

```python
    if oom_state is not None:
        oom_state.step = global_step

    while True:  # outer K-replay loop (spec §4)
        if oom_state is not None:
            effective_K = oom_state.effective_K
        else:
            effective_K = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
        groups = _chunked(classes_in_batch, effective_K)
        G = len(groups)

        global _AUTO_CHUNK_LOGGED
        if len(classes_in_batch) > MULTIPLEX_CAP and not _AUTO_CHUNK_LOGGED:
            _LOG.info(
                "multiplex auto-chunk: classes_in_batch=%d > MULTIPLEX_CAP=%d -> %d groups",
                len(classes_in_batch), MULTIPLEX_CAP, G,
            )
            _AUTO_CHUNK_LOGGED = True

        accum = {"mask": 0.0, "box": 0.0, "obj": 0.0, "presence": 0.0, "total": 0.0}
        finite_group_count = 0
        n_hint_applied = 0
        try:
            for group in groups:
                ... (UNCHANGED per-group body: build prompts_g/hints_g/targets_g,
                     the oom_state-vs-direct forward+backward, is_finite/accum) ...
            break  # all groups processed without B-exhaustion -> done
        except _MicrobatchExhausted as exc:
            if oom_state is None or oom_state.effective_K <= 1:
                raise RuntimeError(
                    f"OOM at step {global_step} after micro_batch=1 and "
                    f"classes_per_forward=1. Use a larger GPU or smaller image_size."
                ) from exc
            # K-rung: discard grads accumulated by groups that backpropped at the
            # larger K, halve effective_K, re-chunk ALL classes, replay from group 0.
            optimizer.zero_grad(set_to_none=True)
            oom_state.effective_K = max(1, oom_state.effective_K // 2)
            oom_state.pending_oom_events.append(
                OomEvent(
                    step=global_step,
                    action="multiplex_halved",
                    new_micro_batch_size=oom_state.micro_batch_size,
                    effective_K=oom_state.effective_K,
                )
            )
            _LOG.warning(
                "OOM at step %d after micro_batch=1 — halving effective_K to %d "
                "(re-chunking %d classes into %d groups; no class dropped)",
                global_step, oom_state.effective_K, len(classes_in_batch),
                len(_chunked(classes_in_batch, oom_state.effective_K)),
            )
            # loop continues -> replays the whole step at the smaller K
```

Notes for the implementer:
- The per-group body is **moved verbatim** inside the `try`; the only change is that it now reads `effective_K`/`G` from the surrounding replay iteration. The B-ladder still lives in `_train_step_with_oom_ladder`; the only difference is its terminal raise is now `_MicrobatchExhausted` (Step 4), which propagates out of the per-group call and is caught by this outer `try`.
- The `/(G * cfg.train.grad_accum_steps)` divisions inside the body already reference the local `G`, so they recompute correctly as K shrinks and G grows (spec §4.3 — effective batch preserved).
- The `ValueError` (NaN-driven group-skip) catch **stays inside the per-group body** exactly as today — it is NOT caught by the new outer `_MicrobatchExhausted` handler. Invariant (c).
- After the `while` loop `break`s, the rest of `train_step` (skipped/new_streak/grad-clip/optimizer.step/return) is unchanged.

- [ ] **Step 6: Edit `train/trainer.py` — initialise `OomState.effective_K`**

At `trainer.py:489`, change:

```python
        oom_state = OomState(micro_batch_size=cfg.train.batch_size)
```

to:

```python
        from custom_sam_peft.models.sam3 import MULTIPLEX_CAP

        oom_state = OomState(
            micro_batch_size=cfg.train.batch_size,
            effective_K=min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP),
        )
```

- [ ] **Step 7: Run to verify they pass**

Run: `uv run pytest tests/unit/test_trainer_oom_retry.py tests/unit/test_train_loop_multiplex.py -v`
Expected: PASS — B-exhaust signals; K-rung re-chunks all classes (none dropped); zero_grad + whole-step replay; `effective_K` sticky; `multiplex_halved` event recorded; final hard-fail only at B=1 ∧ K=1; NaN group-skip untouched.

- [ ] **Step 8: Run the broader train suite to catch regressions**

Run: `uv run pytest tests/unit/test_train_loop_multiplex.py tests/unit/test_train_loop_legacy_k1.py tests/unit/test_trainer_nan_behavior.py tests/unit/test_train_runner.py -q`
Expected: PASS — the K=1 legacy path and the NaN-abort path are unaffected (effective_K never shrinks below 1; non-OOM steps take the `break` on the first pass).

- [ ] **Step 9: Commit**

```bash
git add src/custom_sam_peft/train/loop.py src/custom_sam_peft/train/trainer.py tests/unit/test_trainer_oom_retry.py tests/unit/test_train_loop_multiplex.py
git commit -m "feat(train)!: add B-then-K OOM ladder; re-chunk all classes, replay whole step"
```

---

## REVIEW CHECKPOINT A — Components 1 + 4 complete

Before starting Component 2, verify the formula and ladder are self-consistent:

- [ ] Run: `uv run pytest tests/unit/test_presets.py tests/unit/test_decide_eval_batch_size.py tests/unit/test_trainer_oom_retry.py tests/unit/test_train_loop_multiplex.py tests/unit/test_train_loop_legacy_k1.py -q`
      Expected: all PASS.
- [ ] Run: `! grep -n "_SAM3_PATCH\|_SAM3_HEADS\|_n_tokens" src/custom_sam_peft/presets.py | grep -v "def _attention_bytes_per_example\|^[0-9]*:_SAM3_PATCH = 14\|^[0-9]*:_SAM3_HEADS = 16"` — confirm the inline SDPA locals inside `decide_eval_batch_size` were removed (only the module-scope constants + the helper remain).
- [ ] Dispatch a code-review subagent (min sonnet/high; concurrency-sensitive) over the Component 4 diff: confirm the K-rung calls `optimizer.zero_grad(set_to_none=True)` exactly once per K-halving before replay, that `_MicrobatchExhausted` never escapes `train_step`, that the `ValueError` NaN-skip is NOT swallowed by the new handler, and that no class is dropped when K shrinks.

---

## Phase 3 — Component 2: config-aware opt-in probe (`cli/calibrate_cmd.py`)

> Single file (`calibrate_cmd.py`) + its test. Depends on Task 1 (overhead/attention model). Task 6 (auto-init/skip-init/in-place rewrite) chains after Task 5 (config-aware probe).

### Task 5: Config-aware `_run_probe` + `calibrate` reading the config

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py`
- Test: `tests/unit/test_calibrate_cmd.py`

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_calibrate_cmd.py`, the `_patch_probe` helper currently patches `_run_probe` with `lambda: peak` (no args). Update it to accept the new probe signature and add a test that `calibrate` reads the config's `r`/`method`/`k`. Add:

```python
def test_calibrate_probes_at_config_r_and_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """calibrate reads cfg.peft.r/method and cfg.train.multiplex.classes_per_forward
    and passes them to the probe. Spec §5.1."""
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    # A real config with r=32, qlora, k=8.
    _write_config(tmp_path / "config.yaml", method="qlora", r=32, k=8)
    captured: dict[str, object] = {}

    def _fake_probe(*, method: str, r: int, k_eff: int, batch: int) -> int:
        captured.update(method=method, r=r, k_eff=k_eff, batch=batch)
        return int(38 * _GB)

    monkeypatch.setattr("custom_sam_peft.cli.calibrate_cmd._run_probe", _fake_probe)
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    assert captured == {"method": "qlora", "r": 32, "k_eff": 8, "batch": 1}
```

Add a `_write_config(path, *, method, r, k)` helper to the test file that writes a minimal valid config (reuse the project's `init` output or a hand-written minimal `TrainConfig` YAML — read an existing example under `configs/examples/` for the minimal field set). Update the existing `_patch_probe`'s `_run_probe` patch to `lambda **kw: peak` so the no-config tests still pass.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_calibrate_cmd.py -k "probes_at_config" -v`
Expected: FAIL — `calibrate` has no `--config` option and `_run_probe` takes no kwargs.

- [ ] **Step 3: Rework `_run_probe` to be config-aware**

Rewrite `_run_probe` to accept `method`/`r`/`k_eff`/`batch` and probe at those values (distinct synthetic class prompts, config `r`/`method`, config `batch`):

```python
def _run_probe(*, method: str, r: int, k_eff: int, batch: int) -> int:
    """Run one forward+backward at the config's (method, r, k_eff, batch).

    Returns peak bytes. CUDA only. Spec §5.1.
    """
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE, load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora

    k_eff = max(1, min(k_eff, MULTIPLEX_CAP))
    model_cfg = ModelConfig()
    # No DataConfig in scope; rgb default is the documented exception (spec §5.4).
    wrapper = load_sam31(model_cfg, channels=3, channel_semantics="rgb")
    apply_lora(wrapper, PEFTConfig(method=method, r=r))  # type: ignore[arg-type]

    device = next(wrapper.parameters()).device
    images = torch.zeros(
        batch, 3, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE, dtype=torch.bfloat16, device=device
    )
    from custom_sam_peft.data.base import TextPrompts

    # K_eff distinct synthetic class prompts per image (not a single "thing").
    prompts = [TextPrompts(classes=[f"class_{j}" for j in range(k_eff)]) for _ in range(batch)]

    torch.cuda.reset_peak_memory_stats()
    out = wrapper(images, prompts, support=None)
    loss = torch.zeros((), device=device, dtype=torch.float32)
    for t in out.values():
        if isinstance(t, torch.Tensor):
            loss = loss + t.float().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    return int(torch.cuda.max_memory_allocated())
```

- [ ] **Step 4: Add `--config` to `calibrate` + read representative values + recompute overhead**

Add a `--config` option to `calibrate`. When a config exists, read `cfg.peft.method`/`cfg.peft.r`/`cfg.train.multiplex.classes_per_forward`/`cfg.train.batch_size`; pass them to `_run_probe`. Recompute the overhead with the config's `method`/`r` and the attention term (so the residual activation is the true per-example/K activation):

```python
def calibrate(
    config: Path = typer.Option(
        Path("config.yaml"), "--config", help="Config whose sizing the probe reads + rewrites."
    ),
    output: Path = typer.Option(Path(CACHE_FILENAME), "--output", help="Cache file path."),
    force: bool = typer.Option(False, "--force", help="Re-probe even if the cache is fresh."),
) -> None:
    """Probe peak VRAM at the config's (method, r, k, batch) and cache the result."""
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE
    from custom_sam_peft.presets import _attention_bytes_per_example

    if not torch.cuda.is_available():
        typer.echo(f"ERROR: {_CUDA_HINT}", err=True)
        raise typer.Exit(code=2)

    # (Task 6 inserts the skip-init/auto-init guard here, before reading the config.)
    cfg = load_config(config)
    method = cfg.peft.method
    r = cfg.peft.r
    k_eff = min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)
    batch = cfg.train.batch_size

    gpu_name = torch.cuda.get_device_name(0)
    total = int(torch.cuda.get_device_properties(0).total_memory)
    if not force and _cache_is_fresh(output, gpu_name):
        typer.echo("cache fresh — exiting")
        raise typer.Exit(code=0)

    try:
        peak = _run_probe(method=method, r=r, k_eff=k_eff, batch=batch)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: SAM 3.1 checkpoint not found: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except torch.cuda.OutOfMemoryError as exc:
        typer.echo("ERROR: calibration probe OOMed at the config's sizing — GPU too small", err=True)
        raise typer.Exit(code=5) from exc
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"ERROR: probe failed: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    overhead = (
        _model_bytes(method)
        + _adapter_bytes(r)
        + _optimizer_bytes(r)
        + WORKSPACE_BYTES
        + _attention_bytes_per_example(SAM3_IMAGE_SIZE) * batch
    )
    activation = peak - overhead
    if activation < 0:
        typer.echo(
            f"WARNING: negative activation ({activation} bytes); clamping to 0 — "
            "constants may need recalibration",
            err=True,
        )
        activation = 0
    # Store per-(example*K_eff) so _activation_per_example * k_eff reconstructs it.
    activation_per_example = int(activation / max(1, batch * k_eff))
    # ... payload (Step 5 adds rewrite + cache write) ...
```

Add the necessary imports at the top of `calibrate_cmd.py`: `from custom_sam_peft.config.loader import load_config` (verify the actual loader module path — it is the same `load_config` used by `run_cmd.py`/`setup_wizard.py`).

- [ ] **Step 5: Keep the cache payload writing the recomputed activation**

In the `payload` dict, set `"activation_bytes_per_example": int(activation_per_example)` (the per-example/K value) and keep `"peak_memory_bytes_at_probe": int(peak)`. Leave `schema_version`, `gpu_name`, `sam3_checkpoint_sha`, `torch_version`, `custom_sam_peft_version` as-is. (The in-place config rewrite + `# calibrated` annotation lands in Task 6.)

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/unit/test_calibrate_cmd.py -v`
Expected: PASS — `--config` is read; the probe gets `(method, r, k_eff, batch)`; the no-config tests still pass via the `lambda **kw: peak` patch; negative-activation clamp still fires; atomic-write + cache-fresh tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(calibrate): config-aware probe at (method, r, k, batch); recompute activation residual"
```

### Task 6: In-place config rewrite + `# calibrated` annotation + auto-init/skip-init guard

**Files:**

- Modify: `src/custom_sam_peft/cli/calibrate_cmd.py`
- Test: `tests/unit/test_calibrate_cmd.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_calibrate_cmd.py`:

```python
def test_calibrate_rewrites_config_in_place_annotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a successful probe, the config's sizing block is re-annotated
    '# calibrated <date>' and re-loads via load_config. Spec §5.3."""
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, method="lora", r=16, k=16)
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    body = cfg_path.read_text()
    assert "# calibrated" in body
    from custom_sam_peft.config.loader import load_config

    assert load_config(cfg_path) is not None  # still valid


def test_calibrate_auto_inits_when_no_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No config -> warn + auto-init (formula, no probe) then probe it. Spec §5.4."""
    _patch_probe(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "config.yaml").exists()
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").exists()
    assert "not initialized" in result.output.lower() or "auto" in result.output.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_calibrate_cmd.py -k "rewrites_config or auto_inits" -v`
Expected: FAIL — no annotation written; missing config raises instead of auto-initing.

- [ ] **Step 3: Add the skip-init/auto-init guard to `calibrate`**

At the point marked in Task 5 Step 4 (before `cfg = load_config(config)`), insert: if the config does not exist, warn and call the shared `init` path (formula, no probe), then proceed. Reuse `init_cmd.run_init` to scaffold (it bakes formula-derived sizing as of Task 9):

```python
    if not config.exists():
        typer.echo(
            f"WARNING: {config} not initialized — auto-init (formula, no probe) then probe.",
            err=True,
        )
        from custom_sam_peft.cli.init_cmd import run_init

        run_init("coco-text-lora", config, force=False)
```

- [ ] **Step 4: Add the in-place rewrite + annotation**

After the cache is written (`_atomic_write_json(output, payload)` succeeds), rewrite the config's sizing fields (`peft.method`/`peft.r`, `train.batch_size`/`train.grad_accum_steps`, `model.dtype`) to the calibrated values and annotate. Implement a helper `_rewrite_sizing_block(config_path, *, method, r, batch_size, grad_accum_steps, dtype, annotation)` that does a **targeted, in-place line-rewrite using pyyaml only — no new dependency**: it locates the sizing keys and substitutes their values by line surgery, prepends the annotation comment, and leaves all other lines and comments untouched. This is the shared helper `init` also uses (Task 9; import it from `cli/_config_rewrite.py` rather than duplicating). The annotation string is `f"# calibrated {datetime.now(UTC).date().isoformat()}"`.

**Mechanism (decided — pyyaml only, do NOT add `ruamel.yaml`):** validate with `yaml.safe_load` that the sizing keys exist and that the rewritten file still parses, but perform the value substitution by line surgery so surrounding comments and formatting survive (a full `yaml.safe_dump` would strip them). Add **one** shared rewrite helper used by both `init` (`# formula-derived`) and `calibrate` (`# calibrated`) to keep the annotation logic DRY (YAGNI: no general YAML-comment library).

For the probed config, derive `batch_size`/`grad_accum_steps`/`dtype` from a `decide_preset(k=cfg.train.multiplex.classes_per_forward)` call that now consults the freshly-written cache (the cache makes `provenance == "calibrated"`), so the rewritten sizing reflects the calibrated activation.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/test_calibrate_cmd.py -v`
Expected: PASS — annotation present + config re-loads; auto-init fires when no config.

- [ ] **Step 6: Commit**

```bash
git add src/custom_sam_peft/cli/calibrate_cmd.py tests/unit/test_calibrate_cmd.py
git commit -m "feat(calibrate): rewrite config in place (# calibrated) + auto-init guard"
```

---

## Phase 4 — Component 3: workflow & run-time wiring

> Task 9 (`init_cmd.py`) and Task 10 (`run_cmd.py`) are file-disjoint and both depend on Task 1. Task 11 (verbatim-consume + eval-auto regression tests) follows. Task 12 (wizard offer) chains after Task 6.

### Task 9: `init` bakes concrete formula-derived sizing + annotation (CPU-only fallback)

**Files:**

- Modify: `src/custom_sam_peft/cli/init_cmd.py`
- Create (optional): `src/custom_sam_peft/cli/_config_rewrite.py` (the shared rewrite/annotation helper, if Task 6 did not already place it here)
- Test: `tests/unit/test_cli_init.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_cli_init.py`:

```python
def test_init_bakes_formula_sizing_with_annotation_on_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a (patched) GPU, init bakes concrete decide_preset sizing annotated
    '# formula-derived'. Spec §6.1."""
    # Patch torch.cuda + decide_preset to a known decision.
    from custom_sam_peft.presets import PresetDecision

    fake = PresetDecision(
        method="lora", r=24, batch_size=2, grad_accum_steps=8, dtype="bfloat16",
        headroom_bytes=0, predicted_bytes=0, budget_bytes=0, gpu_name="X",
        provenance="analytic", cache_path=None, calibrated_at=None,
    )
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.decide_preset", lambda **kw: fake)
    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code == 0, result.output
    body = out.read_text()
    assert "# formula-derived" in body
    cfg = load_config(out)
    assert cfg.peft.r == 24
    assert cfg.train.batch_size == 2
    assert cfg.model.dtype == "bfloat16"


def test_init_cpu_only_writes_safe_defaults_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU-only init scaffolds safe defaults + a re-resolve warning; never invents
    GPU numbers. Spec §6.1."""
    monkeypatch.setattr("custom_sam_peft.cli.init_cmd.torch.cuda.is_available", lambda: False)
    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "re-resolve" in result.output.lower() or "gpu" in result.output.lower()
    cfg = load_config(out)  # safe defaults still valid
    assert cfg.peft.method in {"lora", "qlora"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_cli_init.py -k "bakes_formula or cpu_only_writes_safe" -v`
Expected: FAIL — `init`/`run_init` does not call `decide_preset` and writes no `# formula-derived` annotation; the CPU path emits no re-resolve warning.

- [ ] **Step 3: Bake formula sizing into `run_init`**

In `run_init`, after the template is rendered to `output`, resolve concrete sizing and rewrite the sizing block annotated `# formula-derived`. Import `torch` and `decide_preset` at the top of `init_cmd.py`. Add a `cuda_available`-aware step:

```python
    output.write_text(body)

    # Bake concrete formula-derived sizing into the file the user reviews (spec §6.1).
    import torch

    from custom_sam_peft.presets import decide_preset

    cfg = load_config(output)
    k = cfg.train.multiplex.classes_per_forward
    if torch.cuda.is_available():
        try:
            decision = decide_preset(k=k)
        except RuntimeError as exc:
            rprint(f"[yellow]could not auto-size ({exc}); leaving template defaults[/yellow]")
        else:
            _rewrite_sizing_block(
                output,
                method=decision.method,
                r=decision.r,
                batch_size=decision.batch_size,
                grad_accum_steps=decision.grad_accum_steps,
                dtype=decision.dtype,
                annotation="# formula-derived",
            )
    else:
        rprint(
            "[yellow]init ran CPU-only; wrote safe template defaults. Re-run "
            "`custom-sam-peft init` (or `calibrate`) on the GPU to resolve sizing.[/yellow]"
        )
```

`_rewrite_sizing_block` is the shared helper from Task 6 (import from `init_cmd` or `cli/_config_rewrite.py`). It sets `peft.method`/`peft.r`, `train.batch_size`/`train.grad_accum_steps`, `model.dtype` and prepends the annotation comment to the affected blocks. Leave the rest of the rendered template untouched (so manual edits and the comment scaffolding survive — spec §6.1 "manual edits win" / they come last by review, not by `init`).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_cli_init.py -v`
Expected: PASS — GPU path bakes concrete sizing + annotation; CPU path warns + writes safe defaults; existing init tests (template rendering, comprehensiveness) still pass.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/init_cmd.py tests/unit/test_cli_init.py
git commit -m "feat(init): bake concrete formula-derived sizing + annotation (CPU-only safe fallback)"
```

### Task 10: `run` skip-init guard

**Files:**

- Modify: `src/custom_sam_peft/cli/run_cmd.py`
- Test: `tests/integration/test_cli_run.py`

- [ ] **Step 1: Write the failing test**

In `tests/integration/test_cli_run.py`, add a CPU-collectable test (mock `run_training`/`run_eval`/`write_bundle` so no GPU/model is loaded) asserting that `run --config missing.yaml` warns + auto-inits + proceeds. If the existing test file already heavily mocks `_orchestrate`, place the guard test at the `run`/`_orchestrate` boundary:

```python
def test_run_skip_init_guard_warns_and_autoinits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run with no usable config warns 'not initialized', auto-inits (formula, no
    probe), then proceeds. Spec §6.2."""
    monkeypatch.chdir(tmp_path)
    called = {}
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd._orchestrate", lambda *a, **k: called.setdefault("ran", True) or 0
    )
    # Patch run_init so the guard does not need a GPU.
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd.run_init",
        lambda *a, **k: (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG),
    )
    result = runner.invoke(app, ["run", "--config", "config.yaml"])
    assert "not initialized" in (result.output.lower())
    assert called.get("ran") is True
```

Provide `_MINIMAL_CONFIG` (a valid minimal `TrainConfig` YAML) in the test module, or reuse an existing fixture.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/test_cli_run.py -k "skip_init_guard" -v`
Expected: FAIL — `run` raises a Typer/`load_config` error on the missing config instead of auto-initing.

- [ ] **Step 3: Add the guard to `run`**

In `run` (`run_cmd.py:172-201`), before `cfg = load_config(config)`, add:

```python
    from custom_sam_peft.cli.init_cmd import run_init

    if not config.is_file():
        rprint(f"[yellow]{config} not initialized — auto-init (formula, no probe) then run.[/yellow]")
        run_init("coco-text-lora", config, force=False)
```

Leave the rest of `run`/`_orchestrate` unchanged (the train tuple is already consumed verbatim by `run_training(cfg, ...)`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/test_cli_run.py -k "skip_init_guard" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py tests/integration/test_cli_run.py
git commit -m "feat(run): add skip-init guard (warn + auto-init formula, no probe)"
```

### Task 11: Regression guards — verbatim consume + `eval.batch_size: auto` stays run-time

**Files:**

- Test only: `tests/integration/test_cli_run.py` (or `tests/unit/test_train_runner.py`)

- [ ] **Step 1: Write the guard tests**

These assert spec invariants (d), (e), and §6.1's "consumes literals verbatim". They are pure assertions over existing behavior (no production change). Add:

```python
def test_run_consumes_train_tuple_verbatim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """run passes cfg's peft.method/r, train.batch_size/grad_accum, model.dtype to
    run_training unchanged — no run-time resolution. Spec §6.1."""
    seen = {}
    monkeypatch.setattr(
        "custom_sam_peft.cli.run_cmd.run_training",
        lambda cfg, **k: seen.update(
            method=cfg.peft.method, r=cfg.peft.r, bs=cfg.train.batch_size,
            ga=cfg.train.grad_accum_steps, dtype=cfg.model.dtype,
        ) or _stub_train_result(tmp_path),
    )
    # ... build a config with explicit method=qlora,r=32,bs=4,ga=4,dtype=float16 ...
    # invoke run; assert seen == those exact values.


def test_eval_batch_size_auto_still_resolved_at_runtime() -> None:
    """eval.batch_size: 'auto' is NOT baked; decide_eval_batch_size still owns it.
    Spec §6.3 (invariant d)."""
    from custom_sam_peft.config.schema import EvalConfig

    # The schema still accepts "auto" for eval.batch_size (the run-time sentinel).
    assert EvalConfig(batch_size="auto").batch_size == "auto"


def test_train_tuple_has_no_auto_sentinel() -> None:
    """No 'auto' sentinel was added to the train sizing tuple. Spec §6.3 (invariant e)."""
    from custom_sam_peft.config.schema import PEFTConfig, TrainHyperparams

    # method has no "auto" member; batch_size/grad_accum_steps/r are concrete ints.
    import typing

    assert "auto" not in typing.get_args(PEFTConfig.model_fields["method"].annotation)
    # batch_size default is a concrete int, not "auto".
    assert isinstance(TrainHyperparams(epochs=1).batch_size, int)
```

Read `schema.py` to confirm the exact name of the eval-batch-size field and its `"auto"`-accepting type before finalising `test_eval_batch_size_auto_still_resolved_at_runtime` (the assertion must match the real `EvalConfig` shape; adjust the import/field name accordingly).

- [ ] **Step 2: Run to verify they pass (no production change expected)**

Run: `uv run pytest tests/integration/test_cli_run.py tests/unit/test_train_runner.py -k "verbatim or auto_still_resolved or no_auto_sentinel" -v`
Expected: PASS immediately — these guard existing behavior. If any FAIL, that is a real regression introduced earlier; fix the offending task before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_cli_run.py tests/unit/test_train_runner.py
git commit -m "test(run): guard verbatim train-tuple consume + eval auto run-time resolution"
```

### Task 12: Wizard offers `csp calibrate` with consent

**Files:**

- Modify: `src/custom_sam_peft/cli/setup_wizard.py`
- Test: `tests/unit/cli/test_setup_wizard.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/cli/test_setup_wizard.py`:

```python
def test_generate_config_offers_calibrate_with_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After emit, on a CUDA box the wizard OFFERS calibrate; running it requires
    explicit consent (opt-in). Spec §5.2."""
    # Drive run_wizard to a minimal answer set; monkeypatch ask_confirm so the
    # post-emit calibrate prompt returns True, and patch the calibrate invocation.
    invoked = {}
    monkeypatch.setattr(sw, "_invoke_calibrate", lambda output: invoked.setdefault("ran", True))
    # ... patch the wizard steps to a minimal pass + ask_confirm("Run calibrate...") -> True ...
    sw.generate_config(tmp_path / "config.yaml", force=True, cuda_available=True)
    assert invoked.get("ran") is True


def test_generate_config_no_calibrate_when_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the offer (or CPU-only) never runs the probe. Spec §5.2 / §7."""
    invoked = {}
    monkeypatch.setattr(sw, "_invoke_calibrate", lambda output: invoked.setdefault("ran", True))
    # ask_confirm for the calibrate offer -> False.
    sw.generate_config(tmp_path / "config.yaml", force=True, cuda_available=True)
    assert "ran" not in invoked
```

(Match the existing test file's monkeypatching style for the wizard's `ask_*` primitives; read its existing tests first.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -k "offers_calibrate or no_calibrate_when_declined" -v`
Expected: FAIL — `_invoke_calibrate` does not exist and `generate_config` makes no post-emit calibrate offer.

- [ ] **Step 3: Add the opt-in offer to `generate_config`**

In `setup_wizard.py`, add a thin helper and call it after the config is written (end of `generate_config`, after emit succeeds):

```python
def _invoke_calibrate(output: Path) -> None:
    """Run the opt-in config-aware calibration probe on the just-written config."""
    from custom_sam_peft.cli.calibrate_cmd import calibrate

    calibrate(config=output, output=Path(CACHE_FILENAME), force=False)
```

Then at the end of `generate_config`, after the file is written and the launch command is known:

```python
    if cuda_available and ask_confirm(
        "Run `csp calibrate` now to tighten the VRAM sizing with a live GPU probe? "
        "(opt-in; loads the model and runs one forward+backward)",
        default=False,
    ):
        try:
            _invoke_calibrate(output)
        except typer.Exit:
            typer.echo("calibration did not complete; keeping the formula-derived config", err=True)
    return launch, ctx.run_mode
```

The probe is never run implicitly: default is `False`, and it is gated on `cuda_available`. (Spec §5.2 / §7 — keeps OOM-provocation off the default path.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/cli/test_setup_wizard.py -v`
Expected: PASS — calibrate offered with consent; declined/CPU never runs it; existing wizard tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/setup_wizard.py tests/unit/cli/test_setup_wizard.py
git commit -m "feat(wizard): offer opt-in csp calibrate after emit (consent-gated, CUDA-only)"
```

---

## Phase 5 — Component 8: provenance & bundle rendering

### Task 13: Render `multiplex_halved` events in `_oom_edge_note`

**Files:**

- Modify: `src/custom_sam_peft/runs/bundle.py` (`_oom_edge_note` ~333-338)
- Test: `tests/unit/runs/test_bundle.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/runs/test_bundle.py` (read its `_make_ctx`/`_make_decision`/fixtures first):

```python
def test_oom_edge_note_renders_multiplex_halved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OOM edge note reports the final effective_K alongside final micro_batch
    when a multiplex_halved event occurred. Spec §8."""
    events = (
        OomEvent(step=10, action="microbatch_halved", new_micro_batch_size=1),
        OomEvent(step=10, action="multiplex_halved", new_micro_batch_size=1, effective_K=8),
    )
    ctx = _make_ctx(tmp_path, per_example_iou=[], oom_events=events)
    monkeypatch.setattr("custom_sam_peft.runs.bundle._reinfer_one_example", _fake_reinfer)
    write_bundle(ctx, _make_metrics(0.5), val_dataset=_make_dataset(0), model_wrapper=MagicMock())
    summary = (ctx.run_dir / "summary.md").read_text()
    assert "OOM retries: 2" in summary
    assert "final micro_batch=1" in summary
    assert "final classes_per_forward=8" in summary
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/runs/test_bundle.py -k "multiplex_halved" -v`
Expected: FAIL — `_oom_edge_note` reads `events[-1].new_micro_batch_size` and emits no `classes_per_forward` clause; the last event is a `multiplex_halved` whose `new_micro_batch_size` happens to be 1 here, but the note never mentions `effective_K`.

- [ ] **Step 3: Edit `_oom_edge_note`**

Make the note robust to either trailing event type and surface the final `effective_K` when any K-rung fired:

```python
def _oom_edge_note(events: tuple[OomEvent, ...]) -> str | None:
    """Return the OOM-summary line for `## Edge cases`, or None when there were none."""
    if not events:
        return None
    # Final micro_batch is carried on every event; final effective_K only on K-rungs.
    final_mb = events[-1].new_micro_batch_size
    k_events = [e for e in events if e.action == "multiplex_halved" and e.effective_K is not None]
    note = f"OOM retries: {len(events)} — final micro_batch={final_mb}"
    if k_events:
        note += f", final classes_per_forward={k_events[-1].effective_K}"
    return note
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/runs/test_bundle.py -v`
Expected: PASS — the K clause renders; the existing microbatch-only note (no K events) is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/runs/bundle.py tests/unit/runs/test_bundle.py
git commit -m "feat(bundle): render multiplex_halved events (final classes_per_forward) in OOM edge note"
```

---

## REVIEW CHECKPOINT B — full CPU suite + coverage gate

- [ ] Run the FULL suite with coverage (the 80% gate runs on the full suite, not `tests/unit` alone — project memory):
      `uv run pytest -q`
      Expected: all PASS, coverage >= 80%.
- [ ] Run lint/format: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/`
      Expected: clean (fix findings before the ready PR — lint gate).
- [ ] Dispatch a code-review subagent (opus/xhigh; design-sensitive — concurrency in the OOM ladder + the config-rewrite annotation contract) over the full diff.

---

## Phase 6 — GPU validation (run ONE AT A TIME on the 16 GB card)

> Per project memory: reach the GPU only via `--extra gpu-pascal --extra dev`; one gpu run at a time, per-file; no `nvidia-smi` (use torch); checkpoint symlinked from the main repo. These are GPU-only — marked so CPU CI skips them. The 1080 lacks bf16; the formula-accuracy and real-model-peak tests that need bf16-representative numerics are `gpu_t4` (Colab), the rest may be `gpu_local`.

### Task 14: Real config-aware probe, real-model peak, and the 10-vs-22 GiB formula-accuracy validation

**Files:**

- Modify: `tests/gpu/test_calibrate_real.py` (real config-aware probe + real-model peak + clamp-does-not-fire)
- Create: `tests/gpu/test_formula_accuracy.py` (the headline 10-vs-22 GiB validation)

- [ ] **Step 1: Update the real config-aware probe test**

In `tests/gpu/test_calibrate_real.py`, change the probe invocation to write a config first, then run `calibrate --config`, and assert it loads SAM 3.1, attaches the config's `r`/`method`, runs forward+backward at the config's `k_eff`/`batch`, and writes both the cache and the rewritten (annotated) config without crashing. Keep the existing `gpu_t4`/`requires_checkpoint`/`requires_compatible_gpu` markers and the sane-range activation assertion (widen the upper bound if the K-scaled residual exceeds 10 GiB at k=16).

```python
@pytest.mark.gpu_t4
@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_calibrate_real_config_aware(tmp_path: Path) -> None:
    runner = CliRunner()
    os.chdir(tmp_path)
    # init a config (CPU-safe defaults are fine; calibrate rewrites sizing).
    runner.invoke(app, ["init", "--output", "config.yaml", "--force"])
    result = runner.invoke(app, ["calibrate", "--config", "config.yaml", "--force"])
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / ".custom_sam_peft_calibration.json").read_text())
    activation = int(data["activation_bytes_per_example"])
    # Real-model peak in a sane range; clamp must NOT fire under representative settings.
    assert activation > 0, "negative-activation clamp fired — constants need re-deriving"
    assert 5e8 <= activation <= 1.5e10
    assert "# calibrated" in (tmp_path / "config.yaml").read_text()
```

- [ ] **Step 2: Add the formula-accuracy validation test (headline Component 1 check)**

Create `tests/gpu/test_formula_accuracy.py`. It runs the real config-aware probe to get the true peak at the representative config, then asserts the new analytic `_predicted_bytes(..., k_eff=k, mode="train")` for that same config **brackets** the real peak (slightly conservative, never below). This is the 10-vs-22 GiB regression guard — invariant (f).

```python
"""GPU formula-accuracy validation: the new analytic formula no longer
under-predicts the real peak (the 10-vs-22 GiB regression). Spec §3.3 / §9.2.

Run ALONE on the 16 GB card (it loads the full model twice-ish). Marked gpu_t4
because it needs bf16-representative numerics.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu_t4
@pytest.mark.requires_checkpoint
@pytest.mark.requires_compatible_gpu
def test_formula_does_not_underpredict_real_peak(tmp_path: Path) -> None:
    from custom_sam_peft.cli.calibrate_cmd import _run_probe
    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP
    from custom_sam_peft.presets import _predicted_bytes

    method, r, batch, k = "lora", 16, 1, MULTIPLEX_CAP
    real_peak = _run_probe(method=method, r=r, k_eff=k, batch=batch)
    predicted = _predicted_bytes(
        method, r=r, batch=batch, image_size=1008, cache=None, mode="train", k_eff=k
    )
    gib = 1024**3
    # Must not under-predict (invariant f); slightly conservative (within ~1.6x above).
    assert predicted >= real_peak, (
        f"formula UNDER-predicts: predicted {predicted / gib:.1f} GiB < "
        f"real {real_peak / gib:.1f} GiB (the 10-vs-22 regression)"
    )
    assert predicted <= real_peak * 1.6, (
        f"formula too conservative: predicted {predicted / gib:.1f} GiB > "
        f"1.6x real {real_peak / gib:.1f} GiB"
    )
```

- [ ] **Step 3: Run on the GPU (maintainer, one at a time)**

Run (per memory, on the 16 GB card):
`uv run --extra gpu-pascal --extra dev pytest tests/gpu/test_formula_accuracy.py -v -m gpu_t4`
then separately
`uv run --extra gpu-pascal --extra dev pytest tests/gpu/test_calibrate_real.py -v`
Expected: PASS. If `test_formula_does_not_underpredict_real_peak` FAILS with under-prediction, re-derive the seeds via `scripts/_derive_preset_constants.py` and bump `BASE_ACTIVATION_AT_1024` / the per-`K_eff` coefficient in `presets.py` until the formula brackets the real peak (this is the §3.3 re-seed step, validated here). Commit the seed bump.

- [ ] **Step 4: Commit**

```bash
git add tests/gpu/test_calibrate_real.py tests/gpu/test_formula_accuracy.py
git commit -m "test(gpu): real config-aware probe + 10-vs-22 GiB formula-accuracy validation"
```

---

## Final verification

- [ ] `uv run pytest -q` — full suite green, coverage >= 80%.
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy src/` — clean.
- [ ] `! grep -rn "decide_preset(image_size" src/ tests/` — no stale `image_size=` calls to `decide_preset` remain (the param is `k` now).
- [ ] Markdown-lint this plan + the spec before they land on the ready PR (CI lints them): use the project's exact markdownlint config (discover from the lint workflow; do not assume the tool).
- [ ] Confirm the six hard invariants (a)-(f) each have a passing assertion (Tasks 8, 11, 14).

---

## Self-review (writer's pass against the spec)

**Spec coverage:**

- §3 formula (multiplex-`K_eff` + SDPA attention + re-seed) → Task 1 (helper, K-term, k-threaded decide_preset, eval-path refactor), Task 2 (re-derivation script), Task 14 Step 3 (re-seed validated on GPU).
- §4 OOM ladder (B-then-K, no class dropped, zero_grad+replay, OomState/OomEvent, hard-fail) → Task 3 (`OomEvent` widening), Task 4 (`OomState.effective_K`, B-exhaust signal, K-replay loop, hard-fail, trainer init).
- §5 probe (config-aware, opt-in, in-place rewrite+cache, auto-init) → Task 5 (config-aware probe), Task 6 (in-place rewrite + annotation + auto-init), Task 12 (opt-in offer in wizard).
- §6 wiring (init bakes concrete sizing, skip-init guards, verbatim consume, eval-auto stays run-time, no auto sentinel) → Task 9 (`init` bake + CPU fallback), Task 10 (`run` guard), Task 6 (`calibrate` guard), Task 11 (verbatim + eval-auto + no-sentinel guards).
- §7 #148 disposition (no table, no `"table"` provenance) → no task introduces a table or a new provenance value; Task 1 keeps `provenance` as `"calibrated"`/`"analytic"` only. (Disposition is honored by construction — the default is the formula, the probe is opt-in (Task 12), the ladder recovers (Task 4).)
- §8 provenance/bundle (`multiplex_halved` rendering) → Task 13.
- §9 tests (CPU formula math, config rewrite/annotation, workflow+guards, B+K ladder; GPU real probe, real peak, formula accuracy) → Tasks 1, 5, 6, 9, 10, 11, 12 (CPU) + Task 14 (GPU).

**Type consistency:** `_attention_bytes_per_example(image_size: int) -> int` is defined in Task 1 and consumed in Tasks 1, 2, 5, 14 with the same signature. `_predicted_bytes(..., mode, k_eff)` keyword-defaults are consistent across Tasks 1, 5, 14. `_run_probe(*, method, r, k_eff, batch)` is defined in Task 5 and reused in Task 14. `OomState.effective_K: int` (Task 4) and `OomEvent.effective_K: int | None` (Task 3) are distinct-but-consistent (state is concrete; event is optional). `_rewrite_sizing_block(config_path, *, method, r, batch_size, grad_accum_steps, dtype, annotation)` is the single shared helper used by Tasks 6 and 9.

**Placeholder scan:** the Phase 2 Step 2 K-replay tests and the Phase 4 wizard/run tests are sketched with explicit assertions but defer fixture wiring to the implementer because they must bind to the real `train_step`/`generate_config` fixtures already in those test files; each names the exact assertions required. This is intentional (the surrounding test files' fixtures are the source of truth) and is the one place the plan delegates fixture-construction rather than inlining it.

---

## Plan-time findings / open questions

1. **Spec §3.1 says the `K_eff` term *replaces* the K-agnostic `per_example × batch`.** This plan keeps `per_example × batch` and multiplies by `k_eff` (`per * batch * k_eff`), which at `k_eff=1` reduces to the old value — so the eval path (which calls `_activation_bytes` without `k_eff`) is unchanged, and the train path scales up. This satisfies "replaces … now scales with `K_eff`" while keeping the eval branch byte-identical. Flagging in case the intent was a different functional form for the per-example activation under multiplex (e.g., a fixed base + per-class delta). Recommend the GPU re-seed (Task 14 Step 3) confirm the linear-in-`k_eff` shape; if it over/under-shoots, the coefficient (not the structure) is what gets tuned.

2. **§3.3 re-seed timing.** Task 1 deliberately does NOT change the numeric seeds (`MODEL_PARAMS`, `BASE_ACTIVATION_AT_1024`) — only the formula structure. The actual numeric re-seed is gated on the GPU validation (Task 14 Step 3) because the maintainer's 16 GB card is the only data source. This means the CPU formula tests assert *monotonicity and term-presence*, not absolute byte targets; the absolute "no under-predict" check is GPU-only by necessity. If a reviewer expects a CPU test pinning exact predicted bytes, that cannot be done without the GPU-measured seed — noting the constraint rather than fabricating a number.

3. **`_oom_edge_note` and a trailing `multiplex_halved` event.** Every `OomEvent` carries `new_micro_batch_size`, and the K-rung event sets it to the current `micro_batch_size` (1 at the point a K-rung fires). So `final micro_batch=1` is always correct on a K-rung. Confirmed the note logic (Task 13) reads `events[-1].new_micro_batch_size` safely regardless of trailing event type.

4. **Config-rewrite mechanism (Tasks 6/9) — RESOLVED.** Decision: **pyyaml-only targeted line-rewrite; do not add `ruamel.yaml`.** The project depends only on `pyyaml` (confirmed in `pyproject.toml`), whose `safe_dump` would strip comments. The single shared helper (`cli/_config_rewrite.py`, used by both `init` and `calibrate`) validates with `yaml.safe_load` but substitutes the sizing-key values by line surgery and prepends the annotation comment, preserving all other lines/comments. See Task 6 Step 4 and Task 9.

5. **`calibrate --config` default.** The plan defaults `--config` to `config.yaml` (matching `init`'s `--output` default) so `csp calibrate` with no flags targets the canonical file. If the maintainer prefers `calibrate` to require an explicit `--config`, that is a one-line change; flagging the chosen default.
