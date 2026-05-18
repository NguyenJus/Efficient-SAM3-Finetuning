# Colab GPU Remaining Failures Design (spec/colab-gpu-remaining-failures)

**Status:** ready for planning
**Parent specs:**
- [`2026-05-17-colab-gpu-integration-fix-v2-design.md`](2026-05-17-colab-gpu-integration-fix-v2-design.md) — the v2 spec that landed via PR #13. v2 shipped the torchao pin (unblocked 3 LoRA tests) and the `_Sam3ImageAdapter.forward` autocast wrap. This new spec covers the failures that remain on top of v2's merged baseline.
- [`2026-05-17-colab-gpu-integration-fix-design.md`](2026-05-17-colab-gpu-integration-fix-design.md) — original v1 spec; still authoritative for `SCOPE_TARGETS` rationale (§5) and the `_Sam3ImageAdapter.forward` recipe (§4).
- [`2026-05-17-peft-qlora-design.md`](2026-05-17-peft-qlora-design.md) — the QLoRA module contract; this spec changes internals of `_infer_quant_type_from_wrapper` and `merge_lora` but does NOT change any public surface defined there.

**Sibling spec:** [`2026-05-17-training-loop-design.md`](2026-05-17-training-loop-design.md)
**Branch:** TBD by next session — branch off `main` AFTER PR #13 merges. Do NOT branch off `worktree-fix+colab-bpe-gzip`.
**Baseline:** PR #13 merged tip. Local unit suite: 240 passed / 1 skipped. Colab T4 GPU integration suite: 4 of 9 passing post-PR-#13.

---

## 1. Purpose

PR #13 (`docs/superpowers/plans/2026-05-17-colab-gpu-integration-fix-v2.md`) shipped:
- The `torchao>=0.16.0` Colab pin — unblocked 3 LoRA tests (`test_apply_lora_on_real_sam31_under_trainable_budget`, `test_save_load_roundtrip_on_real_sam31`, `test_merge_lora_on_real_sam31`).
- The `_Sam3ImageAdapter.forward` autocast wrap + the empty-`Prompt` fallback's `box_embeddings` dtype handling.

Post-merge Colab T4 run: **4 of 9 GPU integration tests pass; 5 fail.** The 5 failures are mutually independent — they live in 4 different problem domains (sam3-internal fp32 leak, a test predicate that mis-classifies LoRA adapter weights, a bitsandbytes API rename, and a peft 0.19 merge path that does not understand `Linear4bit`). This spec defines the fixes for those 4 problems (covering all 5 failing tests).

After implementation, the Colab GPU integration suite is at 9 of 9 passing, and the unit suite stays at 240 passed / 1 skipped with no regression.

## 2. Constraints

### 2.1 Branch and commit policy

- This work happens on a NEW branch cut from `main` AFTER PR #13 merges. Do NOT base off `worktree-fix+colab-bpe-gzip`.
- No new runtime dependencies in `pyproject.toml`. The fixes below either change source code in `src/esam3/peft_adapters/` or change a test predicate; no installable surface changes.
- No emojis in source, comments, or commit messages.
- Logging: append to `logs/log.md` after each work item using `[<UTC-ISO8601>] [ROLE] action`. Do not read the log during task execution.
- Append-only ticket capture: any newly-discovered out-of-scope work goes in `logs/TODO.md`, NOT this spec.

### 2.2 What must NOT change

- `_Sam3ImageAdapter.forward`'s autocast wrap, signature, or body shape (PR #13 #section 6.2 / #section 6.6 of the v2 spec). Issue 1 below is a NEW failure inside sam3 that surfaces ONLY because the autocast wrap moved the dtype error one level deeper into sam3-internal code; the autocast wrap itself is correct.
- The 3 LoRA integration tests that PR #13 unblocked (`test_apply_lora_on_real_sam31_under_trainable_budget`, `test_save_load_roundtrip_on_real_sam31`, `test_merge_lora_on_real_sam31`) — they are passing and must remain passing.
- `notebooks/colab_gpu_tests.ipynb`'s install cell pins (torchao/numpy/scipy/transformers/huggingface_hub) — re-pinning them is explicitly out of scope.
- The `Sam3Wrapper` / `_build_geometric_prompt` / `_validate_inputs` surface introduced by PR #14 and reconciled by PR #13. Issue 1's fix is allowed to touch `_Sam3ImageAdapter.forward` ONLY if the chosen Option in §3.4 demands it; otherwise it stays untouched.
- The public surface of `src/esam3/peft_adapters/qlora.py` (`apply_qlora`, `save_qlora`, `load_qlora`) and `src/esam3/peft_adapters/lora.py` (`apply_lora`, `merge_lora`, `save_lora`, `load_lora`, `SCOPE_TARGETS`, `_resolve_targets`). Internals are open for editing; signatures and import paths are frozen.

### 2.3 Hardware constraint

- Dev box is GTX 1080 (compute capability 6.1); all `requires_compatible_gpu` tests skip locally.
- Verification of Issue 1, 3, and 4 only happens on Colab T4 via `notebooks/colab_gpu_tests.ipynb` → `bash scripts/run_gpu_tests.sh`. Issue 2's fix (test-predicate adjustment) is verifiable locally to the extent that the predicate's logic is unit-testable on a CPU fixture; the failing assertion itself, however, lives inside a `requires_compatible_gpu` test and re-verifies on Colab.
- The Colab run is the final gate.

## 3. Problem 1: sam3 geometry-encoder fp32 leak

### 3.1 Failing test

`tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical`

### 3.2 Observed trace

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
  src/esam3/models/sam3.py    Sam3Wrapper.forward -> adapter.forward (autocast-wrapped, post-PR #13)
  sam3/model/sam3_image.py:449   self._encode_prompt(...)
  sam3/model/sam3_image.py:189   self.geometry_encoder(...)
  sam3/model/geometry_encoders.py:781   self._encode_points(...)
  sam3/model/geometry_encoders.py:623   proj = self.points_pos_enc_project(enc)   # raises here
```

### 3.3 Why this fires AFTER PR #13's fix

PR #13 §6.2 added a fallback `Prompt(box_embeddings=torch.zeros(0, B, 4, ...), box_mask=torch.zeros(B, 0, ..., bool))` when `_build_geometric_prompt` returns `None`. v2 also addressed the `box_embeddings` dtype so `points_direct_project` (the first projector at `geometry_encoders.py:594`) no longer sees a fp32 input.

The point-embedding tensor itself is now fine, but sam3's `_encode_points` ALSO computes a sinusoidal **positional encoding** (`enc`) for the points and feeds THAT into a SECOND projector — `self.points_pos_enc_project` at line 623. That positional encoding is generated by an internal `PositionEmbeddingSine`-style helper that constructs its frequency tensors in fp32 unconditionally (sam3's internals; not yet read line-by-line by this spec — the planner of the next session MUST confirm by reading `.venv/lib/python3.13/site-packages/sam3/model/geometry_encoders.py` around lines 600-630 before picking a fix). The autocast wrap added in PR #13 either does not engage for this op (because the op is constructed inside a sub-module that disables autocast — see §3.5), or the synthesized fp32 tensor is constructed AFTER the autocast cast-input boundary and never gets downcast.

Net effect: the dtype error moved from `points_direct_project` (line 594, fixed by v2) to `points_pos_enc_project` (line 623). Same class of bug, different sub-module, deeper in sam3.

### 3.4 Options the next session's planner MUST evaluate

DO NOT pick a winner in this spec. The planner will weigh these against current Colab data and pick one.

| Option | Sketch | Pros | Cons |
| --- | --- | --- | --- |
| (A) Wrap only the offending sam3 sub-call in `torch.autocast(bf16)` | Either monkey-patch `sam3.model.geometry_encoders.PointGeometryEncoder._encode_points` to wrap its body in autocast, OR re-introduce an inner autocast scope just before our adapter calls `forward_grounding` AND inside it, with a targeted `enabled=True`. | Surgical; keeps fp32 forward off the table. | High risk: sam3's decoder at `sam3/model/decoder.py:75` (`forward_ffn`) explicitly disables autocast via its own context. Wrapping at the wrong scope re-introduced the v2 bf16-vs-fp32 collision (see §3.5). Monkey-patching a 3rd-party sub-module is fragile across sam3 upgrades. |
| (B) Run forward in fp32 for the integration test (`ModelConfig(..., dtype="float32")`), accept the T4 memory cost just for the smoke test, document that bf16 forward on this sam3 revision needs upstream sam3 patches. | Change `tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical` to use `dtype="float32"`; add a code comment + a `logs/TODO.md` entry pointing at the sam3 internal issue. | Zero source-code risk; bypasses sam3's fp32 leak. The smoke test still demonstrates that `load_sam31` + canonical pipeline are wired. | Test no longer exercises the bf16 path on Colab T4; we lose end-to-end bf16 forward coverage. T4 has 15 GB VRAM — fp32 SAM 3.1 forward on a 1008x1008 image MAY OOM; the planner must verify or budget for downscaling. |
| (C) Monkey-patch sam3's geometry encoder so its sinusoidal pos-enc inherits dtype from input | At import time in `src/esam3/models/sam3.py` (or in a new `src/esam3/models/_sam3_patches.py`), monkey-patch the offending init/forward to do `enc = enc.to(dtype=points.dtype)` before the Linear. | Minimal source delta; explicit; localized. | Crosses the 3rd-party boundary; brittle across sam3 versions; needs unit-test coverage to detect breakage on sam3 bump. |
| (D) Upstream a sam3 PR fixing `_encode_points` to honor input dtype | Submit a PR to Meta's sam3 repo; pin a local fork until merged. | Right long-term answer. | Out of scope for the next sprint; cannot block the Colab green-suite goal on Meta review timeline. Keep as a follow-up ticket. |

The planner SHOULD also consider a hybrid: ship (B) immediately to unblock the suite, while filing (D) as a tracked upstream task. Or ship (C) as a stop-gap with a deletion-when-upstream-lands comment.

### 3.5 Constraint from v2 history: do NOT re-introduce the decoder bug

v2 spent hours on a different dtype collision: `sam3/model/decoder.py:77 self.linear1` inside `forward_ffn`. That sub-module's `forward_ffn` explicitly toggles autocast off (`with torch.autocast(..., enabled=False)`) to keep its math in fp32. An outer autocast wrap that is TOO BROAD (e.g., wrapping the whole `forward_grounding` plus its callees in fp16-mode autocast) causes the decoder to receive bf16 inputs against fp32 weights inside that explicit `enabled=False` region.

Any Option-(A) or Option-(C) fix MUST:
- Read `.venv/lib/python3.13/site-packages/sam3/model/decoder.py:60-95` (the `forward_ffn` region) before writing the fix, and confirm the chosen scope does not re-trigger that collision.
- Add a regression test or documented Colab-trace check that proves both `_encode_points`'s `points_pos_enc_project` AND `decoder.forward_ffn`'s `linear1` are happy.

### 3.6 Verification source paths the planner MUST read

- `.venv/lib/python3.13/site-packages/sam3/model/geometry_encoders.py` lines 580-640 (the `_encode_points` body + the `points_pos_enc_project` Linear + the upstream sinusoidal pos-enc construction).
- `.venv/lib/python3.13/site-packages/sam3/model/geometry_encoders.py` lines 700-800 (the geometry encoder's outer `forward` that calls `_encode_points`).
- `.venv/lib/python3.13/site-packages/sam3/model/decoder.py:60-95` (`forward_ffn`, the autocast-disabled region from v2 history).
- `.venv/lib/python3.13/site-packages/sam3/model/sam3_image.py:440-553` (`forward_grounding`, `_encode_prompt`, `_get_dummy_prompt`).
- Existing autocast wrap in our code: `src/esam3/models/sam3.py` (the post-PR #13 `_Sam3ImageAdapter.forward`).

### 3.7 Acceptance for Problem 1

- `tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical` passes on Colab T4.
- `tests/integration/test_load_sam31_real.py::test_load_sam31_returns_wrapper` (currently passing) still passes.
- No re-introduction of the v2 bf16-vs-fp32 collision in `sam3/model/decoder.py:77` `forward_ffn` (verified by trace inspection on Colab — if the collision returns, the test fails with a recognizable error string).
- If Option (B) is chosen: the test docstring or an in-line comment documents WHY fp32 is used and references the sam3-internal issue; a `logs/TODO.md` entry tracks the bf16 follow-up.
- If Option (A) or (C) is chosen: a unit-level smoke test (CPU) exercises the monkey-patch path or autocast helper so a sam3 version bump cannot silently regress it.

## 4. Problem 2: nn.Linear leak assertion catches LoRA adapters

### 4.1 Failing test

`tests/integration/test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora` (assertion at line 61)

### 4.2 Observed trace

```
AssertionError: plain nn.Linear modules remain after swap
  assert not _has_plain_nn_linear(base)
```

### 4.3 Root cause

The current predicate (`tests/integration/test_peft_qlora_real.py:49-51`) is:

```python
def _has_plain_nn_linear(module: nn.Module) -> bool:
    """True if any nn.Linear remains in the tree (excluding Linear4bit subclasses)."""
    return any(type(m) is nn.Linear for m in module.modules())
```

The test runs `apply_qlora`, which (correctly) replaces every original `nn.Linear` with `bnb.nn.Linear4bit`, then attaches a peft `lora.Linear` wrapper around each Linear4bit. peft's `lora.Linear` contains two ADAPTER weight modules — `lora_A` and `lora_B` — and each of those is a plain `nn.Linear`. They are SUPPOSED to be plain `nn.Linear`s (LoRA adapters live in full precision; quantization is only for the base layer).

So `_has_plain_nn_linear` correctly sees plain `nn.Linear`s, but they belong to LoRA, not to the base. The base 4-bit swap is in fact complete; the test predicate is too coarse.

### 4.4 Fix direction

Tighten the predicate to ignore plain `nn.Linear`s whose qualified name (from `named_modules()`) contains a LoRA adapter sub-path. Specifically: iterate `module.named_modules()` and flag a plain `nn.Linear` ONLY when its name does NOT contain any of:
- `lora_A`
- `lora_B`
- `lora_embedding_A`
- `lora_embedding_B`
- `lora_magnitude_vector` (DoRA — not currently configured but reserved)

The replacement predicate must still catch a regression where the base 4-bit swap is skipped on some submodule. To prove this, the planner SHOULD add a CPU unit test that:
1. Builds a tiny `nn.Sequential([nn.Linear(...), nn.Linear(...)])`.
2. Replaces ONE of the two Linears with a stub `Linear4bit`-shaped sentinel; leaves the other as plain `nn.Linear`.
3. Wraps it in a fake peft-style `lora.Linear` (or a manual `nn.Module` with `lora_A` / `lora_B` `nn.Linear` children).
4. Asserts the tightened predicate flags the leftover plain `nn.Linear` (the one not under a `lora_*` name) AND ignores the `lora_A` / `lora_B` children.

The choice of where to live (helper function in the test file vs a shared helper in `tests/fixtures/peft_helpers.py`) is the planner's call; either is fine.

### 4.5 What MUST NOT change

- `apply_qlora`'s behavior — it is correct.
- The `test_apply_qlora_swaps_every_linear_and_attaches_lora` test's other assertions (the `_has_linear4bit_modules` check, the trainable-ratio check, the vision-encoder / mask-decoder LoRA-target checks).
- `tests/integration/test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata` — fixed by Problem 3.
- `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip` — fixed by Problem 3.
- `tests/integration/test_peft_qlora_real.py::test_merge_lora_dequantizes_qlora_wrapper` — fixed by Problem 4.

### 4.6 Acceptance for Problem 2

- `tests/integration/test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora` passes on Colab T4.
- A new CPU unit test demonstrates the tightened predicate distinguishes a true base-Linear leak from LoRA-adapter Linears.
- The tightened predicate continues to flag a true regression (the unit test enforces this by construction).
- No changes to `apply_qlora`'s production code.

## 5. Problem 3: `Linear4bit.quant_type` -> `weight.quant_state.quant_type`

### 5.1 Failing tests

- `tests/integration/test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata`
- `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip`

### 5.2 Observed trace

```
AttributeError: 'Linear4bit' object has no attribute 'quant_type'. Did you mean: 'quant_state'?
  src/esam3/peft_adapters/qlora.py:170   return cast(str, module.quant_type)
```

The offending line lives in `_infer_quant_type_from_wrapper`:

```python
for module in wrapper.peft_model.modules():
    if isinstance(module, bnb.nn.Linear4bit):
        return cast(str, module.quant_type)
```

### 5.3 Root cause

In newer bitsandbytes (the version currently installed on Colab T4 alongside torch >= 2.4), `quant_type` is no longer an attribute on the `Linear4bit` module itself. It now lives on the `Params4bit` parameter inside the module — specifically at `module.weight.quant_state.quant_type`. The `QuantState` object is constructed lazily when the parameter is first moved to device or first used in a forward pass; the value is a `str` ("nf4" or "fp4").

The same code path also reads `compute_dtype` (`_infer_compute_dtype_from_wrapper`, `qlora.py:177-192`). That attribute is still on the module in current bitsandbytes (confirmed in v2's PR #13 source review), but the planner MUST re-verify against the installed bnb version on Colab and treat it the same way if it has also moved.

### 5.4 Fix direction

Update `_infer_quant_type_from_wrapper` to:
1. Read `module.weight.quant_state.quant_type` as the primary path.
2. Fall back to `module.quant_type` if the new attribute is missing (older bitsandbytes that pre-date the refactor — e.g., the version the test suite was originally written against).
3. If neither path yields a `str`, raise a `RuntimeError` with a diagnostic that names the bnb version and module repr (so future regressions are debuggable on Colab without re-instrumenting).

Read chain to verify before writing the fix:
- `module` is a `bnb.nn.Linear4bit` (subclass of `nn.Linear`).
- `module.weight` is a `bnb.nn.Params4bit` (subclass of `nn.Parameter`).
- `module.weight.quant_state` is a `bnb.functional.QuantState` (populated after the Linear4bit's `to(device)` call OR after the first forward pass — `apply_qlora` calls `.to(old.weight.device)` in `_replace_with_bnb_linear4bit`, which should trigger quantization and populate `quant_state`).
- `quant_state.quant_type` is the `str` we need.

Because `quant_state` is populated lazily, the planner MUST confirm at the call site (inside `save_qlora`) that `apply_qlora`'s `.to(device)` has already fired. If `quant_state` could plausibly be `None` at save time, the fix MUST handle that case explicitly (raise a clear error pointing the user at `apply_qlora` ordering, NOT silently fall back to `"nf4"`).

### 5.5 Same audit for `_infer_compute_dtype_from_wrapper`

`_infer_compute_dtype_from_wrapper` (`qlora.py:177-192`) currently reads `module.compute_dtype`. The planner MUST verify this attribute still exists on the installed bnb version on Colab. If it has moved (e.g., to `module.compute_dtype` -> `module.weight.compute_dtype`), apply the same primary-then-fallback pattern. If it has not moved, leave the function alone.

### 5.6 Suggested test scaffolding

Two layers:
- **Local unit test** (CPU, no bnb): mock a `Linear4bit`-shaped object with a `weight` attr that has a `quant_state` attr that has a `quant_type` attr. Confirm `_infer_quant_type_from_wrapper` returns the right string. Add a fallback case where only `module.quant_type` is set (legacy bnb).
- **Colab integration** (existing): the two failing tests above pass.

### 5.7 Acceptance for Problem 3

- `tests/integration/test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata` passes on Colab T4.
- `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip` passes on Colab T4.
- The CPU unit test demonstrates the primary-then-fallback read chain for `quant_type` (and for `compute_dtype` if the audit shows it has also moved).
- `esam3_qlora.json` written by `save_qlora` contains `"quant_type": "nf4"` and `"compute_dtype": "bfloat16"` (the test asserts this).

## 6. Problem 4: peft 0.19 `merge_and_unload` incompatible with `Linear4bit`

### 6.1 Failing test

`tests/integration/test_peft_qlora_real.py::test_merge_lora_dequantizes_qlora_wrapper`

### 6.2 Observed trace

```
RuntimeError: The size of tensor a (1572864) must match the size of tensor b (3072)
  at non-singleton dimension 0
  peft/tuners/lora/layer.py:871   base_layer.weight.data += delta_weight
  peft/tuners/tuners_utils.py:663 -> 726   (merge_and_unload)
  src/esam3/peft_adapters/lora.py:151   merged = wrapper.peft_model.merge_and_unload()
```

### 6.3 Root cause

For a 4-bit-quantized base layer (`bnb.nn.Linear4bit`), the underlying `weight` parameter is NOT shape `(out_features, in_features)` of dtype fp16/bf16 like a normal `nn.Linear`. It is a 1-D `uint8` blob holding the packed 4-bit quantized values, with metadata (scale/zero) in `weight.quant_state`. For the qkv layer in the failing trace: `3072 * 1024 * 0.5 = 1572864` bytes, so the packed tensor has shape `(1572864,)`.

peft's LoRA merge implementation at `lora/layer.py:871` does:

```python
base_layer.weight.data += delta_weight
```

with `delta_weight` shape `(3072, 1024)` (the dequantized LoRA delta in compute dtype). The shape mismatch (`(1572864,)` vs `(3072, 1024)`) is what raises. peft 0.19 either lost the `Linear4bit` special-case (regression) or never had it for the `merge_and_unload` path on this specific module.

The current `merge_lora` body in `src/esam3/peft_adapters/lora.py:143-154`:

```python
def merge_lora(wrapper: Sam3Wrapper) -> Sam3Wrapper:
    if wrapper.peft_model is None:
        raise RuntimeError("merge_lora: wrapper has no PeftModel; call apply_lora first")
    merged: Any = wrapper.peft_model.merge_and_unload()
    wrapper.model.model = merged
    wrapper.peft_model = None
    return wrapper
```

delegates wholly to peft's `merge_and_unload`. For LoRA-on-fp the delegation is fine; for LoRA-on-Linear4bit it explodes.

### 6.4 Fix direction

Implement an explicit dequant-then-merge path inside `merge_lora` for the QLoRA case. Detection: scan `wrapper.peft_model.modules()` for `bnb.nn.Linear4bit` BEFORE calling `merge_and_unload`. If found, take the explicit path; otherwise delegate as today.

Explicit path algorithm:
1. Walk every `peft.tuners.lora.layer.Linear` in `wrapper.peft_model`.
2. For each wrapper whose `base_layer` is `bnb.nn.Linear4bit`:
   a. Dequantize the 4-bit weight to compute dtype: `dequant = bnb.functional.dequantize_4bit(base_layer.weight.data, base_layer.weight.quant_state)`. The result has shape `(out_features, in_features)` in `compute_dtype`.
   b. Compute the LoRA delta: `delta = lora_B.weight @ lora_A.weight * scaling` (or use peft's `get_delta_weight()` helper if it works without touching `base_layer.weight.data`).
   c. Build a plain `nn.Linear(in_features, out_features, bias=...)` with `.weight.data = dequant + delta` and `.bias` copied from the original `base_layer.bias` (if present).
   d. Locate the parent module via `_resolve_parent`-equivalent walk (the existing helper at `qlora.py:64-70` already does this; either reuse or duplicate inline).
   e. Replace the `peft.tuners.lora.Linear` wrapper on the parent with the merged plain `nn.Linear`.
3. After walking all LoRA-on-4bit layers, the model contains:
   - Plain `nn.Linear`s in place of (LoRA + Linear4bit) wrappers.
   - Any LoRA-on-non-4bit layers (none expected in the QLoRA path, but handle defensively) still wrapped in `peft.tuners.lora.Linear`.
4. Strip remaining LoRA wrappers via the existing peft pattern (either a second walk or call peft's `unload()` on what remains). The acceptance criterion below (no `Linear4bit` modules remain) does NOT mandate that all LoRA wrappers also disappear — but it is the natural endpoint and the existing test `assert w.peft_model is None` will fail if we leave a `PeftModel` reference, so we MUST set `wrapper.peft_model = None` and rebind `wrapper.model.model` to the unwrapped module.

The planner SHOULD prefer reusing peft helpers (`get_delta_weight`, `scaling`) over re-deriving the LoRA math from scratch, but only IF those helpers don't touch the Linear4bit-shape mismatch internally. Read `peft/tuners/lora/layer.py` lines 700-880 before deciding.

### 6.5 Verification source paths the planner MUST read

- `.venv/lib/python3.13/site-packages/peft/tuners/lora/layer.py:700-880` (the merge path, the `get_delta_weight` helper, the existing `Linear4bit` handling — if any).
- `.venv/lib/python3.13/site-packages/peft/tuners/lora/bnb.py` (peft's bitsandbytes-specific LoRA module; this is where Linear4bit support traditionally lives — confirm whether 0.19 has a merge method here).
- `.venv/lib/python3.13/site-packages/bitsandbytes/functional.py` (`dequantize_4bit` signature: `dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=64, quant_type='fp4')` returns a fp tensor of `(out_features, in_features)`).
- `.venv/lib/python3.13/site-packages/bitsandbytes/nn/modules.py` (`Linear4bit`, `Params4bit`, `QuantState`).

### 6.6 What MUST NOT change

- `apply_qlora` and `apply_lora` — they correctly construct the wrapped model.
- The LoRA-only `merge_lora` path (i.e., when no `Linear4bit` is present) — must continue to work for the v1 LoRA tests (`test_merge_lora_on_real_sam31`).
- The `merge_lora` public signature `(wrapper: Sam3Wrapper) -> Sam3Wrapper`.

### 6.7 Acceptance for Problem 4

- `tests/integration/test_peft_qlora_real.py::test_merge_lora_dequantizes_qlora_wrapper` passes on Colab T4.
- `tests/integration/test_peft_lora_real.py::test_merge_lora_on_real_sam31` (currently passing) still passes — confirms the LoRA-only path is not regressed.
- After `merge_lora` returns on a QLoRA wrapper: `wrapper.peft_model is None` AND no `bnb.nn.Linear4bit` modules remain in `wrapper.model.model.modules()` (the test's last assertion).
- The merged model still produces equivalent outputs to `wrapper.peft_model` on a small fixed input. The existing test only checks structural properties — but Problem 4's fix MUST add at least one CPU-unit-test assertion that compares `merged_module(x)` to `pre_merge_wrapper(x)` (within reasonable atol) on a tiny synthetic Linear4bit-with-LoRA setup. This guards against a sign/scaling bug in the dequant-then-merge math.

## 7. Open questions for the planner of the next session

The planner MUST resolve these BEFORE producing the plan. None of them are answerable from this spec alone.

1. **Issue 1, Option choice.** Is the Colab green-suite goal more urgent than bf16-forward coverage in the smoke test? If yes -> Option (B). If no, and the planner is willing to read sam3's internal sub-module hierarchy to pick a precise autocast scope -> Option (A). If neither -> Option (C) as a stop-gap with a deletion-when-upstream-lands plan. Question to answer in the plan's pre-flight: which Option, and why?
2. **Issue 1, OOM budget for Option (B).** If Option (B) is selected, does `dtype="float32"` SAM 3.1 forward on a 1008x1008 input fit in T4's 15 GB VRAM? If not, the test must downscale OR the planner picks a different Option. The planner should sanity-check this against Meta's reference (their `Sam3Processor` ships fp32 by default — see v2 spec §3.3).
3. **Issue 3, `compute_dtype` audit.** Has `compute_dtype` also moved off `Linear4bit` in current bitsandbytes? If yes, fix `_infer_compute_dtype_from_wrapper` in the same PR. If no, leave it.
4. **Issue 3, `quant_state` lazy-population timing.** Does `apply_qlora`'s `_replace_with_bnb_linear4bit` (which calls `.to(old.weight.device)` on the new Linear4bit) ALWAYS populate `quant_state` synchronously, or is it deferred until the first forward? If deferred, `save_qlora` must trigger it (e.g., a no-op forward or explicit quantization call) BEFORE reading `quant_type`. The planner must verify by reading bitsandbytes source.
5. **Issue 4, peft helper reuse.** Does peft 0.19's `lora.Linear.get_delta_weight()` work cleanly when `base_layer` is `Linear4bit`, or does it short-circuit on the same shape assumption? If it works, reuse it; if not, derive the delta manually as `(lora_B.weight @ lora_A.weight) * scaling`.
6. **Issue 4, post-merge wrapper strip strategy.** Should the explicit dequant-then-merge path replace the WHOLE `peft.tuners.lora.Linear` with a plain `nn.Linear` (cleaner), OR keep the LoRA wrapper and just mutate `base_layer` (simpler but leaves a vestigial wrapper)? The acceptance criterion `wrapper.peft_model is None` plus the Linear4bit-absence check together imply the cleaner choice; confirm before coding.
7. **Branching and PR strategy.** Should each Problem (1, 2, 3, 4) be a separate commit on a single PR, or a separate PR per Problem? Recommendation: ONE PR with 4 commits (problems are independent, but verification on Colab is per-suite, not per-test; one PR amortizes the Colab run cost). The plan's TASK structure should reflect this.

## 8. Acceptance criteria

A correct implementation of this spec satisfies:

1. The new branch is cut from `main` AFTER PR #13's merge commit. `git log --oneline origin/main..HEAD` shows 4 (or so) commits, one per Problem.
2. `tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical` passes on Colab T4 (Problem 1).
3. `tests/integration/test_peft_qlora_real.py::test_apply_qlora_swaps_every_linear_and_attaches_lora` passes on Colab T4 (Problem 2).
4. `tests/integration/test_peft_qlora_real.py::test_save_qlora_writes_adapter_and_metadata` passes on Colab T4 (Problem 3).
5. `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip` passes on Colab T4 (Problem 3).
6. `tests/integration/test_peft_qlora_real.py::test_merge_lora_dequantizes_qlora_wrapper` passes on Colab T4 (Problem 4).
7. The 4 currently-passing Colab integration tests (`test_load_sam31_returns_wrapper`, `test_apply_lora_on_real_sam31_under_trainable_budget`, `test_save_load_roundtrip_on_real_sam31`, `test_merge_lora_on_real_sam31`) STILL pass.
8. `uv run pytest tests/unit -q --no-cov` reports `240 passed, 1 skipped` (or the same baseline plus any new unit tests added by this spec's fixes; new tests count UPWARD, never downward). Zero regressions.
9. `ruff check src tests` and `ruff format --check src tests` pass.
10. `pyproject.toml` is byte-identical to its pre-implementation state.
11. `notebooks/colab_gpu_tests.ipynb` is byte-identical to its pre-implementation state.
12. `logs/log.md` contains an entry per work item (one per Problem, plus a final Colab-verification entry).
13. No emojis anywhere in the diff.

## 9. Out of scope

| Item | Why deferred |
| --- | --- |
| Re-doing v2's torchao / numpy / scipy / transformers / huggingface_hub pins. | Already correct on PR #13 baseline; touching them would re-trigger Colab resolver churn. |
| Refactoring the `Sam3Wrapper` / `_Sam3ImageAdapter` architecture (e.g., merging them, exposing a non-adapter forward, plumbing `image_size` via a registry). | The current shape works for v2 + this spec; refactor when a third caller demands a new shape. |
| Any work on the 4 already-passing Colab tests (listed in §8 #7). | Not failing; touching them risks regressing the green slice. |
| Upstreaming the sam3 `_encode_points` fix to Meta. | Option (D) of Problem 1; tracked as a long-tail TODO, not blocking the Colab green suite. |
| Adding numerical-equivalence tests for bf16 vs fp32 forward. | Out of scope; v2 spec §7.3 also deferred this. |
| Supporting LoRA-on-non-4bit and LoRA-on-4bit mixed merge in `merge_lora`. | The QLoRA path is uniformly Linear4bit-based; mixed graphs are theoretical. If a future test introduces mixed mode, add a new ticket. |
| Box-prompt path through `_Sam3ImageAdapter`. | v1 §8; still deferred. |
| Multi-class-per-batch forward. | v1 §8; still deferred. |
| Replacing peft's `merge_and_unload` for all LoRA cases. | Only QLoRA needs the explicit path; LoRA-only continues delegating to peft. |
| Pinning a peft version (e.g. `peft<0.19`) as a workaround. | We're staying on peft 0.19+ — v2 already validated it. |

## 10. References for the implementer

- v2 spec (parent): `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-v2-design.md` — §3 (Problem 2 dtype investigation), §4 (Problem 3 torchao investigation), §6.3 (why autocast wraps the whole adapter body), §7.3 (verification scope).
- v2 plan: `docs/superpowers/plans/2026-05-17-colab-gpu-integration-fix-v2.md` — Task structure and per-task DoD pattern that this spec's plan should mirror.
- v1 spec: `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-design.md` — `SCOPE_TARGETS` rationale, `_Sam3ImageAdapter.forward` recipe.
- QLoRA spec: `docs/superpowers/specs/2026-05-17-peft-qlora-design.md` — `apply_qlora` / `save_qlora` / `load_qlora` contracts; `esam3_qlora.json` schema (v1).
- PEFT-LoRA spec: `docs/superpowers/specs/2026-05-17-peft-lora-design.md` — `merge_lora` contract and `SCOPE_TARGETS`.
- Failing source under our control:
  - `src/esam3/peft_adapters/qlora.py:164-174` (`_infer_quant_type_from_wrapper`).
  - `src/esam3/peft_adapters/qlora.py:177-192` (`_infer_compute_dtype_from_wrapper`, audit target).
  - `src/esam3/peft_adapters/lora.py:143-154` (`merge_lora`).
  - `tests/integration/test_peft_qlora_real.py:49-51` (`_has_plain_nn_linear` predicate).
- 3rd-party source (read but do not modify):
  - `.venv/lib/python3.13/site-packages/sam3/model/geometry_encoders.py:580-800` (Problem 1).
  - `.venv/lib/python3.13/site-packages/sam3/model/decoder.py:60-95` (Problem 1 collision constraint from v2 history).
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/layer.py:700-880` (Problem 4 merge path).
  - `.venv/lib/python3.13/site-packages/peft/tuners/lora/bnb.py` (Problem 4 bnb-specific merge).
  - `.venv/lib/python3.13/site-packages/bitsandbytes/nn/modules.py` (`Linear4bit`, `Params4bit`) for Problems 3 and 4.
  - `.venv/lib/python3.13/site-packages/bitsandbytes/functional.py` (`dequantize_4bit`) for Problem 4.
- Colab test runner: `scripts/run_gpu_tests.sh`, `notebooks/colab_gpu_tests.ipynb`.
