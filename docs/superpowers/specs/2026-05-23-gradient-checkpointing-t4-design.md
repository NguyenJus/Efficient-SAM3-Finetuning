# spec/gradient-checkpointing-t4 — Make Gradient Checkpointing Work on T4

**Status:** Draft (2026-05-23)
**Tracking issue:** #89 (re-scoped by this spec; see §10). Supersedes the deeper investigation that #60 nominally tracked but that auto-closed by accident (see §2.3).
**Scope:** Re-enable `gradient_checkpointing` end-to-end on a Colab T4 across BOTH entry points (static `load_sam31`, dynamic OOM ladder), fixing three coupled breaks in one PR: the activation-checkpoint recompute-metadata mismatch, the dead static entry point, and the dead dynamic OOM-recovery rung. Touches `src/custom_sam_peft/models/` (new patch module + wiring), `src/custom_sam_peft/train/loop.py`, `src/custom_sam_peft/config/schema.py`, shipped YAMLs, CPU unit tests, GPU tests, and three sibling specs. GPU-gated phases run manually on Colab T4.

**Sibling / parent specs:**
- [`2026-05-17-colab-gpu-integration-fix-v2-design.md`](2026-05-17-colab-gpu-integration-fix-v2-design.md) §6.2–6.3 — the adapter forward recipe and the `torch.autocast` reasoning (note: the v2-era adapter autocast was later superseded by the per-module dtype-patch family; see §3.4 of THIS spec for the live state).
- [`2026-05-17-colab-gpu-remaining-failures-design.md`](2026-05-17-colab-gpu-remaining-failures-design.md) §3, §3.5 — sam3's internal `torch.autocast(enabled=False)` fp32 regions (`decoder.forward_ffn` at `sam3/model/decoder.py:77`, `_encode_points` at `geometry_encoders.py:600-630`). These are the load-bearing constraint behind the recompute hypothesis.
- [`2026-05-19-gpu-test-policy-design.md`](2026-05-19-gpu-test-policy-design.md) §5.4 — the T4 VRAM-ceiling policy (ceilings must NOT be raised; the fix reduces usage).
- [`2026-05-18-smoke-test-design.md`](2026-05-18-smoke-test-design.md) §5.1 — the smoke YAMLs and the 14 GB / 10 GB ceilings.
- [`2026-05-21-yaml-config-defaults-audit-design.md`](2026-05-21-yaml-config-defaults-audit-design.md) §5 row 14 — the audit that flipped the default to `False` under #60.

---

## 1. Problem

Issue #89 asks to "re-enable `gradient_checkpointing` defaults once #60 is fixed." #60 is not fixed; it was closed by accident (§2.3). The user directive for this work is unambiguous: **training with gradient checkpointing on must work at all costs; resolve all bugs in this PR.** Brainstorming established that the premise is broken and uncovered two further latent breaks that, together, mean gradient checkpointing is wholly non-functional today through every path that could enable it. This spec covers the full fix as a single PR.

The user has also stated that **OOM is believed already resolved** — the ViT-L trunk at 1008×1008 fits in T4 memory without checkpointing (confirmed by the revert author: the QLoRA fast-smoke forward succeeded with loss=0.2200 at step 0 before the backward `CheckpointError`). Therefore acceptance is **correctness plus a measurable peak-VRAM reduction**, NOT "avoids OOM." Gradient checkpointing must be a working memory lever that demonstrably lowers peak VRAM, not a no-op flag.

### 1.1 The three breaks

| # | Break | Where | Severity |
| --- | --- | --- | --- |
| 1 | **Recompute-metadata mismatch.** Flipping sam3's per-block `use_act_checkpoint` flag on, under non-reentrant `torch.utils.checkpoint`, throws `torch.utils.checkpoint.CheckpointError` on the T4 QLoRA fast-smoke *backward* pass. | sam3 `vitdet.py:982` (external); triggered by our patch + our forward dtype regime. | Hard — blocks all checkpointing. |
| 2 | **Static entry point is dead.** `cfg.gradient_checkpointing=True` routes to a no-op `logger.warning` because sam3 has no `set_grad_checkpointing` on this revision. | `_construct_raw_model`, `src/custom_sam_peft/models/sam3.py:597-610`. | Mechanical once Break 1 is fixed. |
| 3 | **Dynamic entry point is dead.** The OOM ladder sets `state.gradient_checkpointing = True` but never propagates it to the live model, so the retry runs the unchanged model and OOMs identically. | `_train_step_with_oom_ladder`, `src/custom_sam_peft/train/loop.py:126-140`. | Mechanical once Break 1 is fixed. |

Breaks 2 and 3 are mechanical wiring fixes, but they are dead *because* Break 1 was never solved — the patch they would call into was reverted (§2.2). All three must land together for `gradient_checkpointing: true` to be honest.

---

## 2. Dependency history

### 2.1 The feat that added the patch — `889bd2c`

`889bd2c feat(sam3): enable activation checkpointing via _patch_enable_vit_act_checkpoint` added a helper that iterates the model tree and sets `use_act_checkpoint=True` on every submodule exposing that attribute (sam3's ViT-Det blocks), wired it into `load_sam31`'s `cfg.gradient_checkpointing=True` branch (replacing the no-op warning), and added a CPU unit test `tests/unit/test_sam3_act_checkpoint_patch.py`. The commit carried `Closes #60.` in its body. (At that time the package was `esam3`; the test imported `from esam3.models.sam3 import _patch_enable_vit_act_checkpoint`.)

### 2.2 The revert — `2dc4883`

`2dc4883 revert(sam3): unwire _patch_enable_vit_act_checkpoint — recompute mismatch on T4` deleted the helper and its test, restored the no-op warning branch, and flipped the example YAMLs back to `gradient_checkpointing: false`. The reported failure (verbatim from the commit body): on the T4 QLoRA fast-smoke backward pass, `torch.utils.checkpoint.CheckpointError`:

```
Recomputed values for the following tensors have different metadata than during the forward pass.
position 1: shape [256,512] bf16 → recomp [256,256] bf16
position 2: shape [8,32,34] f32  → recomp [256,256] bf16
(positions 3-7: recomp[N] == orig[N-1])
```

The revert author's reading: a **shift-by-one** save pattern fingerprinting a branch divergence in the first op of the checkpointed segment — original forward saved one tensor of shape X; recompute saved two tensors of a different shape; every subsequent save slot shifted by one. Both passes were verified to run under `is_grad_enabled()=True`, so `_patch_addmm_act_grad_safe` is **not** the divergence. The stated hypothesis: sam3's internal `with torch.amp.autocast(enabled=False)` fp32 regions interacting with non-reentrant checkpoint's autocast preservation across recompute. The author's explicit next move: run `torch.utils.checkpoint` with `set_checkpoint_debug_enabled(True)` to capture the per-op recompute trace and pinpoint the divergent op. The commit body closes with: **"#60 stays open for the deeper investigation."**

### 2.3 The accidental closure of #60

#60 was closed `COMPLETED` on **2026-05-21T06:19:40Z**. This was an accident: the `Closes #60.` keyword rode along inside the *reverted* feat commit `889bd2c` and fired when PR #58 (`973164d Manual GPU test pass — drain gpu-marked tests on Colab T4 (issue #44) (#58)`) merged. The revert author explicitly intended #60 to stay open, but the close had already fired. **There is no open issue tracking the real remaining work** — #89 only asks to flip defaults "once #60 is fixed," presuming a fix exists. This spec is the deeper investigation; #89 is re-scoped to own the full fix (§10). The #60 accidental-closure correction is recorded in §10.

---

## 3. Root-cause investigation (HYPOTHESIS — to be confirmed by Phase 0)

This section states the *leading hypothesis* for Break 1. It is a hypothesis, not an established fact. **Phase 0's diagnostic trace is the authority**; if the trace contradicts this section, the fix follows the trace and this section is corrected in the PR.

### 3.1 How sam3 self-checkpoints

sam3's ViT-Det blocks self-checkpoint inside their own forward (`vitdet.py:982`, external sam3 package):

```python
if self.use_act_checkpoint and self.training:
    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
```

`use_act_checkpoint` is read **per-block, per-forward**. The reverted `_patch_enable_vit_act_checkpoint` flipped it to `True` on every exposing block. Two consequences matter:
- Toggling between steps is feasible IF the patch is re-applied to the live model (relevant to Break 3 / Phase 2 dynamic wiring).
- `use_reentrant=False` means PyTorch re-runs the *same* checkpointed callable during backward, recording tensors saved by `torch.autograd.graph.saved_tensors_hooks`, and asserts the recompute saved tensors match the forward's by metadata (shape/dtype/etc.). A divergence raises `CheckpointError`.

### 3.2 The dtype/autocast regime around the trunk forward (live code)

**Correction vs brainstorming.** The brainstorming framing said recompute "fires outside the adapter's `with autocast` block." Verified against the live tree: `_Sam3ImageAdapter.forward` (`src/custom_sam_peft/models/sam3.py:344-398`) **has no `torch.autocast` wrap** today. The v2 spec (§6.2) described an autocast-wrapped adapter forward, but the live tree replaced that approach with the per-module dtype-patch family (`_apply_patches` → `Sam3Patches.apply`, `src/custom_sam_peft/runtime/_patches.py`; modules under `src/custom_sam_peft/models/_patches/`). The only `autocast` mention in `sam3.py` is the comment at lines 605-606.

So the autocast state that governs the trunk forward comes from one of two places depending on PEFT method:
- **LoRA path:** the training loop wraps the forward in `_autocast_ctx` (`src/custom_sam_peft/train/loop.py:179-186`), which returns `torch.autocast(device_type="cuda", dtype=bfloat16)` because `LoraAdapter.disables_outer_autocast()` returns `False` (`src/custom_sam_peft/peft_adapters/__init__.py:80-81`).
- **QLoRA path** (the path that actually produced the `CheckpointError`): `QloraAdapter.disables_outer_autocast()` returns `True` (`__init__.py:103-104`), so `_autocast_ctx` returns `contextlib.nullcontext()`. There is **no outer autocast** at all. Dtype consistency on the QLoRA path comes solely from sam3's internal autocast regions plus the per-module dtype hooks installed by `_apply_patches`.

### 3.3 The refined hypothesis

The original failure was on the **QLoRA** fast-smoke — i.e. the no-outer-autocast path. During the forward pass, the ViT-Det block's SDPA / first-op dispatch resolves under whatever autocast/dtype state is live (sam3's internal `autocast(enabled=False)` fp32 regions — `decoder.forward_ffn` at `sam3/model/decoder.py:77`, `_encode_points`'s positional-encoding projector at `geometry_encoders.py:600-630` — plus the per-module dtype hooks). During **backward recompute**, non-reentrant checkpoint re-runs the block, but the ambient autocast/dispatch context is reconstructed by checkpoint's own preservation logic, which may not reproduce the exact nesting of sam3's internal `autocast(enabled=False)` toggles and the forward-pre-hook dtype casts. If recompute resolves a different SDPA backend (or a different cast) than forward, it saves differently-shaped tensors → the observed shift-by-one metadata mismatch.

The "shift-by-one" save pattern (forward saved 1 tensor at a slot, recompute saved 2) is consistent with a **branch divergence in the first op of the checkpointed segment** — e.g. an SDPA call that took a fused (math-free, fewer saved tensors) backend on forward and a non-fused (more saved tensors) backend on recompute, or an autocast cast that ran on one pass and not the other. This is exactly what `set_checkpoint_debug_enabled(True)` will localize.

### 3.4 Why this is hard, not mechanical

Three properties make Break 1 the nuanced part of this work:
1. **It only reproduces in a real backward pass on T4.** The forward succeeds; CPU stubs do not exercise SDPA backend selection or sam3's fp32 regions. There is no dev-box repro.
2. **sam3 is not locally editable.** sam3 is present only in the Colab T4 environment (not importable on the dev box — confirmed: `import sam3` fails locally). The fix MUST be a load-time monkeypatch/wrapper in the existing `_patch_*` style (thin delegation shim in `sam3.py`, real `apply(model, runtime)` implementation under `models/_patches/`), NOT a sam3 source edit. Unlike the always-on dtype patches in `_ALL_PATCHES`, this patch is **config-gated** on `cfg.gradient_checkpointing` and is therefore invoked conditionally from `_construct_raw_model` and the OOM ladder rather than registered in `_ALL_PATCHES` (see §6, §8).
3. **There is a known adjacent landmine.** Per `2026-05-17-colab-gpu-remaining-failures-design.md` §3.5, an *over-broad* autocast scope re-introduces a bf16-vs-fp32 collision in `decoder.forward_ffn`'s `linear1` (`sam3/model/decoder.py:77`), which explicitly toggles autocast off. Any fix that adds an autocast scope must be narrow enough to not re-trigger that collision. The Phase-0 trace and Phase-3 verification both guard against this.

---

## 4. Goal & acceptance

**Goal.** With `gradient_checkpointing: true`, training runs correctly on T4 via BOTH entry points:
1. No `torch.utils.checkpoint.CheckpointError` on either the LoRA or QLoRA smoke (forward AND backward complete).
2. Convergence identical to checkpointing-off: step-0 loss matches the checkpointing-off reference within a tight tolerance (non-reentrant recompute is numerically exact, so this is a strict expectation, not a loose drop check).
3. **Peak VRAM measurably lower** than checkpointing-off — proving the lever is live.
4. Both static (`load_sam31`) and dynamic (OOM ladder) entry points enable checkpointing on the real model.

**Acceptance is correctness + measurable VRAM reduction.** It is explicitly NOT "avoids OOM" — the model already fits T4 without checkpointing. The T4 VRAM ceilings (14 GB LoRA, 10 GB QLoRA) MUST NOT be raised (gpu-test-policy §5.4 / §1 non-goal); the fix reduces usage under the existing ceilings.

---

## 5. Phase plan (one PR, GPU-gated in phases)

This is a **single PR**. It is decomposed into phases only to make the GPU-gating explicit and to bind the fix-selection (Phase 1) to evidence from the diagnostic spike (Phase 0). The phases are sequential: Phase 1's fix tier is chosen from Phase 0's trace; Phases 2 wiring is independent of which tier Phase 1 lands on; Phase 3 verifies on T4; Phase 4 is mechanical bookkeeping.

**GPU iteration reality.** There is no auto-provisioned GPU yet (auto-provisioning tracked by #124/#125 "soon"; persistent box "near future"; manual Colab T4 per `gpu-test-policy`). Phase 0 and Phase 3 GPU steps MUST be expressed BOTH as portable `pytest -m gpu` commands AND as copy-pasteable Colab cells with explicit "RUN ON COLAB T4" checkpoints, referencing `notebooks/colab_gpu_tests.ipynb`, so they transfer to auto-provisioned GPUs later.

### Phase 0 — Diagnostic spike (Colab T4, throwaway, NOT shipped)

Reproduce and localize Break 1. Output is a **recorded diagnosis** in the PR/spec, not shipped code.

Protocol (RUN ON COLAB T4):
1. Temporarily re-introduce the per-block flag flip (restore `_patch_enable_vit_act_checkpoint`'s `use_act_checkpoint=True` behavior on a throwaway branch or scratch cell) and wire it into the load path.
2. Enable the checkpoint debugger before the smoke:

   ```python
   import torch.utils.checkpoint as ckpt
   ckpt.set_checkpoint_debug_enabled(True)   # captures per-op forward/recompute metadata
   ```

3. Run the QLoRA fast-smoke (the path that originally failed; QLoRA disables outer autocast — see §3.2) under `pytest -m gpu` or the equivalent Colab cell. Capture the full `CheckpointError` trace including the per-op metadata table the debugger emits.
4. Identify the divergent op (the first slot where forward and recompute disagree) and classify the cause into exactly one of:
   - **autocast-only** — recompute resolved a different dtype/backend purely because the ambient autocast state differed → **Fix A**.
   - **needs RNG/full-context control** — divergence also involves RNG (dropout, stochastic-depth) or a context that autocast-pinning alone cannot reproduce → **Fix B**.
   - **benign non-differentiable divergence** — the divergent tensors provably do not participate in the backward gradient (e.g. an integer index buffer, a mask) → **Fix C**.
5. Also confirm the LoRA smoke (outer autocast ON) either reproduces or does not — the fix must satisfy both PEFT paths.

Record in the PR description (and fold the corrected hypothesis back into §3 if it diverges): the divergent op, its forward vs recompute metadata, the classification, and the chosen fix tier. Phase 0 ships **nothing** — the scratch re-introduction is reverted before Phase 1's real patch lands.

### Phase 1 — Fix (decision tree; escalate until checkpointing-on trains correctly)

Implement the lowest tier that Phase 0's trace justifies. Escalate only on evidence (a failing Phase-3 run, or a Phase-0 classification that rules the lower tier out). The escalation criteria are explicit below so "at all costs" does not become "thrash."

#### Fix A (lead) — deterministic-autocast wrapper

New patch module `src/custom_sam_peft/models/_patches/vit_act_checkpoint.py` exposing `apply(model, runtime) -> None`, plus a thin delegation shim `_patch_enable_vit_act_checkpoint(model)` in `src/custom_sam_peft/models/sam3.py` (matching the existing `_patch_*` delegation pattern at lines 402-520). The patch:

(a) **Flips the flag.** Iterates the model and sets `use_act_checkpoint=True` on every submodule that already exposes that attribute (restoring `889bd2c`'s behavior; idempotent via a per-module sentinel like `_custom_sam_peft_act_checkpoint_patched`, matching the `module_input_dtype` pattern at `_patches/module_input_dtype.py:46-49`). Must NOT inject the attribute onto modules that don't already declare it.

(b) **Wraps each such block's `forward`** so that the block runs under an explicit, deterministic `torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16, enabled=runtime.device.type == "cuda")` (dtype from `runtime`). Because non-reentrant checkpoint re-runs the *same wrapped callable* on recompute, the autocast context is reconstructed identically on forward and backward → consistent SDPA backend selection → no metadata divergence. The wrap is per-block (narrow), so it does NOT envelop `decoder.forward_ffn`'s `autocast(enabled=False)` region (§3.4 landmine avoidance — the ViT-Det blocks are in the trunk, upstream of the decoder).

Keep PyTorch's recompute metadata check **ON** (default `determinism_check="default"`). Fix A's correctness rests on the wrapped callable being deterministic, not on disabling the check.

#### Fix B (escalate if A is insufficient) — explicit checkpoint call with controlled context

If Phase 0 classifies the cause as "needs RNG/full-context control," or if a Phase-3 run with Fix A still raises `CheckpointError`: patch the block forward to **bypass sam3's internal `checkpoint.checkpoint(blk, x, use_reentrant=False)` call** and invoke `torch.utils.checkpoint.checkpoint(...)` ourselves with explicit control:

```python
torch.utils.checkpoint.checkpoint(
    blk, x,
    use_reentrant=False,
    context_fn=<deterministic ctx pair>,   # pins autocast AND RNG on recompute
)
```

The `context_fn` returns a `(forward_ctx, recompute_ctx)` pair that both enter the same `torch.autocast(...)` (and, if the trace implicates RNG, the same `torch.random.fork_rng` / preserved RNG state) so forward and recompute are bit-reproducible. This subsumes Fix A (it also pins autocast) and adds RNG/context control.

#### Fix C (last resort — ONLY if the trace proves divergent tensors are non-differentiable)

If and only if Phase 0 proves the divergent tensors do not flow into any gradient: relax the metadata check with `determinism_check="none"` on the affected checkpoint call, AND add a **numerical gradient-parity test** (GPU) proving that gradients with checkpointing-on match gradients with checkpointing-off within a tight tolerance, so "benign" is demonstrated, not asserted. Fix C is explicitly the last resort because it disables a safety check; it is not allowed without the parity evidence.

#### Escalation criteria (explicit)

- Default to **Fix A**.
- Escalate A → B when: Phase 0 classifies "needs RNG/full-context control," OR a Phase-3 T4 run with Fix A still raises `CheckpointError` or fails loss-parity.
- Escalate B → C **only** when: Phase 0 (or a B-tier trace) proves the divergent tensors are non-differentiable AND no autocast/RNG pinning resolves the metadata mismatch. C requires the gradient-parity test as a gate.

### Phase 2 — Re-wire both entry points + flip defaults

Independent of which fix tier Phase 1 landed; all three sub-items are mechanical.

1. **Static entry point.** In `_construct_raw_model` (`src/custom_sam_peft/models/sam3.py:597-610`), replace the no-op `logger.warning` branch (the `else` of the `hasattr(raw_model, "set_grad_checkpointing")` check) with a call to the working Phase-1 patch. Keep the `set_grad_checkpointing` branch as the preferred path if a future sam3 revision grows that method. The patch must run on the raw model before/with `_apply_patches`; choose placement so the deterministic-autocast wrap composes correctly with the existing per-module dtype hooks (the implementer confirms ordering against `_ALL_PATCHES`).
2. **Dynamic entry point.** In `_train_step_with_oom_ladder` (`src/custom_sam_peft/train/loop.py:126-140`), after the existing `state.gradient_checkpointing = True` rung, **actually apply the Phase-1 patch to the live `model`** before the retry `continue`. The helper already receives the live `Sam3Wrapper` as its `model` argument (call site `loop.py:317-318`); the wrapper holds the `_Sam3ImageAdapter`, which holds the raw sam3 model — the patch must reach the raw model's ViT-Det blocks. Because `use_act_checkpoint` is read per-block per-forward, applying the patch between steps takes effect on the very next retry. The patch's idempotency (Fix A item (a)) makes a repeated apply safe. The `OomEvent` "grad_ckpt_enabled" record (`src/custom_sam_peft/train/types.py:30-33`) is unchanged.
3. **Flip defaults.**
   - `ModelConfig.gradient_checkpointing` default `False` → `True` in `src/custom_sam_peft/config/schema.py:118-120`; remove the `# TODO(#60)` comment.
   - Flip the shipped YAMLs that carry the dangling `#60` comment from `false` → `true` and remove (or re-point to #89) the `# see issue #60 …` comment. **Correction vs brainstorming (4 YAMLs cited; actually 6 carry the comment):** `src/custom_sam_peft/cli/templates/coco_text_lora.yaml:21`, `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml:21`, `configs/examples/coco_text_lora.yaml:18`, `configs/examples/coco_text_qlora.yaml:18`, `configs/examples/gpu_smoke_lora.yaml:10`, `configs/examples/gpu_smoke_qlora.yaml:10`.
   - Three further YAMLs ship `gradient_checkpointing: false` with NO `#60` comment: `configs/examples/coco_text_auto_split.yaml:16`, `configs/examples/coco_text_no_val.yaml:18`, `configs/examples/coco_text_lora_subset.yaml:12`. The implementer flips these to `true` as well for consistency with the new default (they are example configs; leaving them `false` would contradict the shipped default). If any of these is a deliberate "checkpointing-off illustrative" config, the implementer documents the exception inline; absent that, flip them.

### Phase 3 — Verify (Colab T4)

With `gradient_checkpointing: true`, RUN ON COLAB T4 (and as `pytest -m gpu`):
1. No `CheckpointError`; backward completes on both the LoRA and QLoRA smokes.
2. Step-0 loss matches the checkpointing-off reference within a tight tolerance (recompute is numerically exact). Capture the off-reference loss first (one run with the flag forced off), then the on run, and assert near-equality.
3. **Peak VRAM measurably lower with checkpointing ON than OFF** — the lever is live. This is a *relative* assertion (on < off by a meaningful margin), distinct from the absolute ceiling assertions.
4. Release-tier GPU suite green under the unchanged ceilings: `tests/gpu/test_real_train_overfits.py` (LoRA, peak VRAM ≤ 14 GB asserted at line 70, `VRAM_CEIL_GB = 14.0` at line 33) and `tests/gpu/test_real_train_qlora.py` (QLoRA, peak VRAM ≤ 10 GB asserted at line 72, `VRAM_CEIL_GB = 10.0` at line 35).
5. The OOM-ladder rung, when exercised, actually enables checkpointing on the model (verified by the new GPU test in §7 that forces an OOM and asserts the live model's blocks have `use_act_checkpoint=True` and the retry proceeds).
6. Re-confirm no re-introduction of the `decoder.forward_ffn` bf16-vs-fp32 collision (§3.4) — a recognizable error string per `2026-05-17-colab-gpu-remaining-failures-design.md` §3.5; the narrow per-block wrap should not touch it, but Phase 3 proves it.

### Phase 4 — Bookkeeping

1. **Re-scope #89** to own the full fix (it currently presumes #60 is fixed); update its body to reference this spec. Record the **#60 accidental-closure correction** (a comment on #60 noting the deeper investigation it tracked is resolved by this PR, since #60 cannot be cleanly "re-fixed" — it is already closed COMPLETED). The recommended record is: a comment on #60 + a note in #89 linking the two, so the audit trail is intact.
2. **Reconcile stale specs** that name gradient checkpointing as the VRAM lever or that show `gradient_checkpointing: true` for a YAML that actually ships `false`:
   - `docs/superpowers/specs/2026-05-18-smoke-test-design.md` — the inline `gpu_smoke_lora.yaml` listing shows `gradient_checkpointing: true` at line 259, but the live `configs/examples/gpu_smoke_lora.yaml:10` ships `false`. After Phase 2 flips it to `true`, this spec listing becomes accurate again; add a dated reconciliation note. Also the VRAM-ceiling rationale (~lines 211-212) can now cite a *working* checkpointing lever.
   - `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md` — §5.4 (~lines 191/193) names "gradient checkpointing knobs" as the remediation when T4 approaches the ceiling; add a dated note that the lever is now functional (it was a dead reference before this PR). The "ceilings must not be raised" policy is unchanged and reaffirmed.

---

## 6. File-level change inventory

| File | Change | Phase |
| --- | --- | --- |
| `src/custom_sam_peft/models/_patches/vit_act_checkpoint.py` | **NEW.** `apply(model, runtime)` — flips `use_act_checkpoint=True` on every exposing block (idempotent) + wraps each block forward in deterministic `torch.autocast` (Fix A). Escalated body for Fix B/C if Phase 0 demands. | 1 |
| `src/custom_sam_peft/models/_patches/__init__.py` | Register the new patch in `_ALL_PATCHES` IF it should run on every load — **but** activation checkpointing is config-gated (`cfg.gradient_checkpointing`), unlike the always-on dtype patches. Decision: do NOT add to `_ALL_PATCHES` (which is unconditional); instead call the patch conditionally from `_construct_raw_model` and from the OOM ladder. Implementer confirms this is the right seam. | 1, 2 |
| `src/custom_sam_peft/models/sam3.py` | Add `_patch_enable_vit_act_checkpoint(model)` delegation shim (lines ~402-520 family). Replace the no-op warning branch in `_construct_raw_model` (lines 597-610) with a call to it. | 1, 2 |
| `src/custom_sam_peft/train/loop.py` | In `_train_step_with_oom_ladder` (lines 126-140), apply the patch to the live `model` after setting `state.gradient_checkpointing = True`, before `continue`. | 2 |
| `src/custom_sam_peft/config/schema.py` | `ModelConfig.gradient_checkpointing` default `False` → `True` (lines 118-120); remove `# TODO(#60)`. | 2 |
| `src/custom_sam_peft/cli/templates/coco_text_lora.yaml` | Line 21: `false` → `true`; drop/redirect `#60` comment. | 2 |
| `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml` | Line 21: `false` → `true`; drop/redirect `#60` comment. | 2 |
| `configs/examples/coco_text_lora.yaml` | Line 18: `false` → `true`; drop/redirect `#60` comment. | 2 |
| `configs/examples/coco_text_qlora.yaml` | Line 18: `false` → `true`; drop/redirect `#60` comment. | 2 |
| `configs/examples/gpu_smoke_lora.yaml` | Line 10: `false` → `true`; drop/redirect `#60` comment. | 2 |
| `configs/examples/gpu_smoke_qlora.yaml` | Line 10: `false` → `true`; drop/redirect `#60` comment. | 2 |
| `configs/examples/coco_text_auto_split.yaml` | Line 16: `false` → `true` (no comment to drop). | 2 |
| `configs/examples/coco_text_no_val.yaml` | Line 18: `false` → `true`. | 2 |
| `configs/examples/coco_text_lora_subset.yaml` | Line 12: `false` → `true`. | 2 |
| `tests/unit/test_sam3_act_checkpoint_patch.py` | **NEW** (re-created, modernized to `custom_sam_peft`). CPU mechanism tests — see §7.1. | 1 |
| `tests/unit/test_oom_ladder*.py` (existing OOM-ladder test file, or a new sibling) | Add a CPU case asserting the OOM-ladder "grad_ckpt_enabled" rung calls the patch on the live model. See §7.1. | 2 |
| `tests/gpu/test_real_train_overfits.py` / `tests/gpu/test_real_train_qlora.py` (or a new `tests/gpu/test_grad_checkpointing.py`) | Add GPU assertions: no `CheckpointError`, loss-parity vs off, VRAM-lower-with-on. See §7.2. | 3 |
| `notebooks/colab_gpu_tests.ipynb` | Add a "RUN ON COLAB T4" cell for the Phase-0 diagnostic protocol and the Phase-3 verification, if not already covered by `pytest -m gpu`. | 0, 3 |
| `docs/superpowers/specs/2026-05-18-smoke-test-design.md` | Dated reconciliation note (§5.1 listing, §5.4 ceiling rationale). | 4 |
| `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md` | Dated note that the checkpointing lever is now functional (§5.4). | 4 |

`_construct_raw_model`'s existing `set_grad_checkpointing` branch (line 598-599) is preserved as the preferred path for any future sam3 revision that grows the method.

---

## 7. Testing strategy

### 7.1 CPU unit tests (mechanism only — sam3-agnostic)

The recompute behavior is GPU-only; CPU tests assert the *mechanism* using synthetic `nn.Module` stand-ins, following the pattern of the deleted `tests/unit/test_sam3_act_checkpoint_patch.py` from `889bd2c` (a `_FakeViTDetBlock` with a `use_act_checkpoint` attribute, a `_FakeNonCheckpointable`, a `_FakeModel` tree). Re-create that test, modernized to `custom_sam_peft`:

- **Flips on every exposing block.** After `apply`, every `_FakeViTDetBlock.use_act_checkpoint is True`.
- **Skips non-exposing modules.** A module without the attribute is left untouched (the attribute is NOT injected).
- **Idempotent.** Double-apply leaves state correct and the per-module sentinel set.
- **Static entry point invokes it.** `load_sam31` / `_construct_raw_model` with `cfg.gradient_checkpointing=True` calls the patch (assert via a spy/monkeypatch on the delegation shim, since the real sam3 model is unavailable on CPU).
- **Dynamic entry point invokes it.** The OOM-ladder "grad_ckpt_enabled" rung applies the patch to the live model on flip (assert the patch is called with the model; a `_FakeModel` substitute for the wrapper's raw model suffices).
- **OOM ladder re-applies on flip.** Forcing `state.micro_batch_size == 1` and an OOM drives the rung; assert the patch ran exactly once and the retry proceeded.

**Honesty note (matches the `_patch_*_dtype` family precedent):** the deterministic-autocast *wrapping behavior itself* (Fix A item (b)) is GPU-only — there is no CPU assertion that the autocast context is reconstructed identically across recompute, exactly as the dtype-patch hooks are CPU-untested for their real-model dtype effects. The CPU tests cover the flag-flip and the wiring, not the autocast-determinism guarantee. This is stated plainly so the coverage is not over-claimed.

**sam3-version fragility guard.** The mechanism tests must fail loudly if sam3's contract changes (the `use_act_checkpoint` attribute renamed or removed). The patch should detect "found zero exposing blocks on a model that should have them" as a distinguishable signal where feasible, and the `_patches/README.md` "When SAM-3 bumps" checklist gains a row for `vit_act_checkpoint.py` pointing at `vitdet.py:982`.

### 7.2 GPU tests (behavior — `@pytest.mark.gpu`, release tier)

Per `gpu-test-policy`, these are release-tier (`-m gpu`), run manually on Colab T4 for now, expressed as portable `pytest -m gpu` commands.

- **No `CheckpointError`.** The LoRA and QLoRA smokes complete forward AND backward with `gradient_checkpointing: true`.
- **Loss parity.** Step-0 loss with checkpointing-on equals the checkpointing-off reference within a tight tolerance (recompute is numerically exact). The test captures both.
- **VRAM-lower-with-on.** Peak VRAM (`torch.cuda.max_memory_allocated`) with checkpointing-on is meaningfully lower than with it off, asserted as a relative inequality with a margin. Distinct from the absolute ceiling assertions in the existing smoke tests (lines 70 / 72), which stay green and unchanged.
- **OOM-ladder rung exercised.** A GPU test that forces an OOM (or simulates one at the `micro_batch_size==1` rung) asserts the live model's ViT-Det blocks have `use_act_checkpoint=True` after the rung fires and the retry proceeds.
- **Release suite green** under unchanged ceilings (14 GB / 10 GB).

If Fix C is selected, the **gradient-parity test** (§5 Phase 1 Fix C) is added as a GPU test and gates the merge.

### 7.3 What CI runs

CPU CI (`ubuntu-latest`) runs §7.1 only; the GPU tests collect-and-skip via the existing `requires_compatible_gpu` / `requires_checkpoint` autoskip (`tests/conftest.py`). `ruff check && ruff format --check && mypy && pytest` green on CPU. The flipped YAMLs must still validate against `TrainConfig` (covered by the existing `tests/unit/test_config_examples.py` parametrized loader).

---

## 8. Risks & open questions

| Risk / question | Mitigation |
| --- | --- |
| **Fix A may not fully resolve the mismatch.** | Escalate to B then C per the explicit §5 Phase 1 criteria, tied to Phase 0's trace. "At all costs" is bounded by the decision tree, not open-ended thrash. |
| **The root cause in §3 is a hypothesis.** | Phase 0's `set_checkpoint_debug_enabled(True)` trace is the authority; the fix follows the trace, and §3 is corrected in the PR if it diverges. |
| **Over-broad autocast re-triggers the `decoder.forward_ffn` bf16-vs-fp32 collision** (`sam3/model/decoder.py:77`, remaining-failures §3.5). | Fix A's wrap is per-ViT-Det-block (trunk, upstream of the decoder), not a global scope. Phase 3 step 6 explicitly re-confirms no collision. |
| **sam3 version fragility** — `use_act_checkpoint` / `vitdet.py:982` could rename across a bump. | CPU mechanism tests fail loudly on a contract change; `_patches/README.md` "When SAM-3 bumps" gains a row for this patch. |
| **Manual-GPU latency** — no auto-provisioned GPU yet. | Phase 0 and Phase 3 ship exact `pytest -m gpu` commands and Colab cells so they transfer to auto-provisioned GPUs (#124/#125) unchanged. |
| **VRAM ceilings must not move.** | Acceptance is correctness + *relative* VRAM reduction under the existing 14/10 GB ceilings; gpu-test-policy §5.4 forbids raising them and this PR reaffirms that. |
| **Patch placement vs `_apply_patches` ordering.** | The activation-checkpoint patch is config-gated, not in `_ALL_PATCHES`; the implementer confirms the deterministic-autocast block wrap composes correctly with the always-on per-module dtype hooks (the wrap is outer to the block forward; the hooks are forward-pre-hooks on inner Linear/LayerNorm/Conv — they compose, but Phase 3 proves it on T4). |
| **OOM-ladder reach into the raw model.** | The ladder holds the `Sam3Wrapper`; the patch must descend `wrapper → _Sam3ImageAdapter → raw sam3 model → ViT-Det blocks`. The implementer confirms the descent path; idempotency makes a redundant apply harmless. |
| **#60 is already closed COMPLETED and cannot be cleanly re-opened-and-fixed in the audit trail.** | Record the correction as a comment on #60 + a cross-link from #89; #89 owns the merge. (§10) |

---

## 9. Non-goals

- Raising the T4 VRAM ceilings (14 GB / 10 GB). The fix reduces usage under them.
- Enabling sam3's other checkpoint flags beyond the ViT-L trunk — `text_encoder_ve`'s `grad_checkpointing` and `sam3_image`'s `use_act_checkpoint_seg_head` (both handle much smaller tensors; deferred per `889bd2c`'s stated scope). If a follow-up wants them, file a separate issue.
- Editing sam3 source. The fix is a load-time monkeypatch/wrapper only.
- Adding a hosted/self-hosted GPU CI runner (tracked by #124/#125).
- Numerical equivalence of the *whole* training run bit-for-bit — the parity check is step-0 loss within tolerance (and gradient-parity only if Fix C is chosen).

---

## 10. Issue bookkeeping (Phase 4 detail)

- **#89** — re-scope from "flip defaults once #60 is fixed" to "fix activation-checkpointing recompute + re-wire both entry points + flip defaults." Update its body to link this spec. #89 is the PR's `Closes` target.
- **#60** — already CLOSED COMPLETED (2026-05-21, accidentally — §2.3). Add a comment: the deeper investigation #60 was meant to track (per `2dc4883`'s "stays open" note) is resolved by the PR closing #89; the accidental closure is acknowledged so the audit trail is correct. Do NOT silently rely on #60's closed state as evidence the work was done.
- New deferred issues (if Phase 0/1 surfaces them): the two other sam3 checkpoint flags (§9) get a follow-up issue only if a future need arises.

---

## 11. References

**Commits.**
- `889bd2c feat(sam3): enable activation checkpointing via _patch_enable_vit_act_checkpoint` — added the helper + CPU test, wired into `load_sam31`, body carried the stray `Closes #60.`.
- `2dc4883 revert(sam3): unwire _patch_enable_vit_act_checkpoint — recompute mismatch on T4` — the revert; carries the `CheckpointError` trace, the shift-by-one analysis, and the "#60 stays open" intent.
- `973164d Manual GPU test pass — drain gpu-marked tests on Colab T4 (issue #44) (#58)` — the merge that fired the stray `Closes #60`.

**Live code (verified at the cited lines on the worktree tip).**
- `src/custom_sam_peft/models/sam3.py:597-610` — the no-op `gradient_checkpointing` warning branch in `_construct_raw_model` (Break 2; the autocast comment is at 605-606).
- `src/custom_sam_peft/models/sam3.py:344-398` — `_Sam3ImageAdapter.forward` (no `torch.autocast` wrap today — §3.2 correction).
- `src/custom_sam_peft/models/sam3.py:402-520` — the `_patch_*` delegation-shim family (the style the new patch follows).
- `src/custom_sam_peft/models/_patches/module_input_dtype.py:42-50` — idempotent per-module sentinel pattern (`_custom_sam_peft_module_input_dtype_patched`).
- `src/custom_sam_peft/models/_patches/__init__.py:18-27` — `_ALL_PATCHES` registry (the new patch is config-gated, NOT added here — §6).
- `src/custom_sam_peft/runtime/_patches.py:25-32` — `Sam3Patches.apply`.
- `src/custom_sam_peft/train/loop.py:126-140` — the OOM-ladder "grad_ckpt_enabled" rung that never reaches the model (Break 3).
- `src/custom_sam_peft/train/loop.py:179-186` — `_autocast_ctx` (LoRA → bf16 autocast; QLoRA → nullcontext).
- `src/custom_sam_peft/train/loop.py:317-318` — the ladder call site (passes the live `model`).
- `src/custom_sam_peft/train/types.py:30-33` — `OomEvent` action `"grad_ckpt_enabled"`.
- `src/custom_sam_peft/peft_adapters/__init__.py:80-81, 103-104` — `disables_outer_autocast()` (LoRA `False`, QLoRA `True`).
- `src/custom_sam_peft/config/schema.py:118-120` — `ModelConfig.gradient_checkpointing = False  # TODO(#60)` (Break 2 default).
- YAMLs with `#60` comment: `src/custom_sam_peft/cli/templates/coco_text_lora.yaml:21`, `.../coco_text_qlora.yaml:21`, `configs/examples/coco_text_lora.yaml:18`, `.../coco_text_qlora.yaml:18`, `.../gpu_smoke_lora.yaml:10`, `.../gpu_smoke_qlora.yaml:10`. Without comment: `configs/examples/coco_text_auto_split.yaml:16`, `.../coco_text_no_val.yaml:18`, `.../coco_text_lora_subset.yaml:12`.
- `tests/gpu/test_real_train_overfits.py:33,70` (`VRAM_CEIL_GB = 14.0`, ≤14 GB assertion); `tests/gpu/test_real_train_qlora.py:35,72` (`VRAM_CEIL_GB = 10.0`, ≤10 GB assertion).
- Deleted CPU test pattern: `tests/unit/test_sam3_act_checkpoint_patch.py` as of `889bd2c` (synthetic `_FakeViTDetBlock` / `_FakeModel` stand-ins).

**sam3 (external — read but never edit; present only on Colab T4).**
- `vitdet.py:982` — `if self.use_act_checkpoint and self.training: x = checkpoint.checkpoint(blk, x, use_reentrant=False)`.
- `sam3/model/decoder.py:77` (`forward_ffn`) — the explicit `autocast(enabled=False)` fp32 region (the §3.4 landmine).
- `sam3/model/geometry_encoders.py:600-630` (`_encode_points`) — fp32 positional-encoding projector.

**PyTorch APIs used.**
- `torch.utils.checkpoint.checkpoint(..., use_reentrant=False, context_fn=..., determinism_check=...)` — Fix B/C.
- `torch.utils.checkpoint.set_checkpoint_debug_enabled(True)` — Phase 0 diagnostic.
- `torch.autocast(device_type=..., dtype=torch.bfloat16, enabled=...)` — Fix A deterministic wrap.
- `torch.nn.attention.sdpa_kernel` — available if Phase 0 shows backend-selection is the divergence and a backend must be pinned explicitly inside the wrapped block.
- `torch.cuda.max_memory_allocated` / `torch.cuda.reset_peak_memory_stats` — Phase 3 VRAM measurement.

**Specs.**
- `docs/superpowers/specs/2026-05-17-colab-gpu-integration-fix-v2-design.md` §6.2-6.3 (adapter recipe / autocast history).
- `docs/superpowers/specs/2026-05-17-colab-gpu-remaining-failures-design.md` §3, §3.5 (sam3 fp32 regions; the decoder landmine).
- `docs/superpowers/specs/2026-05-19-gpu-test-policy-design.md` §5.4 (T4 ceiling policy).
- `docs/superpowers/specs/2026-05-18-smoke-test-design.md` §5.1 (smoke YAMLs; ceilings).
- `docs/superpowers/specs/2026-05-21-yaml-config-defaults-audit-design.md` §5 row 14 (the default flip under #60).
