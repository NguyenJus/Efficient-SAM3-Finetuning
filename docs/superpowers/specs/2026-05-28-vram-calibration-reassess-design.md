# Reassess VRAM calibration: tightened formula + config-aware probe + B-then-K OOM ladder

**Issue:** [#148 — Reassess calibration mode: probing live GPUs risks OOM — consider a VRAM lookup table instead](https://github.com/NguyenJus/custom-sam-peft/issues/148) (reframed — see §1, §10).
**Release:** pre-1.0 minor bump (new behavior + a breaking change to the train-step OOM contract → MINOR).
**Status:** locked design, single PR, no back-compat shims.

The current VRAM estimator under-predicts badly. A run labeled "fits in 10.0 / 24.0 GiB" actually used **22 GiB**. The root cause is structural: the only quantity calibration tunes — the activation term — is derived from a fixed `r=4`, single synthetic class (`"thing"`), zeros-image, batch=1 probe, then scaled as `per_example × batch` with **no multiplex-K term and no attention term**. So roughly 12 GiB of real activation never enters the prediction. The "10 GiB" figure was essentially just the base model (`_model_bytes("lora") = MODEL_PARAMS(5e9) × 2 ≈ 9.3 GiB`) with the activation rounding to ~0 via the negative-activation clamp in `calibrate_cmd.py`.

Two further problems compound it:

1. **No run-time path applies the chosen preset to training.** `decide_preset()` reaches a real run only at config-generation time (the wizard and the notebook). `csp run` calls it solely for the bundle/provenance label (`BundleContext.preset`), never for `run_training(cfg)`. The config a user reviews is what actually trains — but nothing connects the estimator to it.
2. **The original #148 lookup-table proposal cannot be populated.** The maintainer has a single 16 GB GPU and cloud-GPU testing is low priority, so a GPU-keyed measured-peak table has no data source. (See §10 for the disposition.)

**Decision — approximation-first, no GPU table.** Three layers, shipped together:

- **(a) A tightened analytic formula** as the always-available default (Component 1).
- **(b) An opt-in, config-aware calibration probe** as the precision upgrade (Component 2).
- **(c) A run-time OOM ladder extension** as the safety net that makes imperfect estimates safe (Component 4).

Plus the wiring (Component 3) that closes the UX gap so the chosen preset actually reaches a run, and the explicit #148 disposition (Component 5).

---

## §1 Scope & non-goals

### In scope

| File | Component | Change |
|------|-----------|--------|
| `src/custom_sam_peft/presets.py` | 1 | Add a multiplex-`K_eff` term and the SDPA attention term to the **train** branch of `_predicted_bytes` / `_activation_bytes`; re-seed the analytic constants; thread `k` (classes-per-forward) into `decide_preset` so the train formula sees representative K. |
| `scripts/_derive_preset_constants.py` | 1 | **New.** Re-derive the seed constants from a measured probe on the maintainer's 16 GB card; referenced today only in `presets.py` comments. |
| `src/custom_sam_peft/cli/calibrate_cmd.py` | 2 | Rework `_run_probe` + `calibrate` to **read the config** (`image_size`, `k`, `method`/`r`/`batch`) and probe at those representative values; **rewrite `config.yaml` in place** with calibrated sizing + write the cache; **auto-init** when no config exists. |
| `src/custom_sam_peft/cli/run_cmd.py` | 3 | Add a **skip-init guard**: if no usable config, warn and auto-`init` (formula, no probe), then proceed. Leave the train-sizing tuple consumed verbatim (no new run-time resolution). |
| `src/custom_sam_peft/cli/init_cmd.py` | 3 | `init` bakes **concrete** formula-derived sizing values into `config.yaml` (annotated formula-derived); CPU-only fallback writes safe defaults + a warning to re-resolve on the GPU. |
| `src/custom_sam_peft/train/loop.py` | 4 | Extend the OOM ladder: **halve B → 1, then halve `effective_K` → 1, then hard-fail**; K-halving re-chunks all classes (none dropped) and replays the whole step; add `effective_K` as sticky outer state in `OomState`. |
| `src/custom_sam_peft/train/types.py` | 4 | Add the `multiplex_halved` action (carrying the new `effective_K`) to `OomEvent`. |
| `src/custom_sam_peft/runs/bundle.py` | 3, 4 | Render the formula-vs-calibrated provenance in `## Preset` (already partly present); render `multiplex_halved` events in the OOM edge-note. |
| `src/custom_sam_peft/cli/setup_wizard.py` | 2, 3 | The wizard offers `csp calibrate` with consent (opt-in); no probe runs implicitly. |
| Tests (see §9) | all | New CPU coverage for formula math, config rewrite/annotation, the init/calibrate/run workflow + skip-init guards, and the B+K OOM ladder; GPU coverage for the real probe, real-model peak, and formula-accuracy validation. |

### Out of scope

- **No GPU-keyed measured-peak lookup table.** The original #148 proposal is dropped and deferred — see §10. No new `provenance` value for a table.
- **No `auto` sentinel for the train sizing tuple.** `init` always bakes concrete `method`/`r`/`batch_size`/`grad_accum_steps`/`dtype` numbers, so there is never an unresolved train field at run time. The schema gains no train-tuple `"auto"`.
- **`eval.batch_size: "auto"` stays run-time.** It is resolved by `decide_eval_batch_size` (which already owns a batch-halving ladder and the attention ceiling) and is lower-stakes/forward-only. It is NOT baked.
- **No change to the other existing run-time resolutions.** Left exactly as they are: `train.optimizer: auto` → `recommended_optimizer`; `warmup_steps`/`eval_every`/`save_every`/`box_hint.decay_steps: None` → resolved from `steps_per_epoch`; `--resume __latest__` → `find_latest_checkpoint`.
- **No new heavy fudge multiplier.** The formula targets *slightly conservative*; accuracy plus the OOM ladder are the safety mechanism, not a margin.
- **No change to the NaN-driven group-skip.** The existing `finite_group_count` / `is_finite` path (Hungarian-matcher non-finite cost → skip a group) is unchanged and must not be confused with the new OOM-driven K re-chunking (§4).
- **No dynamic LoRA-rank downgrade in the ladder.** That would invalidate optimizer state; the ladder degrades only B then K.

---

## §2 Architectural approach

Three layers. The formula is the floor; the probe is the precision upgrade; the ladder is the safety net. The config.yaml is the single frozen source of truth that ties them together.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  PREDICT — at config-generation time (writes concrete numbers to YAML) │
   └──────────────────────────────────────────────────────────────────────┘

  csp init                                  csp calibrate (opt-in)
    │  decide_preset(k=cfg K, …)              │  config-aware probe at
    │  → tightened analytic formula           │  cfg's (image_size, k, method, r, batch)
    │    (model + adapter + opt +             │  → measured activation
    │     K_eff·activation + attention)       │
    ▼                                         ▼
  config.yaml  (concrete method/r/batch/    config.yaml  (same fields, re-annotated
   grad_accum/dtype, annotated               "# calibrated") + .custom_sam_peft_
   "# formula-derived")                       calibration.json cache
    │                                         │
    └───────────────────────┬─────────────────┘
                            ▼
              user review — manual edits win (they come last)
                            │
   ┌────────────────────────┼─────────────────────────────────────────────┐
   │  RUN — consumes the literals verbatim; no new train-tuple resolution   │
   └────────────────────────┼─────────────────────────────────────────────┘
                            ▼
  csp run --config config.yaml
    skip-init guard: no usable config → warn + auto-init (formula, no probe)
    run_training(cfg) uses cfg.peft.method/r, cfg.train.batch_size/grad_accum,
                      cfg.model.dtype VERBATIM
    eval.batch_size: auto STILL resolved at run time (decide_eval_batch_size)
    provenance (formula vs calibrated) flows to BundleContext.preset label

   ┌──────────────────────────────────────────────────────────────────────┐
   │  RECOVER — during training; B-then-K ladder, no class dropped          │
   └──────────────────────────────────────────────────────────────────────┘

  train/loop.py step:
    try forward+backward at (B, effective_K)
    on OOM:
      if micro_batch B > 1:  halve B; retry (existing rung)
      elif effective_K > 1:  zero_grad; halve effective_K; RE-CHUNK all classes
                             into more, smaller groups; REPLAY whole step from
                             group 0; record OomEvent("multiplex_halved", K)
      else:                  hard-fail with guidance
```

The formula and probe both write **concrete numbers** into `config.yaml`. The run path reads those numbers verbatim — the only run-time sizing decision left is `eval.batch_size: auto`. The ladder absorbs the residual error of an imperfect estimate without ever dropping a class.

---

## §3 Component 1 — Tightened analytic formula (the always-available default)

The estimator lives in `presets.py`. The current train branch of `_predicted_bytes` is:

```
model_bytes(method) + adapter_bytes(r) + optimizer_bytes(r)
  + activation_bytes(image_size, batch, cache)        # per_example × batch, K-agnostic
  + WORKSPACE_BYTES
```

`activation_bytes` is the *only* term calibration tunes, and it omits both the multiplex-K dependence and the SDPA attention cost. Both are added here.

### 3.1 Multiplex-`K_eff` term

Peak memory occurs during **one group's** forward+backward. Groups run sequentially: the trainer chunks `classes_in_batch` into `G = ceil(K_total / effective_K)` groups, accumulating gradients into shared adapter buffers, and the autograd graph frees after each group's backward (verified in `train/loop.py` — the `for group in groups` loop with per-group `backward()` and `total_loss` per group). So activation is a function of the **per-group** class count, not the total.

- Model activation as `f(B, K_eff, image_size)` where `K_eff = min(k, MULTIPLEX_CAP)` and `k = cfg.train.multiplex.classes_per_forward` (schema default 16, capped at `MULTIPLEX_CAP = 16`).
- This **replaces** the current K-agnostic `per_example × batch`. The per-example activation now scales with `K_eff` (the SAM 3.1 multiplex forward materializes per-class mask/box decoder activations within a group).
- `decide_preset` must accept `k` and thread it through to the train-branch formula. The default `k` is `cfg.train.multiplex.classes_per_forward` when a config is in scope (the calibrate/init paths have one); fall back to `MULTIPLEX_CAP` when no config is available (so the conservative worst case is used).

### 3.2 SDPA attention term

The eval path already models the SDPA score-matrix cost (the issue #162 attention ceiling) inside `decide_eval_batch_size`:

```python
_SAM3_PATCH = 14            # vision backbone patch size
_SAM3_HEADS = 16            # vision backbone attention heads
_n_tokens = (image_size // _SAM3_PATCH) ** 2
_attn_per_example = _SAM3_HEADS * _n_tokens * _n_tokens * 4   # fp32 worst case (SDPA math upcast)
```

- **Reuse this exact model** in the train-branch formula (factor it into a shared helper so the train and eval paths cite one definition, not two copies). At SAM 3.1's `image_size = 1008` and `patch = 14`, `N = 5184` tokens, so this `H · N² · 4 B` term is the **dominant** activation contributor — and it is exactly what the train formula omits today. The 10-vs-22 GiB miss is largely this term.
- It enters the **train** branch (forward+backward), so it is not scaled by `forward_only_factor` (that factor stays eval-only).

### 3.3 Re-seed the constants

- Add `scripts/_derive_preset_constants.py` (new; the `presets.py` comments already point at it). It loads SAM 3.1, runs a representative probe, and prints re-derived seed values for `MODEL_PARAMS`, `BASE_ACTIVATION_AT_1024`, and any new per-`K_eff` activation coefficient introduced by §3.1.
- **Validate against the maintainer's 16 GB card using the new config-aware probe** (Component 2): the formula must no longer under-predict the real peak.
- Target **slightly conservative**. **No heavy fudge multiplier** — accuracy plus the OOM ladder (Component 4) are the safety mechanism, not a margin.

### 3.4 What stays the same

- `model_bytes`, `adapter_bytes`, `optimizer_bytes`, `WORKSPACE_BYTES`, `Q_OVERHEAD`, the calibration-cache resolution order (`_load_cache`), `CACHE_SCHEMA_VERSION`, the candidate search and sort, the dtype-by-capability decision (`float16` on `cc < (8,0)`, else `bfloat16`), and the headroom env override are unchanged except where §3.1/§3.2 thread `K_eff` and the attention term into the train branch.

---

## §4 Component 4 — OOM ladder extension: B-then-K, no class dropped

The ladder lives in `train/loop.py`: `OomState`, `_train_step_with_oom_ladder`, and the `for group in groups` block in `train_step`. Today the helper hard-fails at `micro_batch=1` (`loop.py:124`: `"OOM at step {step} after micro_batch=1. Use a larger GPU."`).

### 4.1 New degradation order

```
halve micro-batch B → ... → 1     (existing inner rung; unchanged)
THEN halve effective_K → ... → 1   (new outer rung)
THEN hard-fail                     (replaces the current micro_batch=1 raise)
```

### 4.2 Hard invariant — no class is ever dropped (maintainer's explicit requirement)

Halving `effective_K` **re-chunks all `classes_in_batch` into more, smaller groups** — `G = ceil(K_total / effective_K)` grows. **Every class is still trained; none is bypassed.** An OOM triggers **finer chunking**, never **skipping** a class.

This is distinct from, and must not be confused with, the existing **NaN-driven group-skip** (`finite_group_count` / `is_finite`, where a non-finite Hungarian cost matrix skips a group's backward). That path is **unchanged**. The K-halving rung re-chunks and re-runs every class; it never marks a group non-finite.

### 4.3 Correctness — zero_grad + whole-step replay

A K-halving rung must:

1. `optimizer.zero_grad()` (discard the gradients already accumulated by the groups that backpropped at the larger K), then
2. **replay the whole step from the first group** at the new, smaller `effective_K` — NOT resume mid-step.

Resuming mid-step would double-count groups already backpropped at the larger K. `effective_K` becomes a **sticky outer ladder dimension** in `OomState`, coordinating with the inner B-ladder (the inner B-halving still happens within each replayed group).

**Effective batch / gradient magnitude are preserved** by the existing `/(G · grad_accum_steps)` division (the per-group `total_loss` is divided by `G * cfg.train.grad_accum_steps`). As K shrinks, `G` grows, so the divisor grows in lockstep — throughput drops, but the learned result is unchanged. This holds because there is **no cross-group coupling**: Hungarian matching and `total_loss` operate on per-`(image, class)` rows, so re-chunking the same classes into more groups produces the same summed gradient.

### 4.4 `OomState` and `OomEvent`

- `OomState` gains `effective_K: int` (the current sticky outer-ladder value; initialized to `min(cfg.train.multiplex.classes_per_forward, MULTIPLEX_CAP)`). The inner `micro_batch_size` stays as-is.
- `OomEvent` gains a new `action = "multiplex_halved"` carrying the new `effective_K`, recorded alongside the existing `microbatch_halved` events. (`OomEvent.action` widens from `Literal["microbatch_halved"]` to `Literal["microbatch_halved", "multiplex_halved"]`; the K-event carries `effective_K`; the B-event keeps carrying `new_micro_batch_size`.)

### 4.5 Final hard-fail

When both B and `effective_K` reach 1 and an OOM still fires, raise with guidance, e.g.:

```
OOM at step {step} after micro_batch=1 and classes_per_forward=1. Use a larger GPU or smaller image_size.
```

(replacing the current `loop.py:124` message).

---

## §5 Component 2 — Config-aware opt-in calibration probe

Rework `cli/calibrate_cmd.py`. The fix attacks the under-read at its source: the probe currently uses fixed `r=4`, a single `"thing"` class, a zeros image, and batch=1 — none of which reflect the config that will actually train.

### 5.1 Read the config, probe at representative values

- Read the config being calibrated for `image_size`, `k = cfg.train.multiplex.classes_per_forward`, and the intended `method` / `r` / `batch`, and probe at **those** values:
  - LoRA/QLoRA per `cfg.peft.method` at `cfg.peft.r` (not fixed `r=4`).
  - `K_eff = min(k, MULTIPLEX_CAP)` distinct synthetic class prompts (not a single `"thing"`).
  - `batch = cfg.train.batch_size` (not fixed 1).
  - `image_size = SAM3_IMAGE_SIZE` (the model's fixed native resolution).
- This replaces the fixed-`r=4` / single-`"thing"` / zeros / batch=1 probe and fixes the calibrated activation under-read.

### 5.2 Opt-in only

- The user runs `csp calibrate` explicitly. The wizard offers it **with consent**; it never runs implicitly. This keeps OOM-provocation off the default path — the core #148 concern. (The default path is the analytic formula, which never touches the GPU.)

### 5.3 Rewrite `config.yaml` in place + write the cache

- After a successful probe, **rewrite `config.yaml` in place** with the tighter sizing values, **re-annotated as calibrated** (a comment marking the sizing block `# calibrated <date>`), AND write the calibration cache (`.custom_sam_peft_calibration.json`, `schema_version = CACHE_SCHEMA_VERSION = 2`, same field set as today). Calibration's effect becomes visible in the exact file the user reviews.
- The negative-activation clamp (`calibrate_cmd.py:134-142`) stays as a defensive backstop: if the measured peak comes in below the modeled overhead, clamp to 0 and warn that the constants need re-deriving.

### 5.4 Auto-init when no config exists

- If no config exists, **auto-init** one (formula path, **no probe**) with a warning, then proceed to probe it — mirroring the `run` skip-init guard (§6). The user never has to hand-author a config before calibrating.

---

## §6 Component 3 — Workflow & run-time wiring (closes the UX gap)

The `config.yaml` is the frozen source of truth. Today nothing applies the chosen preset to a real run: `csp run` requires `--config` (mandatory Typer option), and `run_cmd.py:_load_preset_or_fallback` reads a `preset.json` sidecar else calls `decide_preset()` — feeding **only** `BundleContext.preset` (the post-training summary label), never `run_training(cfg)`. This component makes the file the user reviews the file that trains.

### 6.1 Canonical flow

1. **`csp init`** → writes **concrete** formula-derived sizing values into `config.yaml` (`peft.method`/`r`, `train.batch_size`/`grad_accum_steps`, `model.dtype`), annotated `# formula-derived`. The formula needs GPU memory info (`total_memory`, compute capability for the dtype choice). If `init` runs **CPU-only**, scaffold with **safe defaults + a warning** to re-resolve on the GPU (do not invent GPU numbers).
2. **`csp calibrate`** (optional) → config-aware probe (§5); overwrites the sizing block, re-annotated `# calibrated`.
3. **Review** → the user's manual edits are final. They come last; the model is "auto unless you write a value," realized as "init/calibrate fill the fields, your edits win."
4. **`csp run --config …`** → consumes the literals **verbatim**. There is **no new run-time resolution for the train sizing tuple** (`method` / `r` / `batch_size` / `grad_accum_steps` / `dtype`). Provenance (formula vs calibrated) flows through to the `BundleContext.preset` label.

### 6.2 Skip-init guard (on BOTH `run` and `calibrate`)

- If there is no usable config, **warn** ("not initialized"), **auto-`init`** (formula, **no probe**), then **proceed**. This guarantees there is always a concrete config to consume — there is never an unresolved train field at run time.

### 6.3 What stays run-time (explicitly NOT baked)

- **`eval.batch_size: "auto"`** stays run-time, resolved via `decide_eval_batch_size` (it owns its own batch-halving ladder and the SDPA attention ceiling, and is lower-stakes/forward-only). Do NOT bake it.
- The other existing run-time resolutions are left alone: `train.optimizer: auto` → `recommended_optimizer`; `warmup_steps`/`eval_every`/`save_every`/`box_hint.decay_steps: None` → resolved from `steps_per_epoch`; `--resume __latest__` → `find_latest_checkpoint`.
- **No `auto` sentinel** is added to the schema for the train tuple. `init` always bakes concrete numbers.

---

## §7 Component 5 — Issue #148 disposition

This spec **supersedes #148's lookup-table proposal.** The GPU-keyed measured-peak table is **dropped**: it cannot be populated. The maintainer has a single 16 GB GPU, and cloud-GPU testing (T4 / larger tiers) is low priority — so there is no data source for per-GPU pre-measured `activation_bytes_per_example` values across the common cards the table would key on.

The #148 concern (don't provoke OOM on the user's GPU by default) is honored a different way: the **default path is the analytic formula** (§3), which never touches the GPU; the **probe is strictly opt-in** (§5); and the **OOM ladder** (§4) makes any residual estimate error recoverable. No new `provenance` value (`"table"`) is introduced.

The table is **deferred** to if/when cloud-GPU testing infrastructure lands and can seed measured values across GPU tiers. Until then, formula + opt-in probe + ladder cover the need.

---

## §8 Provenance & bundle rendering

`PresetDecision.provenance` already carries `"calibrated"` vs `"analytic"`, and `runs/bundle.py::_preset_block` already renders a `- Source: calibrated <date>` / `- Source: analytic estimate` line. This component keeps that, with two adjustments:

- The label's "analytic" provenance is the **formula-derived** state (Component 1); "calibrated" is the **probe** state (Component 2). The bundle's `## Preset` block reflects which path produced the sizing the run actually used.
- `_oom_edge_note` is extended to render the new `multiplex_halved` events alongside `microbatch_halved` — e.g. the final-`effective_K` reached, in addition to the existing final-`micro_batch`. The note remains a single summary line in `## Edge cases`.

---

## §9 Testing strategy

Per-project rule: CPU-testable cases live on CPU; GPU tests are reserved for real-only failure modes and are run **one at a time** on the 16 GB card.

### 9.1 CPU-testable

| Area | Coverage |
|------|----------|
| Formula math (§3) | `_predicted_bytes(mode="train")` now grows with `K_eff` and includes the attention term; assert it is strictly larger than the old K-agnostic value at the same `(method, r, batch, image_size)`; assert the shared attention helper returns the same bytes for the train and eval paths; assert `decide_preset` threads `k` from the config and that larger `k` → larger predicted bytes (monotone). |
| Config rewrite / annotation (§5, §6) | `calibrate` rewrites `config.yaml` in place with the sizing block re-annotated `# calibrated`; `init` bakes concrete sizing annotated `# formula-derived`; CPU-only `init` writes safe defaults + the re-resolve warning; the rewritten file re-loads via `load_config`. |
| init/calibrate/run workflow + skip-init guards (§6) | `run` with no usable config warns + auto-inits (formula, no probe) then proceeds; `calibrate` with no config auto-inits then probes; `run` consumes the train tuple verbatim (no new resolution of `method`/`r`/`batch`/`grad_accum`/`dtype`); `eval.batch_size: auto` is still resolved at run time. |
| B+K OOM ladder (§4) | Inject a fake `torch.cuda.OutOfMemoryError` into the forward closure. Assert: (1) B halves to 1 first, then `effective_K` halves; (2) on a K-halving rung the classes are **re-chunked into more, smaller groups and EVERY class is still trained** (none dropped); (3) the rung calls `optimizer.zero_grad()` and **replays the whole step from group 0** (not mid-step); (4) `effective_K` is sticky across subsequent steps; (5) an `OomEvent("multiplex_halved", effective_K=…)` is recorded alongside `microbatch_halved`; (6) the final hard-fail fires only when both B and `effective_K` are 1, with the new message; (7) the NaN-driven group-skip path is untouched. |

### 9.2 GPU-only (run one at a time on the 16 GB card)

| Test | Asserts |
|------|---------|
| Real config-aware probe (§5) | the probe loads SAM 3.1, attaches the config's adapter at the config's `r`/`method`, runs forward+backward at the config's `K_eff`/`batch`, and writes a cache + rewritten config without crashing. |
| Real-model peak measurement | measured peak is in a sane range and the negative-activation clamp does not fire under representative settings. |
| **Formula-accuracy validation (the 10-vs-22 regression)** | the new analytic formula **no longer under-predicts** the real peak on the 16 GB card: the predicted bytes for the same config that previously read "10 GiB" now bracket the real ~22 GiB usage (slightly conservative, never below the measured peak). This is the headline correctness test for Component 1. |

---

## §10 Migration & breaking-change stance

**Pre-1.0. Clean breaking changes where they occur; no shims.**

| What changes | Who | How they notice |
|--------------|-----|-----------------|
| The train OOM ladder now adds a K-halving rung before hard-fail; the hard-fail message changes | anyone matching the old `"OOM at step … after micro_batch=1. Use a larger GPU."` string, or expecting a hard-fail at micro_batch=1 | a run that previously crashed at micro_batch=1 now re-chunks classes and continues; the final message names `classes_per_forward=1` too. |
| `OomEvent.action` widens to include `"multiplex_halved"`; the K-event carries `effective_K` | anyone exhaustively matching on `OomEvent.action` | a new literal value appears in the event stream and the bundle edge-note. |
| `init` / `calibrate` now bake concrete sizing into / rewrite `config.yaml` | all users | the emitted/rewritten config carries explicit `method`/`r`/`batch_size`/`grad_accum_steps`/`dtype` with a `# formula-derived` or `# calibrated` annotation; `run` consumes them verbatim. |
| The analytic formula now predicts substantially larger VRAM (multiplex-K + attention) | anyone relying on the old under-prediction to "fit" a config | a config that the old formula called feasible may now be reported as not fitting on a small card — which is correct (it was OOMing). |

The GPU-keyed lookup table from #148 is **not implemented** (§7) and introduces no migration surface.

### Rollback

Revert the PR. The formula change, the probe rework, the init/calibrate/run wiring, and the OOM-ladder extension are one logical change and revert as a unit. Reverting restores the K-agnostic formula, the fixed-`r=4` probe, the label-only `decide_preset` usage in `run`, and the micro_batch=1 hard-fail. The calibration cache and any `# calibrated`/`# formula-derived` annotations in user configs become inert but harmless after a revert (a new init/calibrate regenerates them).
