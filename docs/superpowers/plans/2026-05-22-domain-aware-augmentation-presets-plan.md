# Domain-Aware Augmentation Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-22-domain-aware-augmentation-presets-design.md`](../specs/2026-05-22-domain-aware-augmentation-presets-design.md)
**Issue:** [#75](https://github.com/NguyenJus/custom-sam-peft/issues/75) — *feat(data): domain-aware augmentation presets with safe/medium/aggressive intensity*
**Branch:** `feat/75-aug-presets` (worktree at `/home/justin/projects/custom-sam-peft/.worktrees/feat-75-aug-presets/`)

**Goal:** Replace the two-knob `AugmentationsConfig(hflip, color_jitter)` with a `(preset, intensity, overrides)` triple resolved against a frozen 12-cell preset table, refactor `build_train_transforms` to assemble Albumentations steps from the resolved knobs, add a `StainJitter` HED-space transform, expose `--preset` / `--intensity` to `csp init` and `--config` to `csp doctor`, and persist a per-run `augmentation_pipeline.json` sidecar.

**Architecture:** New module `data/aug_presets.py` (preset table, `LOCKED_OFF` map, `ResolvedAugmentations` dataclass, `resolve`, `_STEP_NAMES_FOR`, `dump_augmentation_pipeline`). Schema break in `config/schema.py` (no aliases). `data/transforms.py` gains `StainJitter`, the new `build_train_transforms` step-assembly, and a `resolve_normalization_with_path` shim. Trainer writes the sidecar. CLI changes are additive (`init` substitutes `${preset}`/`${intensity}`/`${overrides_block}`; `doctor --config` renders two new tables). Tests updated wholesale; no albumentations/pydantic version bump needed.

**Tech Stack:** Python 3.12, pydantic v2, Albumentations ≥1.4 (`albumentations.ImageOnlyTransform`), numpy (HED color deconvolution), Typer + Rich (CLI), pytest + `caplog` + `unittest.mock.patch`, `uv` + `ruff` + `mypy`, `gh` CLI.

---

## File Map

**New files:**

```
src/custom_sam_peft/data/aug_presets.py            CREATE — Preset/Intensity types, PRESET_TABLE, LOCKED_OFF, ResolvedAugmentations, resolve, _STEP_NAMES_FOR, dump_augmentation_pipeline
tests/unit/test_aug_presets.py                     CREATE — resolver + locked-off + sidecar-dump tests
tests/unit/test_stain_jitter.py                    CREATE — StainJitter unit tests
```

**Modified files:**

```
src/custom_sam_peft/config/schema.py                          TOUCHED (delete legacy AugmentationsConfig fields; add AugmentationOverrides + new AugmentationsConfig)
src/custom_sam_peft/data/transforms.py                        TOUCHED (StainJitter class + HED matrices; refactored build_train_transforms; resolve_normalization_with_path wrapper)
src/custom_sam_peft/data/coco.py                              TOUCHED (docstring/comment touch-ups if any reference legacy fields — no Python logic change; model_validate already drives the schema)
src/custom_sam_peft/data/hf.py                                TOUCHED (same — comments only)
src/custom_sam_peft/cli/init_cmd.py                           TOUCHED (add --preset / --intensity Typer options; string.Template substitution with ${overrides_block})
src/custom_sam_peft/cli/doctor_cmd.py                         TOUCHED (add --config Path | None option; render "Resolved augmentations" + "Normalization" tables; inject resolved_config into --json output)
src/custom_sam_peft/cli/templates/coco_text_lora.yaml         TOUCHED (replace augmentations block with ${preset}/${intensity}/${overrides_block} placeholders)
src/custom_sam_peft/cli/templates/coco_text_qlora.yaml        TOUCHED (same)
src/custom_sam_peft/train/trainer.py                          TOUCHED (after config.yaml write, dump augmentation_pipeline.json)
tests/unit/test_config_schema.py                              TOUCHED (replace legacy assertions; add overrides/preset/intensity validation tests)
tests/unit/test_data_transforms.py                            TOUCHED (replace legacy hflip/color_jitter tests with parameterized step-list tests)
tests/unit/test_data_coco.py                                  TOUCHED (fixture dict shape: {hflip, color_jitter} -> {preset, intensity, overrides})
tests/unit/test_data_hf.py                                    TOUCHED (same)
tests/unit/test_trainer_nan_behavior.py                       TOUCHED (one-line callsite migration to preset="none")
tests/unit/test_trainer_run_dir.py                            TOUCHED (callsite migration + new test for augmentation_pipeline.json)
tests/unit/test_cli_init.py                                   TOUCHED (add --preset/--intensity render tests + invalid-value rejection tests + custom-preset overrides scaffold test)
tests/unit/test_cli_doctor.py                                 TOUCHED (add --config table + JSON tests; assert byte-identical default behavior)
tests/integration/test_train_resume.py                        TOUCHED (callsite migration)
tests/integration/test_train_end_to_end.py                    TOUCHED (callsite migration + assert sidecar contents at end of run)
```

No new dependencies. `albumentations>=1.4` already in `pyproject.toml:21`; `numpy` transitively required; `pydantic` v2 already in use.

---

## Assumptions for the cold reader

1. **Working directory.** Every shell command runs with `cwd = /home/justin/projects/custom-sam-peft/.worktrees/feat-75-aug-presets`. Use absolute paths when invoking external tools; use repo-relative paths inside the plan text.
2. **Tooling.** `uv` is on PATH. Run every Python entry via `uv run …`. Pytest invocations use `uv run pytest …`. No bare `python`.
3. **Schema break is one-shot.** No aliases. No deprecation cycle. The PR migrates every callsite in the same diff (8 callsites in 6 test files + 2 templates — enumerated below).
4. **CPU-only.** Every test in this plan runs on CPU. No `@pytest.mark.gpu` markers. No real model load.
5. **`library_version` source.** `src/custom_sam_peft/__init__.py` defines `__version__ = "0.0.1"`. `dump_augmentation_pipeline` reads `custom_sam_peft.__version__` directly; fallback to `"unknown"` if missing. The version string in the sidecar is whatever `__version__` is at write time — tests assert non-empty, not equality.
6. **Logger name.** All resolver warns go to `logging.getLogger("custom_sam_peft.data.aug_presets")`. Test `caplog.set_level(logging.WARNING, logger="custom_sam_peft.data.aug_presets")`.
7. **No new deps.** `albumentations`, `numpy`, `pydantic` already pinned. The plan does not modify `pyproject.toml`. (Confirmed at plan-write time via `grep -n 'albumentations' pyproject.toml` → already present at line 21.)
8. **Mass test-fixture migration.** Eight callsites construct `AugmentationsConfig(hflip=..., color_jitter=...)` across 6 test files. These are deleted/rewritten in Phase G (one mechanical sweep), gated to land before any phase whose own tests would otherwise import the legacy fields.

---

## Parallel groups

The orchestrator dispatches by phase. Phase A is the foundation; everything else depends on it. After A lands, phases B, E, and the doc/migration-only sweep G can fan out together; C depends on A+B; D depends on A; F depends on A+B (touches the same file B owns).

```
Phase A (schema + resolver, NEW files)  [serial; foundation]
   │
   ├─────────────────────────────────────────────────┐
   │                                                 │
   ▼                                                 │
Phase B (StainJitter + resolve_normalization_with_path in transforms.py)  [PARALLEL with E and G]
   │                                                 │
   ├──── Phase E (init_cmd + templates)              │
   │                                                 │
   └──── Phase G (mass test-fixture migration)       │
                                                     │
After B, E, G complete:                              │
   │                                                 │
   ▼                                                 │
Phase C (build_train_transforms refactor +           │
         data/coco.py + data/hf.py comments +        │
         test_data_transforms / test_data_coco /     │
         test_data_hf updates)  [serial after A+B+G] │
   │                                                 │
   ├──── Phase D (trainer sidecar)  [depends on A; PARALLEL with C, F]
   │                                                 │
   └──── Phase F (doctor_cmd + tests)  [depends on A+B; PARALLEL with C, D]
                                                     │
   ▼                                                 │
Phase H (reviewer pass: design-sensitive + general + lint/format)  [serial; final]
```

**Concrete parallel batches the orchestrator can dispatch:**

- **Batch 1 (after A merges):** B, E, G in parallel (3 file-disjoint subagents).
- **Batch 2 (after B+E+G complete):** C, D, F in parallel (3 file-disjoint subagents — see "File-set disjointness verification" below for the proof).
- **Batch 3 (after C, D, F complete):** Phase H sequentially (two reviewers run in parallel; lint/format runs after both return).

### File-set disjointness verification

| Phase | Files touched |
|---|---|
| **B** | `src/custom_sam_peft/data/transforms.py` only |
| **E** | `src/custom_sam_peft/cli/init_cmd.py`, `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`, `tests/unit/test_cli_init.py` |
| **G** | `tests/unit/test_trainer_nan_behavior.py`, `tests/unit/test_trainer_run_dir.py` (callsite-only; the run-dir sidecar test is added in Phase D), `tests/integration/test_train_resume.py`, `tests/integration/test_train_end_to_end.py` (callsite migration only; the sidecar assertion is added in Phase D) |
| **C** | `src/custom_sam_peft/data/transforms.py` (refactor `build_train_transforms` only — B's edits already merged), `src/custom_sam_peft/data/coco.py` (comments), `src/custom_sam_peft/data/hf.py` (comments), `tests/unit/test_data_transforms.py`, `tests/unit/test_data_coco.py`, `tests/unit/test_data_hf.py` |
| **D** | `src/custom_sam_peft/train/trainer.py`, `tests/unit/test_trainer_run_dir.py` (sidecar test only — G's edits already merged), `tests/integration/test_train_end_to_end.py` (sidecar assertion only) |
| **F** | `src/custom_sam_peft/cli/doctor_cmd.py`, `tests/unit/test_cli_doctor.py` |

**Conflict scan:**
- B ∩ E ∩ G: ∅ — clean parallel.
- C ∩ D: `tests/integration/test_train_end_to_end.py` — D appends a sidecar assertion to a test G already migrated; **C does not touch this file**. → disjoint.
- C ∩ D: `tests/unit/test_trainer_run_dir.py` — D adds a sidecar test to a file G already migrated; **C does not touch this file**. → disjoint.
- C ∩ F: ∅.
- D ∩ F: ∅.
- D ∩ G ordering: G migrates the legacy callsites in `test_trainer_run_dir.py` and `test_train_end_to_end.py` first; D then appends the sidecar test/assertion. G must precede D — enforced by the phase ordering.

→ Batch 2 (C, D, F) is fully file-disjoint and dispatchable in parallel.

**Reviewer model floor (per CLAUDE.md):** sonnet/high for every implementer. Design-sensitive reviewer (Phase H1) is opus/xhigh; general code review (Phase H2) is sonnet/high; lint/format (Phase H3) runs the reviewer's tooling directly.

---

## Spec coverage map (every spec §16 deliverable → plan phase)

| Spec §16 row | Phase / step |
|---|---|
| 1. Spec doc | already on disk; no plan action |
| 2. New `AugmentationsConfig` + `aug_presets.py` | Phase A |
| 3. Refactor `build_train_transforms` | Phase C (Step C-3); depends on Phase B (StainJitter) and Phase A (resolver) |
| 4. Update `init_cmd` + templates | Phase E |
| 5. Update `doctor_cmd` + run-metadata sidecar | Phase F + Phase D |
| 6. Update example YAMLs (templates) | Phase E (Step E-2/E-3) |
| 7. Tests | Phases A, B, C, D, E, F all bundle their own tests; mass migration in Phase G |

Cross-check spec §12.1–§12.9:
- §12.1 (`test_aug_presets.py`) → Phase A
- §12.2 (`test_stain_jitter.py`) → Phase B
- §12.3 (`test_data_transforms.py` extend) → Phase C
- §12.4 (`test_config_schema.py` extend) → Phase A (because the new pydantic types are added there)
- §12.5 (`test_data_coco.py`, `test_data_hf.py` extend) → Phase C
- §12.6 (`test_trainer_*` callsite migration) → Phase G; sidecar assertion → Phase D
- §12.7 (integration test migration) → Phase G; sidecar assertion → Phase D
- §12.8 (`test_cli_init.py` extend) → Phase E
- §12.9 (`test_cli_doctor.py` extend) → Phase F

---

## Pre-flight (Phase 0)

**Model/effort:** sonnet / medium. **Parallel:** no. **Blocks:** all later phases.

### Step P0-1: Confirm working tree state

```bash
git -C /home/justin/projects/custom-sam-peft/.worktrees/feat-75-aug-presets status
```

Expected: branch `feat/75-aug-presets`. Untracked file `docs/superpowers/specs/2026-05-22-domain-aware-augmentation-presets-design.md` (the approved spec) and this plan. No staged or modified source files. If the spec is missing, halt.

### Step P0-2: Baseline test sanity

```bash
uv run pytest tests/unit/test_data_transforms.py tests/unit/test_config_schema.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_cli_init.py tests/unit/test_cli_doctor.py tests/unit/test_trainer_run_dir.py tests/unit/test_trainer_nan_behavior.py -q
```

Expected: all green. If anything is red, halt — the baseline is broken and Phase G / Phase C cannot be validated against the post-migration result.

### Step P0-3: Commit spec + plan

```bash
git add docs/superpowers/specs/2026-05-22-domain-aware-augmentation-presets-design.md \
        docs/superpowers/plans/2026-05-22-domain-aware-augmentation-presets-plan.md
git commit -m "docs: add domain-aware augmentation presets spec + plan (#75)"
```

---

## Phase A — `aug_presets.py` module, schema break, and schema tests

**Parallelism:** serial (foundation; blocks all others).
**Files touched:**
- Create: `src/custom_sam_peft/data/aug_presets.py`
- Modify: `src/custom_sam_peft/config/schema.py` (delete legacy `AugmentationsConfig` lines 57-59; add `Preset`, `Intensity`, `AugmentationOverrides`, new `AugmentationsConfig` in the same location)
- Create: `tests/unit/test_aug_presets.py`
- Modify: `tests/unit/test_config_schema.py`

**Spec ref:** §4, §5, §6, §7, §10 (sidecar helper), §12.1, §12.4.

**Verify:** `uv run pytest tests/unit/test_aug_presets.py tests/unit/test_config_schema.py -q`

### Task A1 — Schema break in `config/schema.py`

- [ ] **Step A1-1: Delete the legacy `AugmentationsConfig` and add the new types**

In `src/custom_sam_peft/config/schema.py`, **replace** lines 57-59 (the entire current `AugmentationsConfig` class) with:

```python
Preset = Literal["natural", "medical", "satellite", "microscopy", "none", "custom"]
Intensity = Literal["safe", "medium", "aggressive"]


class AugmentationOverrides(_Strict):
    """Per-knob overrides. All None → inherit from (preset, intensity).

    Setting any field to a non-None value replaces just that field in the
    resolved table. Extra keys are rejected (extra="forbid"); typos surface
    at config-load time.
    """

    hflip: bool | None = None
    vflip: bool | None = None
    rotate90: bool | None = None
    rotate_arbitrary: float | None = Field(default=None, ge=0.0)
    color_jitter: float | None = Field(default=None, ge=0.0)
    stain_jitter: float | None = Field(default=None, ge=0.0)
    blur: float | None = Field(default=None, ge=0.0)
    gauss_noise: float | None = Field(default=None, ge=0.0)


class AugmentationsConfig(_Strict):
    preset: Preset = "natural"
    intensity: Intensity = "medium"
    overrides: AugmentationOverrides = Field(default_factory=AugmentationOverrides)
```

Notes:
- `Field(default=None, ge=0.0)` on the float overrides enforces non-negativity at the schema layer (spec §15 edge-case: "`StainJitter` with negative sigma" — pydantic validators on `AugmentationOverrides.stain_jitter` should also reject negatives).
- The existing `from typing import Literal` import on line 11 already covers `Preset`/`Intensity`.
- The existing line 162 (`augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)`) inside `DataConfig` is unchanged — same field name, same default factory.

- [ ] **Step A1-2: Import-smoke**

```bash
uv run python -c "
from custom_sam_peft.config.schema import AugmentationsConfig, AugmentationOverrides, Preset, Intensity
cfg = AugmentationsConfig()
assert cfg.preset == 'natural'
assert cfg.intensity == 'medium'
assert cfg.overrides.model_dump() == {
    'hflip': None, 'vflip': None, 'rotate90': None,
    'rotate_arbitrary': None, 'color_jitter': None, 'stain_jitter': None,
    'blur': None, 'gauss_noise': None,
}
print('schema OK')
"
```

Expected: `schema OK`. Exit 0.

### Task A2 — Create `data/aug_presets.py`

- [ ] **Step A2-1: Write the module**

Create `src/custom_sam_peft/data/aug_presets.py` with the following content. Do not split into multiple files; this is intentionally a single-file module (~150 LOC) so the resolver + table + sidecar-dump are co-located.

```python
"""Domain-aware augmentation presets — resolver and run-metadata helpers.

Pure-Python (numpy at most). Does NOT import albumentations; the resolver can
be imported into `csp doctor` without dragging Albumentations into the doctor
import graph.

Public API:
  - PRESET_TABLE: dict[(Preset, Intensity), dict[str, bool | float]]
  - LOCKED_OFF:   dict[str, dict[str, str]]
  - ResolvedAugmentations: frozen dataclass with 8 knobs
  - resolve(cfg) -> ResolvedAugmentations
  - dump_augmentation_pipeline(cfg) -> dict  (sidecar helper)
  - _STEP_NAMES_FOR(resolved) -> list[str]   (module-private; consumed by
    trainer + doctor for run-metadata + table display)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from custom_sam_peft.config.schema import AugmentationsConfig, Intensity, Preset

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset × intensity table — spec §5
# ---------------------------------------------------------------------------

# Twelve cells for the four real domains. `none` and `custom` are short-circuited.
PRESET_TABLE: dict[tuple[Preset, Intensity], dict[str, bool | float]] = {
    ("natural", "safe"):       {"hflip": True,  "vflip": False, "rotate90": False, "rotate_arbitrary": 0.0,  "color_jitter": 0.05, "stain_jitter": 0.0,  "blur": 0.0,  "gauss_noise": 0.0},
    ("natural", "medium"):     {"hflip": True,  "vflip": False, "rotate90": False, "rotate_arbitrary": 0.0,  "color_jitter": 0.1,  "stain_jitter": 0.0,  "blur": 0.0,  "gauss_noise": 0.0},
    ("natural", "aggressive"): {"hflip": True,  "vflip": True,  "rotate90": False, "rotate_arbitrary": 10.0, "color_jitter": 0.2,  "stain_jitter": 0.0,  "blur": 0.05, "gauss_noise": 0.02},
    ("medical", "safe"):       {"hflip": False, "vflip": False, "rotate90": False, "rotate_arbitrary": 0.0,  "color_jitter": 0.0,  "stain_jitter": 0.0,  "blur": 0.0,  "gauss_noise": 0.0},
    ("medical", "medium"):     {"hflip": False, "vflip": False, "rotate90": False, "rotate_arbitrary": 5.0,  "color_jitter": 0.0,  "stain_jitter": 0.03, "blur": 0.0,  "gauss_noise": 0.01},
    ("medical", "aggressive"): {"hflip": False, "vflip": False, "rotate90": False, "rotate_arbitrary": 10.0, "color_jitter": 0.0,  "stain_jitter": 0.07, "blur": 0.03, "gauss_noise": 0.03},
    ("satellite", "safe"):     {"hflip": True,  "vflip": True,  "rotate90": True,  "rotate_arbitrary": 0.0,  "color_jitter": 0.0,  "stain_jitter": 0.0,  "blur": 0.0,  "gauss_noise": 0.0},
    ("satellite", "medium"):   {"hflip": True,  "vflip": True,  "rotate90": True,  "rotate_arbitrary": 0.0,  "color_jitter": 0.05, "stain_jitter": 0.0,  "blur": 0.0,  "gauss_noise": 0.0},
    ("satellite", "aggressive"): {"hflip": True, "vflip": True, "rotate90": True,  "rotate_arbitrary": 15.0, "color_jitter": 0.1,  "stain_jitter": 0.0,  "blur": 0.05, "gauss_noise": 0.02},
    ("microscopy", "safe"):    {"hflip": False, "vflip": True,  "rotate90": True,  "rotate_arbitrary": 0.0,  "color_jitter": 0.0,  "stain_jitter": 0.0,  "blur": 0.0,  "gauss_noise": 0.0},
    ("microscopy", "medium"):  {"hflip": False, "vflip": True,  "rotate90": True,  "rotate_arbitrary": 0.0,  "color_jitter": 0.0,  "stain_jitter": 0.0,  "blur": 0.0,  "gauss_noise": 0.0},
    ("microscopy", "aggressive"): {"hflip": False, "vflip": True, "rotate90": True, "rotate_arbitrary": 15.0, "color_jitter": 0.0, "stain_jitter": 0.0, "blur": 0.05, "gauss_noise": 0.02},
}


# ---------------------------------------------------------------------------
# Locked-off knob map — spec §6
# ---------------------------------------------------------------------------

LOCKED_OFF: dict[str, dict[str, str]] = {
    "medical": {
        "hflip":        "laterality (left vs right) is clinically meaningful in most medical modalities (CXR, mammography, derm)",
        "vflip":        "laterality (superior vs inferior) is clinically meaningful in most medical modalities",
        "rotate90":     "laterality is clinically meaningful; arbitrary 90° rotation breaks canonical orientation",
        "color_jitter": "color carries diagnostic signal (e.g. melanoma); use stain_jitter for H&E instead",
    },
    "natural": {
        "rotate90": "arbitrary 90° rotation breaks 'up' for natural photography; use rotate_arbitrary for mild tilt",
    },
    "microscopy": {
        "hflip":        "horizontal flip can break channel-ordering conventions in multiplexed microscopy",
        "color_jitter": "color identifies fluorescence channels and must be preserved",
    },
    "satellite": {
        "stain_jitter": "stain_jitter is H&E-specific (HED color deconvolution); satellite imagery is not H&E",
    },
}


# ---------------------------------------------------------------------------
# Resolved view — spec §7
# ---------------------------------------------------------------------------

_ZERO_BASE: dict[str, bool | float] = {
    "hflip": False, "vflip": False, "rotate90": False,
    "rotate_arbitrary": 0.0, "color_jitter": 0.0, "stain_jitter": 0.0,
    "blur": 0.0, "gauss_noise": 0.0,
}


@dataclass(frozen=True)
class ResolvedAugmentations:
    """Immutable 8-knob view consumed by build_train_transforms and the sidecar."""

    hflip: bool
    vflip: bool
    rotate90: bool
    rotate_arbitrary: float
    color_jitter: float
    stain_jitter: float
    blur: float
    gauss_noise: float


def _is_enabled(v: bool | float | None) -> bool:
    """True if v is a non-False bool or a strictly positive float."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) > 0.0
    return False


def resolve(cfg: AugmentationsConfig) -> ResolvedAugmentations:
    """Resolve (preset, intensity, overrides) into the 8-knob immutable view.

    - For `preset` in {"none", "custom"}: seed all-zero; intensity ignored.
    - Otherwise: seed from PRESET_TABLE[(preset, intensity)].
    - Apply overrides on top. Locked-off knob enabled under a real preset → WARN
      (user override wins; warn is the entire contract — spec §6).
    """
    if cfg.preset in ("none", "custom"):
        base: dict[str, bool | float] = dict(_ZERO_BASE)
    else:
        base = dict(PRESET_TABLE[(cfg.preset, cfg.intensity)])

    for field, override in cfg.overrides.model_dump().items():
        if override is None:
            continue
        base[field] = override
        if cfg.preset in ("none", "custom"):
            continue
        if field in LOCKED_OFF.get(cfg.preset, {}) and _is_enabled(override):
            reason = LOCKED_OFF[cfg.preset][field]
            _LOG.warning(
                "You enabled %s=%s under preset=%s; %s. The override will be applied as-is.",
                field, override, cfg.preset, reason,
            )

    return ResolvedAugmentations(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step-name list — spec §8 assembly mirrored for sidecar + doctor display
# ---------------------------------------------------------------------------

def _STEP_NAMES_FOR(resolved: ResolvedAugmentations) -> list[str]:
    """Ordered Albumentations class-name list produced by build_train_transforms.

    MUST match the conditional emission in
    `custom_sam_peft.data.transforms.build_train_transforms` step-for-step.
    """
    steps: list[str] = ["LongestMaxSize", "PadIfNeeded"]
    if resolved.hflip:
        steps.append("HorizontalFlip")
    if resolved.vflip:
        steps.append("VerticalFlip")
    if resolved.rotate90:
        steps.append("RandomRotate90")
    if resolved.rotate_arbitrary > 0.0:
        steps.append("Affine")
    if resolved.gauss_noise > 0.0:
        steps.append("GaussNoise")
    if resolved.blur > 0.0:
        steps.append("GaussianBlur")
    if resolved.color_jitter > 0.0:
        steps.append("ColorJitter")
    if resolved.stain_jitter > 0.0:
        steps.append("StainJitter")
    steps += ["Normalize", "ToTensorV2"]
    return steps


def dump_augmentation_pipeline(cfg: AugmentationsConfig) -> dict[str, Any]:
    """Build the JSON-shaped sidecar dict for a resolved augmentation config.

    See spec §10 for the exact dict shape. Consumed by the trainer to write
    `run_dir/augmentation_pipeline.json` and by `csp doctor --config` for the
    `resolved_config.augmentations` JSON block.

    For strict reproducibility across library versions, copy the returned
    `resolved` dict verbatim into `overrides:` under `preset: custom` —
    the resolver then returns identical values regardless of future
    PRESET_TABLE shifts.
    """
    try:
        from custom_sam_peft import __version__ as lib_version
    except (ImportError, AttributeError):
        lib_version = "unknown"

    resolved = resolve(cfg)
    return {
        "preset": cfg.preset,
        "intensity": cfg.intensity,
        "resolved": {
            "hflip": resolved.hflip,
            "vflip": resolved.vflip,
            "rotate90": resolved.rotate90,
            "rotate_arbitrary": resolved.rotate_arbitrary,
            "color_jitter": resolved.color_jitter,
            "stain_jitter": resolved.stain_jitter,
            "blur": resolved.blur,
            "gauss_noise": resolved.gauss_noise,
        },
        "steps": _STEP_NAMES_FOR(resolved),
        "library_version": lib_version,
    }
```

- [ ] **Step A2-2: Import-smoke**

```bash
uv run python -c "
from custom_sam_peft.data.aug_presets import (
    PRESET_TABLE, LOCKED_OFF, ResolvedAugmentations, resolve,
    dump_augmentation_pipeline, _STEP_NAMES_FOR,
)
from custom_sam_peft.config.schema import AugmentationsConfig
r = resolve(AugmentationsConfig())
assert r.hflip is True and r.color_jitter == 0.1, r
d = dump_augmentation_pipeline(AugmentationsConfig(preset='medical', intensity='medium'))
assert d['preset'] == 'medical'
assert d['resolved']['rotate_arbitrary'] == 5.0
assert d['resolved']['stain_jitter'] == 0.03
assert d['steps'] == ['LongestMaxSize', 'PadIfNeeded', 'Affine', 'GaussNoise', 'StainJitter', 'Normalize', 'ToTensorV2']
print('aug_presets OK')
"
```

Expected: `aug_presets OK`. Exit 0.

### Task A3 — Resolver tests (`tests/unit/test_aug_presets.py`)

- [ ] **Step A3-1: Create the test file**

Create `tests/unit/test_aug_presets.py`:

```python
"""Tests for custom_sam_peft.data.aug_presets — resolver, locked-off WARN, sidecar dump."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from custom_sam_peft.config.schema import AugmentationsConfig, AugmentationOverrides
from custom_sam_peft.data.aug_presets import (
    LOCKED_OFF,
    PRESET_TABLE,
    ResolvedAugmentations,
    _STEP_NAMES_FOR,
    dump_augmentation_pipeline,
    resolve,
)


_LOGGER = "custom_sam_peft.data.aug_presets"


@pytest.mark.parametrize(
    "preset,intensity",
    sorted(PRESET_TABLE.keys()),
)
def test_resolve_table_exact_values(preset: str, intensity: str) -> None:
    """Every (preset, intensity) cell resolves to its table row."""
    cfg = AugmentationsConfig(preset=preset, intensity=intensity)  # type: ignore[arg-type]
    resolved = resolve(cfg)
    expected = PRESET_TABLE[(preset, intensity)]  # type: ignore[index]
    for k, v in expected.items():
        assert getattr(resolved, k) == v, (preset, intensity, k, v, getattr(resolved, k))


@pytest.mark.parametrize("intensity", ["safe", "medium", "aggressive"])
def test_resolve_none_zeroes_all_knobs(intensity: str) -> None:
    cfg = AugmentationsConfig(preset="none", intensity=intensity)  # type: ignore[arg-type]
    resolved = resolve(cfg)
    assert resolved.hflip is False
    assert resolved.vflip is False
    assert resolved.rotate90 is False
    assert resolved.rotate_arbitrary == 0.0
    assert resolved.color_jitter == 0.0
    assert resolved.stain_jitter == 0.0
    assert resolved.blur == 0.0
    assert resolved.gauss_noise == 0.0


def test_resolve_custom_zeroes_then_overrides_apply() -> None:
    cfg = AugmentationsConfig(
        preset="custom",
        intensity="aggressive",  # ignored
        overrides=AugmentationOverrides(hflip=True, stain_jitter=0.05),
    )
    resolved = resolve(cfg)
    assert resolved.hflip is True
    assert resolved.stain_jitter == 0.05
    # Everything else stays at the all-zero seed.
    assert resolved.vflip is False
    assert resolved.color_jitter == 0.0
    assert resolved.gauss_noise == 0.0


def test_resolve_override_wins_over_table() -> None:
    cfg = AugmentationsConfig(
        preset="natural",
        intensity="medium",
        overrides=AugmentationOverrides(color_jitter=0.5),
    )
    resolved = resolve(cfg)
    assert resolved.color_jitter == 0.5  # override
    # Other fields preserved from the table row.
    assert resolved.hflip is True
    assert resolved.rotate_arbitrary == 0.0  # natural/medium row
    assert resolved.blur == 0.0


def test_resolve_override_zero_disables_table_knob() -> None:
    """Zero is a valid override, not 'inherit'."""
    cfg = AugmentationsConfig(
        preset="natural",
        intensity="medium",
        overrides=AugmentationOverrides(color_jitter=0.0),
    )
    resolved = resolve(cfg)
    assert resolved.color_jitter == 0.0


@pytest.mark.parametrize(
    "preset,knob,value,expected_substr",
    [
        ("medical", "hflip", True, "laterality"),
        ("natural", "rotate90", True, "up"),
        ("microscopy", "color_jitter", 0.1, "fluorescence"),
        ("satellite", "stain_jitter", 0.05, "H&E"),
    ],
)
def test_resolve_locked_off_warns(
    caplog: pytest.LogCaptureFixture,
    preset: str,
    knob: str,
    value: object,
    expected_substr: str,
) -> None:
    cfg = AugmentationsConfig(
        preset=preset,  # type: ignore[arg-type]
        intensity="medium",
        overrides=AugmentationOverrides(**{knob: value}),  # type: ignore[arg-type]
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolved = resolve(cfg)

    # Override applied as-is (not stripped).
    assert getattr(resolved, knob) == value

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1, [r.getMessage() for r in caplog.records]
    msg = warns[0].getMessage()
    assert knob in msg
    assert preset in msg
    assert expected_substr in msg


def test_resolve_locked_off_no_warn_when_disabling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """False/0 override on a locked-off knob does not warn (disabling is always fine)."""
    cfg = AugmentationsConfig(
        preset="medical",
        intensity="medium",
        overrides=AugmentationOverrides(hflip=False),
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolve(cfg)
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_resolve_none_skips_locked_off_check(caplog: pytest.LogCaptureFixture) -> None:
    cfg = AugmentationsConfig(
        preset="none",
        overrides=AugmentationOverrides(hflip=True),
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolved = resolve(cfg)
    assert resolved.hflip is True
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_resolve_custom_skips_locked_off_check(caplog: pytest.LogCaptureFixture) -> None:
    cfg = AugmentationsConfig(
        preset="custom",
        overrides=AugmentationOverrides(hflip=True, stain_jitter=0.1),
    )
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    resolved = resolve(cfg)
    assert resolved.hflip is True
    assert resolved.stain_jitter == 0.1
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_resolved_augmentations_frozen() -> None:
    r = resolve(AugmentationsConfig(preset="none"))
    # replace works.
    r2 = dataclasses.replace(r, hflip=True)
    assert r2.hflip is True
    # Direct mutation forbidden.
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.hflip = True  # type: ignore[misc]


def test_dump_augmentation_pipeline_shape_medical_medium() -> None:
    """Spec §10 literal example."""
    cfg = AugmentationsConfig(preset="medical", intensity="medium")
    d = dump_augmentation_pipeline(cfg)
    assert d["preset"] == "medical"
    assert d["intensity"] == "medium"
    assert d["resolved"] == {
        "hflip": False,
        "vflip": False,
        "rotate90": False,
        "rotate_arbitrary": 5.0,
        "color_jitter": 0.0,
        "stain_jitter": 0.03,
        "blur": 0.0,
        "gauss_noise": 0.01,
    }
    assert d["steps"] == [
        "LongestMaxSize", "PadIfNeeded", "Affine", "GaussNoise", "StainJitter",
        "Normalize", "ToTensorV2",
    ]
    assert isinstance(d["library_version"], str) and d["library_version"]


def test_dump_augmentation_pipeline_steps_empty_for_none() -> None:
    d = dump_augmentation_pipeline(AugmentationsConfig(preset="none"))
    assert d["steps"] == ["LongestMaxSize", "PadIfNeeded", "Normalize", "ToTensorV2"]


def test_step_names_for_natural_aggressive() -> None:
    """Representative non-trivial cell: every knob fires."""
    cfg = AugmentationsConfig(preset="natural", intensity="aggressive")
    resolved = resolve(cfg)
    assert _STEP_NAMES_FOR(resolved) == [
        "LongestMaxSize", "PadIfNeeded",
        "HorizontalFlip", "VerticalFlip",
        "Affine", "GaussNoise", "GaussianBlur", "ColorJitter",
        "Normalize", "ToTensorV2",
    ]
```

- [ ] **Step A3-2: Run the new tests**

```bash
uv run pytest tests/unit/test_aug_presets.py -q
```

Expected: all green. Total: 12 base tests + 12 parametrize cells for `test_resolve_table_exact_values` + 3 for `test_resolve_none_zeroes_all_knobs` + 4 for `test_resolve_locked_off_warns` ≈ 31 cells.

### Task A4 — Extend `tests/unit/test_config_schema.py`

- [ ] **Step A4-1: Update assertions**

In `tests/unit/test_config_schema.py`, find the existing `AugmentationsConfig` reference (line 117 contains the string `"AugmentationsConfig"` — confirm by `grep -n AugmentationsConfig tests/unit/test_config_schema.py`). Replace any test that constructs `AugmentationsConfig(hflip=..., color_jitter=...)` with `AugmentationsConfig(preset="natural", intensity="medium")` and update assertions to match the new defaults.

Then **append** the following tests to the end of the file:

```python
def test_augmentations_default_preset_and_intensity() -> None:
    from custom_sam_peft.config.schema import AugmentationsConfig

    cfg = AugmentationsConfig()
    assert cfg.preset == "natural"
    assert cfg.intensity == "medium"


def test_augmentation_overrides_rejects_unknown_keys() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationOverrides

    with pytest.raises(ValidationError):
        AugmentationOverrides.model_validate({"hfilp": True})  # typo


def test_augmentations_preset_literal_validation() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationsConfig

    with pytest.raises(ValidationError):
        AugmentationsConfig.model_validate({"preset": "mediacl"})  # typo


def test_augmentations_intensity_literal_validation() -> None:
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationsConfig

    with pytest.raises(ValidationError):
        AugmentationsConfig.model_validate({"intensity": "medum"})  # typo


def test_augmentations_overrides_default_factory_isolation() -> None:
    """Two AugmentationsConfig() instances must not share a single overrides object."""
    from custom_sam_peft.config.schema import AugmentationsConfig

    a = AugmentationsConfig()
    b = AugmentationsConfig()
    assert a.overrides is not b.overrides


def test_augmentations_overrides_all_none_by_default() -> None:
    from custom_sam_peft.config.schema import AugmentationsConfig

    dumped = AugmentationsConfig().overrides.model_dump()
    assert all(v is None for v in dumped.values())
    assert set(dumped.keys()) == {
        "hflip", "vflip", "rotate90", "rotate_arbitrary",
        "color_jitter", "stain_jitter", "blur", "gauss_noise",
    }


def test_augmentation_overrides_rejects_negative_floats() -> None:
    """Field(ge=0.0) on float overrides catches negative sigma at load time."""
    from pydantic import ValidationError

    from custom_sam_peft.config.schema import AugmentationOverrides

    with pytest.raises(ValidationError):
        AugmentationOverrides.model_validate({"stain_jitter": -0.1})
```

`pytest` is already imported at the top of the file; if it isn't, add `import pytest` to the top.

- [ ] **Step A4-2: Run schema tests**

```bash
uv run pytest tests/unit/test_config_schema.py -q
```

Expected: all green (existing tests still pass after the legacy-construction edits; new tests pass).

### Task A5 — Commit Phase A

- [ ] **Step A5-1: Commit**

```bash
git add src/custom_sam_peft/config/schema.py \
        src/custom_sam_peft/data/aug_presets.py \
        tests/unit/test_aug_presets.py \
        tests/unit/test_config_schema.py
git commit -m "feat(data): preset/intensity/overrides schema + aug_presets resolver (#75)"
```

---

## Phase B — `StainJitter` Albumentations transform + `resolve_normalization_with_path`

**Parallelism:** parallel-safe with Phase E and Phase G (file-disjoint).
**Depends on:** Phase A (no compile-time dep on aug_presets; this phase only touches `data/transforms.py`).
**Files touched:**
- Modify: `src/custom_sam_peft/data/transforms.py`
- Create: `tests/unit/test_stain_jitter.py`

**Spec ref:** §9, §11.2.2 (resolve_normalization_with_path), §12.2.

**Design decision (resolved at plan-write):** The spec note flags that `doctor_cmd`'s `resolve_normalization_with_path` requirement (§11.2.2) and the StainJitter transform both live in `data/transforms.py`. **Choice:** fold the `resolve_normalization_with_path` addition into this phase. Result: `data/transforms.py` has exactly one phase as its owner across Batch 1, and Phase F (doctor) becomes a pure `doctor_cmd.py` edit. This is cleaner than splitting Phase F into a transforms-edit + a doctor-edit pair.

**Verify:** `uv run pytest tests/unit/test_stain_jitter.py tests/unit/test_data_transforms.py -q` (the existing tests in `test_data_transforms.py` must still pass — Phase C extends them; this phase does not break them).

### Task B1 — Add `StainJitter` and HED matrices to `data/transforms.py`

- [ ] **Step B1-1: Insert module-level HED constants**

In `src/custom_sam_peft/data/transforms.py`, **between** the existing `_STATS_DIVERGENCE_ATOL = 1e-3` block (currently line 36) and the existing `def _stats_diverge(...)` function (currently line 39), insert the following block:

```python
# ---------------------------------------------------------------------------
# StainJitter — HED-space color deconvolution for H&E histopathology
# (Ruifrok & Johnston 2001 / Tellez et al. 2018). Image-only Albumentations
# transform; masks/bboxes/keypoints pass through unchanged.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

# Ruifrok & Johnston 2001 HED basis vectors (rows = stains: H, E, DAB).
_HED_FROM_RGB_MATRIX: np.ndarray = np.array(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ],
    dtype=np.float32,
)
_HED_FROM_RGB_INV: np.ndarray = np.linalg.inv(_HED_FROM_RGB_MATRIX).astype(np.float32)

# Magnitude → Albumentations parameter projection constants — spec §8.1.
_GAUSS_NOISE_MAX_VAR: float = 0.05
_GAUSS_BLUR_MAX_SIGMA: float = 3.0
```

`numpy` is currently only imported inside functions; this lifts it to module scope, which is fine because albumentations itself depends on numpy.

- [ ] **Step B1-2: Add the `StainJitter` class**

Append the following class to the end of `src/custom_sam_peft/data/transforms.py` (after `build_train_transforms`):

```python
class StainJitter:  # type: ignore[misc]
    """HED-space stain jitter for H&E histopathology images.

    Image-only — masks, bboxes, keypoints pass through unchanged. Implements
    the Tellez et al. (2018) / Ruifrok & Johnston (2001) color deconvolution:
    RGB → optical density → HED basis → per-channel affine perturbation →
    back to RGB.

    Identity at sigma=0 (the implementation short-circuits).
    """

    # Declared at class scope so subclassing albumentations.ImageOnlyTransform
    # at module-import time is possible without forcing eager Albumentations
    # import for callers that only resolve presets.
    _ALBU_BASE: type | None = None

    def __new__(cls, *args: object, **kwargs: object) -> "StainJitter":
        # Lazy-mix in albumentations.ImageOnlyTransform so the module can be
        # imported in test contexts that don't have albumentations on PYTHONPATH.
        if cls._ALBU_BASE is None:
            import albumentations as A

            cls._ALBU_BASE = A.ImageOnlyTransform
            # Build a subclass that adds ImageOnlyTransform to MRO once, in
            # place. After this StainJitter ISA ImageOnlyTransform.
            cls.__bases__ = (cls._ALBU_BASE,) + cls.__bases__  # type: ignore[assignment]
        return object.__new__(cls)

    def __init__(self, sigma: float = 0.0, p: float = 0.5) -> None:
        # __new__ wired the base into MRO; call up.
        super().__init__(p=p)  # type: ignore[call-arg]
        if sigma < 0:
            raise ValueError(f"StainJitter sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)

    def apply(self, img: np.ndarray, **params: object) -> np.ndarray:
        # img: uint8 RGB, HWC.
        if self.sigma == 0.0:
            return img
        od = -np.log10((img.astype(np.float32) + 1.0) / 256.0)
        hed = od @ _HED_FROM_RGB_INV
        alpha = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        beta = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        hed = hed * (1.0 + alpha) + beta
        od_back = hed @ _HED_FROM_RGB_MATRIX
        out = 256.0 * np.power(10.0, -od_back) - 1.0
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("sigma",)
```

**Note on the `__new__` mixin pattern:** Albumentations is a heavy import. The lazy-mixin keeps `from custom_sam_peft.data.transforms import StainJitter` cheap when only the class object is needed (e.g. doctor introspection); the first instantiation imports albumentations and wires `ImageOnlyTransform` into the MRO. If the reviewer prefers a straightforward top-level subclass (`class StainJitter(albumentations.ImageOnlyTransform)`), that is acceptable — change the class declaration and drop `__new__`. Either form is spec-compliant; the lazy form is slightly more defensive.

### Task B2 — Add `resolve_normalization_with_path` shim

- [ ] **Step B2-1: Add the 3-tuple variant and wrap the existing 2-tuple**

In `src/custom_sam_peft/data/transforms.py`, replace the existing `resolve_normalization` (lines 57-123) with a 3-tuple-returning variant **plus** a thin 2-tuple shim. The change is internal: the 2-tuple signature consumers depend on (`build_eval_transforms` lines 137, `build_train_transforms` line 173) is preserved.

Replace the entire current `def resolve_normalization(...)` body (lines 57-123) with:

```python
from typing import Literal  # noqa: E402

NormalizationPath = Literal["processor", "table-fallback", "config-fallback"]


def resolve_normalization_with_path(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float], NormalizationPath]:
    """Three-step resolution of (mean, std) plus the path that fired.

    Path codes:
      - "processor":       loaded from AutoImageProcessor
      - "table-fallback":  processor unavailable, model in KNOWN_PROCESSOR_STATS
      - "config-fallback": processor unavailable, no table entry, user fallback

    Logging is unchanged from the legacy `resolve_normalization`: WARN on
    fallback paths, WARN on table-vs-processor divergence, INFO on the happy
    path.
    """
    import transformers

    table_entry = KNOWN_PROCESSOR_STATS.get(model_name)

    try:
        proc = transformers.AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model_name, local_files_only=True
        )
        mean = list(proc.image_mean)
        std = list(proc.image_std)
    except (OSError, AttributeError, ValueError):
        if table_entry is not None:
            table_mean, table_std = table_entry
            _LOG.warning(
                "AutoImageProcessor unavailable for %r; using known-good stats "
                "(mean=%s, std=%s). Populate the HF cache to silence this warning.",
                model_name,
                table_mean,
                table_std,
            )
            return list(table_mean), list(table_std), "table-fallback"
        _LOG.warning(
            "AutoImageProcessor unavailable for %r AND no known-good entry registered; "
            "using NormalizeConfig fallback (mean=%s, std=%s). Verify these are correct "
            "for this backbone.",
            model_name,
            fallback.mean,
            fallback.std,
        )
        return list(fallback.mean), list(fallback.std), "config-fallback"

    if table_entry is not None and _stats_diverge((mean, std), table_entry):
        table_mean, table_std = table_entry
        _LOG.warning(
            "AutoImageProcessor for %r returned stats (mean=%s, std=%s) that diverge "
            "from KNOWN_PROCESSOR_STATS (mean=%s, std=%s) beyond tolerance %g. "
            "Using processor values; update the table if this divergence is expected.",
            model_name,
            mean,
            std,
            table_mean,
            table_std,
            _STATS_DIVERGENCE_ATOL,
        )
    else:
        _LOG.info("Using image_mean/image_std from AutoImageProcessor for %r.", model_name)
    return mean, std, "processor"


def resolve_normalization(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float]]:
    """2-tuple wrapper kept for build_eval_transforms / build_train_transforms.

    Equivalent to dropping the path code from
    :func:`resolve_normalization_with_path`.
    """
    mean, std, _path = resolve_normalization_with_path(model_name, fallback)
    return mean, std
```

This preserves every existing caller's contract while making the path code available to `doctor_cmd` (Phase F).

- [ ] **Step B2-2: Existing test suite still green**

```bash
uv run pytest tests/unit/test_data_transforms.py -q
```

Expected: existing tests pass. The Phase C refactor will replace the legacy `hflip`/`color_jitter` tests; here we only verify the shim doesn't regress them yet.

### Task B3 — `StainJitter` tests

- [ ] **Step B3-1: Create `tests/unit/test_stain_jitter.py`**

```python
"""Tests for custom_sam_peft.data.transforms.StainJitter — HED-space stain jitter."""

from __future__ import annotations

import numpy as np
import pytest

from custom_sam_peft.data.transforms import StainJitter


def _random_uint8_image(shape: tuple[int, int, int] = (32, 32, 3)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_identity_at_sigma_zero() -> None:
    img = _random_uint8_image()
    out = StainJitter(sigma=0.0, p=1.0).apply(img)
    assert np.array_equal(out, img)


def test_dtype_and_shape_preserved() -> None:
    img = _random_uint8_image()
    out = StainJitter(sigma=0.1, p=1.0).apply(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_range_preserved() -> None:
    img = _random_uint8_image()
    out = StainJitter(sigma=0.1, p=1.0).apply(img)
    assert out.min() >= 0
    assert out.max() <= 255


def test_mask_untouched_through_compose() -> None:
    import albumentations as A

    img = _random_uint8_image()
    mask = np.ones((32, 32), dtype=np.uint8) * 7
    compose = A.Compose(
        [StainJitter(sigma=0.1, p=1.0)],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"],
            min_visibility=0.0, min_area=0.0,
        ),
    )
    out = compose(image=img, mask=mask, bboxes=[], class_labels=[])
    assert np.array_equal(out["mask"], mask)


def test_determinism_with_numpy_seed() -> None:
    img = _random_uint8_image()
    np.random.seed(0)
    out1 = StainJitter(sigma=0.1, p=1.0).apply(img)
    np.random.seed(0)
    out2 = StainJitter(sigma=0.1, p=1.0).apply(img)
    assert np.array_equal(out1, out2)


def test_sigma_negative_rejected() -> None:
    with pytest.raises(ValueError, match="sigma must be >= 0"):
        StainJitter(sigma=-0.1)


def test_p_zero_passes_through() -> None:
    import albumentations as A

    img = _random_uint8_image()
    compose = A.Compose(
        [StainJitter(sigma=0.5, p=0.0)],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"],
            min_visibility=0.0, min_area=0.0,
        ),
    )
    out = compose(image=img, bboxes=[], class_labels=[])
    assert np.array_equal(out["image"], img)
```

- [ ] **Step B3-2: Run StainJitter tests**

```bash
uv run pytest tests/unit/test_stain_jitter.py -q
```

Expected: all 7 tests pass.

### Task B4 — Commit Phase B

- [ ] **Step B4-1: Commit**

```bash
git add src/custom_sam_peft/data/transforms.py tests/unit/test_stain_jitter.py
git commit -m "feat(data): StainJitter HED transform + resolve_normalization_with_path shim (#75)"
```

---

## Phase E — `csp init` flags + template substitution

**Parallelism:** parallel-safe with Phase B and Phase G (file-disjoint).
**Depends on:** Phase A (imports `Preset`, `Intensity` from `config.schema`).
**Files touched:**
- Modify: `src/custom_sam_peft/cli/init_cmd.py`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`
- Modify: `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml`
- Modify: `tests/unit/test_cli_init.py`

**Spec ref:** §11.1, §11.1.1, §11.3, §12.8.

**Verify:** `uv run pytest tests/unit/test_cli_init.py -q`

### Task E1 — Add `--preset` / `--intensity` options to `init_cmd.py`

- [ ] **Step E1-1: Edit `init` function signature and body**

In `src/custom_sam_peft/cli/init_cmd.py`:

1. Add `import string` and `from typing import get_args` near the top imports.
2. Import `Intensity, Preset` from `custom_sam_peft.config.schema`.
3. Add two new Typer options to `init(...)` signature **before** `output: Path = …`:

   ```python
       preset: str = typer.Option(
           "natural",
           "--preset",
           case_sensitive=False,
           help="Augmentation domain preset. One of: natural, medical, satellite, microscopy, none, custom.",
       ),
       intensity: str = typer.Option(
           "medium",
           "--intensity",
           case_sensitive=False,
           help="Augmentation intensity tier. One of: safe, medium, aggressive.",
       ),
   ```

4. Inside `init`, after the existing template-validity check (`if template not in TEMPLATES:`), add the preset/intensity validity check:

   ```python
       preset_lc = preset.lower()
       intensity_lc = intensity.lower()
       valid_presets = set(get_args(Preset))
       valid_intensities = set(get_args(Intensity))
       if preset_lc not in valid_presets:
           raise typer.BadParameter(
               f"unknown preset '{preset}'. Available: {sorted(valid_presets)}",
               param_hint="--preset",
           )
       if intensity_lc not in valid_intensities:
           raise typer.BadParameter(
               f"unknown intensity '{intensity}'. Available: {sorted(valid_intensities)}",
               param_hint="--intensity",
           )
   ```

5. Replace the existing template-write line (`body = (files(...) / TEMPLATES[template]).read_text(); output.write_text(body)`, currently lines 61-62) with:

   ```python
       if preset_lc == "custom":
           overrides_block = (
               "overrides: {}  # fill in knobs: hflip, vflip, rotate90, "
               "rotate_arbitrary, color_jitter, stain_jitter, blur, gauss_noise"
           )
       else:
           overrides_block = (
               "# Override individual knobs here; unset keys inherit from (preset, intensity).\n"
               "    # overrides:\n"
               "    #   hflip: false\n"
               "    #   color_jitter: 0.15"
           )

       raw = (files("custom_sam_peft.cli.templates") / TEMPLATES[template]).read_text()
       body = string.Template(raw).substitute(
           preset=preset_lc,
           intensity=intensity_lc,
           overrides_block=overrides_block,
       )
       output.write_text(body)
       rprint(f"[green]wrote[/green] {output}")
   ```

### Task E2 — Edit `templates/coco_text_lora.yaml`

- [ ] **Step E2-1: Replace lines 34-36**

In `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, replace the current augmentations block (lines 34-36):

```yaml
  augmentations:
    hflip: true
    color_jitter: 0.1
```

with:

```yaml
  augmentations:
    preset: ${preset}
    intensity: ${intensity}
    ${overrides_block}
```

The 4-space indent on `${overrides_block}` matches the rest of the `augmentations:` block; the multi-line string built in `init_cmd` already carries the correct leading 4-space indent on each continuation line.

### Task E3 — Edit `templates/coco_text_qlora.yaml`

- [ ] **Step E3-1: Same edit**

Identical replacement in `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml` lines 34-36.

### Task E4 — Extend `tests/unit/test_cli_init.py`

- [ ] **Step E4-1: Append new tests at end of file**

Append to `tests/unit/test_cli_init.py`:

```python
# ---------------------------------------------------------------------------
# spec/domain-aware-augmentation-presets — --preset / --intensity flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset,intensity",
    [
        ("natural", "safe"), ("natural", "medium"), ("natural", "aggressive"),
        ("medical", "safe"), ("medical", "medium"), ("medical", "aggressive"),
        ("satellite", "safe"), ("satellite", "medium"), ("satellite", "aggressive"),
        ("microscopy", "safe"), ("microscopy", "medium"), ("microscopy", "aggressive"),
        ("none", "medium"),
        ("custom", "medium"),
    ],
)
def test_init_renders_preset_intensity(
    tmp_path: Path, preset: str, intensity: str
) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        [
            "init", "--template", "coco-text-lora",
            "--preset", preset, "--intensity", intensity,
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = load_config(out)
    assert cfg.data.augmentations.preset == preset
    assert cfg.data.augmentations.intensity == intensity


def test_init_custom_writes_uncommented_overrides_scaffold(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--template", "coco-text-lora", "--preset", "custom", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    body = out.read_text()
    assert "overrides: {}" in body  # uncommented form
    assert "# overrides:" not in body  # NOT the commented scaffold
    cfg = load_config(out)
    assert cfg.data.augmentations.preset == "custom"


def test_init_default_preset_writes_commented_overrides_scaffold(tmp_path: Path) -> None:
    """Non-custom presets render the commented `# overrides:` scaffold."""
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--template", "coco-text-lora", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    body = out.read_text()
    assert "# overrides:" in body
    assert "overrides: {}" not in body


def test_init_invalid_preset_rejected(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--preset", "typoo", "--output", str(out)],
    )
    assert result.exit_code != 0
    assert "preset" in result.output.lower()


def test_init_invalid_intensity_rejected(tmp_path: Path) -> None:
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--intensity", "huge", "--output", str(out)],
    )
    assert result.exit_code != 0
    assert "intensity" in result.output.lower()


def test_init_other_fields_parse_identically(tmp_path: Path) -> None:
    """Defaults render preserves non-augmentation fields verbatim from the template."""
    _make_data_paths(tmp_path)
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        ["init", "--template", "coco-text-lora", "--output", str(out)],
    )
    assert result.exit_code == 0
    cfg = load_config(out)
    assert cfg.run.name == "my-run"
    assert cfg.model.name == "facebook/sam3.1"
    assert cfg.train.epochs == 10
```

- [ ] **Step E4-2: Run init tests**

```bash
uv run pytest tests/unit/test_cli_init.py -q
```

Expected: all green. The two existing tests (`test_init_writes_lora_template`, `test_init_writes_qlora_template`) still pass because the default preset/intensity render valid YAML.

### Task E5 — Commit Phase E

- [ ] **Step E5-1: Commit**

```bash
git add src/custom_sam_peft/cli/init_cmd.py \
        src/custom_sam_peft/cli/templates/coco_text_lora.yaml \
        src/custom_sam_peft/cli/templates/coco_text_qlora.yaml \
        tests/unit/test_cli_init.py
git commit -m "feat(cli): csp init --preset/--intensity with overrides scaffold (#75)"
```

---

## Phase G — Mass test-fixture migration (legacy callsites)

**Parallelism:** parallel-safe with Phase B and Phase E (file-disjoint).
**Depends on:** Phase A (the new `AugmentationsConfig(preset=..., intensity=..., overrides=...)` API must already exist).
**Files touched:**
- Modify: `tests/unit/test_trainer_nan_behavior.py`
- Modify: `tests/unit/test_trainer_run_dir.py`
- Modify: `tests/integration/test_train_resume.py`
- Modify: `tests/integration/test_train_end_to_end.py`

**Spec ref:** §4.2, §13.

**Verify:** `uv run pytest tests/unit/test_trainer_nan_behavior.py tests/unit/test_trainer_run_dir.py tests/integration/test_train_resume.py tests/integration/test_train_end_to_end.py -q`

This phase is **mechanical**: replace every `AugmentationsConfig(hflip=False, color_jitter=0.0)` with `AugmentationsConfig(preset="none")`. The list of callsites (8 total) was enumerated at plan-write time via `grep -rn 'AugmentationsConfig(hflip' tests/`.

### Task G1 — Migrate the 8 callsites

- [ ] **Step G1-1: Apply the mechanical edit**

Each of the following lines becomes a one-line change. The string `AugmentationsConfig(hflip=False, color_jitter=0.0)` becomes `AugmentationsConfig(preset="none")` everywhere:

```
tests/unit/test_trainer_nan_behavior.py:47
tests/unit/test_trainer_run_dir.py:180
tests/integration/test_train_resume.py:35
tests/integration/test_train_end_to_end.py:37
tests/integration/test_train_end_to_end.py:168
tests/integration/test_train_end_to_end.py:209
tests/integration/test_train_end_to_end.py:274
```

(Seven callsites — the eighth is in `tests/unit/test_data_transforms.py`, owned by Phase C, and is rewritten there as part of the larger refactor; the two callsites in `tests/unit/test_data_coco.py` and `tests/unit/test_data_hf.py` use **dict fixtures**, not the constructor — they belong to Phase C too.)

The mechanical substitution: in each of the seven files above, find the line containing `AugmentationsConfig(hflip=False, color_jitter=0.0)` and replace with `AugmentationsConfig(preset="none")`. The trailing comma and surrounding context are unchanged.

- [ ] **Step G1-2: Verify no stray legacy callsites remain**

```bash
! grep -rn 'AugmentationsConfig(hflip\|AugmentationsConfig(color_jitter' tests/ src/
```

Expected: exit 0 (grep finds nothing → `!` flips to 0).

- [ ] **Step G1-3: Re-run the affected tests**

```bash
uv run pytest tests/unit/test_trainer_nan_behavior.py tests/unit/test_trainer_run_dir.py tests/integration/test_train_resume.py tests/integration/test_train_end_to_end.py -q
```

Expected: all green. These tests previously suppressed augmentation via `hflip=False, color_jitter=0.0`; under the new schema `preset="none"` produces a structurally-equivalent train pipeline (spec §8.3), so the assertions still hold.

### Task G2 — Commit Phase G

- [ ] **Step G2-1: Commit**

```bash
git add tests/unit/test_trainer_nan_behavior.py \
        tests/unit/test_trainer_run_dir.py \
        tests/integration/test_train_resume.py \
        tests/integration/test_train_end_to_end.py
git commit -m "test: migrate AugmentationsConfig callsites to preset=\"none\" (#75)"
```

---

## Phase C — `build_train_transforms` refactor + dataset call-site dict shape + tests

**Parallelism:** serial (depends on A + B + G).
**Files touched:**
- Modify: `src/custom_sam_peft/data/transforms.py` (refactor `build_train_transforms` only — Phase B's edits already merged)
- Modify: `src/custom_sam_peft/data/coco.py` (no logic change; comment touch-up only if any references legacy field names)
- Modify: `src/custom_sam_peft/data/hf.py` (same)
- Modify: `tests/unit/test_data_transforms.py`
- Modify: `tests/unit/test_data_coco.py`
- Modify: `tests/unit/test_data_hf.py`

**Spec ref:** §8, §8.1, §8.2, §8.3, §12.3, §12.5.

**Verify:** `uv run pytest tests/unit/test_data_transforms.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py -q`

### Task C1 — Refactor `build_train_transforms`

- [ ] **Step C1-1: Replace the body**

In `src/custom_sam_peft/data/transforms.py`, **replace** the existing `build_train_transforms` function (currently lines 161-206) with:

```python
def build_train_transforms(
    aug_cfg: AugmentationsConfig,
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> A.Compose:
    """Train pipeline: resolved presets → ordered Albumentations step list.

    See spec §8 for the canonical step ordering. The Albumentations objects
    appear in the compose iff the corresponding knob is enabled / > 0;
    knob = 0/False omits the step entirely (not p=0).
    """
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    from custom_sam_peft.data.aug_presets import resolve

    resolved = resolve(aug_cfg)
    mean, std = resolve_normalization(model_name, normalize)
    steps: list[object] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            position="top_left",
        ),
    ]
    if resolved.hflip:
        steps.append(A.HorizontalFlip(p=0.5))
    if resolved.vflip:
        steps.append(A.VerticalFlip(p=0.5))
    if resolved.rotate90:
        steps.append(A.RandomRotate90(p=0.5))
    if resolved.rotate_arbitrary > 0.0:
        steps.append(
            A.Affine(
                rotate=(-resolved.rotate_arbitrary, resolved.rotate_arbitrary),
                p=0.5,
                fit_output=False,
                fill=0,
                fill_mask=0,
            )
        )
    if resolved.gauss_noise > 0.0:
        steps.append(
            A.GaussNoise(
                var_limit=(0.0, resolved.gauss_noise * _GAUSS_NOISE_MAX_VAR),
                p=0.5,
            )
        )
    if resolved.blur > 0.0:
        steps.append(
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(0.0, resolved.blur * _GAUSS_BLUR_MAX_SIGMA),
                p=0.5,
            )
        )
    if resolved.color_jitter > 0.0:
        v = resolved.color_jitter
        steps.append(
            A.ColorJitter(
                brightness=v, contrast=v, saturation=v, hue=v * 0.5, p=0.5,
            )
        )
    if resolved.stain_jitter > 0.0:
        steps.append(StainJitter(sigma=resolved.stain_jitter, p=0.5))
    steps.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0))
    steps.append(ToTensorV2())
    return A.Compose(
        steps,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )
```

### Task C2 — Touch up dataset builders (comments only)

- [ ] **Step C2-1: `data/coco.py` and `data/hf.py`**

`src/custom_sam_peft/data/coco.py` and `src/custom_sam_peft/data/hf.py` both validate the augmentation dict via `AugmentationsConfig.model_validate(cfg.get("augmentations", {}))` — the Python is **unchanged**. Confirm by `grep -n 'model_validate' src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py`. If either file's docstring or comment references the legacy fields, update it; otherwise leave the file alone.

Specifically: `src/custom_sam_peft/data/coco.py:294` and `src/custom_sam_peft/data/hf.py:307` already use `AugmentationsConfig.model_validate(cfg.get("augmentations", {}))` — no edit needed beyond confirming the imports still resolve. **No commit required for this step** unless a docstring change is made; if so, batch it into Task C5's commit.

### Task C3 — Rewrite `tests/unit/test_data_transforms.py`

- [ ] **Step C3-1: Delete legacy tests, add parameterized step-list tests**

In `tests/unit/test_data_transforms.py`:

1. **Delete** `test_train_transforms_hflip_disabled` (around line 252) and `test_train_transforms_color_jitter_zero_preserves_color` (around line 271). The test at line 235 (`test_train_transforms_includes_color_jitter_when_enabled`, or similar — confirm by reading lines 230-250 before editing) is also replaced.

2. **Append** the following block at the end of the file:

```python
# ---------------------------------------------------------------------------
# spec/domain-aware-augmentation-presets — pipeline step assembly
# ---------------------------------------------------------------------------


def _class_names(compose: object) -> list[str]:
    """Return the ordered class-name list of an A.Compose's .transforms."""
    return [type(t).__name__ for t in compose.transforms]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "preset,intensity,expected_optional",
    [
        ("natural", "medium", ["HorizontalFlip", "ColorJitter"]),
        ("medical", "medium", ["Affine", "StainJitter", "GaussNoise"]),
        ("medical", "safe", []),  # all-zero → no optional steps
        ("satellite", "aggressive",
         ["HorizontalFlip", "VerticalFlip", "RandomRotate90", "Affine",
          "GaussNoise", "GaussianBlur", "ColorJitter"]),
        ("microscopy", "safe", ["VerticalFlip", "RandomRotate90"]),
    ],
)
def test_pipeline_step_list_per_preset_intensity(
    preset: str, intensity: str, expected_optional: list[str]
) -> None:
    from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig
    from custom_sam_peft.data.transforms import build_train_transforms

    compose = build_train_transforms(
        AugmentationsConfig(preset=preset, intensity=intensity),  # type: ignore[arg-type]
        image_size=32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    names = _class_names(compose)
    # First two and last two steps are constant.
    assert names[:2] == ["LongestMaxSize", "PadIfNeeded"]
    assert names[-2:] == ["Normalize", "ToTensorV2"]
    assert names[2:-2] == expected_optional


def test_pipeline_preset_none_equals_eval_steps() -> None:
    """preset=none → train pipeline contains only the eval steps."""
    from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms

    train = build_train_transforms(
        AugmentationsConfig(preset="none"),
        image_size=32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    eval_ = build_eval_transforms(
        image_size=32,
        model_name="facebook/sam3.1",
        normalize=NormalizeConfig(),
    )
    assert _class_names(train) == _class_names(eval_)


def test_pipeline_custom_with_overrides_step_list() -> None:
    from custom_sam_peft.config.schema import (
        AugmentationOverrides,
        AugmentationsConfig,
        NormalizeConfig,
    )
    from custom_sam_peft.data.transforms import build_train_transforms

    cfg = AugmentationsConfig(
        preset="custom",
        overrides=AugmentationOverrides(hflip=True, stain_jitter=0.05),
    )
    compose = build_train_transforms(
        cfg, image_size=32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    names = _class_names(compose)
    assert names == [
        "LongestMaxSize", "PadIfNeeded",
        "HorizontalFlip", "StainJitter",
        "Normalize", "ToTensorV2",
    ]


def test_pipeline_omits_step_at_zero_magnitude() -> None:
    """A knob at 0 omits the step entirely (not p=0)."""
    import albumentations as A

    from custom_sam_peft.config.schema import (
        AugmentationOverrides,
        AugmentationsConfig,
        NormalizeConfig,
    )
    from custom_sam_peft.data.transforms import build_train_transforms

    cfg = AugmentationsConfig(
        preset="natural",
        intensity="medium",
        overrides=AugmentationOverrides(color_jitter=0.0),
    )
    compose = build_train_transforms(
        cfg, image_size=32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    assert not any(isinstance(t, A.ColorJitter) for t in compose.transforms)  # type: ignore[attr-defined]


def test_pipeline_step_names_match_aug_presets_helper() -> None:
    """The Albumentations compose's class-name list matches _STEP_NAMES_FOR(resolved)."""
    from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig
    from custom_sam_peft.data.aug_presets import _STEP_NAMES_FOR, resolve
    from custom_sam_peft.data.transforms import build_train_transforms

    cfg = AugmentationsConfig(preset="natural", intensity="aggressive")
    compose = build_train_transforms(
        cfg, image_size=32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    assert _class_names(compose) == _STEP_NAMES_FOR(resolve(cfg))
```

### Task C4 — Migrate dict fixtures in `test_data_coco.py` and `test_data_hf.py`

- [ ] **Step C4-1: Edit `tests/unit/test_data_coco.py`**

In `tests/unit/test_data_coco.py`:
- Line 707: `"augmentations": {"hflip": True, "color_jitter": 0.1},` → `"augmentations": {"preset": "natural", "intensity": "medium"},`
- Line 733: `"augmentations": {"hflip": False, "color_jitter": 0.0},` → `"augmentations": {"preset": "none"},`

- [ ] **Step C4-2: Edit `tests/unit/test_data_hf.py`**

In `tests/unit/test_data_hf.py`:
- Line 364: `"augmentations": {"hflip": False, "color_jitter": 0.0},` → `"augmentations": {"preset": "none"},`
- Line 463: `"augmentations": {"hflip": False, "color_jitter": 0.0},` → `"augmentations": {"preset": "none"},`

- [ ] **Step C4-3: Run the data tests**

```bash
uv run pytest tests/unit/test_data_transforms.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py -q
```

Expected: all green. The transforms tests use the new parameterized step-list assertions; the dataset tests use the new dict shape.

### Task C5 — Commit Phase C

- [ ] **Step C5-1: Commit**

```bash
git add src/custom_sam_peft/data/transforms.py \
        tests/unit/test_data_transforms.py \
        tests/unit/test_data_coco.py \
        tests/unit/test_data_hf.py
# Add src/custom_sam_peft/data/coco.py and src/custom_sam_peft/data/hf.py ONLY if
# Task C2 changed a docstring; otherwise omit.
git commit -m "feat(data): build_train_transforms emits preset-resolved step list (#75)"
```

---

## Phase D — Trainer sidecar (`augmentation_pipeline.json`)

**Parallelism:** parallel-safe with Phase C and Phase F (file-disjoint).
**Depends on:** Phase A (imports `dump_augmentation_pipeline`) and Phase G (sidecar tests append to migrated test files).
**Files touched:**
- Modify: `src/custom_sam_peft/train/trainer.py`
- Modify: `tests/unit/test_trainer_run_dir.py` (append sidecar test)
- Modify: `tests/integration/test_train_end_to_end.py` (append sidecar assertion to the existing end-to-end test)

**Spec ref:** §10, §10.1, §10.2, §10.3, §12.6 (sidecar test), §12.7 (e2e sidecar assertion).

**Verify:** `uv run pytest tests/unit/test_trainer_run_dir.py -q` and (slower) `uv run pytest tests/integration/test_train_end_to_end.py -q`.

### Task D1 — Wire `dump_augmentation_pipeline` into `Trainer.fit`

- [ ] **Step D1-1: Edit `trainer.py`**

In `src/custom_sam_peft/train/trainer.py`, find the `config.yaml` write (currently line 166):

```python
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
```

**Insert immediately after that line**:

```python
        from custom_sam_peft.data.aug_presets import dump_augmentation_pipeline

        (run_dir / "augmentation_pipeline.json").write_text(
            json.dumps(
                dump_augmentation_pipeline(cfg.data.augmentations),
                indent=2,
                sort_keys=False,
            )
        )
```

`json` is already imported at the top of `trainer.py` (line 5).

### Task D2 — Sidecar unit test in `test_trainer_run_dir.py`

- [ ] **Step D2-1: Append a new test**

Append to the end of `tests/unit/test_trainer_run_dir.py`:

```python
def test_run_dir_writes_augmentation_pipeline_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After fit constructs run_dir, augmentation_pipeline.json is present and shaped."""
    # Reuse the file's existing trainer-construction helpers — find the closest
    # existing test in this file that calls Trainer(...).fit(run_dir=...) and
    # mirror its setup. The minimal additional assertion is:
    import json as _json

    from custom_sam_peft.config.schema import (
        AugmentationsConfig, DataConfig, DataSplit, PEFTConfig, RunConfig,
        TextPromptConfig, TrainConfig, TrainHyperparams,
    )
    from custom_sam_peft.data.coco import COCODataset
    from custom_sam_peft.data.transforms import build_eval_transforms, build_train_transforms
    from custom_sam_peft.train.trainer import Trainer
    from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper

    tiny_coco_dir = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_coco"
    from custom_sam_peft.config.schema import NormalizeConfig
    transforms_t = build_train_transforms(
        AugmentationsConfig(preset="medical", intensity="medium"),
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig(),
    )
    transforms_v = build_eval_transforms(
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig(),
    )
    ds_train = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text", transforms=transforms_t, text_prompt=TextPromptConfig(),
    )
    ds_val = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text", transforms=transforms_v, text_prompt=TextPromptConfig(),
    )
    wrapper = make_stub_wrapper(dim=8, working=True)
    cfg = TrainConfig(
        run=RunConfig(name="sidecar", output_dir=str(tmp_path), seed=0),
        data=DataConfig(
            format="coco",
            train=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            val=DataSplit(
                annotations=str(tiny_coco_dir / "annotations.json"),
                images=str(tiny_coco_dir / "images"),
            ),
            prompt_mode="text",
            image_size=32,
            augmentations=AugmentationsConfig(preset="medical", intensity="medium"),
        ),
        peft=PEFTConfig(method="lora"),
        train=TrainHyperparams(epochs=1),
    )

    from tests.fixtures.null_tracker import NullTracker  # if available; otherwise use the file's existing tracker stub

    run_dir = tmp_path / "sidecar-run"
    trainer = Trainer(wrapper, ds_train, ds_val, NullTracker(), cfg)
    trainer.fit(run_dir=run_dir)

    sidecar = run_dir / "augmentation_pipeline.json"
    assert sidecar.exists()
    blob = _json.loads(sidecar.read_text())
    assert set(blob.keys()) == {"preset", "intensity", "resolved", "steps", "library_version"}
    assert blob["preset"] == "medical"
    assert blob["intensity"] == "medium"
    assert set(blob["resolved"].keys()) == {
        "hflip", "vflip", "rotate90", "rotate_arbitrary",
        "color_jitter", "stain_jitter", "blur", "gauss_noise",
    }
    assert isinstance(blob["library_version"], str) and blob["library_version"]
```

**Implementer note:** the existing `test_trainer_run_dir.py` already has helpers / fixtures for the tracker stub. Inspect lines 1-100 before this step and reuse whatever stub the existing tests use (`NullTracker`, `_DummyTracker`, etc.) rather than importing a name that may not exist. The shape above is illustrative; copy the existing test's tracker source verbatim.

### Task D3 — End-to-end sidecar assertion

- [ ] **Step D3-1: Append sidecar assertions to the existing e2e test**

In `tests/integration/test_train_end_to_end.py`, in the main e2e test that asserts `result.run_dir.exists()` etc. (around line 102), add the following assertions immediately after the existing `assert (result.run_dir / "adapter" / "adapter_config.json").exists()` line:

```python
    import json as _json

    sidecar = result.run_dir / "augmentation_pipeline.json"
    assert sidecar.exists()
    blob = _json.loads(sidecar.read_text())
    assert blob["preset"] == "none"  # this test uses preset=none (post-Phase-G migration)
    assert blob["steps"][:2] == ["LongestMaxSize", "PadIfNeeded"]
    assert blob["steps"][-2:] == ["Normalize", "ToTensorV2"]
    assert blob["library_version"]
```

(The e2e test was already migrated by Phase G to `preset="none"`; the sidecar therefore records that preset.)

### Task D4 — Run trainer tests

- [ ] **Step D4-1:**

```bash
uv run pytest tests/unit/test_trainer_run_dir.py tests/unit/test_trainer_nan_behavior.py -q
uv run pytest tests/integration/test_train_end_to_end.py tests/integration/test_train_resume.py -q
```

Expected: all green.

### Task D5 — Commit Phase D

- [ ] **Step D5-1: Commit**

```bash
git add src/custom_sam_peft/train/trainer.py \
        tests/unit/test_trainer_run_dir.py \
        tests/integration/test_train_end_to_end.py
git commit -m "feat(train): write augmentation_pipeline.json sidecar next to config.yaml (#75)"
```

---

## Phase F — `csp doctor --config` (resolved augmentations + normalization tables)

**Parallelism:** parallel-safe with Phase C and Phase D (file-disjoint — only touches `doctor_cmd.py` + its test, since `resolve_normalization_with_path` is already in `data/transforms.py` from Phase B).
**Depends on:** Phase A (`resolve`, `_STEP_NAMES_FOR`, `dump_augmentation_pipeline`) and Phase B (`resolve_normalization_with_path`).
**Files touched:**
- Modify: `src/custom_sam_peft/cli/doctor_cmd.py`
- Modify: `tests/unit/test_cli_doctor.py`

**Spec ref:** §11.2, §11.2.1, §11.2.2, §11.2.3, §12.9.

**Verify:** `uv run pytest tests/unit/test_cli_doctor.py -q`

### Task F1 — Add `--config` option to `doctor_cmd.py`

- [ ] **Step F1-1: Edit the function**

In `src/custom_sam_peft/cli/doctor_cmd.py`:

1. Add to imports near the top:

   ```python
   from custom_sam_peft.config.loader import load_config
   from custom_sam_peft.config.schema import NormalizeConfig, TrainConfig
   from custom_sam_peft.data.aug_presets import (
       _STEP_NAMES_FOR,
       dump_augmentation_pipeline,
       resolve,
   )
   from custom_sam_peft.data.transforms import resolve_normalization_with_path
   ```

2. Add a third option to the `doctor(...)` signature:

   ```python
       config: Path | None = typer.Option(
           None,
           "--config",
           help=(
               "Path to a training config YAML. When set, doctor also renders the "
               "resolved augmentations and normalization derived from it."
           ),
       ),
   ```

3. Add helper functions **before** `doctor(...)`:

   ```python
   def _render_resolved_config_tables(cfg: TrainConfig) -> None:
       """Spec §11.2.1 + §11.2.2 — two additional tables when --config is set."""
       console = Console()

       resolved = resolve(cfg.data.augmentations)
       aug = Table(title="Resolved augmentations", show_header=False, box=None)
       aug.add_row("preset", cfg.data.augmentations.preset)
       aug.add_row("intensity", cfg.data.augmentations.intensity)
       aug.add_row("hflip", str(resolved.hflip))
       aug.add_row("vflip", str(resolved.vflip))
       aug.add_row("rotate90", str(resolved.rotate90))
       aug.add_row("rotate_arbitrary", str(resolved.rotate_arbitrary))
       aug.add_row("color_jitter", str(resolved.color_jitter))
       aug.add_row("stain_jitter", str(resolved.stain_jitter))
       aug.add_row("blur", str(resolved.blur))
       aug.add_row("gauss_noise", str(resolved.gauss_noise))
       aug.add_row("steps", ", ".join(_STEP_NAMES_FOR(resolved)))
       console.print(aug)

       mean, std, path = resolve_normalization_with_path(
           cfg.model.name, cfg.data.normalize
       )
       norm = Table(title="Normalization", show_header=False, box=None)
       norm.add_row("model.name", cfg.model.name)
       norm.add_row("mean", str(mean))
       norm.add_row("std", str(std))
       norm.add_row("resolution path", path)
       console.print(norm)


   def _build_resolved_config_json(cfg: TrainConfig) -> dict[str, object]:
       """Spec §11.2.3 — additive `resolved_config` block injected into --json."""
       aug_dump = dump_augmentation_pipeline(cfg.data.augmentations)
       mean, std, path = resolve_normalization_with_path(
           cfg.model.name, cfg.data.normalize
       )
       return {
           "augmentations": {
               "preset": aug_dump["preset"],
               "intensity": aug_dump["intensity"],
               "resolved": aug_dump["resolved"],
               "steps": aug_dump["steps"],
           },
           "normalize": {
               "model_name": cfg.model.name,
               "mean": mean,
               "std": std,
               "resolution_path": path,
           },
       }
   ```

4. In the existing `doctor(...)` body, **after** the existing `_render_table(report)` / `print(json.dumps(...))` branches, add the config-derived branch:

   Replace the existing body (lines 80-85):

   ```python
       report = run_doctor(weights_path=weights_path)
       if json_output:
           print(json.dumps(dataclasses.asdict(report), default=str, indent=2))
       else:
           _render_table(report)
   ```

   with:

   ```python
       report = run_doctor(weights_path=weights_path)
       cfg = load_config(config) if config is not None else None

       if json_output:
           blob = dataclasses.asdict(report)
           if cfg is not None:
               blob["resolved_config"] = _build_resolved_config_json(cfg)
           print(json.dumps(blob, default=str, indent=2))
       else:
           _render_table(report)
           if cfg is not None:
               _render_resolved_config_tables(cfg)
   ```

### Task F2 — Tests in `test_cli_doctor.py`

- [ ] **Step F2-1: Append doctor-with-config tests**

Append to `tests/unit/test_cli_doctor.py`:

```python
# ---------------------------------------------------------------------------
# spec/domain-aware-augmentation-presets — doctor --config
# ---------------------------------------------------------------------------


def _write_doctor_config(tmp_path) -> str:
    """Write a minimal valid TrainConfig YAML for doctor --config tests."""
    import yaml

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text("{}")
    (data_dir / "val.json").write_text("{}")
    (data_dir / "train").mkdir()
    (data_dir / "val").mkdir()
    cfg = {
        "run": {"name": "doctor-cfg"},
        "model": {"name": "facebook/sam3.1"},
        "data": {
            "format": "coco",
            "train": {"annotations": str(data_dir / "train.json"), "images": str(data_dir / "train")},
            "val": {"annotations": str(data_dir / "val.json"), "images": str(data_dir / "val")},
            "prompt_mode": "text",
            "augmentations": {"preset": "medical", "intensity": "medium"},
        },
        "peft": {"method": "lora"},
        "train": {"epochs": 1},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_doctor_with_config_renders_resolved_augmentations(tmp_path) -> None:
    cfg_path = _write_doctor_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", cfg_path])
    assert result.exit_code == 0, result.output
    text = _plain(result.stdout)
    assert "Resolved augmentations" in text
    assert "preset" in text
    assert "medical" in text
    assert "intensity" in text


def test_doctor_with_config_renders_normalization(tmp_path) -> None:
    cfg_path = _write_doctor_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", cfg_path])
    assert result.exit_code == 0
    text = _plain(result.stdout)
    assert "Normalization" in text
    assert "mean" in text
    assert "std" in text
    assert "resolution path" in text


def test_doctor_json_no_config_no_resolved_block() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "resolved_config" not in blob


def test_doctor_json_with_config_has_resolved_block(tmp_path) -> None:
    cfg_path = _write_doctor_config(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", cfg_path, "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    assert "resolved_config" in blob
    rc = blob["resolved_config"]
    assert set(rc.keys()) == {"augmentations", "normalize"}
    assert rc["augmentations"]["preset"] == "medical"
    assert rc["augmentations"]["intensity"] == "medium"
    assert set(rc["augmentations"]["resolved"].keys()) == {
        "hflip", "vflip", "rotate90", "rotate_arbitrary",
        "color_jitter", "stain_jitter", "blur", "gauss_noise",
    }
    assert isinstance(rc["augmentations"]["steps"], list)
    assert rc["normalize"]["model_name"] == "facebook/sam3.1"
    assert rc["normalize"]["resolution_path"] in {"processor", "table-fallback", "config-fallback"}


def test_doctor_no_config_byte_identical_to_pre_config_behavior() -> None:
    """Without --config, JSON output has the same top-level keys as the legacy run."""
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    blob = json.loads(_plain(result.stdout))
    # Top-level keys are the DoctorReport dataclass fields; resolved_config is absent.
    assert "torch_version" in blob
    assert "hf_auth" in blob
    assert "resolved_config" not in blob
```

- [ ] **Step F2-2: Run doctor tests**

```bash
uv run pytest tests/unit/test_cli_doctor.py -q
```

Expected: all green. The existing 8 tests still pass (the `--config` branch is purely additive).

### Task F3 — Commit Phase F

- [ ] **Step F3-1: Commit**

```bash
git add src/custom_sam_peft/cli/doctor_cmd.py tests/unit/test_cli_doctor.py
git commit -m "feat(cli): csp doctor --config renders resolved aug + normalization tables (#75)"
```

---

## Phase H — Final reviewer pass + lint/format

**Parallelism:** Reviewers H1 + H2 dispatch in parallel; H3 runs serially after both return.
**Depends on:** all of A–G complete.

### Task H1 — Design-sensitive review (opus / xhigh)

**Focus files:**
- `src/custom_sam_peft/data/aug_presets.py` — resolver semantics, locked-off WARN behavior, sidecar dict shape (cross-check against spec §10's literal example for `medical/medium`).
- `src/custom_sam_peft/data/transforms.py` — `StainJitter.apply` math (Ruifrok-Johnston HED basis matrix orientation, OD ↔ RGB roundtrip, identity at σ=0); `build_train_transforms` step order vs spec §8.
- `src/custom_sam_peft/train/trainer.py` — sidecar write atomicity (no half-written file on a `dump_augmentation_pipeline` exception).

**Areas to interrogate:**
- Does `_is_enabled(False)` correctly return `False`? (Python `bool` is `int`; the float branch must not fire on bool.)
- Does the WARN message format exactly match spec §6.1?
- Does `_STEP_NAMES_FOR` exactly match the `build_train_transforms` step assembly? Any reordering, missing knob, or missing constant step would silently corrupt the sidecar.
- Is `_HED_FROM_RGB_INV` actually `np.linalg.inv(_HED_FROM_RGB_MATRIX)` (not transpose / pseudo-inverse / a hand-typed value)? The Tellez paper uses the inverse.

**Action:** dispatch a single opus/xhigh reviewer subagent with this section as its brief.

### Task H2 — General code review (sonnet / high)

**Focus files:**
- `src/custom_sam_peft/cli/init_cmd.py` — `string.Template.substitute` indentation, `--preset custom` rendering, Typer error messages.
- `src/custom_sam_peft/cli/doctor_cmd.py` — Rich table column ordering, JSON-vs-table parity (the `resolved_config` block in JSON mirrors the two tables).
- All test files for: parametrization coverage, log-capture logger name correctness, no flaky np.random seeding gaps.

**Action:** dispatch a sonnet/high reviewer subagent with this section as its brief. May dispatch in parallel with H1.

### Task H3 — Lint / format / type / full-test gate

This is the canonical CI mirror. Run **after** H1 + H2 return.

- [ ] **Step H3-1: Auto-fix lint**

```bash
uv run ruff check . --fix
```

- [ ] **Step H3-2: Apply format**

```bash
uv run ruff format .
```

- [ ] **Step H3-3: Confirm format is stable**

```bash
uv run ruff format --check .
```

Expected: exit 0.

- [ ] **Step H3-4: Type check**

```bash
uv run mypy src/custom_sam_peft tests
```

Expected: exit 0. If `StainJitter`'s `__new__` MRO trick fails mypy, switch to a plain `class StainJitter(albumentations.ImageOnlyTransform):` declaration (the spec allows either form — see Phase B note).

- [ ] **Step H3-5: Full unit + integration test run**

```bash
uv run pytest tests/unit tests/integration -q
```

Expected: green. Total new tests: roughly 30 (resolver) + 7 (StainJitter) + 5 (transforms refactor) + 7 (config_schema additions) + 6 (cli_init) + 5 (cli_doctor) + 1 (trainer_run_dir sidecar) + 1 (e2e sidecar assertion) ≈ 62 new test cells. No tests should be skipped.

- [ ] **Step H3-6: Commit any lint/format fixups (if non-empty)**

```bash
if ! git diff --quiet; then
  git add -u
  git commit -m "chore: ruff fixups for #75"
fi
```

---

## Final reviewer agenda

When the orchestrator dispatches the design-sensitive reviewer for Task H1, give it these explicit files and questions:

1. **`src/custom_sam_peft/data/aug_presets.py`**:
   - Does `PRESET_TABLE` match spec §5 row-for-row for all 12 cells? (Don't trust the implementer — diff the table in the file against the spec table by hand.)
   - Does `LOCKED_OFF` match spec §6 verbatim, including the reason strings?
   - Does `_STEP_NAMES_FOR` produce the same list `build_train_transforms` would emit, in the same order, for at least these three cells: `(natural, aggressive)`, `(medical, medium)`, `(microscopy, safe)`?
   - Does the WARN log call use `_LOG.warning("... %s ...", ...)` (lazy formatting) and emit at most one record per `resolve()` call per overridden knob?

2. **`src/custom_sam_peft/data/transforms.py`**:
   - Is the HED basis matrix numerically correct (Ruifrok-Johnston 2001 values)?
   - Does the σ=0 short-circuit fire **before** the np.random calls, so a σ=0 call is genuinely deterministic with no RNG advancement?
   - Does `build_train_transforms` emit exactly the 12-step order in spec §8 (constants + 8 optional + 2 constants)?
   - Does `_GAUSS_NOISE_MAX_VAR = 0.05` and `_GAUSS_BLUR_MAX_SIGMA = 3.0` match spec §8.1?
   - Does `resolve_normalization_with_path` preserve every log call and every fall-through path the original `resolve_normalization` had?

3. **`src/custom_sam_peft/train/trainer.py`**:
   - Is `dump_augmentation_pipeline` called on `cfg.data.augmentations` (the validated pydantic instance), not on a raw dict?
   - Does the sidecar write happen on the same path inside `fit()` that the existing `config.yaml` write is on, so a single failure mode (run_dir nonexistent) catches both?

When the orchestrator dispatches the general code reviewer for Task H2, give it the CLI files and the test files as the focus, and ask it to verify:

- `csp init --preset custom` produces YAML that round-trips through `load_config` (already tested, but reviewer eyeballs the rendered indent).
- `csp doctor` (no `--config`) JSON output has the same top-level shape as before the PR.
- Every new test uses `caplog.set_level(...)` with the correct logger name; no test calls `caplog.set_level` with the bare root logger.

---

## Post-merge close-out (orchestrator only — not implementer)

Per CLAUDE.md §5: tag + log-fold + worktree-remove + sign-off. This phase **is not part of the plan**; it's the orchestrator's responsibility after the user merges the PR.
