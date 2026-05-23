# spec/domain-aware-augmentation-presets — Domain-aware augmentation presets with intensity tiers (issue #75)

**Status:** Draft (2026-05-22)
**Tracking:** [#75](https://github.com/NguyenJus/custom-sam-peft/issues/75) — *feat(data): domain-aware augmentation presets (natural / medical / satellite / …) with safe|medium|aggressive intensity*
**Scope:** Replace the two-knob `AugmentationsConfig` (`hflip`, `color_jitter`) with a `(preset, intensity, overrides)` triple that resolves to eight typed knobs, wire those knobs into a refactored `build_train_transforms` step-assembly, add a CPU-only custom `StainJitter` Albumentations transform for H&E histopathology, expose `--preset` / `--intensity` to `csp init` and `--config` to `csp doctor`, and persist a `run_dir/augmentation_pipeline.json` sidecar so a finished run records exactly what was applied. Clean schema break (no back-compat aliases; pre-1.0 per #70).

**Builds on:**
[`2026-05-21-yaml-config-defaults-audit-design.md`](2026-05-21-yaml-config-defaults-audit-design.md) — that audit closed the normalization correctness hole (#69) and ratified the schema-as-source-of-truth contract. This spec keeps `resolve_normalization` and the `KNOWN_PROCESSOR_STATS` table untouched; the new augmentation surface is orthogonal.

---

## 1. Goals

- Replace the legacy two-field `AugmentationsConfig` with a higher-level `(preset, intensity, overrides)` API so users pick a domain ("medical", "satellite", "microscopy", "natural") and an intensity tier ("safe", "medium", "aggressive") instead of hand-tuning eight knobs.
- Define eight magnitude-valued augmentation knobs (`hflip`, `vflip`, `rotate90`, `rotate_arbitrary`, `color_jitter`, `stain_jitter`, `blur`, `gauss_noise`) with a frozen lookup table for each `(preset, intensity)` combination.
- Make domain safety explicit: knobs that are semantically wrong for a domain (e.g. `hflip` for medical chest X-rays — laterality matters) are *locked off* — users who enable them via `overrides` get a `logging.WARNING` naming the knob, the preset, and the reason. **Locked-off overrides are NOT stripped** — the user's explicit override always wins; the warning is the contract.
- Ship a custom `StainJitter` Albumentations transform implementing HED-space color deconvolution (Ruifrok & Johnston 2001 / Tellez et al. 2018) so the `medical` preset has a meaningful H&E-aware knob.
- Persist the resolved pipeline to `run_dir/augmentation_pipeline.json` next to `config.yaml` so each run is fully reproducible from the surface fields, and cross-version reproducible by copying `resolved` into `overrides` under `preset: custom`.
- Expose presets in the CLI: `csp init --preset X --intensity Y` renders the chosen pair into the starter template; `csp doctor --config path.yaml` shows a "Resolved augmentations" table and a "Normalization" table so users can dry-run their config without launching training.
- 100% CPU-testable. No real model load, no GPU.

## 2. Non-goals

- **Sub-presets** (`medical_xray`, `medical_histopathology`, `microscopy_fluorescence`) — single-level taxonomy in v1. Filed as a follow-up if requested.
- **Raw Albumentations escape hatch** (`albumentations_pipeline: [list of dicts]`) — out of scope; `preset: custom` + `overrides:` is the configured-knob escape hatch.
- **3D / volumetric augmentations** — RGB-2D only.
- **Loss / model changes for non-RGB modalities** — augmentation surface only; model is unchanged.
- **Domain-specific normalization stats** — covered by #69 (closed); this PR does not touch `resolve_normalization` or `KNOWN_PROCESSOR_STATS`.
- **Per-knob application-probability overrides** — every step that runs is `p=0.5`. Magnitude-only API for v1.
- **Augmenting eval transforms** — `build_eval_transforms` is unchanged; `csp predict` (issue #74) is unaffected.

## 3. Current state

The legacy surface is two booleans-ish fields:

```python
# src/custom_sam_peft/config/schema.py
class AugmentationsConfig(_Strict):
    hflip: bool = True
    color_jitter: float = Field(default=0.1, ge=0.0, le=1.0)
```

`build_train_transforms` in `src/custom_sam_peft/data/transforms.py` consumes these directly: optional `A.HorizontalFlip(p=0.5)` if `hflip`, always-on `A.ColorJitter(brightness/contrast/saturation=v, hue=v*0.5, p=0.5)` even at `color_jitter=0`. The call sites that construct `AugmentationsConfig` from YAML dicts via pydantic's `model_validate` live at:

- `src/custom_sam_peft/data/coco.py` (COCO dataset builder)
- `src/custom_sam_peft/data/hf.py` (HF datasets builder)

The two shipped starter templates `src/custom_sam_peft/cli/templates/coco_text_lora.yaml` and `coco_text_qlora.yaml` carry `data.augmentations: {hflip: true, color_jitter: 0.1}` verbatim. Multiple unit + integration tests construct `AugmentationsConfig(hflip=..., color_jitter=...)` directly.

The trainer (`src/custom_sam_peft/train/trainer.py`) already dumps the validated `cfg.model_dump(mode="json")` to `run_dir/config.yaml`. There is no per-run augmentation provenance file today.

`csp init` (`src/custom_sam_peft/cli/init_cmd.py`) is purely file-copy + optional weights-download; it does not perform template substitution. `csp doctor` (`src/custom_sam_peft/cli/doctor_cmd.py`) is environment-only; it does not accept a config path.

## 4. Schema

New definitions in `src/custom_sam_peft/config/schema.py`. The legacy `AugmentationsConfig` is **deleted** (not aliased, not deprecated) — clean break under #70's pre-1.0 schema-break allowance.

```python
Preset    = Literal["natural", "medical", "satellite", "microscopy", "none", "custom"]
Intensity = Literal["safe", "medium", "aggressive"]


class AugmentationOverrides(_Strict):
    """Per-knob overrides. All None → inherit from (preset, intensity).

    Setting any field to a non-None value replaces just that field in the
    resolved table. Extra keys are rejected (extra="forbid"); typos surface
    at config-load time.
    """

    hflip:            bool  | None = None
    vflip:            bool  | None = None
    rotate90:         bool  | None = None
    rotate_arbitrary: float | None = None   # max degrees θ; samples uniform [-θ, θ]; 0 → step omitted
    color_jitter:     float | None = None   # magnitude in [0, 1]; 0 → step omitted
    stain_jitter:     float | None = None   # HED-space σ; 0 → step omitted
    blur:             float | None = None   # magnitude in [0, 1]; 0 → step omitted
    gauss_noise:      float | None = None   # magnitude in [0, 1]; 0 → step omitted


class AugmentationsConfig(_Strict):
    preset:    Preset    = "natural"
    intensity: Intensity = "medium"
    overrides: AugmentationOverrides = Field(default_factory=AugmentationOverrides)
```

`_Strict` (`extra="forbid"`) is the existing base in `schema.py`; both new models inherit it. The `DataConfig` field stays at `augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)` — exactly one field name change for downstream consumers (none, since it's the same field).

### 4.1 Knob semantics

Each continuous knob is a **magnitude**, not a probability. The application probability is hard-coded at `p=0.5` for every randomized step. A knob value of `0.0` causes the corresponding Albumentations step to be **omitted from the pipeline entirely** — not just `p=0`. Booleans (`hflip`, `vflip`, `rotate90`) include their step at `p=0.5` when `True`, omit it when `False`.

This means an all-zero / all-false resolved configuration produces a pipeline structurally equivalent to `build_eval_transforms` — just `LongestMaxSize → PadIfNeeded → Normalize → ToTensorV2`. That is the expected behavior for `preset: none` and for tight presets like `medical/safe`.

### 4.2 Migration (clean break)

There is no back-compat alias. Every callsite is migrated in the same PR:

- Two templates: `cli/templates/coco_text_lora.yaml`, `cli/templates/coco_text_qlora.yaml` (lines carrying `hflip:` / `color_jitter:` are replaced with `preset: ${preset}\n  intensity: ${intensity}`).
- Two dataset builders: `data/coco.py`, `data/hf.py` (fixture-dict shape changes from `{"hflip": ..., "color_jitter": ...}` to `{"preset": ..., "intensity": ..., "overrides": {...}}` — these files use `model_validate`, so the change is in the YAML/dict shape not the Python).
- Six unit-test files: `test_data_transforms.py`, `test_config_schema.py`, `test_data_coco.py`, `test_data_hf.py`, `test_trainer_nan_behavior.py`, `test_trainer_run_dir.py`.
- Two integration-test files: `test_train_resume.py`, `test_train_end_to_end.py`.

Tests that previously passed `AugmentationsConfig(hflip=False, color_jitter=0.0)` (to suppress randomness) migrate to `AugmentationsConfig(preset="none")`.

## 5. Preset × intensity table

Single frozen source of truth, lives as `PRESET_TABLE` in the new module `src/custom_sam_peft/data/aug_presets.py`. Twelve `(preset, intensity)` cells for the four real domains; `none` and `custom` are handled by short-circuit (§7), not by table lookup.

| knob | nat/safe | nat/med | nat/agg | med/safe | med/med | med/agg | sat/safe | sat/med | sat/agg | mic/safe | mic/med | mic/agg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hflip`            | on   | on   | on   | off  | off  | off  | on   | on   | on   | off  | off  | off  |
| `vflip`            | off  | off  | on   | off  | off  | off  | on   | on   | on   | on   | on   | on   |
| `rotate90`         | off  | off  | off  | off  | off  | off  | on   | on   | on   | on   | on   | on   |
| `rotate_arbitrary` | 0    | 0    | 10   | 0    | 5     | 10   | 0    | 0    | 15   | 0    | 0    | 15   |
| `color_jitter`     | 0.05 | 0.1  | 0.2  | 0    | 0    | 0    | 0    | 0.05 | 0.1  | 0    | 0    | 0    |
| `stain_jitter`     | 0    | 0    | 0    | 0    | 0.03  | 0.07 | 0    | 0    | 0    | 0    | 0    | 0    |
| `blur`             | 0    | 0    | 0.05 | 0    | 0    | 0.03 | 0    | 0    | 0.05 | 0    | 0    | 0.05 |
| `gauss_noise`      | 0    | 0    | 0.02 | 0    | 0.01  | 0.03 | 0    | 0    | 0.02 | 0    | 0    | 0.02 |

`preset: none` → all knobs zero/off, **every intensity** (intensity is ignored, no warn).
`preset: custom` → all knobs zero/off as the seed, intensity is ignored, overrides apply on top, locked-off checks are skipped (user is explicit; no warns).

### 5.1 Rationale for the table values

- **natural**: hflip universally on (typical photographic content); vflip only at aggressive (upside-down photos are unusual but tolerated under heavy augmentation); arbitrary rotation grows linearly with intensity; color/blur/noise also grow.
- **medical**: no flips, no 90° rotation — laterality matters across modalities (chest X-ray L vs R, mammography, derm); small arbitrary rotation (acquisition jitter); no `color_jitter` (semantically meaningful); `stain_jitter` only — the v1 medical preset implicitly targets H&E histopathology since that's the modality the available HED-space knob serves.
- **satellite**: all four flip/rotation symmetries on (no canonical "up"); arbitrary rotation at aggressive only; mild color_jitter; no stain_jitter (not H&E).
- **microscopy**: vflip + rotate90 (no canonical orientation); no hflip (some channel conventions are L→R order); no color_jitter (fluorescence channel identity is meaningful); blur + noise only at aggressive.

## 6. Locked-off rules and warn behavior

For each preset, a fixed set of knobs is **locked off** — they are 0/off at *every* intensity tier in the table, and if a user explicitly enables them via `overrides`, the resolver emits a `logging.WARNING`. The override is **NOT stripped**: per the issue's explicit requirement ("Do not silently strip user overrides"), the user's value wins; the warn is the entire contract.

Module-level constant in `data/aug_presets.py`:

```python
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
```

`preset: none` and `preset: custom` are **not** keyed in `LOCKED_OFF` — they bypass the check, by design (none = no augmentation period; custom = user is fully explicit).

### 6.1 Warning message format

Exactly one `logging.WARNING` per locked-off override, per `resolve()` call. Format:

> `You enabled <knob>=<value> under preset=<preset>; <reason>. The override will be applied as-is.`

Examples:
- `You enabled hflip=True under preset=medical; laterality (left vs right) is clinically meaningful in most medical modalities (CXR, mammography, derm). The override will be applied as-is.`
- `You enabled color_jitter=0.15 under preset=microscopy; color identifies fluorescence channels and must be preserved. The override will be applied as-is.`

Logger: `logging.getLogger("custom_sam_peft.data.aug_presets")` (module-level via `_LOG = logging.getLogger(__name__)`).

Dedup is automatic because `resolve()` is called exactly once per `build_train_transforms` call (which is called once per training run). The warns are reproducible across runs given the same config — no inadvertent multi-emission.

## 7. Resolution algorithm

Pseudocode (the real implementation is in `data/aug_presets.py::resolve`):

```python
@dataclass(frozen=True)
class ResolvedAugmentations:
    hflip: bool
    vflip: bool
    rotate90: bool
    rotate_arbitrary: float
    color_jitter: float
    stain_jitter: float
    blur: float
    gauss_noise: float


def resolve(cfg: AugmentationsConfig) -> ResolvedAugmentations:
    # 1. Seed from the preset table.
    if cfg.preset in ("none", "custom"):
        base = _ZERO_BASE                       # all knobs 0/False
    else:
        base = dict(PRESET_TABLE[(cfg.preset, cfg.intensity)])

    # 2. Apply overrides; warn if a locked-off knob is enabled.
    for field, override in cfg.overrides.model_dump().items():
        if override is None:
            continue
        base[field] = override
        if cfg.preset in ("none", "custom"):
            continue                            # explicit-user paths skip locked-off warns
        if field in LOCKED_OFF.get(cfg.preset, {}) and _is_enabled(override):
            reason = LOCKED_OFF[cfg.preset][field]
            _LOG.warning(
                "You enabled %s=%s under preset=%s; %s. The override will be applied as-is.",
                field, override, cfg.preset, reason,
            )

    # 3. Build the immutable resolved view.
    return ResolvedAugmentations(**base)
```

`_is_enabled(v)` is:
- `True` for booleans equal to `True`
- `True` for floats `> 0.0`
- `False` otherwise (so an override that sets a knob to `False` / `0.0` does NOT trigger a warn — turning a locked-off knob off-er is always fine).

`_ZERO_BASE` is the 8-field dict with `False` for the three boolean knobs and `0.0` for the five float knobs.

Module placement: `src/custom_sam_peft/data/aug_presets.py`. The module owns:

- `PRESET_TABLE: dict[tuple[Preset, Intensity], dict[str, bool | float]]` — frozen.
- `LOCKED_OFF: dict[str, dict[str, str]]` — frozen.
- `ResolvedAugmentations` dataclass (frozen, hashable).
- `resolve(cfg) -> ResolvedAugmentations`.
- `dump_augmentation_pipeline(cfg) -> dict` — see §10.
- A private `_STEP_NAMES_FOR(resolved)` helper returning the ordered list of Albumentations class names that would be emitted by §8, used by `dump_augmentation_pipeline` and by `csp doctor`.

## 8. Pipeline step assembly

`build_train_transforms` in `data/transforms.py` is refactored to call `aug_presets.resolve(aug_cfg)` once, then assemble Albumentations steps in the order below. `build_eval_transforms` is **not** modified.

```
1.  A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR)
2.  A.PadIfNeeded(min_height=image_size, min_width=image_size,
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
                  position="top_left")
3.  A.HorizontalFlip(p=0.5)                       # iff resolved.hflip
4.  A.VerticalFlip(p=0.5)                         # iff resolved.vflip
5.  A.RandomRotate90(p=0.5)                       # iff resolved.rotate90
6.  A.Affine(rotate=(-θ, θ), p=0.5,               # iff resolved.rotate_arbitrary > 0
             fit_output=False, fill=0, fill_mask=0)
        where θ = resolved.rotate_arbitrary
7.  A.GaussNoise(var_limit=(0.0, v * MAX_VAR),    # iff resolved.gauss_noise > 0
                 p=0.5)
        where v = resolved.gauss_noise, MAX_VAR = 0.05
8.  A.GaussianBlur(blur_limit=(3, 7),             # iff resolved.blur > 0
                   sigma_limit=(0.0, v * MAX_SIGMA),
                   p=0.5)
        where v = resolved.blur, MAX_SIGMA = 3.0
9.  A.ColorJitter(brightness=v, contrast=v,       # iff resolved.color_jitter > 0
                  saturation=v, hue=v * 0.5,
                  p=0.5)
        where v = resolved.color_jitter
10. StainJitter(sigma=resolved.stain_jitter,      # iff resolved.stain_jitter > 0
                p=0.5)
11. A.Normalize(mean=mean, std=std,
                max_pixel_value=255.0)
12. ToTensorV2()
```

### 8.1 Constants

`MAX_VAR = 0.05` and `MAX_SIGMA = 3.0` are module-level constants in `data/transforms.py`, named `_GAUSS_NOISE_MAX_VAR` and `_GAUSS_BLUR_MAX_SIGMA`. They define how the unit-interval magnitude `v` projects onto the Albumentations parameter range — `MAX_VAR = 0.05` corresponds to a max variance of 5% of the [0, 1]-normalized pixel range (after `A.Normalize`, the input is float in roughly the unit range, so this is a meaningful absolute scale); `MAX_SIGMA = 3.0` matches Albumentations' default `sigma_limit` upper bound.

### 8.2 Order justification

- **Geometry before pixel ops** (steps 3-6 before 7-10): flips and rotations resample pixels, so any subsequent pixel-domain perturbation observes the geometrically perturbed image.
- **Noise / blur before color / stain** (steps 7-8 before 9-10): sensor-style noise/blur is conceptually pre-color-pipeline. Color and stain operate on the noisy-blurry pixels — closer to what the downstream backbone will see.
- **Color before stain**: `ColorJitter` is a generic photometric perturbation; `StainJitter` is HED-deconvolution-based and assumes well-formed H&E-like pixels. Doing color first injects the perturbation; stain refines it.
- **Normalize then ToTensor last**: matches `build_eval_transforms`.

### 8.3 Empty-pipeline equivalence

If `resolve()` returns the all-zero/all-false state (e.g. `preset: none`, or `medical/safe`), the assembled train pipeline contains exactly steps 1, 2, 11, 12 — structurally equivalent to `build_eval_transforms` (which has bbox_params identical to the train pipeline). This is the test for "no augmentation".

## 9. `StainJitter` custom Albumentations transform

Lives in `src/custom_sam_peft/data/transforms.py`, **not** in `aug_presets.py` — it is an Albumentations transform, co-located with the other Albumentations code. `aug_presets.py` stays pure-Python (numpy at most) so the resolver can be imported without dragging Albumentations.

### 9.1 Class signature

```python
class StainJitter(albumentations.ImageOnlyTransform):
    """HED-space stain jitter for H&E histopathology images.

    Image-only transform — masks, bboxes, and keypoints pass through unchanged.
    Implements the Tellez et al. (2018) / Ruifrok & Johnston (2001)
    color deconvolution: RGB -> optical density -> HED basis -> per-channel
    affine perturbation -> back to RGB.

    Identity at sigma=0 (within float roundtrip tolerance).
    """

    def __init__(self, sigma: float = 0.0, p: float = 0.5):
        super().__init__(p=p)
        if sigma < 0:
            raise ValueError(f"StainJitter sigma must be >= 0, got {sigma}")
        self.sigma = float(sigma)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        ...

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("sigma",)
```

### 9.2 Algorithm

Module-level constants in `transforms.py`:

```python
# Ruifrok & Johnston 2001 HED basis vectors (rows = stains: H, E, DAB).
_HED_FROM_RGB_MATRIX: np.ndarray = np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78],
], dtype=np.float32)
_HED_FROM_RGB_INV: np.ndarray = np.linalg.inv(_HED_FROM_RGB_MATRIX).astype(np.float32)
```

```python
def apply(self, img: np.ndarray, **params) -> np.ndarray:
    # img: uint8 RGB, HWC.
    if self.sigma == 0.0:
        return img
    od = -np.log10((img.astype(np.float32) + 1.0) / 256.0)         # optical density
    hed = od @ _HED_FROM_RGB_INV
    alpha = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
    beta  = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
    hed = hed * (1.0 + alpha) + beta
    od_back = hed @ _HED_FROM_RGB_MATRIX
    out = 256.0 * np.power(10.0, -od_back) - 1.0
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
```

### 9.3 Properties

- **Image-only**: subclasses `albumentations.ImageOnlyTransform`. Masks, bboxes, keypoints pass through unchanged — Albumentations' Compose handles routing.
- **Identity at `sigma=0`**: when σ=0, `alpha` and `beta` are zero vectors → `hed * (1.0 + 0) + 0 == hed` → roundtrip through the HED basis. Float roundtrip noise + the +1/-1 OD shift means the test tolerance is ±1 LSB on uint8, not exact equality. The implementation short-circuits at `sigma=0` to make this exact, but the test still uses ±1 to guard against future refactors.
- **Dtype preserved**: input uint8 → output uint8.
- **Shape preserved**: output `.shape == input.shape`.
- **Output range**: clipped to `[0, 255]` after the back-transform.
- **Determinism**: uses `np.random.uniform`, so seeding `numpy.random.seed(...)` (or `numpy.random.default_rng(seed)`-backed flow) reproduces outputs. Albumentations 1.4+ routes RNG via its own seed; since we use the bare `np.random` module here, the test fixes `np.random.seed` directly. This matches the rest of the codebase, which does not yet rely on Albumentations' newer per-transform RNG.

## 10. Run-metadata sidecar

`src/custom_sam_peft/train/trainer.py` already writes `run_dir/config.yaml` from the validated config. After that write, it now also writes `run_dir/augmentation_pipeline.json`:

```json
{
  "preset": "medical",
  "intensity": "medium",
  "resolved": {
    "hflip": false,
    "vflip": false,
    "rotate90": false,
    "rotate_arbitrary": 5.0,
    "color_jitter": 0.0,
    "stain_jitter": 0.03,
    "blur": 0.0,
    "gauss_noise": 0.01
  },
  "steps": [
    "LongestMaxSize",
    "PadIfNeeded",
    "Affine",
    "GaussNoise",
    "StainJitter",
    "Normalize",
    "ToTensorV2"
  ],
  "library_version": "0.6.0"
}
```

The example above is the literal expected output for `preset: medical, intensity: medium` with no overrides — and it is internally consistent with §5 (medical/medium row: `vflip=false, rotate90=false, rotate_arbitrary=5, color_jitter=0, stain_jitter=0.03, blur=0, gauss_noise=0.01`) and with §8's step-assembly order (`Affine` from `rotate_arbitrary=5`, `GaussNoise` from `gauss_noise=0.01`, `StainJitter` from `stain_jitter=0.03`; no `ColorJitter` because color_jitter=0; no flips/rotate90).

### 10.1 Helper

`dump_augmentation_pipeline(cfg: AugmentationsConfig) -> dict` lives in `data/aug_presets.py`. It calls `resolve(cfg)`, derives the step-name list from the resolved knobs using the same conditional logic as §8 (factored into a `_STEP_NAMES_FOR(resolved)` helper so the trainer and `csp doctor` see exactly the same ordered list), and assembles the dict above. `library_version` is sourced from `custom_sam_peft.__version__`.

### 10.2 Trainer wire-up

`trainer.py` imports `from custom_sam_peft.data.aug_presets import dump_augmentation_pipeline`, calls it on `cfg.data.augmentations`, and writes the result with `json.dump(..., indent=2, sort_keys=False)` (preserving the key order in §10's example) to `run_dir / "augmentation_pipeline.json"`. This happens immediately after the existing `run_dir/config.yaml` write so a failure in either does not leave a partial run dir.

### 10.3 Cross-version reproducibility escape

Documented in the spec and as a docstring on `dump_augmentation_pipeline`: the `resolved` dict is the surface that pins behavior across library versions. A user who needs strict reproducibility against a future preset-table change can copy `resolved` verbatim into `overrides:` under `preset: custom` (intensity ignored) and the resolver will return the same 8 values — even if the table for, say, `medical/medium` shifts. Same-library-version reproducibility is automatic: `(preset, intensity, overrides)` is a pure function of the resolved view.

## 11. CLI changes

### 11.1 `csp init --preset --intensity`

Two new Typer options on `cli/init_cmd.py::init`:

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

Validation: `init_cmd` checks both values against the `Preset` / `Intensity` literal tuples (imported from `config/schema.py`) and raises `typer.BadParameter` on mismatch — matching the existing `--template` validation pattern. Typer 0.9 does not auto-generate Choice from `Literal`, so explicit validation is needed.

Template substitution: the template `.read_text()` body is wrapped in `string.Template(...).substitute(preset=preset, intensity=intensity)` before `output.write_text(body)`. Templates carry `${preset}` and `${intensity}` placeholders inside the `augmentations:` block. Any literal `$` in the templates must be escaped as `$$` — the audit confirms no existing literal `$` in either template, so this is forward-only protection.

Rendered template snippet (replacing the current lines 34-36):

```yaml
  augmentations:
    preset: ${preset}
    intensity: ${intensity}
    # Override individual knobs here; unset keys inherit from (preset, intensity).
    # overrides:
    #   hflip: false
    #   color_jitter: 0.15
```

#### 11.1.1 `--preset custom` branch

When `--preset custom` is passed, the rendered template also includes a non-commented empty overrides scaffold so the user has a clear hook to fill. The substitution machinery achieves this by rendering a `${overrides_block}` placeholder, with two values:

- For `preset != "custom"`: the commented scaffold above (the `# overrides:` block).
- For `preset == "custom"`: an uncommented `overrides: {}` with a single inline comment pointing at the 8 knob names.

Concretely, `init_cmd` constructs the substitution dict:

```python
if preset == "custom":
    overrides_block = (
        "overrides: {}  # fill in knobs: hflip, vflip, rotate90, rotate_arbitrary, "
        "color_jitter, stain_jitter, blur, gauss_noise"
    )
else:
    overrides_block = (
        "# Override individual knobs here; unset keys inherit from (preset, intensity).\n"
        "    # overrides:\n"
        "    #   hflip: false\n"
        "    #   color_jitter: 0.15"
    )
```

The template uses `${overrides_block}` indented under `augmentations:`:

```yaml
  augmentations:
    preset: ${preset}
    intensity: ${intensity}
    ${overrides_block}
```

Indentation note: because `string.Template.substitute` does not re-indent multi-line values, the `overrides_block` string above is constructed with the correct leading spaces on each continuation line (matching the 4-space indent of the `augmentations:` block). The implementation is one `textwrap.indent` call or a hand-formatted constant — the planner picks; both work.

### 11.2 `csp doctor --config`

One new option on `cli/doctor_cmd.py::doctor`:

```python
config: Path | None = typer.Option(
    None, "--config",
    help="Path to a training config YAML. When set, doctor also "
         "renders the resolved augmentations and normalization derived from it.",
),
```

When unset, doctor output is **byte-identical to today** (existing tests in `test_cli_doctor.py` must continue to pass without snapshot updates; the new branch is purely additive).

When set, after the existing tables render, doctor renders two additional tables:

#### 11.2.1 "Resolved augmentations" table

Rendered via `rich.table.Table(title="Resolved augmentations", show_header=False, box=None)`. Rows:
- `preset` → `cfg.data.augmentations.preset`
- `intensity` → `cfg.data.augmentations.intensity`
- 8 knob rows, one per `ResolvedAugmentations` field, in the §4 declaration order.
- `steps` → comma-joined Albumentations step-name list from `_STEP_NAMES_FOR(resolved)`.

Loading: `cfg = load_config(config_path)` using the existing `custom_sam_peft.config.loader.load_config`. Resolver: `aug_presets.resolve(cfg.data.augmentations)`. Step list: `aug_presets._STEP_NAMES_FOR(resolved)` — the helper is intentionally module-private (underscore-prefixed) but the `doctor_cmd` import is allowed; alternatively the planner can promote it to public if cleaner.

#### 11.2.2 "Normalization" table

Rendered via `rich.table.Table(title="Normalization", show_header=False, box=None)`. Rows:
- `model.name` → `cfg.model.name`
- `mean` → returned mean vector
- `std`  → returned std vector
- `resolution path` → one of `"processor"`, `"table-fallback"`, `"config-fallback"`

**Resolution-path discovery (pick one):** the existing `resolve_normalization` returns `(mean, std)` without indicating which of its three paths fired. The spec adopts **option A** (small return-shape change with a back-compat shim):

- Introduce `resolve_normalization_with_path(model_name, fallback) -> tuple[list[float], list[float], Literal["processor", "table-fallback", "config-fallback"]]` next to the existing `resolve_normalization`.
- `resolve_normalization(...)` becomes a thin wrapper that drops the third value: `return resolve_normalization_with_path(...)[:2]`.
- `build_eval_transforms` and `build_train_transforms` keep using the 2-tuple wrapper — they don't need the path.
- `doctor_cmd` calls the 3-tuple variant.

This avoids touching every call site of `resolve_normalization` (there are two in `transforms.py` and tests in `test_data_transforms.py`). The wrapper is unconditional and zero-cost.

Rejected alternative B (separate "re-derive path" helper that runs the resolver logic twice): brittle and duplicates the three-path logic; spec explicitly picks A.

#### 11.2.3 `--json` mode

When `--config` is **not** set, JSON output is byte-identical to today.

When `--config` is set, the top-level JSON dict gains an optional key `resolved_config`:

```json
{
  "python_version": "...",
  "...": "existing fields",
  "resolved_config": {
    "augmentations": {
      "preset": "medical",
      "intensity": "medium",
      "resolved": { /* 8 knobs */ },
      "steps": ["LongestMaxSize", "..."]
    },
    "normalize": {
      "model_name": "facebook/sam3.1",
      "mean": [0.485, 0.456, 0.406],
      "std":  [0.229, 0.224, 0.225],
      "resolution_path": "processor"
    }
  }
}
```

Implementation: `doctor_cmd` builds the `resolved_config` dict from the same helpers as the table rendering, then injects it into the dataclasses-asdict output. The DoctorReport dataclass and `diagnostics.run_doctor` are **not** modified — config-derived data is purely a `doctor_cmd.py` concern. This matches the existing separation of concerns (`diagnostics.py` is environment-only; `doctor_cmd.py` is presentation).

### 11.3 Template updates

Both `cli/templates/coco_text_lora.yaml` and `cli/templates/coco_text_qlora.yaml` have lines 34-36 replaced with the §11.1 `${preset} / ${intensity} / ${overrides_block}` block.

## 12. Test plan

CPU-only. New tests use the existing `caplog` convention for log-message assertions and the existing `Typer` runner convention for CLI tests (mirroring `test_cli_init.py`, `test_cli_doctor.py`).

### 12.1 New: `tests/unit/test_aug_presets.py`

- `test_resolve_table_exact_values` — parameterized over all 12 `(preset, intensity)` pairs for the four real domains; for each, build `AugmentationsConfig(preset=p, intensity=i)`, call `resolve`, and assert each of the 8 resolved fields matches the §5 table.
- `test_resolve_none_zeroes_all_knobs` — for each intensity, `preset="none"` → all-zero/false; intensity is ignored.
- `test_resolve_custom_zeroes_then_overrides_apply` — `preset="custom"` seed is all-zero; overrides apply; intensity ignored.
- `test_resolve_override_wins_over_table` — for a representative subset (e.g. `(natural, medium, color_jitter=0.5)`), assert the override field replaces the table value and other fields keep their table values.
- `test_resolve_override_zero_disables_table_knob` — e.g. `(natural, medium, color_jitter=0.0)` → resolved `color_jitter == 0.0` (zero is a valid override, not "inherit").
- `test_resolve_locked_off_warns_medical_hflip` — `caplog`, level WARNING: `AugmentationsConfig(preset="medical", overrides={"hflip": True})` emits one warning naming `hflip`, `medical`, and the word "laterality".
- `test_resolve_locked_off_warns_natural_rotate90` — same pattern for natural + rotate90 (mentions "up" or "natural").
- `test_resolve_locked_off_warns_microscopy_color_jitter` — same for microscopy + color_jitter=0.1 (mentions "fluorescence" or "channel").
- `test_resolve_locked_off_warns_satellite_stain_jitter` — same for satellite + stain_jitter=0.05 (mentions "H&E").
- `test_resolve_locked_off_no_warn_when_disabling` — medical + `overrides={"hflip": False}` emits **no** warning (False/0 is "always fine").
- `test_resolve_none_skips_locked_off_check` — `preset="none"` with `overrides={"hflip": True}` emits no warning; `resolved.hflip is True`.
- `test_resolve_custom_skips_locked_off_check` — same with `preset="custom"`.
- `test_resolved_augmentations_frozen` — `dataclasses.replace` works; direct mutation raises `dataclasses.FrozenInstanceError`.
- `test_dump_augmentation_pipeline_shape` — `dump_augmentation_pipeline(AugmentationsConfig(preset="medical", intensity="medium"))` returns the §10 dict structure; `steps` matches §10's literal example; `library_version` is a non-empty string.
- `test_dump_augmentation_pipeline_steps_empty_for_none` — `preset="none"` → `steps == ["LongestMaxSize", "PadIfNeeded", "Normalize", "ToTensorV2"]`.
- `test_step_names_match_assembly` — for a representative cell (e.g. natural/aggressive), the `_STEP_NAMES_FOR(resolved)` list equals the names of the actual Albumentations objects in `build_train_transforms(...).transforms` (calls Albumentations and inspects the compose's `transforms` attribute).

### 12.2 New: `tests/unit/test_stain_jitter.py`

- `test_identity_at_sigma_zero` — random uint8 RGB image (e.g. shape `(32, 32, 3)`); `StainJitter(sigma=0.0, p=1.0).apply(img) == img` (exact equality, because the implementation short-circuits at σ=0).
- `test_dtype_and_shape_preserved` — σ=0.1, random image; output `.shape == input.shape`; `output.dtype == np.uint8`.
- `test_range_preserved` — σ=0.1, random image; `output.min() >= 0` and `output.max() <= 255`.
- `test_mask_untouched_through_compose` — wrap `StainJitter(sigma=0.1, p=1.0)` in `A.Compose([...], bbox_params=...)`; pass `image=img, mask=mask, bboxes=[], class_labels=[]`; assert `out["mask"] is mask` or `np.array_equal(out["mask"], mask)`.
- `test_determinism_with_numpy_seed` — same input, `np.random.seed(0)`, two calls of `StainJitter(sigma=0.1, p=1.0).apply(img)` reseeded each time → identical output.
- `test_sigma_negative_rejected` — `StainJitter(sigma=-0.1)` raises `ValueError`.
- `test_p_zero_passes_through` — wrap in compose with `p=0.0`; output equals input (Albumentations skips it).

### 12.3 Extend: `tests/unit/test_data_transforms.py`

- Delete the legacy tests that constructed `AugmentationsConfig(hflip=True, color_jitter=0.1)` directly.
- Add `test_pipeline_step_list_per_preset_intensity` — parameterized over a representative subset: `(natural, medium)`, `(medical, medium)`, `(medical, safe)`, `(satellite, aggressive)`, `(microscopy, safe)`, `none`, and a `custom` with `overrides={"hflip": True, "stain_jitter": 0.05}`. For each, build the train compose and assert the ordered class-name list of `compose.transforms` matches §8's conditional emission.
- `test_pipeline_eval_equivalent_when_all_zero` — for `preset="none"`, the train pipeline's step list (excluding bbox_params machinery) equals the eval pipeline's step list.
- `test_pipeline_omit_vs_p_zero` — when `gauss_noise=0`, `A.GaussNoise` is **not** in the compose at all (omitted), as opposed to being present with `p=0`. Verified by `not any(isinstance(t, A.GaussNoise) for t in compose.transforms)`.

### 12.4 Extend: `tests/unit/test_config_schema.py`

- Replace any `AugmentationsConfig(hflip=..., color_jitter=...)` assertion with `AugmentationsConfig(preset="natural", intensity="medium")` and check defaults.
- `test_augmentation_overrides_rejects_unknown_keys` — `AugmentationOverrides(hfilp=True)` raises `ValidationError` (extra="forbid").
- `test_augmentations_preset_literal_validation` — `AugmentationsConfig(preset="mediacl")` raises `ValidationError`.
- `test_augmentations_intensity_literal_validation` — `AugmentationsConfig(intensity="medum")` raises `ValidationError`.
- `test_augmentations_overrides_default_factory` — two independent `AugmentationsConfig()` instances do not share the same `overrides` object.
- `test_augmentations_overrides_all_none_by_default` — `AugmentationsConfig().overrides.model_dump()` has every field = `None`.

### 12.5 Extend: `tests/unit/test_data_coco.py`, `tests/unit/test_data_hf.py`

Fixture YAML/dict shapes for `data.augmentations:` change from the legacy two-key form to the new triple. Tests that wanted "no augmentation" use `{"preset": "none"}`; tests that wanted "default" use `{"preset": "natural", "intensity": "medium"}` (or rely on the field's default factory).

### 12.6 Extend: `tests/unit/test_trainer_nan_behavior.py`, `tests/unit/test_trainer_run_dir.py`

Migrate every `AugmentationsConfig(hflip=False, color_jitter=0.0)` callsite to `AugmentationsConfig(preset="none")`.

Add to `test_trainer_run_dir.py` (or whichever already asserts on run-dir artifacts):
- `test_run_dir_writes_augmentation_pipeline_json` — after `Trainer(...)` constructs `run_dir`, assert `run_dir/augmentation_pipeline.json` exists, parses as JSON, has top-level keys `{"preset", "intensity", "resolved", "steps", "library_version"}`, and `resolved` has all 8 knob keys.

### 12.7 Extend: `tests/integration/test_train_resume.py`, `tests/integration/test_train_end_to_end.py`

- Migrate `AugmentationsConfig(...)` callsites as above.
- In `test_train_end_to_end.py`, after the run finishes, assert `run_dir/augmentation_pipeline.json` contains the expected `preset`, the expected step list, and a non-empty `library_version`.

### 12.8 Extend: `tests/unit/test_cli_init.py`

- `test_init_renders_preset_intensity` — invoke `csp init --preset medical --intensity safe --output tmp.yaml`; `load_config(tmp.yaml)` succeeds; `cfg.data.augmentations.preset == "medical"` and `intensity == "safe"`.
- Parameterized version of the above over all `4 real presets × 3 intensities = 12` combinations plus `none × medium` and `custom × medium` → 14 cases. (We skip `none/safe`, `none/aggressive`, etc. because intensity is ignored for none/custom — one representative each is enough; the resolver tests already cover the matrix.)
- `test_init_custom_writes_empty_overrides_scaffold` — `csp init --preset custom --output tmp.yaml`; the rendered file contains the `overrides: {}` scaffold (not the commented form); `load_config(tmp.yaml)` succeeds.
- `test_init_invalid_preset_rejected` — `csp init --preset typoo` exits non-zero; stderr mentions "preset".
- `test_init_invalid_intensity_rejected` — `csp init --intensity huge` exits non-zero; stderr mentions "intensity".
- `test_init_other_fields_parse_identically` — render with defaults; load config; assert non-augmentation fields (`run.name`, `model.name`, `train.epochs`) equal the values in today's templates — guards against the template-substitution accidentally corrupting unrelated YAML.

### 12.9 Extend: `tests/unit/test_cli_doctor.py`

- `test_doctor_no_config_byte_identical` — `csp doctor` (no `--config`) output equals today's output (either captured snapshot or compared against `_render_table` call count / specific table titles — the existing test convention dictates this; planner picks).
- `test_doctor_with_config_renders_resolved_augmentations` — `csp doctor --config <good.yaml>`; stdout contains the literal title string `"Resolved augmentations"`; contains a row labeled `preset`; the value of that row matches the config.
- `test_doctor_with_config_renders_normalization` — same; contains `"Normalization"`; contains rows `mean`, `std`, `resolution path`; for `facebook/sam3.1` with the standard normalize block, the resolution path is one of the three known strings.
- `test_doctor_json_no_config_no_resolved_block` — `csp doctor --json`; parsed JSON has no `"resolved_config"` key.
- `test_doctor_json_with_config_has_resolved_block` — `csp doctor --config <good.yaml> --json`; parsed JSON has `"resolved_config"` with sub-keys `"augmentations"` and `"normalize"`; augmentations sub-dict has `preset`, `intensity`, `resolved`, `steps`.

## 13. Migration (clean break)

No back-compat aliases. No deprecation cycle. The schema break is explicit:

1. **Delete** the old `AugmentationsConfig` definition (`schema.py` lines 57-59).
2. **Add** `Preset`, `Intensity`, `AugmentationOverrides`, and the new `AugmentationsConfig` in its place.
3. **Update** every fixture YAML / dict that constructed the old shape (six unit-test files, two integration-test files, both starter templates — enumerated in §4.2).
4. **Update** any docstring or comment in `data/coco.py`, `data/hf.py` referencing the legacy fields.

A user with a pre-existing config carrying `augmentations: {hflip: true, color_jitter: 0.1}` will fail `load_config` with a pydantic `ValidationError` ("extra fields not permitted: hflip, color_jitter"). The PR description includes a single-line migration recipe: `{hflip: X, color_jitter: Y}` → `{preset: natural, intensity: medium, overrides: {hflip: X, color_jitter: Y}}` (or just `{preset: natural, intensity: medium}` to inherit defaults).

#70 (v1.0 criteria) is the umbrella issue tracking acceptable pre-1.0 schema breaks. This PR's `Related issues` section in the PR body adds a one-line note that this break is gated under #70.

## 14. In / out of scope

### 14.1 In scope (v1)

- New `AugmentationsConfig` + `AugmentationOverrides` in `schema.py` (clean break, no aliases).
- `data/aug_presets.py` module with `PRESET_TABLE`, `LOCKED_OFF`, `ResolvedAugmentations`, `resolve`, `_STEP_NAMES_FOR`, `dump_augmentation_pipeline`.
- `StainJitter` custom Albumentations transform in `data/transforms.py`.
- Refactor `build_train_transforms` to consume `ResolvedAugmentations` and emit the §8 step list.
- `csp init --preset --intensity` with template substitution; `${overrides_block}` placeholder; `--preset custom` writes the uncommented overrides scaffold.
- `csp doctor --config` with two new tables and an additive `resolved_config` JSON block.
- `resolve_normalization_with_path` helper + wrapping the existing `resolve_normalization` (zero-cost shim).
- `trainer.py` writes `run_dir/augmentation_pipeline.json` next to `config.yaml`.
- Both starter templates updated.
- Tests as enumerated in §12.

### 14.2 Out of scope (file as follow-up issues only if explicitly requested)

- Sub-presets (`medical_xray`, `medical_histopathology`, `microscopy_fluorescence`).
- Raw Albumentations pipeline escape hatch (`albumentations_pipeline: [...]`).
- 3D / volumetric augmentations.
- Loss / model changes for non-RGB modalities.
- Domain-specific normalization stats (covered by #69, closed).
- Per-knob `p` (application-probability) overrides — magnitude-only for v1; `p=0.5` hard-coded.
- Albumentations Compose-level seeding (uses the codebase's existing `np.random` reproducibility flow).

## 15. Edge cases

- **`preset="none"`** → all knobs zero/off; intensity ignored; no warns even if locked-off knobs are enabled via overrides.
- **`preset="custom"`** → seed all-zero; overrides apply; intensity ignored; no warns (user is fully explicit).
- **All-zero resolved knobs** (e.g. `medical/safe`, `microscopy/safe`, `microscopy/medium`) → train pipeline equals eval pipeline structurally (`LongestMaxSize → PadIfNeeded → Normalize → ToTensorV2`); explicitly tested in §12.3.
- **Override sets a locked-off knob to False/0** → no warn (the user is *disabling* the locked-off knob; never problematic).
- **Override sets a non-locked-off knob to its table value** → no warn; behaves as if no override were given. (The resolver does not optimize this case; the override path still runs and the value is identical to the seed.)
- **WARN dedup** → automatic; `resolve()` is called once per `build_train_transforms` which is called once per training run.
- **`extra="forbid"` on `AugmentationOverrides`** → catches typos like `hfilp: true` at `load_config` time with a clear error message naming the bad key.
- **`extra="forbid"` on `AugmentationsConfig`** → catches typos like `presset: medical` at `load_config` time.
- **`StainJitter` with negative sigma** → constructor raises `ValueError`; pydantic validators on `AugmentationOverrides.stain_jitter` should also reject negatives (a `Field(ge=0)` validator on the float fields would close this — the planner adds it explicitly).
- **`rotate_arbitrary > 360`** → permitted by schema (no upper bound enforced); the resolver passes through; `A.Affine` handles the wraparound. Documented as acceptable v1 behavior.
- **`color_jitter > 1.0`** → permitted; the implementation passes the value through to `A.ColorJitter` which accepts any non-negative float. Documented as acceptable v1 behavior.
- **Empty overrides** (`AugmentationsConfig(preset="natural")` with default factory) → resolver returns exactly the `PRESET_TABLE` row for `(natural, medium)`.
- **`library_version` derivation** → reads `custom_sam_peft.__version__`; if missing (shouldn't happen — set in `pyproject.toml`), falls back to the string `"unknown"`.
- **Step-name match between sidecar and runtime compose** → guaranteed by both code paths calling the same `_STEP_NAMES_FOR(resolved)` helper; tested in §12.1.

## 16. Deliverables-to-issue mapping

| Issue item | Where in this spec | Implementation file(s) |
|---|---|---|
| 1. Spec | This document | `docs/superpowers/specs/2026-05-22-domain-aware-augmentation-presets-design.md` |
| 2. New `AugmentationsConfig` | §4 | `src/custom_sam_peft/config/schema.py` + `src/custom_sam_peft/data/aug_presets.py` |
| 3. Refactor `build_train_transforms` | §8 | `src/custom_sam_peft/data/transforms.py` |
| 4. Update `init_cmd` | §11.1 | `src/custom_sam_peft/cli/init_cmd.py` + `src/custom_sam_peft/cli/templates/*.yaml` |
| 5. Update `doctor_cmd` + run metadata | §11.2 + §10 | `src/custom_sam_peft/cli/doctor_cmd.py` + `src/custom_sam_peft/train/trainer.py` |
| 6. Update example YAMLs | §11.3 | `src/custom_sam_peft/cli/templates/coco_text_lora.yaml`, `src/custom_sam_peft/cli/templates/coco_text_qlora.yaml` |
| 7. Tests | §12 | `tests/unit/test_aug_presets.py` (new), `tests/unit/test_stain_jitter.py` (new), and the eight existing files enumerated in §4.2 |

## 17. Related issues

- **#70** — pre-1.0 / v1.0 criteria; this PR's schema break is gated under #70's "breaking changes acceptable before v1.0" allowance. Add a line in the PR description referencing #70 so the umbrella issue's checklist captures this change.
- **#33** — folder-format dataset adapter; domain presets are especially useful for the small custom datasets that adapter will enable. No code dependency between the two PRs.
- **#69** — normalization fallback (closed by the 2026-05-21 audit). This PR does not modify `resolve_normalization` or `KNOWN_PROCESSOR_STATS`; the only normalization touch is the additive `resolve_normalization_with_path` wrapper used by `doctor`.
- **#74** — predict CLI. Inference uses `build_eval_transforms` only; no augmentation. This PR does not alter predict.
- **#60** — sam3 gradient-checkpointing recompute mismatch. Unrelated; mentioned only because it's referenced in the template comments that this PR preserves verbatim.
