# N-channel input support (1–16 channels) — Design Specification

**Issue:** #111
**Status:** Locked design (do not redesign; this is the source of truth for the planner and implementers).
**Worktree branch:** `n-channel-input-111`
**Anchors verified against:** `91bbe3d` (worktree HEAD; line numbers re-confirmed below — the line numbers in the original brainstorm had drifted).

---

## 1. Overview / Goal

Today both dataset loaders force every image to 3-channel RGB via `pil_img.convert("RGB")`:

- `src/custom_sam_peft/data/coco.py:213` — `CocoDataset._decode_image` returns `np.asarray(pil_img.convert("RGB"))`.
- `src/custom_sam_peft/data/hf.py:213` — `HFDataset._decode_image` returns `np.asarray(img_obj.convert("RGB"))` (plus an array branch at `hf.py:214-217` that np.stacks a 2-D array to 3 channels).

Consequences: grayscale is silently triplicated, and stacks with more than 4 channels have no ingestion path at all (PIL caps at RGBA). `NormalizeConfig` (`schema.py:248`) hardcodes 3-element ImageNet stats. The model wrapper hardcodes a `(B, 3, H, W)` input contract (`sam3.py:230`).

**Goal:** support any input channel count `1 ≤ N ≤ 16` end-to-end, from data ingestion through normalization, augmentation, and the model forward, without regressing the existing N=3 RGB path. This is the **channel** axis only — NOT volumetric/temporal data (that is issue #110, disjoint from this work).

The design uses a single learned **N→3 channel adapter** ("bridge A") inserted before the frozen pretrained SAM 3.1 patch-embed. We do **not** replace the patch-embed itself (that is "bridge B", out of scope — see §14).

### 1.1 Core concept: channel COUNT vs channel SEMANTICS are decoupled

The central design principle of this revision: **channel count and channel semantics are two independent axes.**

- **Channel count** (`data.channels`, `1 ≤ N ≤ 16`) drives *tensor shapes and per-channel lengths*: the reader's output `(H, W, C)`, the adapter's `in_channels`, the normalize stats length, the `_validate_inputs` channel check.
- **Channel semantics** (`data.channel_semantics`, one of `rgb`/`rgba`/`grayscale`/`freeform`) drives *how the channels are treated*: whether an adapter is built and how it is initialized, whether the `AutoImageProcessor` is consulted for normalization, and which augmentation regime applies.

Crucially, **the semantic — not the count — drives the adapter, normalization, and augmentation.** The same channel count can carry different semantics, and the treatment must follow the semantic:

- A **3-channel** stack may be photometric RGB (`rgb`) OR a `grayscale+SAR+height` measurement stack (`freeform`). Same count (3), opposite treatment: `rgb` gets passthrough + full photometric augs + ImageNet stats; `freeform` gets a learned 3→3 adapter + geometry-only augs + required explicit stats.
- A **1-channel** image may be a photometric X-ray/luminance image (`grayscale`) OR a single SAR band (`freeform`). Same count (1): `grayscale` triplicates into the pretrained luminance prior and allows intensity augs; `freeform` gets average-broadcast init and geometry-only augs.

Existing behavior is preserved exactly by the default `channel_semantics = "rgb"` (which requires `channels == 3`): the RGB path is byte-for-byte unchanged.

### 1.2 The channel-semantics profile table

The four shipped semantics and their full treatment (this table is LOCKED — implement faithfully):

| Semantic | channels | Adapter | Adapter init | Normalize default | Augmentations |
|---|---|---|---|---|---|
| `rgb` (DEFAULT) | `== 3` | NONE (passthrough) | n/a | ImageNet via `AutoImageProcessor` (unchanged) | FULL — `ColorJitter` (bright/contrast/sat/hue), `StainJitter`, `GaussNoise`, `GaussianBlur`, geometry (unchanged from today) |
| `rgba` | `== 4` | `Conv2d(4,3,1)` learned | `identity_passthrough`: identity on RGB + zero on the 4th (alpha) channel (`weight[o,i]=1` if `o==i` for `o,i ∈ {0,1,2}` else `0`; alpha column `= 0`; `bias = 0` → step-0 output ≡ RGB, alpha ignored, then learns) | ImageNet-3 mean/std + `[0.5]` for alpha (length 4) | photometric non-3ch: `RandomBrightnessContrast` (driven by `color_jitter` knob), `GaussNoise`, `GaussianBlur`, geometry. NO sat/hue, NO `StainJitter` (need exactly 3ch). |
| `grayscale` | `== 1` | `Conv2d(1,3,1)` learned | `average_broadcast`: `weight = 1.0`, `bias = 0` → EXACT triplication | ImageNet luminance `mean=[0.449] std=[0.226]` (length 1) | same regime as `rgba` (`RandomBrightnessContrast` + `GaussNoise` + `GaussianBlur` + geometry) |
| `freeform` | `1..16` | `Conv2d(N,3,1)` learned | `average_broadcast`: `weight = 1/N`, `bias = 0` | NONE — explicit per-channel stats REQUIRED (loud error if omitted/mismatched) | GEOMETRY-ONLY (hard-disable all 4 value-altering augs) |

**Derived predicates (extensibility-friendly — state these, not hardcoded names):**

- `photometric = semantic in {rgb, rgba, grayscale}` → intensity augs allowed. `freeform` is **non-photometric** → geometry-only.
- "sat/hue + `StainJitter` allowed" ⟺ `photometric AND channels == 3` (i.e. only `rgb`).
- "no adapter (passthrough)" ⟺ `semantic == rgb`. **ALL other semantics get a learned N→3 adapter, INCLUDING `freeform` with `channels == 3`.**

### 1.3 Extensibility registry

Future semantics are intended to be added (the user explicitly wants this). The four profiles above are entries in a registry, and all per-semantic logic (adapter construction/init, normalize default, augmentation regime) reads the **profile**, never a hardcoded semantic name.

Introduce a frozen dataclass and a registry dict (recommended location: a new `src/custom_sam_peft/data/channel_semantics.py` — planner's final call):

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ChannelSemanticsProfile:
    allowed_channels: ...          # e.g. a frozenset {3}, {4}, {1}, or a range 1..16
    use_adapter: bool              # False only for rgb
    adapter_init: Literal["average_broadcast", "identity_passthrough"]
    photometric: bool              # True for rgb/rgba/grayscale; False for freeform
    normalize_default: ...         # (mean, std) tuple, or None when explicit stats are required (freeform)

CHANNEL_SEMANTICS: dict[str, ChannelSemanticsProfile] = {
    "rgb":       ChannelSemanticsProfile(allowed_channels={3}, use_adapter=False, adapter_init="average_broadcast", photometric=True,  normalize_default=<ImageNet-3>),
    "rgba":      ChannelSemanticsProfile(allowed_channels={4}, use_adapter=True,  adapter_init="identity_passthrough", photometric=True,  normalize_default=<ImageNet-3 + [0.5]>),
    "grayscale": ChannelSemanticsProfile(allowed_channels={1}, use_adapter=True,  adapter_init="average_broadcast",   photometric=True,  normalize_default=<luminance [0.449]/[0.226]>),
    "freeform":  ChannelSemanticsProfile(allowed_channels=range(1, 17), use_adapter=True, adapter_init="average_broadcast", photometric=False, normalize_default=None),
}
```

Notes for the planner:

- `adapter_init` value semantics: `identity_passthrough` = identity on the **first 3** input channels and **zero** on the rest (covers the rgba identity-on-RGB + zero-alpha init; generalizes to any `N ≥ 3` where the first 3 channels map straight through). `average_broadcast` = every output channel is `1/N` over all N inputs (covers grayscale's exact triplication and freeform's mean-of-stack).
- The config field validates membership against `CHANNEL_SEMANTICS.keys()`.
- **Adding a future semantic = one registry entry.** The augmentation / normalize / adapter logic reads the profile (`use_adapter`, `adapter_init`, `photometric`, `normalize_default`, `allowed_channels`), so no new branches are needed elsewhere. `rgb`'s `use_adapter=False` (passthrough) is the lone special case keyed on the profile flag, not the name.

---

## 2. Non-goals (v1)

Each non-goal below should be filed as a follow-up GitHub issue (see §14).

1. **Bridge B** — replacing the SAM 3.1 patch-embed with an `in_chans=N` stem for large-N / true hyperspectral input (>16 channels). The `N≤16` cap exists precisely because beyond ~16 channels the 3-channel bottleneck of bridge A becomes lossy.
2. **GDAL / rasterio georeferencing** — geospatial CRS, clinical DICOM. v1 reads multi-band TIFF pixel bands only, not georeferencing metadata.
3. **Per-channel learned augmentation.**
4. **Channel-count auto-detection.** `data.channels` is explicit only.

---

## 3. Config changes (`src/custom_sam_peft/config/schema.py`)

### 3.1 New field `data.channels`

Add to `DataConfig` (class at `schema.py:368`):

```python
channels: int = Field(
    default=3,
    ge=1,
    le=16,
    description=(
        "Number of input image channels (1..16). The N->3 channel adapter "
        "(a 1x1 conv inserted before the frozen SAM3.1 patch-embed) bridges "
        "N channels down to the pretrained 3-channel stem. The cap of 16 is "
        "deliberate: beyond ~16 channels the 3-channel bottleneck becomes "
        "lossy, at which point a future 'bridge B' (replacing the patch-embed "
        "with an in_chans=N stem; issue follow-up) would be warranted instead. "
        "Explicit only — no auto-detection."
    ),
)
```

The inline comment/description MUST document both the cap and *why* (the lossy-bottleneck rationale above). This is a hard requirement.

### 3.2 New field `data.channel_semantics`

Add to `DataConfig` alongside `channels`:

```python
channel_semantics: Literal["rgb", "rgba", "grayscale", "freeform"] = Field(
    default="rgb",
    description=(
        "How the input channels are interpreted (independent of the channel "
        "COUNT in `channels`). Drives the channel adapter (build + init), the "
        "normalization default, and the augmentation regime. See the "
        "CHANNEL_SEMANTICS registry (src/custom_sam_peft/data/channel_semantics.py) "
        "for the per-semantic profile. Default 'rgb' reproduces today's behavior "
        "exactly. Add new semantics by adding a registry entry."
    ),
)
```

The `Literal` values MUST match `CHANNEL_SEMANTICS.keys()` (§1.3). The default `"rgb"` preserves existing behavior. The description points users at the registry as the extension surface.

### 3.3 Cross-field validation in `DataConfig`

Add a `@model_validator(mode="after")` to `DataConfig` that enforces, in order:

**(a) semantic ↔ channels match** — look up the profile `CHANNEL_SEMANTICS[self.channel_semantics]` and require `self.channels` ∈ `profile.allowed_channels`. Per the registry: `rgb ⟹ channels == 3`, `rgba ⟹ channels == 4`, `grayscale ⟹ channels == 1`, `freeform ⟹ channels ∈ 1..16`. Loud, specific error on mismatch, e.g.:

> `data.channel_semantics='rgba' requires data.channels=4, but data.channels=3.`

**(b) normalize length cross-check** — once normalization is resolved (see (c)), enforce:

```
len(self.normalize.mean) == self.channels
len(self.normalize.std)  == self.channels
```

with a loud, specific error on mismatch, e.g.:

> `data.normalize.mean has 3 entries but data.channels=5; provide exactly 5 per-channel mean values (and 5 std values).`

**(c) per-semantic normalize default / required-stats** — resolve the normalize stats with this precedence:

1. **Explicit user stats win.** If the user supplied `normalize.mean`/`std`, use them as-is (still subject to the (b) length check).
2. **Else profile default.** For `rgb`/`rgba`/`grayscale`, when the user omits `normalize`, fill `mean`/`std` from `profile.normalize_default` (rgb → ImageNet-3; rgba → ImageNet-3 + `[0.5]`; grayscale → `[0.449]`/`[0.226]`).
3. **`freeform` has NO default.** `profile.normalize_default is None`, so explicit per-channel stats are REQUIRED. If the user omits `normalize` for `freeform`, raise a loud error, e.g.:

> `data.channel_semantics='freeform' requires explicit data.normalize.mean/std (one value per channel; no default exists for freeform). Provide N=5 mean and 5 std values.`

Rationale for placing all three checks here rather than in `NormalizeConfig`: `NormalizeConfig` knows neither `channels` nor `channel_semantics`; `DataConfig` owns all the sub-configs and the registry lookup.

> Implementation note for the planner: how to distinguish "user omitted `normalize`" from "user supplied stats equal to the default" matters for steps (c.1)/(c.2). Use a sentinel (e.g. `normalize: NormalizeConfig | None = None` defaulting to `None`, then materialize the profile default in the validator) rather than a `default_factory` that always produces ImageNet-3 — otherwise a non-rgb semantic can't tell an omitted normalize from a deliberate ImageNet-3 override. The planner finalizes the mechanism; the observable contract is the precedence in (c).

### 3.4 `channel_semantics == "rgb"` default behavior

The shipped default `channel_semantics="rgb"` requires `channels == 3` (§3.3a) and, when the user does not override `normalize`, fills the existing ImageNet-3 default (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) — staying valid and byte-for-byte unchanged. For `rgba`/`grayscale` the profile default fills in if omitted; for `freeform` the user MUST provide explicit per-channel stats (§3.3c), and the length must match `channels` (§3.3b).

---

## 4. `NormalizeConfig` changes (`schema.py:248`)

### 4.1 Relax the length constraints

Today (`schema.py:270-275`):

```python
mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406], min_length=3, max_length=3)
std:  list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225], min_length=3, max_length=3)
```

Change to allow `1 ≤ len ≤ 16` (i.e. `min_length=1, max_length=16`). The exact-channel-count match is enforced cross-field in `DataConfig` (§3.3b), not here, because `NormalizeConfig` is channel-agnostic and semantic-agnostic in isolation.

**Per-semantic defaults come from the registry, applied in `DataConfig`.** `NormalizeConfig` no longer carries the only default: the per-semantic defaults (rgb → ImageNet-3, rgba → ImageNet-3 + `[0.5]`, grayscale → `[0.449]`/`[0.226]`) live in `CHANNEL_SEMANTICS[...].normalize_default` and are materialized by `DataConfig`'s validator when the user omits `normalize` (§3.3c). **`freeform` has no default** — explicit stats are required. See §3.3's implementation note on using a sentinel (`normalize: NormalizeConfig | None = None`) so an omitted normalize is distinguishable from a deliberate override; the `rgb`/ImageNet-3 path remains the zero-config default for backward compatibility.

### 4.2 Keep the per-value range checks

The existing `_check_ranges` validator (`schema.py:277-285`: each mean in `[0,1]`, each std `> 0`) is channel-count-agnostic and is retained as-is — it already iterates over arbitrary-length lists.

---

## 5. Channel-adapter design (bridge A)

### 5.1 Where it lives

The adapter is a `nn.Conv2d(in_channels=N, out_channels=3, kernel_size=1)` (3·N weights + 3 bias when an adapter is built; **no adapter / zero new params only for `semantic == rgb`**). It is inserted **before** the frozen SAM 3.1 patch-embed.

The patch-embed lives inside `self.model.backbone.forward_image(images)`, called at `src/custom_sam_peft/models/sam3.py:360` inside `_Sam3ImageAdapter.forward` (class at `sam3.py:322`, `forward` at `sam3.py:344`). The channel adapter MUST run on `images` immediately before this call:

```python
# _Sam3ImageAdapter.forward, before line 360:
images = self.channel_adapter(images)  # (B, N, H, W) -> (B, 3, H, W); identity (None) only when semantic == rgb
backbone_out = self.model.backbone.forward_image(images)
```

**Module ownership (load-bearing for the cross-cutting requirements in §10):** the adapter is registered as a submodule of `_Sam3ImageAdapter` (e.g. `self.channel_adapter`). Structurally:

- `Sam3Wrapper.model` → `_Sam3ImageAdapter` (the channel adapter lives here)
- `_Sam3ImageAdapter.model` → raw SAM 3.1 model, which `apply_lora` replaces with a `PeftModel` (`lora.py:117`, `base = wrapper.model.model`).

Because the channel adapter is a sibling of `_Sam3ImageAdapter.model` (not inside it), it survives PEFT wrapping untouched and is reached by `wrapper.parameters()` and `wrapper.model.state_dict()`, but is **not** reached by `wrapper.peft_model.save_pretrained()`. This distinction drives §10.

### 5.2 Construction predicate: adapter is None ⟺ `semantic == rgb`

The decision to build an adapter is keyed on the **semantic profile flag**, NOT on channel count:

```
adapter is None  ⟺  profile.use_adapter == False  ⟺  semantic == rgb
otherwise        →  build nn.Conv2d(channels, 3, 1) with profile.adapter_init
```

For `semantic == rgb` (the shipped default, which requires `channels == 3`): **no adapter is constructed** and **no new parameters exist**. The forward path is byte-for-byte identical to today's RGB path. Implementation: `self.channel_adapter` is `None` (or `nn.Identity`) and the forward skips it entirely. A test MUST assert this (§12, CPU case C3 "passthrough keyed on `semantic == rgb`").

> **Zero-regression now attaches to `semantic == rgb`, NOT to `channels == 3`.** This is the key change. A `freeform` config with `channels == 3` does **NOT** get a passthrough — it builds a learned 3→3 `Conv2d(3,3,1)` with `average_broadcast` init (non-zero, trainable new params). The only passthrough is the default `rgb` semantic.
>
> Note: prefer `None` over `nn.Identity` if `nn.Identity` would alter `state_dict` keys or break the "params unchanged" assertion; the planner picks the mechanism, but the observable contract is "no new params, output identical to pre-feature RGB path" for `semantic == rgb`.

### 5.3 Non-rgb semantics — learned channel adapter

For every semantic with `profile.use_adapter == True` (`rgba`, `grayscale`, `freeform`), build a `nn.Conv2d(channels, 3, kernel_size=1)`, **full-rank and fully trainable** (NOT a LoRA adapter — brand-new weights, not low-rank deltas on an existing layer). `requires_grad=True` for both weight and bias. The init is selected by `profile.adapter_init`.

**Init kind `average_broadcast` (grayscale, freeform).** Each of the 3 output channels is initialized so its output equals the mean over the N input channels:

- `weight[o, i, 0, 0] = 1/N` for all output channels `o ∈ {0,1,2}` and all input channels `i ∈ {0..N-1}`.
- `bias[o] = 0`.

At step 0 this makes all 3 output channels equal to the per-pixel mean of the N input channels — i.e. "grayscale-of-the-stack" fed through the pretrained patch-embed. Training is then free to specialize the 3·N weights.

- **grayscale (N=1) — exact triplication.** With N=1, `weight = 1/1 = 1.0` on every output channel and `bias = 0`, so the adapter output is exactly the single input channel replicated 3×. This is the grayscale init property. A test MUST assert the grayscale (channels=1) adapter output equals `torch.cat([x, x, x], dim=1)` exactly (§12).
- **freeform — mean of stack.** For arbitrary `N ∈ 1..16`, `weight = 1/N`, so step-0 output is the per-pixel mean over all N channels. Note `freeform` with `channels == 3` uses this `average_broadcast` init (NOT a passthrough), producing non-zero trainable params.

**Init kind `identity_passthrough` (rgba).** Identity on the first 3 input channels, zero on the rest:

- `weight[o, i, 0, 0] = 1` if `o == i` for `o, i ∈ {0,1,2}`, else `0`. For rgba (N=4) this means the alpha column (`i == 3`) is all zeros.
- `bias[o] = 0`.

At step 0 the adapter output equals the first 3 (RGB) input channels exactly, with the alpha channel ignored (zeroed) — so step-0 forward through the patch-embed is the RGB image, and training then learns to incorporate alpha. A test MUST assert the rgba adapter's step-0 output equals the input's first 3 channels (alpha dropped) — see §12. (Generalizes: `identity_passthrough` maps the first 3 channels straight through and zeros channels `3..N-1`; only used by `rgba` in the shipped registry.)

**Channel order is intentionally not preserved (freeform / average_broadcast).** The adapter is trained from scratch and learns channel-specific filters in the user's *fixed* channel order. This is "freedom, not permutation-invariance": there is no guarantee or requirement that permuting input channels yields equivalent outputs. Do not add any ordering/permutation-invariance constraint. (`identity_passthrough` does ascribe meaning to the first-3 ordering by construction — that is intentional for rgba, where RGB are the first 3.)

### 5.4 Construction & plumbing

The adapter is constructed from `config.channels` AND `config.channel_semantics` at model load (the semantic selects `use_adapter` and `adapter_init` from the registry; the count sizes the conv). **Plumbing problem (verified):** `load_sam31(cfg: ModelConfig)` (`sam3.py:671`) currently receives only `ModelConfig`, but both `channels` and `channel_semantics` live in `DataConfig`. The planner must thread both into model construction. Recommended approach (least invasive): add `channels: int = 3` and `channel_semantics: str = "rgb"` parameters to `load_sam31`, `_Sam3ImageAdapter.__init__` (`sam3.py:339`), and `Sam3Wrapper.__init__` (`sam3.py:206`) — OR pass a resolved `ChannelSemanticsProfile` object alongside `channels`; each `load_sam31(...)` call site passes `cfg.data.channels` and `cfg.data.channel_semantics`. (Planner picks "two scalars" vs "channels + profile"; the profile-object form keeps the registry as the single source of init/adapter logic.)

**All seven `load_sam31` call sites must be updated** to pass BOTH `channels` and `channel_semantics` (verified via grep):

| File:line | Caller | Passes |
| --- | --- | --- |
| `src/custom_sam_peft/cli/run_cmd.py:91` | `run` command | `cfg.data.channels`, `cfg.data.channel_semantics` |
| `src/custom_sam_peft/cli/calibrate_cmd.py:76` | `calibrate` command | `cfg.data.channels`, `cfg.data.channel_semantics` |
| `src/custom_sam_peft/eval/runner.py:130` | eval runner | `cfg.data.channels`, `cfg.data.channel_semantics` |
| `src/custom_sam_peft/runs/bundle.py:67` | `run_export` | `cfg.data.channels`, `cfg.data.channel_semantics` |
| `src/custom_sam_peft/predict/runner.py:286` | predict runner | resolved `channels`, `channel_semantics` (§11) |
| `src/custom_sam_peft/train/runner.py:114` | train runner | `cfg.data.channels`, `cfg.data.channel_semantics` |
| (definition) `src/custom_sam_peft/models/sam3.py:671` | `load_sam31` itself | — |

Several of these have `cfg: TrainConfig` (so `cfg.data.channels`/`cfg.data.channel_semantics` are available); `calibrate_cmd.py` and `predict/runner.py` pass a bare `model_cfg`. **All relevant call sites are plumbed with real `channels` AND `channel_semantics` values — none defaults to a workaround, and predict is NOT excluded:**

- `predict/runner.py:286` now receives both `channels` and `channel_semantics` like the others. The values are read from the bundle config's `data.channels`/`data.channel_semantics` in `_resolve_config` and threaded through `_ResolvedConfig` (see §11 for the exact plumbing).
- `calibrate_cmd.py` similarly carries config; the planner confirms `data.channels`/`data.channel_semantics` are reachable there and passes them (defaults `3`/`"rgb"` only if the calibrate config genuinely lacks a `data` section, and document if so).

Note `_freeze_base` (`sam3.py:655`) is a no-op today; the channel adapter is created already-trainable, so freezing is not its concern — PEFT factories freeze only the base.

---

## 6. Data-ingestion reader

### 6.1 New helper `read_image(path, channels) -> np.ndarray (H, W, C)`

Add a channel-aware reader (recommended location: a new function in `src/custom_sam_peft/data/`, e.g. `data/io.py` or appended to an existing data module — planner's choice) that replaces the inline `convert("RGB")` in both loaders. Signature:

```python
def read_image(path: str | Path, channels: int) -> np.ndarray:  # returns (H, W, C) with C == channels
```

Dispatch on file extension:

| Extension(s) | Backend | Behavior |
| --- | --- | --- |
| standard raster (`.png/.jpg/.jpeg/.bmp/...`) | PIL | `channels==1 → convert("L")`, `==3 → convert("RGB")`, `==4 → convert("RGBA")`. Other N for raster formats is an error (PIL caps at RGBA). |
| `.npy`, `.npz` | `np.load` | Accept `(H, W, C)` or `(C, H, W)`; transpose CHW→HWC as needed; validate the channel dim `== channels` (loud error on mismatch). |
| `.tif`, `.tiff` | **`tifffile`** | Multi-band read; pixel bands only (no georeferencing). Validate band count `== channels`. |

All branches MUST return `(H, W, C)` with `C == channels`, raising a clear error on any mismatch (e.g. `read_image: <path> has 4 channels but data.channels=3`).

**The reader keys off channel COUNT only — `channel_semantics` does NOT reach the reader.** The PIL mode dispatch (1→"L", 3→"RGB", 4→"RGBA") and the array/tiff branches are driven entirely by `channels`. The semantic never overrides the reader: a `freeform` 3-channel array is read as-is `(H, W, 3)`, identical to how an `rgb` 3-channel image is read. Concretely, `rgba ⟹ channels == 4 ⟹` PIL `convert("RGBA")` is already covered by the existing 4→"RGBA" rule — no new reader branch is introduced by adding semantics. **No new reader logic comes from semantics.**

### 6.2 New dependency: `tifffile`

`tifffile` is a NEW runtime dependency (lightweight; no GDAL/rasterio in v1). Add it to `pyproject.toml`'s `dependencies` list (`pyproject.toml:9-26`), e.g. `"tifffile>=2024.1"` (planner pins an appropriate floor). Note: standard single-/3-band TIFFs would also work via PIL, but multi-band (>4) TIFFs require `tifffile`; route all `.tif/.tiff` through `tifffile` for consistency.

### 6.3 Wire into both loaders

- **COCO** (`coco.py:204-213`): `_decode_image` currently opens via PIL and returns `convert("RGB")`. Replace the body with `return read_image(img_path, self._channels)`, where `self._channels` is plumbed from `DataConfig.channels` into `CocoDataset.__init__`. Update the docstring (`coco.py:207`, "(H, W, 3)" → "(H, W, C)").
- **HF** (`hf.py:207-217`): `_decode_image` has two branches — a PIL-image branch (`hf.py:212-213`) and an array branch (`hf.py:214-217`) that `np.stack`s an `ndim==2` array to 3. **Reconcile with the new reader/channels semantics:**
  - PIL-image branch: route through `read_image`-style logic (convert to the configured channel mode) OR, since HF rows already carry decoded objects rather than file paths, apply the same channel-mode conversion inline. The planner picks the cleanest factoring (HF rows are objects/arrays, not always paths, so a path-based `read_image` may not fit directly — a shared `_coerce_to_channels(array_or_pil, channels)` core that `read_image` also calls is the recommended refactor).
  - Array branch: replace the unconditional "stack ndim==2 to 3" with channel-aware logic — for `channels==1` keep a single channel, for `channels==3` triplicate a 2-D array, accept `(H,W,C)`/`(C,H,W)` arrays and validate `C == channels`. Update the docstring (`hf.py:208`).

### 6.4 Annotations unchanged; no new `DataFormat`

Boxes/masks/classes continue to use the existing COCO/HF machinery unchanged. There is **no** new `DataFormat` literal (the existing `Literal["coco", "hf"]` at `schema.py:90` stays). The reader dispatches on file extension / array shape, not on a config format flag. "Multi-band" simply means existing COCO/HF annotations that point at multi-band image files or arrays.

---

## 7. Normalize resolution changes (`src/custom_sam_peft/data/transforms.py`)

### 7.1 `resolve_normalization` — consult the processor ONLY for `semantic == rgb`

`resolve_normalization` (`transforms.py:147`) and `resolve_normalization_with_path` (`transforms.py:84`) consult `AutoImageProcessor` first (`transforms.py:103`), which returns 3-channel RGB ImageNet stats. That processor is correct ONLY for `semantic == rgb`: for `rgba`/`grayscale`/`freeform` the frozen patch-embed's RGB stats do not apply to the *raw input channels* the adapter consumes (normalization happens upstream of the adapter, on the N-channel input), and the resolved config already carries the right per-channel stats (profile default or explicit, per §3.3c).

Change: thread `channel_semantics` (or a `use_processor: bool` derived from it — `use_processor == (semantic == rgb)`) into `resolve_normalization` / `resolve_normalization_with_path`. When `semantic == rgb`, the existing 3-step resolution (AutoImageProcessor → table → config) is unchanged. For `rgba`/`grayscale`/`freeform`, **skip the AutoImageProcessor and table-fallback paths entirely** and use the resolved config-provided per-channel `mean`/`std` directly (the "config-fallback" path code).

> Predicate change: the processor-skip now keys on `semantic == rgb`, NOT on `channels == 3`. (`grayscale` has `channels == 3`'s old behavior gated out — it has `channels == 1` anyway, but the point is the gate is the semantic. `freeform` with `channels == 3` ALSO skips the processor, unlike the old `channels == 3` rule.)

Both functions are called from `build_eval_transforms` (`transforms.py:170`), `build_train_transforms` (`transforms.py:214`), and `predict/runner.py:188`. Thread `channel_semantics` (or the derived `use_processor`) through all of these call sites. The predict call site (`predict/runner.py:188`) passes the `channel_semantics` resolved by `_resolve_config` (§11); a `channel_semantics="rgb"` default on the function signature preserves behavior only for any future caller that does not yet supply it.

### 7.2 `max_pixel_value` for non-uint8 multi-band arrays

Both transform builders pass `A.Normalize(..., max_pixel_value=255.0)` (`transforms.py:182`, `transforms.py:276`), assuming uint8 input. Multi-band SAR/height arrays are often float with non-`[0,255]` ranges. v1 requirement: **flag this and propose handling** rather than fully solve it. Recommended handling for the planner to wire:

- Add an optional `max_pixel_value` knob to `NormalizeConfig` (default `255.0`), thread it into `build_eval_transforms`/`build_train_transforms`, and pass to `A.Normalize`.
- Document that for float multi-band input the user sets `max_pixel_value` to match their data range (e.g. `1.0` for already-`[0,1]` data), and that mean/std must be expressed in the same units.

This is the minimum bar: the knob must exist and be documented; deeper per-channel range normalization is out of scope.

---

## 8. Augmentation changes (`src/custom_sam_peft/data/transforms.py`)

The `build_train_transforms` pipeline is fully knob-driven: each Albumentations step is appended only when its resolved knob is `> 0`/enabled. The knobs split into two families:

- **GEOMETRIC** (value-preserving, semantics-agnostic, safe for any N/semantic — they only rearrange space, leaving every per-channel value intact; masks/boxes co-transform correctly): `LongestMaxSize` + `PadIfNeeded` (mandatory resize/pad), `HorizontalFlip` (hflip), `VerticalFlip` (vflip), `RandomRotate90` (rotate90), `Affine` (rotate_arbitrary).
- **VALUE-ALTERING** (assume photometric continuity): `A.ColorJitter` (`transforms.py:263-273`; saturation/hue defined only for 3-channel RGB), `StainJitter` (`transforms.py:289`; HED stain basis, RGB-only — `StainJitter.apply` at `transforms.py:315` assumes a 3-channel image and breaks otherwise), `A.GaussNoise` (additive intensity), `A.GaussianBlur` (spatial averaging — corrupts categorical/measurement channels).

**Locked decision:** the augmentation regime is driven by the **semantic profile** (`profile.photometric`) plus a `channels == 3` test for the 3ch-only transforms — NOT by a bare `channels == 3` vs `!= 3` split. There are now THREE regimes:

### 8.1 Regime 1 — `rgb` (photometric, channels == 3): FULL family, UNCHANGED

`semantic == rgb` ⟹ `photometric AND channels == 3`. All knob-driven steps apply exactly as today, including `A.ColorJitter` (with saturation/hue), `StainJitter`, `A.GaussNoise`, and `A.GaussianBlur` (each appended when its knob is enabled). Byte-for-byte unchanged from today.

### 8.2 Regime 2 — `rgba` / `grayscale` (photometric, channels != 3): substitute brightness/contrast, keep noise/blur/geometry

These semantics are photometric but not 3-channel, so the 3ch-only transforms (sat/hue and `StainJitter`) are impossible, while plain intensity perturbation is still meaningful:

- **Substitute `A.ColorJitter` → `A.RandomBrightnessContrast(brightness_limit=color_jitter, contrast_limit=color_jitter, p=0.5)`** — driven by the same `color_jitter` knob, applied when that knob is `> 0`. This keeps brightness/contrast jitter (channel-agnostic) while dropping the saturation/hue components (RGB-only).
- **KEEP `A.GaussNoise`, `A.GaussianBlur`, and all geometric steps** per their configured knobs.
- **SKIP saturation/hue** (only meaningful for 3ch RGB) and **SKIP `StainJitter`** (HED basis needs exactly 3 channels).
- Emit a **one-time** WARNING (module-level guard, logs once per process) that saturation/hue and `StainJitter` are skipped (RGB-3ch-only) and that brightness/contrast is substituted via `A.RandomBrightnessContrast`.

### 8.3 Regime 3 — `freeform` (non-photometric): GEOMETRY-ONLY

`freeform` has `profile.photometric == False`. We make no assumption about photometric continuity of arbitrary channels, so every value-altering transform is hard-disabled — "a very unassuming basic layer":

- **Apply ONLY the geometric steps** (per their configured knobs): `LongestMaxSize`, `PadIfNeeded`, `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `Affine`. These preserve every per-channel value and only rearrange space; masks/boxes co-transform correctly.
- **HARD-DISABLE all four value-altering augs** — `A.ColorJitter`, `StainJitter`, `A.GaussNoise`, `A.GaussianBlur` — by **omitting the step even when its knob is `> 0`**. This gating **overrides** the knobs: `resolved.color_jitter`/`stain_jitter`/`gauss_noise`/`blur > 0` does NOT append the step for `freeform`.
- **Full freeform augmentation set (explicit, unambiguous):** `LongestMaxSize` (resize) → `PadIfNeeded` (pad) → `HorizontalFlip` → `VerticalFlip` → `RandomRotate90` → `Affine` (whichever flips/rotate90/affine are configured) → `A.Normalize` → `ToTensorV2`. Resize/pad are mandatory; hflip/vflip/rotate90/affine are appended per their knobs; no value-altering step ever appears.
- Emit a **one-time** WARNING (module-level guard) that names all four disabled transforms — `A.ColorJitter`, `StainJitter`, `A.GaussNoise`, `A.GaussianBlur` — and explains that they assume photometric continuity and are therefore skipped for non-photometric (freeform) input; only geometric augmentations apply.

### 8.4 Driving the gating

Drive the three regimes from `profile.photometric` plus a `channels == 3` test, threaded into `build_train_transforms` (pass `channel_semantics` or the resolved profile + `channels`). Decision logic:

```
photometric = profile.photometric
if photometric and channels == 3:        # rgb
    full family (sat/hue + StainJitter + noise + blur + geometry)
elif photometric:                        # rgba / grayscale
    RandomBrightnessContrast(color_jitter) + noise + blur + geometry; warn (sat/hue + StainJitter skipped)
else:                                     # freeform (non-photometric)
    geometry only; warn (all 4 value-altering disabled)
```

`AugmentationsConfig` (`schema.py:214`) itself does not change.

**Float-range interaction (FLAG, do not solve here):** `A.RandomBrightnessContrast` and `A.GaussNoise` are intensity transforms whose magnitude is implicitly tied to the pixel value range. For float multi-band input this interacts with the `max_pixel_value` knob (§7.2). The planner must keep `RandomBrightnessContrast`'s brightness scaling consistent with the §7.2 `max_pixel_value` semantics (e.g. via Albumentations' `brightness_by_max` / range handling) so substituted intensity augs do not misbehave on non-`[0,255]` data. This is flagged for the planner, not solved in this spec.

---

## 9. Model-wiring changes (`src/custom_sam_peft/models/sam3.py`)

1. **`Sam3Wrapper.__init__` (`sam3.py:206`)** — accept `channels: int = 3` and `channel_semantics: str = "rgb"` (or a resolved `ChannelSemanticsProfile`), store them, and pass to the `_Sam3ImageAdapter`. (Whether the adapter is owned by `Sam3Wrapper` or `_Sam3ImageAdapter` is the planner's choice, but §5.1's ownership constraint — adapter outside `_Sam3ImageAdapter.model` — must hold.)
2. **`_Sam3ImageAdapter.__init__` (`sam3.py:339`)** — accept `channels` and `channel_semantics`, look up `CHANNEL_SEMANTICS[channel_semantics]`, and construct `self.channel_adapter` per §5.2/§5.3: `None` when `profile.use_adapter == False` (i.e. `semantic == rgb`); otherwise a `Conv2d(channels, 3, 1)` initialized per `profile.adapter_init` (`average_broadcast` or `identity_passthrough`).
3. **`_Sam3ImageAdapter.forward` (`sam3.py:344`)** — apply `self.channel_adapter(images)` immediately before `self.model.backbone.forward_image(images)` (`sam3.py:360`) when not None.
4. **`_validate_inputs` (`sam3.py:224`)** — relax the hardcoded `(B, 3, H, W)` assumption. Line `sam3.py:230` currently raises `"images must be (B, 3, H, W)"` only on `images.ndim != 4`; it never actually checks channel count. Update to (a) keep the `ndim == 4` check, (b) add `images.shape[1] == channels` with a clear error, and (c) fix the error message to reference the configured `C` rather than literal `3`. The channel check uses `channels` (the count) and is independent of `channel_semantics`. `_validate_inputs` is currently a `@staticmethod`; it must gain access to `channels` (make it an instance method or pass `channels`).
5. **`load_sam31` (`sam3.py:671`)** — accept `channels` and `channel_semantics` and plumb both to the wrapper/adapter constructors (see §5.4 for all call sites).

---

## 10. Cross-cutting touchpoints (adapter trainable-weights requirement)

The adapter is brand-new trainable weights (not LoRA). The locked decision requires it to be (a) added to the optimizer's trainable params, (b) saved/loaded in checkpoints, and (c) included in the exported bundle. Each seam, verified against source:

### 10.1 Optimizer & grad-clip param collection — works automatically (verify with a test)

- `Trainer._build_optimizer` (`trainer.py:215-218`): `trainable = [p for p in self.model.parameters() if p.requires_grad]`.
- Grad-clip in the step loop (`loop.py:355-361`): `[p for p in model.parameters() if p.requires_grad]`.

Because the channel adapter is a registered submodule reachable from `wrapper.parameters()` (see §5.1) and is created with `requires_grad=True`, it is **automatically** collected by both. **No code change is required here, BUT a test must assert the adapter's params appear in the optimizer's param set** (otherwise a future refactor could silently drop them). Caveat for the planner: confirm the adapter is constructed *before* `_build_optimizer` runs and that nothing in the freeze/PEFT path flips its `requires_grad` to False.

### 10.2 Checkpoint save/load — THE GAP; requires new code

This is the load-bearing finding. The checkpoint path (`src/custom_sam_peft/train/checkpoint.py`) persists adapter weights via **PEFT only**:

- `save_adapter` (`checkpoint.py:66`) → `save_lora`/`save_qlora` → `wrapper.peft_model.save_pretrained(...)` (`lora.py:145`, `qlora.py:353`).
- `load_adapter` (`checkpoint.py:76`) → `load_lora`/`load_qlora` → `PeftModel.from_pretrained` / `peft_model.load_adapter(...)`.

`save_pretrained` serializes only the PeftModel's LoRA weights. The channel adapter is a plain `nn.Conv2d` outside the PeftModel, so **it is NOT saved or loaded by the current checkpoint machinery.** Without a fix, a resumed/exported run with a non-rgb semantic silently re-initializes the adapter to its profile init, discarding training.

**Required:** the channel adapter's `state_dict` must be persisted alongside the PEFT adapter and reloaded on resume/export. Recommended approach (planner finalizes the exact mechanism):

- In `save_full_state` (`checkpoint.py:96`) and `save_adapter`/`save_merged`, also dump the channel adapter's `state_dict` to a sibling file in the adapter dir (e.g. `channel_adapter.pt`). Skip the file entirely when there is no adapter (i.e. `semantic == rgb`).
- In `load_full_state` (`checkpoint.py:129`) and `load_adapter`, reload it after `load_adapter` runs. When `semantic == rgb` (no file present), no-op.
- Decide save granularity: it must be written by every code path that writes an adapter checkpoint — `save_full_state` (resume checkpoints), the standalone `save_adapter` (`trainer.py:399`), and `save_merged`. Note `save_merged` (`checkpoint.py:83-93`) dumps `wrapper.model.state_dict()`, which DOES already include the channel adapter (it is a submodule of `wrapper.model`); confirm whether merged exports therefore already carry it — but the non-merged adapter export does not, and resume via `save_full_state`/`load_full_state` does not.
- Round-trip integrity is a GPU-only test (real `state_dict`); see §12.

Edge: the `peft_method` mismatch / `detect_method_from_checkpoint` logic in `load_full_state` (`checkpoint.py:150-170`) is about LoRA vs QLoRA detection and is orthogonal to the channel adapter — do not entangle the channel-adapter reload with that check.

### 10.3 Export bundle — requires the §10.2 fix to flow through

- `run_export` (`runs/bundle.py:47-81`): non-merge path calls `save_adapter` (`bundle.py:79`); merge path calls `save_merged` (`bundle.py:72`). The export reloads via `load_sam31` + `load_adapter` (`bundle.py:67-68`).
- For the exported bundle to be reloadable with a working adapter, `save_adapter`/`load_adapter` must round-trip the channel adapter (the §10.2 fix), and `run_export`'s `load_sam31(cfg.model)` call must construct the adapter with the right `channels` AND `channel_semantics` (it has `cfg`, so pass `cfg.data.channels` and `cfg.data.channel_semantics`).
- The bundle README writers (`bundle.py:352-384`, `bundle.py:494-528`) reference the adapter dir; no semantic change needed beyond ensuring the channel-adapter file ships in that dir. Consider noting the channel adapter in the README text (optional, planner's call).

### 10.4 Summary table

| Seam | File:line | Change needed |
| --- | --- | --- |
| Optimizer params | `trainer.py:217` | None (auto via `requires_grad`); add a test |
| Grad-clip params | `loop.py:358` | None (auto); covered by same test |
| Save (resume) | `checkpoint.py:96` `save_full_state` | Dump `channel_adapter.pt` |
| Load (resume) | `checkpoint.py:129` `load_full_state` | Reload `channel_adapter.pt` |
| Save (standalone adapter) | `checkpoint.py:66` `save_adapter` | Dump channel adapter |
| Load (standalone adapter) | `checkpoint.py:76` `load_adapter` | Reload channel adapter |
| Save (merged) | `checkpoint.py:83` `save_merged` | Likely already included via `wrapper.model.state_dict()`; verify |
| Export | `runs/bundle.py:47` `run_export` | Pass `cfg.data.channels` + `cfg.data.channel_semantics` to `load_sam31`; relies on save/load fix |

---

## 11. Inference path (predict is N-channel-aware — in v1 scope)

The predict path is **fully in scope** for N-channel input in v1. Prediction on non-rgb-semantic / multi-channel images is supported, provided the loaded model's channel count matches the input image's channel count. Three changes are required:

1. **Image reader (`predict/runner.py:385`).** The per-image `_PILImage.open(img_path).convert("RGB")` becomes a channel-aware read using the same logic the training loaders use — call the shared `read_image(img_path, channels)` (or, for the PIL/array core, `_coerce_to_channels`) introduced in §6, with the configured `channels`. This returns an `(H, W, C)` array with `C == channels`. (Reader keys off `channels` only; semantic does not reach it — §6.1.)
2. **Model construction (`predict/runner.py:286`).** `load_sam31(model_cfg)` becomes `load_sam31(model_cfg, channels=..., channel_semantics=...)` so the correct adapter is constructed (per §5.2/§5.3, the semantic decides whether an adapter exists and how it inits; a wrong-channel adapter would otherwise be built and the forward would reject the input).
3. **Normalization (`predict/runner.py:188`).** The `resolve_normalization(model_name, NormalizeConfig())` call passes `channel_semantics` (or the derived `use_processor`); for non-rgb semantics this skips `AutoImageProcessor` and uses the config-provided per-channel stats directly, per §7.1.

**Source of `channels` and `channel_semantics` (plumbing — verified against source).** The predict path resolves its config in `_resolve_config` (`predict/runner.py:110`), which parses the `--config` YAML and currently extracts only `model.name` (`predict/runner.py:132-136`) and `data.image_size` (`predict/runner.py:137-141`). It does **not** read `data.channels` or `data.channel_semantics` today, and it constructs a *bare* `ModelConfig(name=rcfg.model_name)` (`predict/runner.py:284`) rather than carrying the full config object. The exported bundle ships a config that carries both `data.channels` and `data.channel_semantics` (the fields added in §3.1/§3.2). **Minimal plumbing:** in `_resolve_config`, read BOTH `data.channels` (default 3 when absent) AND `data.channel_semantics` (default `"rgb"` when absent) from the parsed YAML's `data` section, mirroring the existing `image_size` extraction; store them on `_ResolvedConfig` (add `channels: int` and `channel_semantics: str` fields at `predict/runner.py:96-102`), and thread them into the three call sites above. Both are therefore read once in `_resolve_config` and flow through `rcfg` to the reader (`channels`), `load_sam31` (`channels` + `channel_semantics`), and `resolve_normalization` (`channel_semantics`).

**Correctness contract (input must fit the model's channel count — enforced by construction, no separate guard).** The model's adapter is built for the bundle's `data.channels`/`data.channel_semantics`; predict reads images at that *same* `channels`; and `read_image`'s `C == channels` validation (§6.1) rejects any input image whose channel count differs from the configured `channels` with a clear error. Because the reader and the adapter are driven by the *same* `channels`/`channel_semantics` values, "the input must fit the model's channel count" is guaranteed by construction plus the reader's validation — there is **no separate fail-fast guard and no silent-mispredict path.** (The model's own `_validate_inputs` channel check from §9 is a redundant backstop, not the primary enforcement.)

---

## 12. Testing matrix (CPU-first; standing project policy)

Push every CPU-testable case to CPU; reserve GPU for real-only failure modes (real SAM 3.1, real `state_dict`, peak VRAM).

### 12.1 CPU tests

| # | Case | Assertion |
| --- | --- | --- |
| C1 | Adapter init = average_broadcast | A fresh `average_broadcast` `Conv2d(N,3,1)` adapter has `weight == 1/N` everywhere and `bias == 0`; output for a known input equals the per-pixel mean over N channels, broadcast to 3. |
| C2 | grayscale triplication identity | For `channel_semantics="grayscale"` (`channels=1`), the `average_broadcast` adapter output equals `torch.cat([x, x, x], dim=1)` exactly (triplication identity). |
| C3 | passthrough keyed on `semantic == rgb` | For `channel_semantics="rgb"` (`channels=3`), no adapter is constructed (`channel_adapter is None`); **zero** new params vs. the pre-feature param count; forward output identical to the RGB path. **Paired case:** `channel_semantics="freeform", channels=3` DOES construct a learned `Conv2d(3,3,1)` adapter (non-`None`, **non-zero** new params, `average_broadcast` init) — confirming passthrough is keyed on the semantic, not on `channels == 3`. |
| C3b | rgba identity_passthrough init | For `channel_semantics="rgba"` (`channels=4`), the adapter's `identity_passthrough` init has `weight[o,o]=1` for `o ∈ {0,1,2}`, the alpha column (`i=3`) all zeros, `bias=0`; step-0 output for a known input equals the input's first 3 (RGB) channels exactly, alpha dropped. |
| C4 | Config: channels range + semantic membership | `data.channels` accepts 1 and 16, rejects 0 and 17; `data.channel_semantics` accepts the four registry keys and rejects an unknown value. |
| C5 | Config: semantic↔channels + normalize cross-checks | Semantic↔channels enforced (`rgb⟹3`, `rgba⟹4`, `grayscale⟹1`, `freeform⟹1..16`; mismatches like `rgba`+`channels=3` rejected with a clear error). `len(mean)==len(std)==channels` enforced. `freeform` without explicit `normalize` is rejected (loud error). For `rgb`/`rgba`/`grayscale` the profile default fills in when `normalize` is omitted and validates (rgb→ImageNet-3, rgba→ImageNet-3+`[0.5]` len-4, grayscale→`[0.449]`/`[0.226]` len-1). |
| C6 | Reader dispatch + C-validation | `read_image` returns `(H,W,C)` with correct C for PIL (`L`/`RGB`/`RGBA`), `.npy`/`.npz` (HWC and CHW), and `.tif` multi-band; mismatched channel count raises. (Reader keyed on `channels` only; semantic not involved.) |
| C7 | THREE augmentation regimes | `build_train_transforms` produces: (a) `rgb` (`channels=3`) → FULL family (`A.ColorJitter` present when knob `> 0`, plus `StainJitter`/`GaussNoise`/`GaussianBlur` if configured); (b) `rgba`/`grayscale` → pipeline contains `A.RandomBrightnessContrast` (substituted from the `color_jitter` knob), `A.GaussNoise`, `A.GaussianBlur`, geometry, and does **NOT** contain `A.ColorJitter` or `StainJitter`; the one-time "sat/hue + StainJitter skipped, brightness/contrast substituted" warning logs; (c) `freeform` → only geometric steps (resize/pad + enabled flips/rotate90/affine), and does **NOT** contain `A.ColorJitter`, `StainJitter`, `A.GaussNoise`, or `A.GaussianBlur` **even when their knobs are `> 0`**; the one-time "all four value-altering disabled" warning logs. |
| C8 | `_validate_inputs` channel check | Accepts `(B, C, H, W)` for the configured C; rejects wrong C and non-4-D with clear messages. (Driven by `channels`, semantic-independent.) |
| C9 | resolve_normalization consults processor only for `rgb` | With `channel_semantics="rgb"`, `resolve_normalization` consults `AutoImageProcessor` (existing 3-step resolution). With `rgba`/`grayscale`/`freeform`, it returns the config stats via the config-fallback path WITHOUT consulting `AutoImageProcessor`. |
| C10 | Optimizer collects adapter params (mockable) | The trainable param set passed to the optimizer includes the channel adapter's weight+bias when an adapter exists (non-rgb semantic; can be exercised CPU-only with a lightweight stand-in wrapper if a full real model is not needed). |
| C11 | Predict reader returns correct C | The predict image-read path (`predict/runner.py:385`, via `read_image`/`_coerce_to_channels` with the configured `channels`) returns `(H, W, C)` with `C == channels`; a mismatched input image raises the clear `read_image` channel error. |
| C12 | Predict `_resolve_config` reads channels + semantics | `_resolve_config` parses BOTH `data.channels` (default 3) AND `data.channel_semantics` (default `"rgb"`) from the `--config` YAML and exposes them on `_ResolvedConfig`; the resolved values reach `load_sam31` (channels + semantics) and `resolve_normalization` (semantics). |
| C13 | Predict resolves normalization without AutoImageProcessor for non-rgb | The predict normalization step (`predict/runner.py:188`) with a non-rgb semantic returns the config-provided per-channel stats via the config-fallback path without consulting `AutoImageProcessor`. |

### 12.2 GPU-only tests

| # | Case | Why GPU-only |
| --- | --- | --- |
| G1 | Real SAM 3.1 forward through the adapter | Requires the real backbone/patch-embed. Assert an N-channel batch flows end-to-end and the adapter feeds 3 channels into `forward_image`. |
| G2 | Checkpoint round-trip of adapter weights | Train the adapter a few steps, `save_full_state` → `load_full_state`, assert the channel-adapter weights are restored bit-for-bit inside a real `state_dict`. |
| G3 | Export-bundle reload with adapter | `run_export` then reload via `load_sam31` + `load_adapter`; assert the reloaded adapter weights match. |
| G4 | Real-model N-channel predict forward | Requires the real backbone. Run `run_predict` end-to-end on a non-rgb-semantic / multi-channel image with a matching model/bundle; assert the predict reader feeds N channels through the adapter into `forward_image` and produces predictions without error. |

---

## 13. Open risks

1. **Checkpoint gap (highest risk).** §10.2 — the PEFT-only checkpoint path silently omits the channel adapter today. If the §10.2 fix is incomplete, runs with a non-rgb semantic lose their adapter on resume/export with no error. The GPU round-trip tests (G2/G3) are the guard; ensure they fail before the fix and pass after.
2. **`load_sam31` signature change blast radius.** Seven call sites (§5.4), all plumbed with real `channels` AND `channel_semantics` values. `predict/runner.py` reads both from the bundle config in `_resolve_config` (§11). Only `calibrate_cmd.py` may need `3`/`"rgb"` fallbacks if its config genuinely lacks a `data` section; document if so.
3. **Semantic ↔ channels / stats misconfiguration.** The decoupling of count and semantics introduces a new misconfiguration surface: a semantic that does not match the channel count (e.g. `rgba` with `channels=3`), or a `freeform` config that omits the required explicit per-channel stats. Mitigated by the §3.3 cross-validation (semantic↔channels match + normalize-length check) and the §3.3c `freeform`-requires-explicit-stats loud error, all enforced at config-load time before any run starts.
4. **Predict input/model channel-count match (correctness note, not an open risk).** §11 — predict requires the input image's channel count to equal the model's trained `data.channels`. This is enforced by construction: the channel adapter and the image reader are both driven by the *same* `channels`/`channel_semantics` values (read from the bundle config), and `read_image`'s `C == channels` validation rejects mismatched inputs with a clear error. There is **no silent-mispredict path** and no separate guard to maintain. (Listed here only to make the invariant explicit; it is not a residual risk.)
5. **`max_pixel_value` / float-range interaction for intensity augs.** §7.2 ships only a knob + docs, not full range normalization; float SAR/height ranges remain a sharp edge for users who misconfigure. §8.4 additionally flags that the `rgba`/`grayscale` substituted `RandomBrightnessContrast` and the retained `GaussNoise` must keep their magnitude consistent with `max_pixel_value` for non-`[0,255]` float data.
6. **Augmentation regime expectations (three-regime model).** §8 — augmentation is now driven by the semantic profile, not a bare `channels==3` split. `rgb` keeps the full family; `rgba`/`grayscale` substitute `RandomBrightnessContrast` and keep noise/blur/geometry while dropping sat/hue + `StainJitter`; `freeform` is geometry-only with all four value-altering augs hard-disabled (knob-overriding). A `freeform` user expecting noise/blur will find it disabled, and an `rgba`/`grayscale` user loses sat/hue + stain — both mitigated by the one-time warnings and the §14 follow-up (opt-in channel-aware intensity augmentation for freeform).
7. **HF reader reconciliation.** §6.3 — HF rows carry decoded objects/arrays, not always file paths, so a path-based `read_image` may not fit the HF branch directly; a shared `_coerce_to_channels` core is recommended to avoid divergent channel logic between the two loaders.

---

## 14. Out-of-scope / follow-up issues to file

1. **Bridge B**: replace the SAM 3.1 patch-embed with an `in_chans=N` stem for large-N / true hyperspectral (>16 channels).
2. **GDAL/rasterio georeferencing**: geospatial CRS, clinical DICOM metadata.
3. **Per-channel learned augmentation.**
4. **Channel-count auto-detection** (deliberately rejected for v1; record the decision in the issue).
5. **Opt-in intensity/photometric augmentation for `freeform` data that tolerates it** (channel-aware noise/blur/brightness), re-enabling the value-altering family for users who confirm their channels are photometric-continuous.

> **Note (not a follow-up — an extensibility property of this design):** new channel semantics are added by adding ONE entry to the `CHANNEL_SEMANTICS` registry (§1.3), not by reopening this feature. The adapter, normalize, and augmentation logic read the profile (`use_adapter`, `adapter_init`, `photometric`, `normalize_default`, `allowed_channels`), so a new semantic needs no new branches in those code paths — only a registry entry and (if the config `Literal` is hand-maintained rather than derived from the registry keys) an added literal value.

---

## Appendix A — Verified anchors (worktree HEAD `91bbe3d`)

| Symbol / site | Original (brainstorm) | Verified (this worktree) |
| --- | --- | --- |
| COCO `convert("RGB")` | `coco.py:176` | `coco.py:213` (`_decode_image`) |
| HF `convert("RGB")` | `hf.py:171` | `hf.py:213`; array branch `hf.py:214-217` |
| `DataConfig` | `schema.py:154` | `schema.py:368` |
| `data.channel_semantics` (NEW field) | — | NEW — no existing anchor; added to `DataConfig` (`schema.py:368`) per §3.2; backed by new `CHANNEL_SEMANTICS` registry in a new `src/custom_sam_peft/data/channel_semantics.py` (planner's final call on location) |
| `NormalizeConfig` length fields | `schema.py:111-116` | `schema.py:270-275` |
| `AugmentationsConfig` | `schema.py:57` | `schema.py:214` |
| `resolve_normalization` | `transforms.py:57` | `transforms.py:147` (and `_with_path` at `:84`) |
| `build_train_transforms` ColorJitter | `transforms.py:187-195` | `transforms.py:263-273` |
| `A.Normalize` max_pixel_value | — | `transforms.py:182`, `transforms.py:276` |
| `_validate_inputs` | `sam3.py:209`, `(B,3,H,W)` at `:216` | `sam3.py:224`; message at `:230` (note: only checks `ndim`, not channel count) |
| Wrapper `__init__` | `sam3.py:192` | `sam3.py:206` |
| `_Sam3ImageAdapter.forward` | `sam3.py:287+` | class `sam3.py:322`, `forward` `sam3.py:344`, patch-embed call `sam3.py:360` |
| `load_sam31` | — | `sam3.py:671` (takes `ModelConfig` only) |
| Optimizer param collection | — | `trainer.py:217`; grad-clip `loop.py:358` |
| Checkpoint save/load | — | `checkpoint.py:66/76/96/129`; PEFT `save_pretrained` at `lora.py:145`, `qlora.py:353` |
| Export | — | `runs/bundle.py:47` (`run_export`), `bundle.py:67-79` |
| `pyproject` dependencies | — | `pyproject.toml:9-26` |
