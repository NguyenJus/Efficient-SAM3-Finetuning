# N-channel Input Support (1–16 channels) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support any input channel count `1 ≤ N ≤ 16` end-to-end (ingestion → normalize → augment → model forward → checkpoint → export → predict) via a learned N→3 channel adapter, with the default `channel_semantics="rgb"` path byte-for-byte unchanged.

**Architecture:** A new `channel_semantics` registry (`ChannelSemanticsProfile` dataclass + `CHANNEL_SEMANTICS` dict) decouples channel *count* from channel *semantics*. The semantic profile drives whether a `nn.Conv2d(N,3,1)` channel adapter is built (only `rgb` skips it), how it inits (`average_broadcast` / `identity_passthrough`), the normalize default, and the augmentation regime. The adapter is a sibling submodule of `_Sam3ImageAdapter.model`, so it is reached by `wrapper.parameters()` and `wrapper.model.state_dict()` but NOT by PEFT's `save_pretrained` — requiring an explicit `channel_adapter.pt` save/load fix.

**Tech Stack:** PyTorch, Pydantic v2, Albumentations, PIL, tifffile (new dep), pytest. Source spec: `docs/superpowers/specs/2026-05-23-n-channel-input-design.md` (anchors verified against worktree HEAD `91bbe3d`).

---

## Anchor verification notes (read before starting)

All spec Appendix-A anchors were re-verified against HEAD `91bbe3d` and are correct, with these clarifications the implementer must honor:

- **Class name correction:** the COCO dataset class is `COCODataset` (spec §6.3 / §5.1 write `CocoDataset`); the HF class is `HFDataset`. Use the real names.
- **Dataset plumbing path:** `cfg.data.model_dump()` feeds `build_coco`/`build_hf` as a dict (`train/runner.py:75,96`; `eval/runner.py` similar). Adding `channels`/`channel_semantics` as `DataConfig` fields makes them appear automatically in that dict — the builders read them from the dict.
- **`load_sam31` call-site config availability:** `run_cmd.py:91`, `eval/runner.py:130`, `train/runner.py:114`, `runs/bundle.py:67` all have a `TrainConfig` (`cfg`) in scope → pass `cfg.data.channels`, `cfg.data.channel_semantics`. `predict/runner.py:286` resolves both via `_resolve_config` (Task 14). `calibrate_cmd.py:76` (`_run_probe`) has **only** `image_size` and builds a bare `ModelConfig()` for a VRAM probe — it has no `DataConfig`. Pass the defaults `channels=3, channel_semantics="rgb"`; this is the documented exception (spec §5.4, risk #2). Document it inline.

## OPEN QUESTIONS (spec gaps flagged during planning — resolve before/at the relevant task)

1. **`§8.4 float-range interaction for substituted intensity augs` is flagged, not specified.** The spec (§7.2, §8.4, risk #5) requires `RandomBrightnessContrast` (rgba/grayscale substitute) and the retained `GaussNoise` to keep magnitude consistent with the `§7.2 max_pixel_value` semantics, but does NOT specify the exact Albumentations mechanism. **Plan decision (Task 12):** wire `max_pixel_value` into both transform builders and pass it to `A.Normalize`; for `A.RandomBrightnessContrast` set `brightness_by_max=False` so brightness/contrast scale relative to the actual data range rather than a hardcoded 255. This is the minimum bar the spec mandates (knob exists + documented); deeper per-channel range normalization is out of scope. **If the reviewer disagrees with `brightness_by_max=False`, escalate per the Design-Ambiguity ladder.**
2. **`save_merged` already-carries-the-adapter claim needs runtime confirmation (spec §10.2/§10.4).** The spec asserts `save_merged` likely already persists the channel adapter because it dumps `wrapper.model.state_dict()` (the adapter is a submodule of `wrapper.model`). Task 13 adds a `channel_adapter.pt` to `save_merged` anyway for symmetry with the load path, and Task 22 (GPU G3) is the runtime confirmation. If the GPU test shows the merged `state_dict` ALREADY round-trips the adapter and the extra file is redundant/harmful, drop it from `save_merged` only. Flagged so the implementer does not silently assume.
3. **Adapter ownership (`Sam3Wrapper` vs `_Sam3ImageAdapter`).** Spec §9.1 leaves this to the planner but mandates the §5.1 constraint (adapter OUTSIDE `_Sam3ImageAdapter.model`). **Plan decision:** own it on `_Sam3ImageAdapter` as `self.channel_adapter` (sibling of `self.model`). This keeps the forward call-site (§5.1) local and satisfies the ownership constraint. `Sam3Wrapper.__init__` still accepts + forwards `channels`/`channel_semantics` so the public constructor signature is uniform.

---

## File Structure

**New files:**
- `src/custom_sam_peft/data/channel_semantics.py` — `ChannelSemanticsProfile` frozen dataclass + `CHANNEL_SEMANTICS` registry + `CHANNEL_SEMANTIC_NAMES` literal-source. (Tasks 1)
- `src/custom_sam_peft/data/io.py` — `read_image(path, channels)` + shared `_coerce_to_channels(array_or_pil, channels)` core. (Task 7)
- `tests/unit/test_channel_semantics.py` — registry/profile tests.
- `tests/unit/test_channel_adapter.py` — adapter init/forward tests (C1, C2, C3, C3b).
- `tests/unit/test_data_io.py` — `read_image` dispatch tests (C6).
- `tests/gpu/test_channel_adapter_gpu.py` — G1, G2, G3 (real model / state_dict / export).
- `tests/gpu/test_predict_nchannel_gpu.py` — G4.

**Modified files:**
- `src/custom_sam_peft/config/schema.py` — `NormalizeConfig` relax + `max_pixel_value`; `DataConfig` `channels`/`channel_semantics` fields + cross-validation.
- `src/custom_sam_peft/models/sam3.py` — adapter construction/forward, `_validate_inputs`, `load_sam31` signature.
- `src/custom_sam_peft/data/transforms.py` — `resolve_normalization*` semantic-skip, `build_*_transforms` three-regime gating + `max_pixel_value`.
- `src/custom_sam_peft/data/coco.py`, `src/custom_sam_peft/data/hf.py` — reader wiring + `channels` plumbing through `COCODataset`/`HFDataset`/`build_coco`/`build_hf`.
- `src/custom_sam_peft/train/checkpoint.py` — `channel_adapter.pt` save/load in `save_full_state`/`load_full_state`/`save_adapter`/`load_adapter`/`save_merged`.
- `src/custom_sam_peft/cli/run_cmd.py`, `cli/calibrate_cmd.py`, `eval/runner.py`, `runs/bundle.py`, `train/runner.py`, `predict/runner.py` — `load_sam31` call sites.
- `pyproject.toml` — add `tifffile`.

---

## Landing order rationale (TDD + zero-regression-first)

The spec's hard requirement is **zero regression for `channel_semantics="rgb"`** (§1, §5.2). The plan front-loads the proof of inertness:

1. **Tasks 1–6 (config + registry):** add the registry and config fields. These are additive — `rgb` default validates exactly as today. A regression test (Task 6) asserts existing example configs still load unchanged.
2. **Tasks 7–8 (reader):** `read_image` with `channels=3` returns the same array `convert("RGB")` did. Loaders keep RGB behavior.
3. **Tasks 9–11 (adapter + model wiring):** `rgb` ⟹ `channel_adapter is None`, forward identical, zero new params. The FIRST adapter test (Task 9 C3) proves inertness before any non-rgb adapter exists.
4. **Tasks 12 (transforms):** `rgb` keeps the full augmentation family + processor-consulted normalize unchanged; non-rgb regimes layer on top.
5. **Tasks 13 (checkpoint gap fix):** depends on the adapter existing (Task 10).
6. **Tasks 14 (predict):** depends on reader (Task 7) + adapter wiring (Task 11) + normalize skip (Task 12).
7. **Tasks 15 (load_sam31 call sites):** threads the now-existing signature through all 7 sites.
8. **Tasks 16–22 (cross-cutting tests + GPU):** optimizer-collection test, then GPU-gated real-model/round-trip/export/predict.

**Cross-file dependency invariants:**
- Registry (Task 1) must land before config validation (Tasks 3–5) and adapter construction (Task 10) — both read the profile.
- The adapter submodule (Task 10) must exist before checkpoint round-trip (Task 13) and the optimizer-collection test (Task 16).
- `read_image` / `_coerce_to_channels` (Task 7) must exist before both loaders (Task 8) and the predict reader (Task 14).
- `load_sam31`'s new signature (Task 11) must exist before the call-site sweep (Task 15) — but Task 11 adds defaulted params (`channels=3, channel_semantics="rgb"`) so existing callers keep working until Task 15.

---

## Task 1: Channel-semantics registry

**Files:**
- Create: `src/custom_sam_peft/data/channel_semantics.py`
- Test: `tests/unit/test_channel_semantics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_channel_semantics.py
import pytest

from custom_sam_peft.data.channel_semantics import (
    CHANNEL_SEMANTICS,
    CHANNEL_SEMANTIC_NAMES,
    ChannelSemanticsProfile,
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def test_registry_has_four_shipped_semantics():
    assert set(CHANNEL_SEMANTICS) == {"rgb", "rgba", "grayscale", "freeform"}
    assert set(CHANNEL_SEMANTIC_NAMES) == set(CHANNEL_SEMANTICS)


def test_rgb_profile_passthrough_imagenet():
    p = CHANNEL_SEMANTICS["rgb"]
    assert p.allowed_channels == {3}
    assert p.use_adapter is False
    assert p.photometric is True
    assert p.normalize_default == (_IMAGENET_MEAN, _IMAGENET_STD)


def test_rgba_profile_identity_passthrough_alpha_default():
    p = CHANNEL_SEMANTICS["rgba"]
    assert p.allowed_channels == {4}
    assert p.use_adapter is True
    assert p.adapter_init == "identity_passthrough"
    assert p.photometric is True
    mean, std = p.normalize_default
    assert mean == _IMAGENET_MEAN + [0.5]
    assert std == _IMAGENET_STD + [0.5]


def test_grayscale_profile_luminance_default():
    p = CHANNEL_SEMANTICS["grayscale"]
    assert p.allowed_channels == {1}
    assert p.use_adapter is True
    assert p.adapter_init == "average_broadcast"
    assert p.photometric is True
    assert p.normalize_default == ([0.449], [0.226])


def test_freeform_profile_no_default_range_channels():
    p = CHANNEL_SEMANTICS["freeform"]
    assert set(p.allowed_channels) == set(range(1, 17))
    assert p.use_adapter is True
    assert p.adapter_init == "average_broadcast"
    assert p.photometric is False
    assert p.normalize_default is None


def test_profile_is_frozen():
    with pytest.raises(Exception):
        CHANNEL_SEMANTICS["rgb"].use_adapter = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "semantic,channel,ok",
    [("rgb", 3, True), ("rgb", 4, False), ("rgba", 4, True), ("rgba", 3, False),
     ("grayscale", 1, True), ("grayscale", 3, False), ("freeform", 1, True),
     ("freeform", 16, True), ("freeform", 17, False)],
)
def test_allowed_channels_membership(semantic, channel, ok):
    assert (channel in CHANNEL_SEMANTICS[semantic].allowed_channels) is ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_channel_semantics.py -q`
Expected: FAIL — `ModuleNotFoundError: custom_sam_peft.data.channel_semantics`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/custom_sam_peft/data/channel_semantics.py
"""Channel-semantics registry: decouples channel COUNT from channel SEMANTICS.

The semantic profile (NOT the count) drives whether a channel adapter is built,
how it is initialized, the normalize default, and the augmentation regime.
Add a new semantic by adding ONE entry here (spec §1.3, §14 note); the adapter,
normalize, and augmentation logic read the profile flags, never a hardcoded name.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ChannelSemanticsProfile:
    """Per-semantic treatment profile. See spec §1.2 / §1.3."""

    allowed_channels: Collection[int]
    use_adapter: bool  # False only for rgb (passthrough)
    adapter_init: Literal["average_broadcast", "identity_passthrough"]
    photometric: bool  # True for rgb/rgba/grayscale; False for freeform
    # (mean, std) tuple, or None when explicit stats are required (freeform).
    normalize_default: tuple[list[float], list[float]] | None


CHANNEL_SEMANTICS: dict[str, ChannelSemanticsProfile] = {
    "rgb": ChannelSemanticsProfile(
        allowed_channels=frozenset({3}),
        use_adapter=False,
        adapter_init="average_broadcast",
        photometric=True,
        normalize_default=(list(_IMAGENET_MEAN), list(_IMAGENET_STD)),
    ),
    "rgba": ChannelSemanticsProfile(
        allowed_channels=frozenset({4}),
        use_adapter=True,
        adapter_init="identity_passthrough",
        photometric=True,
        normalize_default=(_IMAGENET_MEAN + [0.5], _IMAGENET_STD + [0.5]),
    ),
    "grayscale": ChannelSemanticsProfile(
        allowed_channels=frozenset({1}),
        use_adapter=True,
        adapter_init="average_broadcast",
        photometric=True,
        normalize_default=([0.449], [0.226]),
    ),
    "freeform": ChannelSemanticsProfile(
        allowed_channels=range(1, 17),
        use_adapter=True,
        adapter_init="average_broadcast",
        photometric=False,
        normalize_default=None,
    ),
}

# Tuple of the registry keys, for the schema Literal and membership checks.
CHANNEL_SEMANTIC_NAMES: tuple[str, ...] = tuple(CHANNEL_SEMANTICS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_channel_semantics.py -q`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/channel_semantics.py tests/unit/test_channel_semantics.py
git commit -m "feat(data): add channel-semantics registry (spec §1.3)"
```

---

## Task 2: Add `tifffile` dependency

**Files:**
- Modify: `pyproject.toml:9-27` (dependencies list)

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml` `[project].dependencies` — add after the `pillow>=10` line:

```toml
  "tifffile>=2024.1",
```

- [ ] **Step 2: Refresh the lockfile**

Run: `uv lock`
Expected: `uv.lock` updated with a `tifffile` entry; exit 0.

- [ ] **Step 3: Verify it imports**

Run: `uv run python -c "import tifffile; print(tifffile.__version__)"`
Expected: prints a version string `>= 2024.1`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add tifffile runtime dependency for multi-band TIFF reads (spec §6.2)"
```

---

## Task 3: `NormalizeConfig` length relax + `max_pixel_value` knob

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py:270-275` (mean/std fields), `:248` (class)
- Test: `tests/unit/test_config_schema.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config_schema.py  (append)
from custom_sam_peft.config.schema import NormalizeConfig


def test_normalize_accepts_length_one_and_sixteen():
    NormalizeConfig(mean=[0.5], std=[0.2])
    NormalizeConfig(mean=[0.5] * 16, std=[0.2] * 16)


def test_normalize_rejects_length_zero_and_seventeen():
    import pytest

    with pytest.raises(Exception):
        NormalizeConfig(mean=[], std=[])
    with pytest.raises(Exception):
        NormalizeConfig(mean=[0.5] * 17, std=[0.2] * 17)


def test_normalize_max_pixel_value_default_and_override():
    assert NormalizeConfig().max_pixel_value == 255.0
    assert NormalizeConfig(max_pixel_value=1.0).max_pixel_value == 1.0


def test_normalize_keeps_per_value_range_checks():
    import pytest

    with pytest.raises(ValueError, match=r"normalize\.mean values must be in"):
        NormalizeConfig(mean=[1.5], std=[0.2])
    with pytest.raises(ValueError, match=r"normalize\.std values must be > 0"):
        NormalizeConfig(mean=[0.5], std=[0.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config_schema.py -q -k "normalize"`
Expected: FAIL — length-1 rejected (`min_length=3`); `max_pixel_value` attribute missing.

- [ ] **Step 3: Edit the fields**

In `schema.py`, replace the `mean`/`std` field declarations (`:270-275`) with relaxed bounds and add `max_pixel_value`:

```python
    mean: list[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406], min_length=1, max_length=16
    )
    std: list[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225], min_length=1, max_length=16
    )
    max_pixel_value: float = Field(
        default=255.0,
        gt=0.0,
        description=(
            "Divisor applied by A.Normalize before subtracting mean / dividing by "
            "std. Default 255.0 assumes uint8 input. For float multi-band input "
            "(e.g. SAR/height already in [0,1]), set this to your data's max (e.g. "
            "1.0); mean/std must be expressed in the same units. See spec §7.2."
        ),
    )
```

The `_check_ranges` validator (`:277-285`) is unchanged — it already iterates arbitrary-length lists.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config_schema.py -q -k "normalize"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_config_schema.py
git commit -m "feat(config): relax NormalizeConfig length to 1..16 + add max_pixel_value knob (spec §4,§7.2)"
```

---

## Task 4: `DataConfig.channels` + `channel_semantics` fields (no cross-validation yet)

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py:368` (`DataConfig`), import block (`:19`)
- Test: `tests/unit/test_data_schema_extensions.py` (append)

This task adds the FIELDS only (membership/range), so the `rgb` default is provably valid before cross-validation is layered. Cross-field checks come in Task 5.

- [ ] **Step 1: Write the failing test (C4 — range + membership)**

```python
# tests/unit/test_data_schema_extensions.py  (append)
import pytest

from custom_sam_peft.config.schema import DataConfig


def _minimal_data(**kw):
    base = dict(
        format="coco",
        train={"annotations": "a.json", "images": "imgs"},
        prompt_mode="text",
    )
    base.update(kw)
    return DataConfig.model_validate(base)


def test_channels_defaults_to_three_and_semantic_rgb():
    d = _minimal_data()
    assert d.channels == 3
    assert d.channel_semantics == "rgb"


def test_channels_accepts_1_and_16_rejects_0_and_17():
    _minimal_data(channels=1, channel_semantics="grayscale")
    _minimal_data(channels=16, channel_semantics="freeform",
                  normalize={"mean": [0.5] * 16, "std": [0.2] * 16})
    with pytest.raises(Exception):
        _minimal_data(channels=0)
    with pytest.raises(Exception):
        _minimal_data(channels=17)


def test_channel_semantics_membership():
    with pytest.raises(Exception):
        _minimal_data(channel_semantics="hyperspectral")
```

(Note: the `channels=1/16` cases also exercise Task-5 cross-validation; they will fully pass only after Task 5. For THIS task, narrow the run to membership/range with `-k "membership or defaults or rejects"`. The full set greens at Task 5.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_data_schema_extensions.py -q -k "channel_semantics_membership or channels_defaults"`
Expected: FAIL — `channels`/`channel_semantics` not fields of `DataConfig`.

- [ ] **Step 3: Add the fields**

In `schema.py`, ensure the registry literal is importable. Add near the top imports (`after :31`):

```python
from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTICS
```

In `DataConfig` (`:368`), add after `format: DataFormat` (keep alongside `image_size`):

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

Also change `normalize` to a sentinel for Task 5's precedence logic (spec §3.3 note):

```python
    normalize: NormalizeConfig | None = None
```

(The old default was `Field(default_factory=NormalizeConfig)`. The materialization of the per-semantic default moves into Task 5's validator. Until Task 5 lands, downstream code that reads `cfg.data.normalize` may see `None` — Task 5 lands in the SAME PR before any green claim, and Task 6 regression-tests existing configs.)

- [ ] **Step 4: Run test to verify it passes (membership/defaults subset)**

Run: `uv run pytest tests/unit/test_data_schema_extensions.py -q -k "channel_semantics_membership or channels_defaults"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_data_schema_extensions.py
git commit -m "feat(config): add data.channels + data.channel_semantics fields (spec §3.1,§3.2)"
```

---

## Task 5: `DataConfig` cross-field validation (semantic↔channels, normalize default/required, length)

**Files:**
- Modify: `src/custom_sam_peft/config/schema.py` (`DataConfig`, add `@model_validator(mode="after")`)
- Test: `tests/unit/test_data_schema_extensions.py` (append — C5)

- [ ] **Step 1: Write the failing test (C5)**

```python
# tests/unit/test_data_schema_extensions.py  (append)
def test_semantic_channels_mismatch_rejected():
    with pytest.raises(ValueError, match=r"channel_semantics='rgba' requires .*channels=4"):
        _minimal_data(channels=3, channel_semantics="rgba")
    with pytest.raises(ValueError, match=r"channel_semantics='grayscale' requires .*channels=1"):
        _minimal_data(channels=3, channel_semantics="grayscale")


def test_rgb_default_fills_imagenet_when_omitted():
    d = _minimal_data()  # rgb, channels=3, no normalize
    assert d.normalize.mean == [0.485, 0.456, 0.406]
    assert d.normalize.std == [0.229, 0.224, 0.225]


def test_rgba_default_fills_imagenet_plus_alpha_len4():
    d = _minimal_data(channels=4, channel_semantics="rgba")
    assert d.normalize.mean == [0.485, 0.456, 0.406, 0.5]
    assert len(d.normalize.mean) == 4 == len(d.normalize.std)


def test_grayscale_default_fills_luminance_len1():
    d = _minimal_data(channels=1, channel_semantics="grayscale")
    assert d.normalize.mean == [0.449]
    assert d.normalize.std == [0.226]


def test_freeform_without_explicit_stats_rejected():
    with pytest.raises(ValueError, match=r"channel_semantics='freeform' requires explicit"):
        _minimal_data(channels=5, channel_semantics="freeform")


def test_freeform_with_explicit_stats_ok():
    d = _minimal_data(
        channels=5, channel_semantics="freeform",
        normalize={"mean": [0.1, 0.2, 0.3, 0.4, 0.5], "std": [0.1] * 5},
    )
    assert len(d.normalize.mean) == 5


def test_normalize_length_must_match_channels():
    with pytest.raises(ValueError, match=r"normalize\.mean has 3 entries but .*channels=5"):
        _minimal_data(
            channels=5, channel_semantics="freeform",
            normalize={"mean": [0.1, 0.2, 0.3], "std": [0.1, 0.2, 0.3]},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_data_schema_extensions.py -q -k "semantic_channels or default_fills or freeform or length_must_match"`
Expected: FAIL — no cross-validation yet; `normalize` is `None`.

- [ ] **Step 3: Add the validator**

In `DataConfig`, add (place AFTER `_check_format_specific` so registry lookups run on a fully-built model):

```python
    @model_validator(mode="after")
    def _check_channels_semantics_normalize(self) -> DataConfig:
        from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTICS

        profile = CHANNEL_SEMANTICS[self.channel_semantics]

        # (a) semantic <-> channels match
        if self.channels not in profile.allowed_channels:
            allowed = sorted(profile.allowed_channels)
            allowed_str = (
                f"{allowed[0]}" if len(allowed) == 1 else f"{allowed[0]}..{allowed[-1]}"
            )
            raise ValueError(
                f"data.channel_semantics={self.channel_semantics!r} requires "
                f"data.channels={allowed_str}, but data.channels={self.channels}."
            )

        # (c) resolve normalize: explicit wins; else profile default; freeform requires explicit.
        if self.normalize is None:
            if profile.normalize_default is None:
                raise ValueError(
                    f"data.channel_semantics={self.channel_semantics!r} requires explicit "
                    f"data.normalize.mean/std (one value per channel; no default exists for "
                    f"freeform). Provide N={self.channels} mean and {self.channels} std values."
                )
            mean, std = profile.normalize_default
            object.__setattr__(
                self, "normalize", NormalizeConfig(mean=list(mean), std=list(std))
            )

        # (b) length cross-check (after default materialization)
        if len(self.normalize.mean) != self.channels or len(self.normalize.std) != self.channels:
            raise ValueError(
                f"data.normalize.mean has {len(self.normalize.mean)} entries but "
                f"data.channels={self.channels}; provide exactly {self.channels} per-channel "
                f"mean values (and {self.channels} std values)."
            )
        return self
```

(`object.__setattr__` is used because `_Strict` models may be frozen-ish; if `DataConfig` is mutable, plain `self.normalize = ...` is fine — implementer confirms `_Strict`'s `model_config`. If mutation is disallowed even via `object.__setattr__`, use `mode="before"` to inject the default into the raw dict instead. Observable contract is the precedence in spec §3.3c.)

- [ ] **Step 4: Run all DataConfig extension tests (including Task 4's full set)**

Run: `uv run pytest tests/unit/test_data_schema_extensions.py -q`
Expected: PASS (all, including Task 4's `channels=1/16` cases now that cross-validation + defaults exist).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/config/schema.py tests/unit/test_data_schema_extensions.py
git commit -m "feat(config): cross-validate semantic<->channels + normalize default/required/length (spec §3.3)"
```

---

## Task 6: Backward-compat regression — existing configs still load

**Files:**
- Test: `tests/unit/test_config_examples.py` (append) and/or `tests/unit/test_config_schema.py`

This task PROVES the rgb default is inert at config level before any model/data behavior changes.

- [ ] **Step 1: Write the regression test**

```python
# tests/unit/test_config_examples.py  (append)
def test_existing_rgb_config_unchanged(tmp_path):
    """An rgb config that omits channels/channel_semantics resolves to the
    pre-feature ImageNet-3 normalize default, byte-for-byte (spec §3.4)."""
    from custom_sam_peft.config.schema import DataConfig

    d = DataConfig.model_validate(
        dict(format="coco", train={"annotations": "a.json", "images": "i"}, prompt_mode="text")
    )
    assert d.channels == 3
    assert d.channel_semantics == "rgb"
    assert d.normalize.mean == [0.485, 0.456, 0.406]
    assert d.normalize.std == [0.229, 0.224, 0.225]
    assert d.normalize.max_pixel_value == 255.0
```

- [ ] **Step 2: Run the full config + example suite**

Run: `uv run pytest tests/unit/test_config_examples.py tests/unit/test_config_loader.py tests/unit/test_cli_doctor_config.py -q`
Expected: PASS — shipped example YAMLs and the doctor still validate (the sentinel `normalize=None` materializes to the same ImageNet-3 stats).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_config_examples.py
git commit -m "test(config): regression — rgb default config resolves unchanged (spec §3.4)"
```

---

## Task 7: `read_image` reader + `_coerce_to_channels` core

**Files:**
- Create: `src/custom_sam_peft/data/io.py`
- Test: `tests/unit/test_data_io.py`

- [ ] **Step 1: Write the failing test (C6 — dispatch + C-validation)**

```python
# tests/unit/test_data_io.py
import numpy as np
import pytest
from PIL import Image

from custom_sam_peft.data.io import _coerce_to_channels, read_image


def _save_png(tmp_path, arr, name="x.png"):
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def test_pil_grayscale_rgb_rgba(tmp_path):
    rgb = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = _save_png(tmp_path, rgb)
    assert read_image(p, 3).shape == (8, 10, 3)
    assert read_image(p, 1).shape == (8, 10, 1)
    assert read_image(p, 4).shape == (8, 10, 4)


def test_pil_unsupported_channel_count_errors(tmp_path):
    rgb = (np.random.rand(8, 10, 3) * 255).astype(np.uint8)
    p = _save_png(tmp_path, rgb)
    with pytest.raises(ValueError, match=r"channels=5"):
        read_image(p, 5)  # PIL caps at RGBA


def test_npy_hwc_and_chw(tmp_path):
    hwc = (np.random.rand(8, 10, 5)).astype(np.float32)
    chw = np.transpose(hwc, (2, 0, 1)).copy()
    p_hwc = tmp_path / "hwc.npy"
    p_chw = tmp_path / "chw.npy"
    np.save(p_hwc, hwc)
    np.save(p_chw, chw)
    assert read_image(p_hwc, 5).shape == (8, 10, 5)
    assert read_image(p_chw, 5).shape == (8, 10, 5)


def test_npy_channel_mismatch_errors(tmp_path):
    p = tmp_path / "m.npy"
    np.save(p, np.zeros((8, 10, 3), np.float32))
    with pytest.raises(ValueError, match=r"has 3 channels but .*channels=4"):
        read_image(p, 4)


def test_tiff_multiband(tmp_path):
    import tifffile

    arr = (np.random.rand(6, 8, 7)).astype(np.float32)  # H,W,C=7
    p = tmp_path / "mb.tif"
    tifffile.imwrite(p, np.transpose(arr, (2, 0, 1)))  # tifffile writes C,H,W as pages
    out = read_image(p, 7)
    assert out.shape == (6, 8, 7)


def test_coerce_pil_2d_array_triplicate_and_keep1():
    arr2d = (np.random.rand(8, 10) * 255).astype(np.uint8)
    assert _coerce_to_channels(arr2d, 3).shape == (8, 10, 3)
    assert _coerce_to_channels(arr2d, 1).shape == (8, 10, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_data_io.py -q`
Expected: FAIL — `custom_sam_peft.data.io` does not exist.

- [ ] **Step 3: Write the implementation**

```python
# src/custom_sam_peft/data/io.py
"""Channel-aware image reader (spec §6). Keys off channel COUNT only —
channel_semantics never reaches this module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PIL_MODE = {1: "L", 3: "RGB", 4: "RGBA"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def _coerce_to_channels(obj: object, channels: int) -> np.ndarray:
    """Coerce a PIL image OR an ndarray to (H, W, C) with C == channels.

    PIL path uses mode conversion (1->L, 3->RGB, 4->RGBA). Array path accepts
    2-D (H,W -> triplicate/keep), (H,W,C), or (C,H,W); validates C == channels.
    """
    from PIL import Image as PILImage

    if isinstance(obj, PILImage.Image):
        mode = _PIL_MODE.get(channels)
        if mode is None:
            raise ValueError(
                f"read_image: PIL/raster input cannot produce channels={channels} "
                f"(PIL supports 1=L, 3=RGB, 4=RGBA only). Use a .npy/.npz/.tif source."
            )
        out = np.asarray(obj.convert(mode))
        if out.ndim == 2:  # mode "L"
            out = out[:, :, None]
        return out

    arr = np.asarray(obj)
    if arr.ndim == 2:
        if channels == 1:
            return arr[:, :, None]
        return np.repeat(arr[:, :, None], channels, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"read_image: expected 2-D or 3-D array, got ndim={arr.ndim}")
    # Resolve channel axis: prefer HWC; transpose CHW when the leading dim matches.
    if arr.shape[2] == channels:
        hwc = arr
    elif arr.shape[0] == channels:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        found = arr.shape[2] if arr.shape[2] <= arr.shape[0] else arr.shape[0]
        raise ValueError(
            f"read_image: array has {found} channels but data.channels={channels}"
        )
    return np.ascontiguousarray(hwc)


def read_image(path: str | Path, channels: int) -> np.ndarray:
    """Read an image file to (H, W, C) with C == channels. Dispatch on extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _RASTER_EXTS:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            return _coerce_to_channels(im, channels)
    if ext in {".npy", ".npz"}:
        loaded = np.load(path)
        if ext == ".npz":
            loaded = loaded[loaded.files[0]]
        return _coerce_to_channels(loaded, channels)
    if ext in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(path)  # (C,H,W) for multipage, or (H,W) / (H,W,C)
        return _coerce_to_channels(arr, channels)
    raise ValueError(f"read_image: unsupported file extension {ext!r} for {path}")
```

(Note: the raster-mismatch error message references `channels=` per the spec; the array-mismatch path uses the `has N channels but data.channels=M` form. Both satisfy spec §6.1's "clear error" requirement and the C6/C11 assertions.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_data_io.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/io.py tests/unit/test_data_io.py
git commit -m "feat(data): channel-aware read_image + _coerce_to_channels reader (spec §6.1)"
```

---

## Task 8: Wire reader into COCO + HF loaders (channels plumbing)

**Files:**
- Modify: `src/custom_sam_peft/data/coco.py` — `COCODataset.__init__:123`, `_decode_image:204-213`, `build_coco:360`
- Modify: `src/custom_sam_peft/data/hf.py` — `HFDataset.__init__:129`, `_decode_image:207-217`, `build_hf:401`
- Test: `tests/unit/test_data_coco.py`, `tests/unit/test_data_hf.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_data_coco.py  (append)
def test_coco_decode_image_uses_channels(tmp_path, monkeypatch):
    """COCODataset._decode_image routes through read_image with self._channels."""
    import numpy as np
    from custom_sam_peft.data import coco as coco_mod

    captured = {}

    def fake_read_image(path, channels):
        captured["channels"] = channels
        return np.zeros((4, 5, channels), np.uint8)

    monkeypatch.setattr(coco_mod, "read_image", fake_read_image, raising=False)
    # Build a minimal COCODataset stub-style instance is heavy; instead assert the
    # body via a lightweight object carrying _image_root and _channels.
    obj = coco_mod.COCODataset.__new__(coco_mod.COCODataset)
    obj._image_root = tmp_path
    obj._channels = 5
    raw = (1, {"file_name": "a.png"}, [])
    out = coco_mod.COCODataset._decode_image(obj, raw)
    assert out.shape == (4, 5, 5)
    assert captured["channels"] == 5
```

```python
# tests/unit/test_data_hf.py  (append)
def test_hf_decode_image_array_branch_channel_aware():
    """HFDataset._decode_image coerces an array row to self._channels."""
    import numpy as np
    from custom_sam_peft.data import hf as hf_mod

    obj = hf_mod.HFDataset.__new__(hf_mod.HFDataset)
    obj._channels = 1
    obj._field_map = type("FM", (), {"image": "image"})()

    # ndim==2 array -> (H,W,1) for channels=1
    raw = {"image": np.zeros((4, 6), np.uint8)}
    out = hf_mod.HFDataset._decode_image(obj, raw)
    assert out.shape == (4, 6, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_coco.py tests/unit/test_data_hf.py -q -k "channels or channel_aware"`
Expected: FAIL — `read_image` not imported in coco; `_channels` not set; HF array branch hardcodes 3.

- [ ] **Step 3a: COCO changes**

In `coco.py`, add import at top: `from custom_sam_peft.data.io import read_image`.

`COCODataset.__init__` (`:123-133`) — add a keyword param:

```python
    def __init__(
        self,
        annotations: str,
        images: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPropmtConfig,  # (keep existing type name TextPromptConfig)
        seed: int = 0,
        image_ids: Iterable[int] | None = None,
        channels: int = 3,
    ) -> None:
```

(Correct the typo above to `text_prompt: TextPromptConfig` — keep the existing signature, only ADD `channels: int = 3` as the last keyword.) In the body, store `self._channels = channels`.

Replace `_decode_image` (`:204-213`) body:

```python
    def _decode_image(
        self, raw: tuple[int, dict[str, Any], list[dict[str, Any]]]
    ) -> np.ndarray[Any, Any]:
        """Load and decode the image for *raw* to an (H, W, C) uint8/float ndarray."""
        _image_id, rec, _anns = raw
        img_path = self._image_root / rec["file_name"]
        return read_image(img_path, self._channels)
```

`build_coco` (`:393-400`) — pass `channels`:

```python
    return COCODataset(
        annotations=split["annotations"],
        images=split["images"],
        prompt_mode=cfg["prompt_mode"],
        transforms=transforms,
        text_prompt=text_prompt,
        image_ids=[int(s) for s in resolved] if resolved is not None else None,
        channels=int(cfg.get("channels", 3)),
    )
```

- [ ] **Step 3b: HF changes**

In `hf.py`, add import: `from custom_sam_peft.data.io import _coerce_to_channels`.

`HFDataset.__init__` (`:129-140`) — add `channels: int = 3` last keyword; store `self._channels = channels`.

Replace `_decode_image` (`:207-217`):

```python
    def _decode_image(self, raw: dict[str, Any]) -> np.ndarray[Any, Any]:
        """Decode a raw HF row's image field to an (H, W, C) ndarray (C == self._channels)."""
        img_obj = _resolve_field(raw, self._field_map.image)
        return _coerce_to_channels(img_obj, self._channels)
```

`build_hf` (`:430-434`) — pass `channels=int(cfg.get("channels", 3))` as a keyword to the `HFDataset(...)` constructor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_coco.py tests/unit/test_data_hf.py -q`
Expected: PASS (new + existing — existing RGB rows with `channels=3` default behave as `convert("RGB")` did).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py tests/unit/test_data_coco.py tests/unit/test_data_hf.py
git commit -m "feat(data): wire channel-aware reader into COCO + HF loaders (spec §6.3)"
```

---

## Task 9: Channel adapter — `rgb` passthrough is inert (C3 first)

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py` — add a `_build_channel_adapter(channels, channel_semantics)` helper
- Test: `tests/unit/test_channel_adapter.py`

This task proves the zero-regression contract for `rgb` BEFORE any non-rgb adapter math. We add a pure builder helper, tested in isolation (no real SAM model needed).

- [ ] **Step 1: Write the failing test (C3 — passthrough keyed on semantic + freeform-3ch paired case)**

```python
# tests/unit/test_channel_adapter.py
import torch

from custom_sam_peft.models.sam3 import _build_channel_adapter


def test_rgb_builds_no_adapter():
    assert _build_channel_adapter(channels=3, channel_semantics="rgb") is None


def test_freeform_3ch_builds_learned_adapter_not_passthrough():
    adapter = _build_channel_adapter(channels=3, channel_semantics="freeform")
    assert adapter is not None
    assert isinstance(adapter, torch.nn.Conv2d)
    assert adapter.in_channels == 3 and adapter.out_channels == 3
    # average_broadcast init for N=3 => weight == 1/3 everywhere, bias == 0
    assert torch.allclose(adapter.weight, torch.full_like(adapter.weight, 1.0 / 3.0))
    assert torch.allclose(adapter.bias, torch.zeros_like(adapter.bias))
    assert adapter.weight.requires_grad and adapter.bias.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_channel_adapter.py -q`
Expected: FAIL — `_build_channel_adapter` not defined.

- [ ] **Step 3: Write the builder helper**

In `sam3.py` (near `_Sam3ImageAdapter`, before the class), add:

```python
def _build_channel_adapter(channels: int, channel_semantics: str) -> nn.Conv2d | None:
    """Build the N->3 channel adapter per the semantic profile (spec §5.2/§5.3).

    Returns None for semantic=='rgb' (passthrough, zero new params). Otherwise a
    fully-trainable Conv2d(channels, 3, 1) initialized per profile.adapter_init.
    """
    from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTICS

    profile = CHANNEL_SEMANTICS[channel_semantics]
    if not profile.use_adapter:
        return None
    conv = nn.Conv2d(channels, 3, kernel_size=1, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        if profile.adapter_init == "average_broadcast":
            conv.weight.fill_(1.0 / channels)
        elif profile.adapter_init == "identity_passthrough":
            # Identity on first 3 input channels, zero on the rest.
            for o in range(3):
                if o < channels:
                    conv.weight[o, o, 0, 0] = 1.0
        else:  # pragma: no cover - registry guards this
            raise ValueError(f"unknown adapter_init: {profile.adapter_init!r}")
    conv.weight.requires_grad_(True)
    conv.bias.requires_grad_(True)
    return conv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_channel_adapter.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/test_channel_adapter.py
git commit -m "feat(model): _build_channel_adapter — rgb passthrough + learned non-rgb (spec §5.2,§5.3)"
```

---

## Task 10: Channel adapter init math — average_broadcast (C1, C2) + identity_passthrough (C3b)

**Files:**
- Test: `tests/unit/test_channel_adapter.py` (append) — C1, C2, C3b

The math is already implemented in Task 9's builder; this task adds the assertions the spec mandates.

- [ ] **Step 1: Write the failing/confirming tests (C1, C2, C3b)**

```python
# tests/unit/test_channel_adapter.py  (append)
def test_C1_average_broadcast_mean_of_stack():
    adapter = _build_channel_adapter(channels=4, channel_semantics="freeform")
    assert torch.allclose(adapter.weight, torch.full_like(adapter.weight, 1.0 / 4.0))
    assert torch.allclose(adapter.bias, torch.zeros_like(adapter.bias))
    x = torch.randn(2, 4, 5, 6)
    out = adapter(x)
    expected_mean = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    assert torch.allclose(out, expected_mean, atol=1e-5)


def test_C2_grayscale_triplication_identity():
    adapter = _build_channel_adapter(channels=1, channel_semantics="grayscale")
    x = torch.randn(2, 1, 5, 6)
    out = adapter(x)
    assert torch.allclose(out, torch.cat([x, x, x], dim=1), atol=1e-6)


def test_C3b_rgba_identity_passthrough_drops_alpha():
    adapter = _build_channel_adapter(channels=4, channel_semantics="rgba")
    w = adapter.weight  # (3,4,1,1)
    for o in range(3):
        assert torch.isclose(w[o, o, 0, 0], torch.tensor(1.0))
    assert torch.allclose(w[:, 3, 0, 0], torch.zeros(3))  # alpha column zero
    assert torch.allclose(adapter.bias, torch.zeros_like(adapter.bias))
    x = torch.randn(2, 4, 5, 6)
    out = adapter(x)
    assert torch.allclose(out, x[:, :3], atol=1e-6)  # first 3 (RGB) exactly, alpha dropped
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/test_channel_adapter.py -q -k "C1 or C2 or C3b"`
Expected: PASS (builder from Task 9 already produces these inits).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_channel_adapter.py
git commit -m "test(model): assert average_broadcast (C1,C2) + identity_passthrough (C3b) init math (spec §5.3)"
```

---

## Task 11: Model wiring — adapter into `_Sam3ImageAdapter`, `_validate_inputs`, `load_sam31` signature

**Files:**
- Modify: `src/custom_sam_peft/models/sam3.py` — `_Sam3ImageAdapter.__init__:339`, `.forward:344-360`, `Sam3Wrapper.__init__:206`, `_validate_inputs:223-230`, `load_sam31:671-684`
- Test: `tests/unit/test_sam3_wrapper.py` (append) — C8 + adapter-presence

- [ ] **Step 1: Write the failing tests (C8 + wiring)**

```python
# tests/unit/test_sam3_wrapper.py  (append)
import pytest
import torch
import torch.nn as nn

from custom_sam_peft.models.sam3 import _Sam3ImageAdapter, Sam3Wrapper


class _StubBackbone(nn.Module):
    def forward_image(self, images):  # pragma: no cover - shape probe
        return {"_chans": images.shape[1]}


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _StubBackbone()


def test_validate_inputs_accepts_configured_channels_rejects_wrong():
    w = Sam3Wrapper(_Sam3ImageAdapter(_StubModel(), channels=5, channel_semantics="freeform"),
                    channels=5, channel_semantics="freeform")
    # 5-channel ok (ndim==4, C==5)
    w._validate_inputs(torch.zeros(1, 5, 8, 8), [], None)
    with pytest.raises(ValueError, match=r"\(B, 5, H, W\)"):
        w._validate_inputs(torch.zeros(1, 3, 8, 8), [], None)
    with pytest.raises(ValueError):
        w._validate_inputs(torch.zeros(5, 8, 8), [], None)  # ndim != 4


def test_rgb_adapter_is_none_zero_new_params():
    ad = _Sam3ImageAdapter(_StubModel(), channels=3, channel_semantics="rgb")
    assert ad.channel_adapter is None
    base = sum(p.numel() for p in _StubModel().parameters())
    total = sum(p.numel() for p in ad.parameters())
    assert total == base  # zero new params for rgb


def test_freeform_adapter_present_and_trainable():
    ad = _Sam3ImageAdapter(_StubModel(), channels=4, channel_semantics="freeform")
    assert isinstance(ad.channel_adapter, nn.Conv2d)
    assert any(p.requires_grad for p in ad.channel_adapter.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_sam3_wrapper.py -q -k "validate_inputs_accepts or adapter_is_none or adapter_present"`
Expected: FAIL — constructors do not accept `channels`/`channel_semantics`; `_validate_inputs` is a staticmethod with no channel check.

- [ ] **Step 3a: `_Sam3ImageAdapter`**

Update `__init__` (`:339`):

```python
    def __init__(
        self,
        model: nn.Module,
        image_size: int = 1008,
        *,
        channels: int = 3,
        channel_semantics: str = "rgb",
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.channel_semantics = channel_semantics
        self.channel_adapter = _build_channel_adapter(channels, channel_semantics)
```

In `forward` (`:344`), immediately BEFORE the `forward_image` call (`:360`):

```python
        if self.channel_adapter is not None:
            images = self.channel_adapter(images)  # (B, N, H, W) -> (B, 3, H, W)
        backbone_out = self.model.backbone.forward_image(images)  # type: ignore[union-attr, operator]
```

- [ ] **Step 3b: `Sam3Wrapper`**

Update `__init__` (`:206`):

```python
    def __init__(
        self,
        model: nn.Module,
        image_size: int = 1008,
        mask_size: int = 288,
        *,
        channels: int = 3,
        channel_semantics: str = "rgb",
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.mask_size = mask_size
        self.channels = channels
        self.channel_semantics = channel_semantics
        self.peft_model: PeftModel | None = None
```

(`Sam3Wrapper` stores `channels`/`channel_semantics` for `_validate_inputs` and for checkpoint logic in Task 13; the adapter itself lives on the `_Sam3ImageAdapter` passed as `model`.)

- [ ] **Step 3c: `_validate_inputs`**

Change from `@staticmethod` to an instance method and add the channel check (`:223-230`):

```python
    def _validate_inputs(
        self,
        images: Tensor,
        prompts: list[Prompts],
        box_hints: list[Tensor | None] | None,
    ) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must be (B, {self.channels}, H, W); got shape {tuple(images.shape)}"
            )
        if images.shape[1] != self.channels:
            raise ValueError(
                f"images must be (B, {self.channels}, H, W); got "
                f"{images.shape[1]} channels in shape {tuple(images.shape)}"
            )
        b = images.shape[0]
        ...  # rest unchanged
```

Confirm the only caller is `self._validate_inputs(...)` at `:219` (already `self.`), so no other call-site change is needed.

- [ ] **Step 3d: `load_sam31`**

Update (`:671-684`):

```python
def load_sam31(
    cfg: ModelConfig,
    *,
    channels: int = 3,
    channel_semantics: str = "rgb",
) -> Sam3Wrapper:
    ...
    adapter = _Sam3ImageAdapter(
        raw_model, image_size=1008, channels=channels, channel_semantics=channel_semantics
    )
    return Sam3Wrapper(
        adapter, image_size=1008, mask_size=288,
        channels=channels, channel_semantics=channel_semantics,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_sam3_wrapper.py tests/unit/test_sam3_adapter.py -q`
Expected: PASS (new + existing — existing tests use defaults `channels=3, channel_semantics="rgb"` → `channel_adapter is None`, forward unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/models/sam3.py tests/unit/test_sam3_wrapper.py
git commit -m "feat(model): wire channel adapter + channel-aware _validate_inputs + load_sam31 signature (spec §9)"
```

---

## Task 12: Transforms — processor-skip for non-rgb + three augmentation regimes + max_pixel_value

**Files:**
- Modify: `src/custom_sam_peft/data/transforms.py` — `resolve_normalization_with_path:84`, `resolve_normalization:147`, `build_eval_transforms:159`, `build_train_transforms:194-277`
- Modify: `src/custom_sam_peft/data/coco.py:388-392`, `hf.py:425-429` (pass `channel_semantics`/`channels`)
- Test: `tests/unit/test_data_transforms.py` (append) — C7, C9

- [ ] **Step 1: Write the failing tests (C9 processor-skip + C7 regimes)**

```python
# tests/unit/test_data_transforms.py  (append)
from custom_sam_peft.config.schema import AugmentationsConfig, NormalizeConfig
from custom_sam_peft.data import transforms as T


def _names(compose):
    return [type(s).__name__ for s in compose.transforms]


def test_C9_processor_consulted_only_for_rgb(monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("AutoImageProcessor must NOT be consulted for non-rgb")

    import transformers
    monkeypatch.setattr(transformers, "AutoImageProcessor",
                        type("X", (), {"from_pretrained": staticmethod(boom)}))
    mean, std = T.resolve_normalization(
        "facebook/sam3.1", NormalizeConfig(mean=[0.1, 0.2, 0.3, 0.4], std=[0.1] * 4),
        channel_semantics="rgba",
    )
    assert mean == [0.1, 0.2, 0.3, 0.4]
    assert calls["n"] == 0


def test_C7_rgb_full_family():
    aug = AugmentationsConfig.model_validate({"preset": "natural", "intensity": "aggressive"})
    c = T.build_train_transforms(aug, 64, model_name="facebook/sam3.1",
                                 normalize=NormalizeConfig(),
                                 channel_semantics="rgb", channels=3)
    names = _names(c)
    assert "ColorJitter" in names  # full family (assumes preset enables color_jitter)


def test_C7_rgba_substitutes_brightness_contrast_no_colorjitter(caplog):
    aug = AugmentationsConfig.model_validate({"preset": "natural", "intensity": "aggressive"})
    c = T.build_train_transforms(aug, 64, model_name="facebook/sam3.1",
                                 normalize=NormalizeConfig(mean=[0.1] * 4, std=[0.1] * 4),
                                 channel_semantics="rgba", channels=4)
    names = _names(c)
    assert "RandomBrightnessContrast" in names
    assert "ColorJitter" not in names
    assert "StainJitter" not in names


def test_C7_freeform_geometry_only_even_with_knobs():
    aug = AugmentationsConfig.model_validate({"preset": "natural", "intensity": "aggressive"})
    c = T.build_train_transforms(aug, 64, model_name="facebook/sam3.1",
                                 normalize=NormalizeConfig(mean=[0.1] * 5, std=[0.1] * 5),
                                 channel_semantics="freeform", channels=5)
    names = _names(c)
    for forbidden in ("ColorJitter", "StainJitter", "GaussNoise",
                      "GaussianBlur", "RandomBrightnessContrast"):
        assert forbidden not in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_transforms.py -q -k "C7 or C9"`
Expected: FAIL — `resolve_normalization` / `build_train_transforms` do not accept `channel_semantics`/`channels`.

- [ ] **Step 3a: `resolve_normalization*` — processor-skip**

`resolve_normalization_with_path` (`:84`): add `*, channel_semantics: str = "rgb"`. At the top, if `channel_semantics != "rgb"`, short-circuit to the config-fallback path WITHOUT consulting `AutoImageProcessor`:

```python
def resolve_normalization_with_path(
    model_name: str, fallback: NormalizeConfig, *, channel_semantics: str = "rgb"
) -> tuple[list[float], list[float], NormalizationPath]:
    if channel_semantics != "rgb":
        # Frozen patch-embed RGB stats do not apply to raw N-channel input
        # (normalization happens upstream of the adapter). Use config stats. (spec §7.1)
        return list(fallback.mean), list(fallback.std), "config-fallback"
    import transformers
    ...  # existing 3-step body unchanged
```

`resolve_normalization` (`:147`): add `*, channel_semantics: str = "rgb"` and forward it.

- [ ] **Step 3b: `build_eval_transforms` (`:159`)**

Add params `*, channel_semantics: str = "rgb"`; forward `channel_semantics` to `resolve_normalization`; replace `max_pixel_value=255.0` with `max_pixel_value=normalize.max_pixel_value`.

- [ ] **Step 3c: `build_train_transforms` (`:194`)**

Add params `*, channel_semantics: str = "rgb", channels: int = 3`. Compute the regime:

```python
    from custom_sam_peft.data.channel_semantics import CHANNEL_SEMANTICS

    profile = CHANNEL_SEMANTICS[channel_semantics]
    photometric = profile.photometric
    rgb_like = photometric and channels == 3
    max_pixel = normalize.max_pixel_value
    mean, std = resolve_normalization(model_name, normalize, channel_semantics=channel_semantics)
```

Replace the value-altering append block (`gauss_noise`/`blur`/`color_jitter`/`stain_jitter`, `:242-275`) with regime-gated logic. Geometric steps (`:226-241`) stay as-is for all regimes:

```python
    # --- value-altering steps: gated by regime (spec §8) ---
    if not photometric:
        _warn_freeform_augs_once()  # names all 4 disabled augs; module-level guard
    else:
        if resolved.gauss_noise > 0.0:
            steps.append(A.GaussNoise(
                std_range=(0.0, resolved.gauss_noise * _GAUSS_NOISE_MAX_VAR), p=0.5))
        if resolved.blur > 0.0:
            steps.append(A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(0.0, resolved.blur * _GAUSS_BLUR_MAX_SIGMA), p=0.5))
        if rgb_like:
            if resolved.color_jitter > 0.0:
                v = resolved.color_jitter
                steps.append(A.ColorJitter(brightness=v, contrast=v, saturation=v, hue=v * 0.5, p=0.5))
            if resolved.stain_jitter > 0.0:
                steps.append(StainJitter(sigma=resolved.stain_jitter, p=0.5))
        else:
            _warn_non3ch_photometric_augs_once()  # sat/hue + StainJitter skipped; B/C substituted
            if resolved.color_jitter > 0.0:
                v = resolved.color_jitter
                steps.append(A.RandomBrightnessContrast(
                    brightness_limit=v, contrast_limit=v, brightness_by_max=False, p=0.5))
    steps.append(A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel))
    steps.append(ToTensorV2())
```

(`brightness_by_max=False` resolves OPEN QUESTION 1 — keeps brightness scaling consistent with float `max_pixel_value`. The original geometric block ordering — flips/rotate90/affine — is unchanged; only the value-altering tail is restructured. Note `freeform` keeps GaussNoise/Blur OUT because `not photometric` short-circuits before them, satisfying §8.3's hard-disable.)

Add module-level one-time warning guards near the top of `transforms.py`:

```python
_warned_non3ch_photometric = False
_warned_freeform = False


def _warn_non3ch_photometric_augs_once() -> None:
    global _warned_non3ch_photometric
    if not _warned_non3ch_photometric:
        _LOG.warning(
            "Non-3ch photometric semantic: saturation/hue and StainJitter are "
            "skipped (RGB-3ch-only); brightness/contrast substituted via "
            "A.RandomBrightnessContrast. (spec §8.2)"
        )
        _warned_non3ch_photometric = True


def _warn_freeform_augs_once() -> None:
    global _warned_freeform
    if not _warned_freeform:
        _LOG.warning(
            "freeform (non-photometric) semantic: A.ColorJitter, StainJitter, "
            "A.GaussNoise, and A.GaussianBlur are disabled (they assume photometric "
            "continuity); only geometric augmentations apply. (spec §8.3)"
        )
        _warned_freeform = True
```

(Use the module's existing logger — confirm its name; the file uses `_LOG`.)

- [ ] **Step 3d: thread through `build_coco` / `build_hf`**

`coco.py:388-392` and `hf.py:425-429`: pass `channel_semantics=cfg.get("channel_semantics", "rgb")` to BOTH `build_train_transforms` and `build_eval_transforms`, and `channels=int(cfg.get("channels", 3))` to `build_train_transforms`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_transforms.py -q`
Expected: PASS (new C7/C9 + existing RGB-pipeline tests — `rgb` default keeps the full family + processor consult).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/data/transforms.py src/custom_sam_peft/data/coco.py src/custom_sam_peft/data/hf.py tests/unit/test_data_transforms.py
git commit -m "feat(data): processor-skip for non-rgb + three augmentation regimes + max_pixel_value (spec §7,§8)"
```

---

## Task 13: Checkpoint gap fix — `channel_adapter.pt` save/load round-trip

**Files:**
- Modify: `src/custom_sam_peft/train/checkpoint.py` — `save_adapter:66`, `load_adapter:76`, `save_merged:83`, `save_full_state:96-126`, `load_full_state:171`
- Test: `tests/unit/test_train_checkpoint.py` (append) — CPU round-trip with a stub wrapper

- [ ] **Step 1: Write the failing test (CPU stub round-trip)**

```python
# tests/unit/test_train_checkpoint.py  (append)
def test_channel_adapter_file_written_and_skipped_for_rgb(tmp_path):
    """save_adapter writes channel_adapter.pt iff the adapter exists; rgb skips it."""
    import torch
    import torch.nn as nn
    from custom_sam_peft.train import checkpoint as C

    class _StubWrapper:
        def __init__(self, adapter):
            self._adapter = adapter
            self.peft_model = _StubPeft()

        @property
        def channel_adapter(self):
            return self._adapter

    class _StubPeft:
        def save_pretrained(self, p):
            from pathlib import Path
            Path(p).mkdir(parents=True, exist_ok=True)
            (Path(p) / "adapter_model.safetensors").write_bytes(b"x")

    # helper the implementer adds: _wrapper_channel_adapter(wrapper) -> nn.Module|None
    conv = nn.Conv2d(4, 3, 1)
    with torch.no_grad():
        conv.weight.normal_()
    w_has = _StubWrapper(conv)
    C.save_adapter(w_has, tmp_path / "a")
    assert (tmp_path / "a" / C._CHANNEL_ADAPTER_FILENAME).exists()

    # reload into a fresh conv via load_adapter's channel-adapter restore
    fresh = nn.Conv2d(4, 3, 1)
    w_fresh = _StubWrapper(fresh)
    # stub load_adapter PEFT side is exercised by integration; here assert the
    # channel-adapter restore helper round-trips bit-for-bit:
    C._save_channel_adapter(w_has, tmp_path / "rt")
    C._load_channel_adapter(w_fresh, tmp_path / "rt")
    assert torch.allclose(fresh.weight, conv.weight)

    # rgb: no adapter -> no file
    w_rgb = _StubWrapper(None)
    C.save_adapter(w_rgb, tmp_path / "b")
    assert not (tmp_path / "b" / C._CHANNEL_ADAPTER_FILENAME).exists()
```

(The implementer factors the channel-adapter persistence into `_save_channel_adapter(wrapper, dir)` / `_load_channel_adapter(wrapper, dir)` helpers and a `_wrapper_channel_adapter(wrapper)` accessor that reaches `wrapper.model.channel_adapter` on the real `Sam3Wrapper` — the stub above mirrors that via a `channel_adapter` property. Real-`state_dict` bit-for-bit round-trip is GPU test G2.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_checkpoint.py -q -k "channel_adapter"`
Expected: FAIL — `_CHANNEL_ADAPTER_FILENAME` / helpers do not exist.

- [ ] **Step 3: Implement persistence helpers + wire into save/load paths**

In `checkpoint.py` add near the filename constants (`:33-35`):

```python
_CHANNEL_ADAPTER_FILENAME = "channel_adapter.pt"


def _wrapper_channel_adapter(wrapper: Sam3Wrapper) -> Any:
    """Return the channel adapter module if present, else None.

    The adapter lives on _Sam3ImageAdapter (wrapper.model), sibling of the raw
    SAM model — outside the PeftModel, so PEFT save/load never touches it (spec §5.1).
    """
    return getattr(wrapper.model, "channel_adapter", None)


def _save_channel_adapter(wrapper: Sam3Wrapper, adapter_dir: Path) -> None:
    """Dump the channel adapter's state_dict to channel_adapter.pt. No-op when None (rgb)."""
    ca = _wrapper_channel_adapter(wrapper)
    if ca is None:
        return
    adapter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ca.state_dict(), adapter_dir / _CHANNEL_ADAPTER_FILENAME)


def _load_channel_adapter(wrapper: Sam3Wrapper, adapter_dir: Path) -> None:
    """Restore the channel adapter from channel_adapter.pt. No-op when the file is
    absent (rgb) or the wrapper has no adapter."""
    ca = _wrapper_channel_adapter(wrapper)
    path = adapter_dir / _CHANNEL_ADAPTER_FILENAME
    if ca is None or not path.exists():
        return
    state = torch.load(path, weights_only=True, map_location="cpu")
    ca.load_state_dict(state)
```

Wire into the write paths:
- `save_adapter` (`:66-73`) — after the PEFT save dispatch, call `_save_channel_adapter(wrapper, Path(path))`.
- `save_full_state` (`:108`) — `save_adapter(...)` already runs at `:108`; it now writes the channel adapter into `state_dir / "adapter"`. No extra call needed (it flows through `save_adapter`).
- `save_merged` (`:83-93`) — after `torch.save(wrapper.model.state_dict(), ...)`, ALSO call `_save_channel_adapter(wrapper, path)` for symmetry (see OPEN QUESTION 2; the merged `state_dict` likely already carries it, G3 confirms).

Wire into the read paths:
- `load_adapter` (`:76-80`) — after the PEFT load dispatch returns, call `_load_channel_adapter(wrapper, Path(path))` then return `wrapper`.
- `load_full_state` (`:171`) — `load_adapter(wrapper, adapter_dir)` already runs at `:171`, which now restores the channel adapter. No extra call needed (flows through `load_adapter`). Keep the channel-adapter restore ORTHOGONAL to the `detect_method_from_checkpoint` LoRA/QLoRA check (`:150-170`) — do not entangle (spec §10.2 edge).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_train_checkpoint.py tests/unit/test_checkpoint_roundtrip.py -q`
Expected: PASS (new + existing LoRA-only round-trips, which have no channel adapter → file skipped).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/train/checkpoint.py tests/unit/test_train_checkpoint.py
git commit -m "fix(checkpoint): persist channel_adapter.pt across save/load (spec §10.2 gap)"
```

---

## Task 14: Predict path — channels/semantics aware

**Files:**
- Modify: `src/custom_sam_peft/predict/runner.py` — `_ResolvedConfig:92-102`, `_resolve_config:110-198`, model construction `:284-286`, normalize `:188`, image reader `:385-396`
- Test: `tests/predict/test_config_layering.py` (append) — C12; `tests/predict/test_preprocessing_parity.py` or new — C11, C13

- [ ] **Step 1: Write the failing tests (C11, C12, C13)**

```python
# tests/predict/test_config_layering.py  (append)
def test_C12_resolve_config_reads_channels_and_semantics(tmp_path):
    from custom_sam_peft.predict.runner import _resolve_config
    from custom_sam_peft.predict.options import PredictOptions  # confirm actual import path

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model:\n  name: facebook/sam3.1\n"
        "data:\n  image_size: 512\n  channels: 4\n  channel_semantics: rgba\n"
    )
    opts = PredictOptions(images=[tmp_path], prompts="cat", config=cfg, checkpoint=None)
    rcfg = _resolve_config(opts)
    assert rcfg.channels == 4
    assert rcfg.channel_semantics == "rgba"


def test_C12_defaults_when_absent(tmp_path):
    from custom_sam_peft.predict.runner import _resolve_config
    from custom_sam_peft.predict.options import PredictOptions

    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  name: facebook/sam3.1\ndata:\n  image_size: 512\n")
    opts = PredictOptions(images=[tmp_path], prompts="cat", config=cfg, checkpoint=None)
    rcfg = _resolve_config(opts)
    assert rcfg.channels == 3
    assert rcfg.channel_semantics == "rgb"
```

```python
# tests/predict/test_preprocessing_parity.py  (append) — C11 + C13
def test_C13_predict_normalize_skips_processor_for_non_rgb(monkeypatch):
    import transformers
    from custom_sam_peft.config.schema import NormalizeConfig
    from custom_sam_peft.data.transforms import resolve_normalization

    def boom(*a, **k):
        raise AssertionError("processor must not be consulted")

    monkeypatch.setattr(transformers, "AutoImageProcessor",
                        type("X", (), {"from_pretrained": staticmethod(boom)}))
    mean, std = resolve_normalization(
        "facebook/sam3.1", NormalizeConfig(mean=[0.1] * 4, std=[0.1] * 4),
        channel_semantics="rgba")
    assert mean == [0.1, 0.1, 0.1, 0.1]


def test_C11_predict_reader_returns_correct_C(tmp_path):
    import numpy as np
    from custom_sam_peft.data.io import read_image

    p = tmp_path / "x.npy"
    np.save(p, np.zeros((8, 10, 4), np.float32))
    assert read_image(p, 4).shape == (8, 10, 4)
    import pytest
    with pytest.raises(ValueError):
        read_image(p, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/predict/test_config_layering.py tests/predict/test_preprocessing_parity.py -q -k "C11 or C12 or C13"`
Expected: FAIL — `_ResolvedConfig` has no `channels`/`channel_semantics`; `resolve_normalization` already accepts the kwarg from Task 12 so C13/C11 may pass once those exist (run after Task 12). The C12 cases fail until predict plumbing lands.

- [ ] **Step 3a: `_ResolvedConfig`**

Add fields (`:92-102`):

```python
    channels: int
    channel_semantics: str
```

- [ ] **Step 3b: `_resolve_config` — parse both from YAML `data` section**

In the `data_section` parse block (`:137-141`), after `image_size`:

```python
                ch = data_section.get("channels")
                if ch is not None:
                    config_channels = int(ch)
                sem = data_section.get("channel_semantics")
                if sem is not None:
                    config_channel_semantics = str(sem)
```

Initialize `config_channels: int | None = None` and `config_channel_semantics: str | None = None` near `:124-125`. Resolve defaults near `:168`:

```python
    channels = config_channels if config_channels is not None else 3
    channel_semantics = (
        config_channel_semantics if config_channel_semantics is not None else "rgb"
    )
```

Pass `channel_semantics` to the `resolve_normalization` call (`:188`):

```python
    mean, std = resolve_normalization(
        model_name, NormalizeConfig(), channel_semantics=channel_semantics
    )
```

Add to the `_ResolvedConfig(...)` return (`:190-198`): `channels=channels, channel_semantics=channel_semantics`.

- [ ] **Step 3c: model construction (`:284-286`)**

```python
    model_cfg = ModelConfig(name=rcfg.model_name)
    model: torch.nn.Module = load_sam31(
        model_cfg, channels=rcfg.channels, channel_semantics=rcfg.channel_semantics
    )
```

- [ ] **Step 3d: build_eval_transforms call (`:310-314`)** — pass `channel_semantics=rcfg.channel_semantics`.

- [ ] **Step 3e: image reader (`:381-396`)** — replace the per-image PIL open:

```python
        for img_path in chunk_paths:
            try:
                from custom_sam_peft.data.io import read_image

                img_np = read_image(img_path, rcfg.channels)  # (H, W, C)
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", img_path, exc)
                continue
            orig_h, orig_w = img_np.shape[0], img_np.shape[1]
            ...
            transformed = transforms(image=img_np, bboxes=[], class_labels=[])
```

(Replace `pil_img.height/width` reads and `np.array(pil_img)` with `img_np.shape`. The warmup tensor at `:344-346` uses `1, 3, ...` — change the channel dim to `rcfg.channels` so warmup feeds the adapter the right shape; the adapter then maps to 3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/predict/ -q -k "C11 or C12 or C13 or smoke or layering"`
Expected: PASS (and existing predict smoke/dry-run tests with rgb defaults unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/predict/runner.py tests/predict/
git commit -m "feat(predict): N-channel-aware reader + _resolve_config channels/semantics + normalize skip (spec §11)"
```

---

## Task 15: Thread channels/semantics through all 7 `load_sam31` call sites

**Files:**
- Modify: `src/custom_sam_peft/cli/run_cmd.py:91`, `cli/calibrate_cmd.py:76`, `eval/runner.py:130`, `runs/bundle.py:67`, `train/runner.py:114`
- (predict/runner.py:286 done in Task 14; definition done in Task 11)
- Test: `tests/integration/test_cli_run.py` or targeted call-site assertions; lean on existing integration coverage

- [ ] **Step 1: Write the failing test (call sites pass real values)**

```python
# tests/unit/test_load_sam31_callsites.py  (new)
def test_train_runner_passes_data_channels(monkeypatch):
    """train.runner.run_training calls load_sam31 with cfg.data.channels/semantics."""
    import custom_sam_peft.train.runner as R

    captured = {}

    def fake_load_sam31(model_cfg, *, channels=3, channel_semantics="rgb"):
        captured["channels"] = channels
        captured["semantics"] = channel_semantics
        raise SystemExit("stop after load")  # short-circuit the heavy path

    monkeypatch.setattr(R, "load_sam31", fake_load_sam31)
    # Build a minimal TrainConfig with data.channels=4/channel_semantics=rgba and
    # invoke the smallest entry that reaches the load_sam31 line; assert captured.
    # (Implementer: reuse an existing TrainConfig fixture; mark xfail->pass once wired.)
```

(This is a wiring test; the implementer adapts it to whatever TrainConfig fixture the integration suite already provides. The substantive guard is that each call site passes `cfg.data.channels` + `cfg.data.channel_semantics`. If a full `run_training` invocation is too heavy for unit scope, assert at the integration layer instead and keep this as a focused monkeypatch test for the train + eval runners.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_load_sam31_callsites.py -q`
Expected: FAIL — call sites still pass only `cfg.model`.

- [ ] **Step 3: Update each call site**

- `run_cmd.py:91` → `load_sam31(cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics)`
- `eval/runner.py:130` → `load_sam31(cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics)`
- `train/runner.py:114` → `load_sam31(cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics)`
- `runs/bundle.py:67` → `load_sam31(cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics)`
- `calibrate_cmd.py:76` → `load_sam31(model_cfg, channels=3, channel_semantics="rgb")` with an inline comment:
  ```python
  # calibrate is a VRAM probe with no DataConfig in scope; the rgb default is
  # the documented exception (spec §5.4 / risk #2): probe RAM is for the base
  # model, not channel-adapter sizing.
  ```

- [ ] **Step 4: Run the affected suites**

Run: `uv run pytest tests/unit/test_load_sam31_callsites.py tests/integration/test_cli_run.py -q`
Expected: PASS. Also run `uv run pytest tests/unit -q` to confirm nothing regressed.

- [ ] **Step 5: Commit**

```bash
git add src/custom_sam_peft/cli/run_cmd.py src/custom_sam_peft/cli/calibrate_cmd.py src/custom_sam_peft/eval/runner.py src/custom_sam_peft/runs/bundle.py src/custom_sam_peft/train/runner.py tests/unit/test_load_sam31_callsites.py
git commit -m "feat: thread channels+channel_semantics through all load_sam31 call sites (spec §5.4)"
```

---

## Task 16: Optimizer collects adapter params (C10, CPU)

**Files:**
- Test: `tests/unit/test_train_checkpoint.py` or new `tests/unit/test_optimizer_collects_adapter.py`

The collection works automatically via `requires_grad` (spec §10.1 — no code change); this test is the guard against a future refactor silently dropping it.

- [ ] **Step 1: Write the test (CPU stub)**

```python
# tests/unit/test_optimizer_collects_adapter.py
import torch
import torch.nn as nn

from custom_sam_peft.models.sam3 import _Sam3ImageAdapter


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        for p in self.parameters():
            p.requires_grad_(False)  # base frozen


def test_C10_optimizer_includes_channel_adapter_params():
    adapter = _Sam3ImageAdapter(_StubModel(), channels=4, channel_semantics="freeform")
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    ca_params = set(map(id, adapter.channel_adapter.parameters()))
    assert ca_params.issubset(set(map(id, trainable)))
    assert len(trainable) == 2  # exactly the adapter weight + bias


def test_C10_rgb_has_no_trainable_adapter_params():
    adapter = _Sam3ImageAdapter(_StubModel(), channels=3, channel_semantics="rgb")
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    assert trainable == []  # rgb: no adapter, base frozen
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/unit/test_optimizer_collects_adapter.py -q`
Expected: PASS (adapter created `requires_grad=True` in Task 9; collection works).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_optimizer_collects_adapter.py
git commit -m "test(train): channel-adapter params collected by trainable set (spec §10.1 C10)"
```

---

## Task 17: Lint + full CPU suite gate

**Files:** none (gate)

- [ ] **Step 1: Run formatter + linter**

Run: `uv run ruff format src tests && uv run ruff check --fix src tests && uv run ruff check src tests`
Expected: clean (exit 0).

- [ ] **Step 2: Type-check**

Run: `uv run mypy src`
Expected: no new errors. Fix any introduced by the signature changes (e.g. `_validate_inputs` instance-method conversion, optional `nn.Conv2d | None`).

- [ ] **Step 3: Full CPU suite**

Run: `uv run pytest tests/unit tests/predict tests/integration -q -m "not gpu and not gpu_inspection and not requires_compatible_gpu"`
Expected: PASS; coverage `>= 80`.

- [ ] **Step 4: Commit any lint/type fixups**

```bash
git add -A
git commit -m "chore: lint/format/type fixups for n-channel support"
```

---

## Task 18: GPU test G1 — real SAM 3.1 forward through the adapter

**Files:**
- Create: `tests/gpu/test_channel_adapter_gpu.py`

- [ ] **Step 1: Write the GPU-gated test**

```python
# tests/gpu/test_channel_adapter_gpu.py
from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


def test_G1_real_forward_nchannel(tmp_path):
    """A freeform N-channel batch flows end-to-end; adapter feeds 3 ch to forward_image."""
    from custom_sam_peft.config.schema import ModelConfig
    from custom_sam_peft.data.base import TextPrompts
    from custom_sam_peft.models.sam3 import load_sam31

    n = 5
    wrapper = load_sam31(ModelConfig(), channels=n, channel_semantics="freeform").cuda()
    images = torch.randn(1, n, 1008, 1008, device="cuda", dtype=torch.bfloat16)
    prompts = [TextPrompts(classes=["thing"])]
    with torch.no_grad():
        out = wrapper(images, prompts, box_hints=None)
    assert "pred_masks" in out  # forward produced native output dict
```

- [ ] **Step 2: Run (skips on CPU host)**

Run: `uv run pytest tests/gpu/test_channel_adapter_gpu.py::test_G1_real_forward_nchannel -q`
Expected: SKIP on CPU; PASS on a real GPU host.

- [ ] **Step 3: Commit**

```bash
git add tests/gpu/test_channel_adapter_gpu.py
git commit -m "test(gpu): G1 real SAM3.1 N-channel forward through adapter (spec §12 G1)"
```

---

## Task 19: GPU test G2 — checkpoint round-trip of adapter weights

**Files:** Modify `tests/gpu/test_channel_adapter_gpu.py` (append)

- [ ] **Step 1: Write G2**

```python
# tests/gpu/test_channel_adapter_gpu.py  (append)
def test_G2_checkpoint_roundtrip_adapter_weights(tmp_path):
    """save_full_state -> load_full_state restores channel-adapter weights bit-for-bit."""
    import torch
    from custom_sam_peft.config.schema import ModelConfig
    from custom_sam_peft.models.sam3 import load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora
    from custom_sam_peft.config.schema import PEFTConfig
    from custom_sam_peft.train.checkpoint import _save_channel_adapter, _load_channel_adapter

    w = load_sam31(ModelConfig(), channels=4, channel_semantics="rgba").cuda()
    apply_lora(w, PEFTConfig(method="lora", r=4))
    # perturb the adapter to simulate training
    with torch.no_grad():
        w.model.channel_adapter.weight.add_(torch.randn_like(w.model.channel_adapter.weight))
    before = w.model.channel_adapter.weight.detach().clone()

    _save_channel_adapter(w, tmp_path)
    with torch.no_grad():
        w.model.channel_adapter.weight.zero_()
    _load_channel_adapter(w, tmp_path)
    assert torch.equal(w.model.channel_adapter.weight.cpu(), before.cpu())
```

(If a full `save_full_state`/`load_full_state` cycle is feasible on the GPU host, prefer that for a true end-to-end guard; the helper-level test above is the minimum guaranteed real-`state_dict` check.)

- [ ] **Step 2: Run** — Run: `uv run pytest tests/gpu/test_channel_adapter_gpu.py::test_G2_checkpoint_roundtrip_adapter_weights -q` — SKIP on CPU; PASS on GPU.

- [ ] **Step 3: Commit**

```bash
git add tests/gpu/test_channel_adapter_gpu.py
git commit -m "test(gpu): G2 channel-adapter checkpoint round-trip (spec §12 G2, risk #1)"
```

---

## Task 20: GPU test G3 — export-bundle reload with adapter

**Files:** Modify `tests/gpu/test_channel_adapter_gpu.py` (append)

- [ ] **Step 1: Write G3**

```python
# tests/gpu/test_channel_adapter_gpu.py  (append)
def test_G3_export_bundle_reload_adapter(tmp_path):
    """run_export then reload via load_sam31 + load_adapter restores adapter weights."""
    import torch
    from custom_sam_peft.config.schema import ModelConfig
    from custom_sam_peft.models.sam3 import load_sam31
    from custom_sam_peft.train.checkpoint import save_adapter, load_adapter
    from custom_sam_peft.peft_adapters.lora import apply_lora
    from custom_sam_peft.config.schema import PEFTConfig

    w = load_sam31(ModelConfig(), channels=4, channel_semantics="rgba").cuda()
    apply_lora(w, PEFTConfig(method="lora", r=4))
    with torch.no_grad():
        w.model.channel_adapter.weight.normal_()
    before = w.model.channel_adapter.weight.detach().cpu().clone()
    save_adapter(w, tmp_path / "exp")

    w2 = load_sam31(ModelConfig(), channels=4, channel_semantics="rgba").cuda()
    load_adapter(w2, tmp_path / "exp")
    assert torch.equal(w2.model.channel_adapter.weight.detach().cpu(), before)
```

- [ ] **Step 2: Run** — SKIP on CPU; PASS on GPU.

- [ ] **Step 3: Commit**

```bash
git add tests/gpu/test_channel_adapter_gpu.py
git commit -m "test(gpu): G3 export-bundle adapter reload (spec §12 G3, §10.3)"
```

---

## Task 21: GPU test G4 — real-model N-channel predict forward

**Files:** Create `tests/gpu/test_predict_nchannel_gpu.py`

- [ ] **Step 1: Write G4**

```python
# tests/gpu/test_predict_nchannel_gpu.py
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.requires_compatible_gpu,
    pytest.mark.requires_checkpoint,
]


def test_G4_real_nchannel_predict(tmp_path):
    """run_predict on a non-rgb multi-channel image produces predictions without error."""
    from custom_sam_peft.predict.options import PredictOptions  # confirm path
    from custom_sam_peft.predict.runner import run_predict

    # 4-channel npy image matching an rgba-config bundle.
    img = (np.random.rand(256, 256, 4)).astype(np.float32)
    img_path = tmp_path / "img.npy"
    np.save(img_path, img)

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model:\n  name: facebook/sam3.1\n"
        "data:\n  image_size: 1008\n  channels: 4\n  channel_semantics: rgba\n"
    )
    opts = PredictOptions(images=[img_path], prompts="thing", config=cfg, checkpoint=None)
    report = run_predict(opts)
    assert report.n_images >= 1
```

- [ ] **Step 2: Run** — SKIP on CPU; PASS on GPU.

- [ ] **Step 3: Commit**

```bash
git add tests/gpu/test_predict_nchannel_gpu.py
git commit -m "test(gpu): G4 real-model N-channel predict forward (spec §12 G4, §11)"
```

---

## Task 22: Final verification + GPU confirmation of OPEN QUESTION 2

**Files:** none (gate)

- [ ] **Step 1: Re-run lint + full CPU suite** — `uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy src && uv run pytest tests/unit tests/predict tests/integration -q -m "not gpu and not gpu_inspection and not requires_compatible_gpu"`. Expected: clean, PASS, coverage `>= 80`.

- [ ] **Step 2 (on a GPU host, if available):** Run `uv run pytest tests/gpu/test_channel_adapter_gpu.py tests/gpu/test_predict_nchannel_gpu.py -q`. Confirm G1–G4 pass. While doing so, **resolve OPEN QUESTION 2**: inspect whether `save_merged`'s merged `state_dict` already carries `channel_adapter.*` keys (it should, being a submodule of `wrapper.model`). If the extra `channel_adapter.pt` in `save_merged` is redundant, leave it (harmless symmetry) OR remove it from `save_merged` only — record the decision in the PR description.

- [ ] **Step 3:** Confirm no follow-up issues are needed beyond spec §14 (Bridge B, GDAL/rasterio, per-channel aug, auto-detection, freeform opt-in intensity augs). File them per the user's CLAUDE.md out-of-scope rule if not already tracked.

---

## Self-Review (planner)

**Spec coverage (every section mapped to a task):**
- §1.3 registry + profile dataclass → Task 1. §3.1/§3.2 config fields → Task 4. §3.3 cross-validation (a/b/c) + §3.4 rgb default → Task 5; regression → Task 6.
- §4 NormalizeConfig relax + §7.2 max_pixel_value knob → Task 3; max_pixel_value threading → Tasks 3 (field), 12 (Normalize call).
- §5.1/§5.2/§5.3 adapter (both init kinds, no-adapter-for-rgb) → Tasks 9, 10; ownership constraint (sibling of `_Sam3ImageAdapter.model`) → Task 11 + OPEN Q3.
- §5.4/§9 model wiring (`_validate_inputs`, signature) → Task 11; all 7 `load_sam31` call sites → Task 11 (definition + predict via Task 14) + Task 15.
- §6 reader + tifffile + both loaders + HF reconciliation (`_coerce_to_channels`) → Tasks 2, 7, 8.
- §7.1 processor-skip for non-rgb → Task 12 (C9).
- §8 three augmentation regimes → Task 12 (C7); §8.4 float-range interaction → OPEN Q1 (resolved with `brightness_by_max=False`).
- §10.1 optimizer collection (test-only) → Task 16 (C10); §10.2 checkpoint gap → Task 13; §10.3 export → Tasks 13 + 15 (bundle call site) + GPU G3.
- §11 predict `_resolve_config` + reader + normalize + construction → Task 14.
- §12 test matrix: C1→T10, C2→T10, C3→T9, C3b→T10, C4→T4, C5→T5, C6→T7, C7→T12, C8→T11, C9→T12, C10→T16, C11→T14, C12→T14, C13→T14; G1→T18, G2→T19, G3→T20, G4→T21. All present.
- §14 follow-ups → Task 22 step 3.

**Type/name consistency:** `read_image(path, channels)` and `_coerce_to_channels(obj, channels)` used consistently (Tasks 7, 8, 14). `_build_channel_adapter(channels, channel_semantics)` (Tasks 9–11). `channel_adapter` attribute name on `_Sam3ImageAdapter` (Tasks 11, 13, 16, 19, 20). `_CHANNEL_ADAPTER_FILENAME` / `_save_channel_adapter` / `_load_channel_adapter` / `_wrapper_channel_adapter` (Task 13). `_ResolvedConfig.channels`/`.channel_semantics` (Task 14). Real class names `COCODataset`/`HFDataset` (Task 8). `load_sam31(..., *, channels, channel_semantics)` keyword-only, defaulted — consistent across definition (T11) and call sites (T15).

**Ordering hazards addressed:** registry before validation/adapter; reader before loaders/predict; adapter before checkpoint/optimizer-test; `load_sam31` signature (defaulted) before the call-site sweep. The `normalize: NormalizeConfig | None = None` sentinel (Task 4) is materialized in Task 5's validator IN THE SAME PR before any green claim; Task 6 regression-guards existing configs. Lint/type/CPU gate (Task 17) runs before GPU tasks; final gate (Task 22) re-runs everything.

**Flagged open questions:** (1) `brightness_by_max=False` mechanism for float-range intensity augs — plan-decided, escalate if reviewer disagrees. (2) `save_merged` redundancy of `channel_adapter.pt` — confirmed at GPU G3 (Task 22). (3) Adapter ownership on `_Sam3ImageAdapter` — plan-decided per §5.1 constraint.
