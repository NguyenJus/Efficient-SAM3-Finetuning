# esam3 data-loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the data-loading subsystem per `docs/superpowers/specs/2026-05-16-data-loading-design.md` — extend `config/schema.py` with five new models, replace four `data/` stubs (`coco.py`, `hf.py`, `transforms.py`, `collate.py`), add core deps (`albumentations`, `opencv-python-headless`, promote `pillow`), and ship ~48 unit tests against the existing `tests/fixtures/tiny_coco/` fixture.

**Architecture:** Two `@register("dataset", ...)` adapters convert COCO instance JSON or HuggingFace `datasets` rows into `Example` dataclasses (defined in the immutable `data/base.py`). Albumentations runs resize-longest-edge + zero-pad-to-square + normalize geometry; `transformers.AutoImageProcessor` is the first-try source of normalization stats, with `NormalizeConfig` defaults as fallback. `collate_batch` stacks images into `(B,3,H,W)` and keeps ragged prompts/instances as Python lists.

**Tech Stack:** Python 3.13, pydantic v2, pycocotools, datasets, transformers (`AutoImageProcessor` only), albumentations >=1.4, opencv-python-headless >=4.10, pillow >=10, torch, pytest, pytest-cov.

**Commit-message convention:** Conventional Commits — `feat(data):`, `feat(config):`, `test(data):`, `chore(deps):`, `docs(data):`, `refactor(tests):`. One short imperative subject (<=72 chars), lowercase body after the scope prefix. Every commit on `spec/data-loading`.

---

## File Structure

### Modified

- `src/esam3/config/schema.py` — add `TextPromptMode` alias, `TextPromptConfig`, `NormalizeConfig`, `HFFieldMap`, `HFDatasetConfig`; extend `DataConfig` with `hf`, `text_prompt`, `normalize`, plus `_check_format_specific` model validator.
- `src/esam3/data/coco.py` — replace stub with full `COCODataset` + module-private helpers + `build_coco` builder (now keyword-only `model_name`, `pipeline`).
- `src/esam3/data/hf.py` — replace stub with full `HFDataset` + module-private helpers + `HFFieldError` + `build_hf` builder.
- `src/esam3/data/transforms.py` — replace stubs with `build_eval_transforms`, `build_train_transforms`, `resolve_normalization`.
- `src/esam3/data/collate.py` — replace stub with full `collate_batch`.
- `pyproject.toml` — add `albumentations>=1.4`, `opencv-python-headless>=4.10`, promote `pillow>=10` to core deps (remove from `[dependency-groups].dev`).
- `tests/conftest.py` — update `tiny_coco_dataset` fixture to pass the new keyword-only `transforms` and `text_prompt` args.
- `tests/unit/test_stubs_raise.py` — drop `test_data_stubs` (the data layer is implemented; stubs no longer raise).
- `configs/examples/coco_text_lora.yaml` — append `text_prompt` and `normalize` blocks (defaults; behavior unchanged).
- `configs/examples/coco_bbox_qlora.yaml` — append `text_prompt` and `normalize` blocks (defaults; behavior unchanged).
- `ARCHITECTURE.md` — one-line edit on the `data/coco.py / hf.py` line to mention `pipeline` and `model_name` keyword args.
- `logs/TODO.md` — append two `[DEFERRED]` entries.
- `logs/log.md` — appended once per task by the implementer per the global CLAUDE.md convention.

### Created

- `tests/unit/test_data_schema_extensions.py` — 10 schema tests.
- `tests/unit/test_data_transforms.py` — 7 transforms tests.
- `tests/unit/test_data_collate.py` — 6 collate tests.
- `tests/unit/test_data_coco.py` — 17 COCO tests.
- `tests/unit/test_data_hf.py` — 8 HF tests.
- `tests/unit/test_data_import_boundary.py` — 1 AST scan test verifying the data layer does not import `TrainConfig`.

### Untouched (do NOT modify)

- `src/esam3/data/base.py` — the protocols and dataclasses are the stable seam.
- `src/esam3/_registry.py` — used as-is.
- `src/esam3/config/loader.py` — `_PATH_KEYS` is left alone (assumption in spec §3.6 / §10.1).

---

## Task ordering rationale

Bottom-up, TDD throughout. Build deps first so all tests can import the same modules. Then schema (no upstream deps), then the lowest-coupling data utilities (`collate.py`, `transforms.py`), then COCO, then HF. End with example YAML refresh, docs touch-up, the boundary test, the stub-test cleanup, and a final clean sweep. Total: 22 tasks. Each task ends with one commit.

---

### Task 1: Add core dependencies (difficulty: L)

**Files:**
- Modify: `pyproject.toml:9-20` (add deps), `pyproject.toml:27-36` (remove pillow from dev)
- Test: `tests/unit/test_data_deps_smoke.py` (created)

- [ ] **Step 1: Write the failing import-smoke test**

Create `tests/unit/test_data_deps_smoke.py`:

```python
"""Smoke-import the new data-loading deps so missing-dep failures surface here."""

from __future__ import annotations


def test_albumentations_imports() -> None:
    import albumentations as A

    assert hasattr(A, "Compose")
    assert hasattr(A, "LongestMaxSize")
    assert hasattr(A, "PadIfNeeded")
    assert hasattr(A, "Normalize")
    assert hasattr(A, "HorizontalFlip")
    assert hasattr(A, "ColorJitter")
    assert hasattr(A, "BboxParams")


def test_albumentations_to_tensor_v2_imports() -> None:
    from albumentations.pytorch import ToTensorV2

    assert ToTensorV2 is not None


def test_cv2_imports_headless() -> None:
    import cv2

    assert hasattr(cv2, "INTER_LINEAR")
    assert hasattr(cv2, "BORDER_CONSTANT")


def test_pycocotools_mask_imports() -> None:
    from pycocotools import mask as coco_mask

    assert hasattr(coco_mask, "frPyObjects")
    assert hasattr(coco_mask, "decode")


def test_datasets_imports() -> None:
    import datasets

    assert hasattr(datasets, "Dataset")
    assert hasattr(datasets, "load_dataset")


def test_pillow_imports() -> None:
    from PIL import Image

    assert hasattr(Image, "open")
```

- [ ] **Step 2: Run test to confirm albumentations and cv2 are missing**

```bash
uv run pytest tests/unit/test_data_deps_smoke.py -v
```

Expected: `test_albumentations_imports` FAIL with `ModuleNotFoundError: No module named 'albumentations'`; `test_cv2_imports_headless` FAIL with `ModuleNotFoundError: No module named 'cv2'`. Other four PASS (deps already present).

- [ ] **Step 3: Edit `pyproject.toml` — add two deps to `[project].dependencies`**

Replace the `dependencies` block at `pyproject.toml:9-20` with:

```toml
dependencies = [
  "torch>=2.4",
  "transformers>=4.50",
  "peft>=0.13",
  "datasets>=3.0",
  "pydantic>=2.7",
  "typer>=0.12",
  "pyyaml>=6.0",
  "pycocotools>=2.0",
  "numpy>=1.26",
  "rich>=13",
  "pillow>=10",
  "albumentations>=1.4",
  "opencv-python-headless>=4.10",
]
```

- [ ] **Step 4: Edit `pyproject.toml` — remove `pillow` from dev group**

Replace the `[dependency-groups]` block at `pyproject.toml:27-36` with:

```toml
[dependency-groups]
dev = [
  "ruff>=0.7",
  "mypy>=1.13",
  "pytest>=8",
  "pytest-cov>=5",
  "pre-commit>=4",
  "types-PyYAML>=6",
]
```

- [ ] **Step 5: Sync lockfile**

```bash
uv sync --all-extras --dev
```

Expected: `Resolved N packages` (N grows by ~5), exit 0. `uv.lock` updated in-place. Don't manually edit `uv.lock`.

- [ ] **Step 6: Re-run the smoke test**

```bash
uv run pytest tests/unit/test_data_deps_smoke.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Add mypy override for albumentations**

`albumentations` ships no type stubs. Add to `pyproject.toml` under the existing `[[tool.mypy.overrides]]` block at `pyproject.toml:66-68`:

Replace:

```toml
[[tool.mypy.overrides]]
module = ["pycocotools.*", "bitsandbytes.*", "wandb.*", "tensorboard.*"]
ignore_missing_imports = true
```

With:

```toml
[[tool.mypy.overrides]]
module = [
  "pycocotools.*",
  "bitsandbytes.*",
  "wandb.*",
  "tensorboard.*",
  "albumentations.*",
  "cv2",
  "datasets.*",
]
ignore_missing_imports = true
```

- [ ] **Step 8: Confirm mypy still clean**

```bash
uv run mypy
```

Expected: `Success: no issues found in N source files`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock tests/unit/test_data_deps_smoke.py
git commit -m "chore(deps): add albumentations, opencv-headless; promote pillow to core"
```

---

### Task 2: Append deferred-work TODO entries (difficulty: L)

**Files:**
- Modify: `logs/TODO.md` (append two lines)

- [ ] **Step 1: Append the two `[DEFERRED]` lines**

Append to `logs/TODO.md` (file currently contains only the header comment):

```
[2026-05-16] [planner] [DEFERRED] revisit iscrowd handling after first real eval pass — v0 drops iscrowd=1 annotations entirely
[2026-05-16] [planner] [DEFERRED] named transform suites — let users pick "default" / "augmentation_heavy" / "geometric_only" from a menu instead of editing aug params
```

- [ ] **Step 2: Commit**

```bash
git add logs/TODO.md
git commit -m "docs(data): record deferred iscrowd + named-transform-suite work"
```

---

### Task 3: `TextPromptMode` + `TextPromptConfig` schema (difficulty: L)

**Files:**
- Modify: `src/esam3/config/schema.py:1-26` (imports + Literal aliases) and bottom of file (new class)
- Test: `tests/unit/test_data_schema_extensions.py` (created)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_data_schema_extensions.py` with the first two cases:

```python
"""Tests for the new data-loading config schema additions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esam3.config.schema import TextPromptConfig


def test_text_prompt_config_defaults() -> None:
    cfg = TextPromptConfig()
    assert cfg.mode == "present"
    assert cfg.negatives_per_image == 0
    assert cfg.k == 16


def test_text_prompt_config_k_bounded() -> None:
    with pytest.raises(ValidationError):
        TextPromptConfig(k=17)
    with pytest.raises(ValidationError):
        TextPromptConfig(k=0)
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py -v
```

Expected: `ImportError: cannot import name 'TextPromptConfig' from 'esam3.config.schema'`.

- [ ] **Step 3: Add the type alias and class to `schema.py`**

In `src/esam3/config/schema.py`, add `TextPromptMode` next to the other `Literal` aliases at the top (after line 22 `TrackerBackend = ...`):

```python
TextPromptMode = Literal["present", "all", "present_plus_negatives", "sampled_fixed_k"]
```

Append `TextPromptConfig` to the end of the file (after `class TrainConfig` block):

```python
class TextPromptConfig(_Strict):
    """How TextPrompts.classes is populated for each image when prompt_mode='text'.

    - present:                Use exactly the categories present in the image's
                              annotations (post-iscrowd filter). Default.
    - all:                    Use the full dataset class vocabulary every time.
    - present_plus_negatives: Use the present categories plus N randomly-sampled
                              negative class names per image.
    - sampled_fixed_k:        Use exactly k class names: all positives, plus
                              negatives sampled to reach k. If positives exceed
                              k, positives are truncated (kept in dense-id
                              ascending order). Deterministic given (seed, image_id).
    """

    mode: TextPromptMode = "present"
    negatives_per_image: int = Field(default=0, ge=0)
    k: int = Field(default=16, ge=1, le=16)
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Type-check**

```bash
uv run mypy
```

Expected: `Success`.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_data_schema_extensions.py
git commit -m "feat(config): add TextPromptMode + TextPromptConfig schema"
```

---

### Task 4: `NormalizeConfig` schema (difficulty: L)

**Files:**
- Modify: `src/esam3/config/schema.py` (append class + validator)
- Test: `tests/unit/test_data_schema_extensions.py` (append cases)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_data_schema_extensions.py`:

```python
from esam3.config.schema import NormalizeConfig


def test_normalize_config_defaults() -> None:
    cfg = NormalizeConfig()
    assert cfg.mean == [0.485, 0.456, 0.406]
    assert cfg.std == [0.229, 0.224, 0.225]
    assert len(cfg.mean) == 3 and len(cfg.std) == 3


def test_normalize_config_validation_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[0.1, 0.2], std=[0.1, 0.1, 0.1])


def test_normalize_config_validation_rejects_nonpositive_std() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[0.1, 0.1, 0.1], std=[0.0, 0.1, 0.1])


def test_normalize_config_validation_rejects_mean_out_of_range() -> None:
    with pytest.raises(ValidationError):
        NormalizeConfig(mean=[1.5, 0.1, 0.1], std=[0.1, 0.1, 0.1])
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py::test_normalize_config_defaults -v
```

Expected: `ImportError: cannot import name 'NormalizeConfig'`.

- [ ] **Step 3: Add `NormalizeConfig` to `schema.py`**

At the top of `schema.py`, expand the existing pydantic import line (`from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt`) to include `model_validator`:

```python
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator
```

Append below `TextPromptConfig`:

```python
class NormalizeConfig(_Strict):
    """Normalization stats used when AutoImageProcessor cannot be loaded.

    Resolution order at dataset construction:
      1. AutoImageProcessor.from_pretrained(model.name, local_files_only=True)
         and read image_mean/image_std.
      2. On OSError/AttributeError/ValueError, fall back to (mean, std) here.
    """

    mean: list[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406], min_length=3, max_length=3
    )
    std: list[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225], min_length=3, max_length=3
    )

    @model_validator(mode="after")
    def _check_ranges(self) -> "NormalizeConfig":
        for m in self.mean:
            if not (0.0 <= m <= 1.0):
                raise ValueError(f"normalize.mean values must be in [0, 1]; got {m}")
        for s in self.std:
            if s <= 0.0:
                raise ValueError(f"normalize.std values must be > 0; got {s}")
        return self
```

- [ ] **Step 4: Run all schema tests**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_data_schema_extensions.py
git commit -m "feat(config): add NormalizeConfig with range validator"
```

---

### Task 5: `HFFieldMap` schema (difficulty: L)

**Files:**
- Modify: `src/esam3/config/schema.py` (append class)
- Test: `tests/unit/test_data_schema_extensions.py` (append cases)

- [ ] **Step 1: Append failing test**

Append to `tests/unit/test_data_schema_extensions.py`:

```python
from esam3.config.schema import HFFieldMap


def test_hf_field_map_defaults() -> None:
    fm = HFFieldMap()
    assert fm.image == "image"
    assert fm.bbox == "objects.bbox"
    assert fm.category == "objects.category"
    assert fm.segmentation == "objects.segmentation"
    assert fm.categories_feature == "categories"
    assert fm.bbox_format == "xyxy"


def test_hf_field_map_segmentation_can_be_none() -> None:
    fm = HFFieldMap(segmentation=None)
    assert fm.segmentation is None


def test_hf_field_map_rejects_invalid_bbox_format() -> None:
    with pytest.raises(ValidationError):
        HFFieldMap(bbox_format="cxcywh")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py::test_hf_field_map_defaults -v
```

Expected: `ImportError: cannot import name 'HFFieldMap'`.

- [ ] **Step 3: Add `HFFieldMap` to `schema.py`**

Append below `NormalizeConfig`:

```python
class HFFieldMap(_Strict):
    """Optional overrides for HuggingFace dataset field names.

    Defaults match a conventional schema: top-level `image`, nested `objects.bbox`,
    `objects.category`, optional `objects.segmentation`; class names from the
    top-level `categories` feature.
    """

    image: str = "image"
    bbox: str = "objects.bbox"
    category: str = "objects.category"
    segmentation: str | None = "objects.segmentation"
    categories_feature: str = "categories"
    bbox_format: Literal["xywh", "xyxy"] = "xyxy"
```

- [ ] **Step 4: Run all schema tests**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_data_schema_extensions.py
git commit -m "feat(config): add HFFieldMap schema with conventional defaults"
```

---

### Task 6: `HFDatasetConfig` schema (difficulty: L)

**Files:**
- Modify: `src/esam3/config/schema.py` (append class)
- Test: `tests/unit/test_data_schema_extensions.py` (append cases)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_data_schema_extensions.py`:

```python
from esam3.config.schema import HFDatasetConfig


def test_hf_dataset_config_required_name() -> None:
    with pytest.raises(ValidationError):
        HFDatasetConfig()  # type: ignore[call-arg]


def test_hf_dataset_config_defaults() -> None:
    cfg = HFDatasetConfig(name="my-org/my-ds")
    assert cfg.name == "my-org/my-ds"
    assert cfg.split_train == "train"
    assert cfg.split_val == "validation"
    assert cfg.field_map.bbox == "objects.bbox"


def test_hf_dataset_config_name_min_length() -> None:
    with pytest.raises(ValidationError):
        HFDatasetConfig(name="")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py::test_hf_dataset_config_defaults -v
```

Expected: `ImportError: cannot import name 'HFDatasetConfig'`.

- [ ] **Step 3: Add `HFDatasetConfig` to `schema.py`**

Append below `HFFieldMap`:

```python
class HFDatasetConfig(_Strict):
    """HuggingFace dataset specification (used when DataConfig.format == 'hf')."""

    name: str = Field(min_length=1)
    split_train: str = "train"
    split_val: str = "validation"
    field_map: HFFieldMap = Field(default_factory=HFFieldMap)
```

- [ ] **Step 4: Run all schema tests**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_data_schema_extensions.py
git commit -m "feat(config): add HFDatasetConfig schema"
```

---

### Task 7: Extend `DataConfig` with new fields + format validator (difficulty: M)

**Files:**
- Modify: `src/esam3/config/schema.py:51-58` (DataConfig)
- Test: `tests/unit/test_data_schema_extensions.py` (append cases); existing `tests/unit/test_config_schema.py` must still pass unchanged.

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_data_schema_extensions.py`:

```python
from pathlib import Path

from esam3.config.schema import DataConfig, DataSplit, TrainConfig


def _minimal_data(format: str = "coco") -> dict[str, object]:
    return {
        "format": format,
        "train": {"annotations": "a.json", "images": "imgs/"},
        "val": {"annotations": "a.json", "images": "imgs/"},
        "prompt_mode": "bbox",
    }


def test_data_config_accepts_coco_without_hf() -> None:
    cfg = DataConfig.model_validate(_minimal_data("coco"))
    assert cfg.hf is None
    assert cfg.text_prompt.mode == "present"
    assert cfg.normalize.mean == [0.485, 0.456, 0.406]


def test_data_config_requires_hf_when_format_hf() -> None:
    with pytest.raises(ValidationError) as exc:
        DataConfig.model_validate(_minimal_data("hf"))
    assert "data.hf" in str(exc.value)


def test_data_config_accepts_hf_with_hf_block() -> None:
    d = _minimal_data("hf")
    d["hf"] = {"name": "cppe-5"}
    cfg = DataConfig.model_validate(d)
    assert cfg.hf is not None
    assert cfg.hf.name == "cppe-5"


def test_data_config_accepts_text_prompt_override() -> None:
    d = _minimal_data("coco")
    d["text_prompt"] = {"mode": "present_plus_negatives", "negatives_per_image": 3}
    cfg = DataConfig.model_validate(d)
    assert cfg.text_prompt.mode == "present_plus_negatives"
    assert cfg.text_prompt.negatives_per_image == 3


def test_existing_example_yaml_still_validates() -> None:
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    for name in ("coco_text_lora.yaml", "coco_bbox_qlora.yaml"):
        p = repo_root / "configs" / "examples" / name
        raw = yaml.safe_load(p.read_text())
        TrainConfig.model_validate(raw)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py::test_data_config_accepts_coco_without_hf -v
```

Expected: `AttributeError: 'DataConfig' object has no attribute 'hf'` (or similar — the new fields aren't on the model yet).

- [ ] **Step 3: Replace `DataConfig` in `schema.py`**

Replace the `DataConfig` class at `src/esam3/config/schema.py:51-58` with:

```python
class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit
    val: DataSplit
    hf: HFDatasetConfig | None = None
    prompt_mode: PromptMode
    image_size: PositiveInt = 1024
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)
    text_prompt: TextPromptConfig = Field(default_factory=TextPromptConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)

    @model_validator(mode="after")
    def _check_format_specific(self) -> "DataConfig":
        if self.format == "hf" and self.hf is None:
            raise ValueError("data.hf is required when data.format == 'hf'")
        return self
```

Note: `TextPromptConfig`, `NormalizeConfig`, `HFDatasetConfig`, and `HFFieldMap` are defined below `DataConfig` in the current file. Python evaluates pydantic default-factories lazily, but the type annotations are resolved at class-body time. Since `from __future__ import annotations` is in effect on line 8, annotations are strings and resolution is deferred — both orderings work. Leave the new classes where Tasks 3-6 placed them (at end of file) and accept this forward-reference.

- [ ] **Step 4: Run all schema tests + existing schema tests + config loader tests**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v
```

Expected: every test passes — 12 (this file's previous total) + 5 new = 17 in `test_data_schema_extensions.py`; `test_config_schema.py` and `test_config_loader.py` unchanged. mypy still clean.

- [ ] **Step 5: Type-check**

```bash
uv run mypy
```

Expected: `Success`.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/config/schema.py tests/unit/test_data_schema_extensions.py
git commit -m "feat(config): extend DataConfig with hf, text_prompt, normalize"
```

---

### Task 8: `resolve_normalization` helper (difficulty: M)

**Files:**
- Modify: `src/esam3/data/transforms.py` (replace stub)
- Test: `tests/unit/test_data_transforms.py` (created)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_data_transforms.py`:

```python
"""Tests for data/transforms.py."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest

from esam3.config.schema import NormalizeConfig
from esam3.data.transforms import resolve_normalization


def test_resolve_normalization_uses_image_processor_when_available(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from transformers import AutoImageProcessor

    fake_proc = SimpleNamespace(image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])

    def fake_from_pretrained(name: str, **kwargs: object) -> object:
        assert name == "facebook/sam3.1"
        assert kwargs.get("local_files_only") is True
        return fake_proc

    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", fake_from_pretrained)
    caplog.set_level(logging.INFO, logger="esam3.data.transforms")
    mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())
    assert mean == [0.1, 0.2, 0.3]
    assert std == [0.4, 0.5, 0.6]
    assert any(
        re.search(r"Using image_mean/image_std from AutoImageProcessor", rec.message)
        for rec in caplog.records
    )


def test_resolve_normalization_falls_back_on_oserror(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from transformers import AutoImageProcessor

    def boom(name: str, **kwargs: object) -> object:
        raise OSError("no cache")

    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", boom)
    caplog.set_level(logging.INFO, logger="esam3.data.transforms")
    mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())
    assert mean == [0.485, 0.456, 0.406]
    assert std == [0.229, 0.224, 0.225]
    assert any(
        re.search(r"AutoImageProcessor cache miss", rec.message) for rec in caplog.records
    )


def test_resolve_normalization_falls_back_on_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoImageProcessor

    monkeypatch.setattr(
        AutoImageProcessor,
        "from_pretrained",
        lambda name, **kwargs: SimpleNamespace(),  # missing image_mean/image_std
    )
    mean, std = resolve_normalization("facebook/sam3.1", NormalizeConfig())
    assert mean == [0.485, 0.456, 0.406]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_transforms.py -v
```

Expected: 3 failures — current `transforms.py` does not export `resolve_normalization`.

- [ ] **Step 3: Replace `src/esam3/data/transforms.py`**

Overwrite the file with this content. (Later tasks expand it; for now, write only what's tested.)

```python
"""Image augmentation + normalization pipelines (Albumentations).

Public API:
  - resolve_normalization(model_name, fallback) -> (mean, std)
  - build_eval_transforms(image_size, *, model_name, normalize) -> A.Compose
  - build_train_transforms(aug_cfg, image_size, *, model_name, normalize) -> A.Compose
"""

from __future__ import annotations

import logging

from esam3.config.schema import AugmentationsConfig, NormalizeConfig

_LOG = logging.getLogger(__name__)


def resolve_normalization(
    model_name: str, fallback: NormalizeConfig
) -> tuple[list[float], list[float]]:
    """Try `AutoImageProcessor.from_pretrained(model_name, local_files_only=True)`.

    On success, read `image_mean` / `image_std`. On any of `(OSError, AttributeError,
    ValueError)`, return the fallback's (mean, std). Emits exactly one INFO log line.
    """
    from transformers import AutoImageProcessor

    try:
        proc = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        mean = list(proc.image_mean)
        std = list(proc.image_std)
    except (OSError, AttributeError, ValueError):
        _LOG.info(
            "AutoImageProcessor cache miss for %r; falling back to NormalizeConfig "
            "(mean=%s, std=%s).",
            model_name,
            fallback.mean,
            fallback.std,
        )
        return list(fallback.mean), list(fallback.std)
    else:
        _LOG.info(
            "Using image_mean/image_std from AutoImageProcessor for %r.", model_name
        )
        return mean, std


def build_eval_transforms(
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> object:
    raise NotImplementedError("filled in by Task 9")


def build_train_transforms(
    aug_cfg: AugmentationsConfig,
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> object:
    raise NotImplementedError("filled in by Task 10")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_data_transforms.py -v
```

Expected: 3 passed.

- [ ] **Step 5: mypy**

```bash
uv run mypy
```

Expected: `Success`.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/data/transforms.py tests/unit/test_data_transforms.py
git commit -m "feat(data): resolve_normalization with AutoImageProcessor + fallback"
```

---

### Task 9: `build_eval_transforms` (difficulty: M)

**Files:**
- Modify: `src/esam3/data/transforms.py` (replace stub body)
- Test: `tests/unit/test_data_transforms.py` (append cases)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_data_transforms.py`:

```python
import numpy as np
import torch

from esam3.data.transforms import build_eval_transforms


def _patch_proc_to_imagenet(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoImageProcessor

    monkeypatch.setattr(
        AutoImageProcessor,
        "from_pretrained",
        lambda name, **kwargs: (_ for _ in ()).throw(OSError("no cache")),
    )


def test_eval_transforms_resizes_to_square(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_proc_to_imagenet(monkeypatch)
    compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    masks = [np.ones((40, 80), dtype=np.uint8)]
    out = compose(image=img, bboxes=[[0.0, 0.0, 80.0, 40.0]], masks=masks, class_labels=[0])
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 64, 64)
    assert out["image"].dtype == torch.float32
    bx = out["bboxes"][0]
    assert 0 <= bx[0] <= 1 and 0 <= bx[1] <= 1
    assert 60 <= bx[2] <= 64 and 28 <= bx[3] <= 36
    assert out["masks"][0].shape == (64, 64)


def test_eval_transforms_pad_position_top_left(monkeypatch: pytest.MonkeyPatch) -> None:
    """The right/bottom region should be zero-padded (top-left preserves original)."""
    _patch_proc_to_imagenet(monkeypatch)
    compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
    img = np.full((32, 64, 3), 255, dtype=np.uint8)
    out = compose(image=img, bboxes=[], masks=[], class_labels=[])
    # After normalize: padded zeros become -mean/std (negative). Original 255 -> positive.
    # Top-left row should be > 0; bottom-right row should be < 0.
    top_row = out["image"][0, 0, :]
    bottom_row = out["image"][0, 60, :]
    assert top_row.mean().item() > 0
    assert bottom_row.mean().item() < 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_transforms.py::test_eval_transforms_resizes_to_square -v
```

Expected: `NotImplementedError: filled in by Task 9`.

- [ ] **Step 3: Implement `build_eval_transforms` in `transforms.py`**

Replace the `build_eval_transforms` stub body with:

```python
def build_eval_transforms(
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> "A.Compose":
    """Deterministic eval pipeline: longest-edge resize -> top-left pad -> normalize -> ToTensor."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    mean, std = resolve_normalization(model_name, normalize)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                position="top_left",
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )
```

Add the `TYPE_CHECKING` import above so `A.Compose` resolves under mypy without runtime import cost at module import:

Top of file (modify the existing imports block):

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from esam3.config.schema import AugmentationsConfig, NormalizeConfig

if TYPE_CHECKING:
    import albumentations as A
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_data_transforms.py -v
```

Expected: 5 passed (3 from Task 8 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/esam3/data/transforms.py tests/unit/test_data_transforms.py
git commit -m "feat(data): build_eval_transforms (resize+pad+normalize+ToTensor)"
```

---

### Task 10: `build_train_transforms` (difficulty: M)

**Files:**
- Modify: `src/esam3/data/transforms.py` (replace stub body)
- Test: `tests/unit/test_data_transforms.py` (append cases)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_data_transforms.py`:

```python
import random

from esam3.config.schema import AugmentationsConfig
from esam3.data.transforms import build_train_transforms


def test_train_transforms_deterministic_with_seeded_global_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proc_to_imagenet(monkeypatch)
    aug = AugmentationsConfig(hflip=True, color_jitter=0.1)

    def run() -> torch.Tensor:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        compose = build_train_transforms(aug, 64, model_name="x", normalize=NormalizeConfig())
        img = np.arange(40 * 80 * 3, dtype=np.uint8).reshape(40, 80, 3)
        return compose(image=img, bboxes=[], masks=[], class_labels=[])["image"]

    a = run()
    b = run()
    assert torch.equal(a, b)


def test_train_transforms_hflip_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_proc_to_imagenet(monkeypatch)
    aug = AugmentationsConfig(hflip=False, color_jitter=0.0)
    compose = build_train_transforms(aug, 64, model_name="x", normalize=NormalizeConfig())
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    img[:, :8, 0] = 200  # strong left column marker
    flips = 0
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        out = compose(image=img.copy(), bboxes=[], masks=[], class_labels=[])
        left = out["image"][0, :32, :8].mean().item()
        right = out["image"][0, :32, 56:64].mean().item()
        if right > left:
            flips += 1
    assert flips == 0


def test_train_transforms_color_jitter_zero_preserves_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_proc_to_imagenet(monkeypatch)
    eval_compose = build_eval_transforms(64, model_name="x", normalize=NormalizeConfig())
    train_compose = build_train_transforms(
        AugmentationsConfig(hflip=False, color_jitter=0.0),
        64,
        model_name="x",
        normalize=NormalizeConfig(),
    )
    img = np.random.RandomState(0).randint(0, 256, size=(40, 60, 3), dtype=np.uint8)
    e_out = eval_compose(image=img, bboxes=[], masks=[], class_labels=[])
    t_out = train_compose(image=img, bboxes=[], masks=[], class_labels=[])
    # With p=0.5 ColorJitter but jitter magnitudes all 0, the result must equal eval.
    assert torch.allclose(e_out["image"], t_out["image"], atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_transforms.py::test_train_transforms_hflip_disabled -v
```

Expected: `NotImplementedError: filled in by Task 10`.

- [ ] **Step 3: Implement `build_train_transforms`**

Replace the `build_train_transforms` stub body in `transforms.py`:

```python
def build_train_transforms(
    aug_cfg: AugmentationsConfig,
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> "A.Compose":
    """Train pipeline: resize+pad geometry + optional hflip + color jitter + normalize."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    mean, std = resolve_normalization(model_name, normalize)
    steps: list[object] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            position="top_left",
        ),
    ]
    if aug_cfg.hflip:
        steps.append(A.HorizontalFlip(p=0.5))
    steps.append(
        A.ColorJitter(
            brightness=aug_cfg.color_jitter,
            contrast=aug_cfg.color_jitter,
            saturation=aug_cfg.color_jitter,
            hue=aug_cfg.color_jitter * 0.5,
            p=0.5,
        )
    )
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

- [ ] **Step 4: Run all transform tests**

```bash
uv run pytest tests/unit/test_data_transforms.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/data/transforms.py tests/unit/test_data_transforms.py
git commit -m "feat(data): build_train_transforms with hflip + color jitter"
```

---

### Task 11: `collate_batch` (difficulty: L)

**Files:**
- Modify: `src/esam3/data/collate.py` (replace stub)
- Test: `tests/unit/test_data_collate.py` (created)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_data_collate.py`:

```python
"""Tests for data/collate.py."""

from __future__ import annotations

import pytest
import torch

from esam3.data.base import BoxPrompts, Example, Instance, TextPrompts
from esam3.data.collate import collate_batch


def _ex(image_id: str, shape: tuple[int, int, int] = (3, 64, 64)) -> Example:
    return Example(
        image=torch.zeros(shape, dtype=torch.float32),
        image_id=image_id,
        prompts=TextPrompts(classes=["a"]),
        instances=[
            Instance(
                mask=torch.zeros((shape[1], shape[2]), dtype=torch.bool),
                class_id=0,
                box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
            )
        ],
    )


def test_collate_stacks_images() -> None:
    batch = collate_batch([_ex("a"), _ex("b"), _ex("c")])
    assert batch["images"].shape == (3, 3, 64, 64)
    assert batch["images"].dtype == torch.float32


def test_collate_keeps_prompts_as_list() -> None:
    a = _ex("a")
    b = Example(
        image=torch.zeros((3, 64, 64)),
        image_id="b",
        prompts=BoxPrompts(
            boxes=torch.zeros((2, 4)), class_ids=torch.tensor([0, 1], dtype=torch.int64)
        ),
        instances=[],
    )
    c = _ex("c")
    batch = collate_batch([a, b, c])
    assert isinstance(batch["prompts"], list)
    assert len(batch["prompts"]) == 3
    assert isinstance(batch["prompts"][0], TextPrompts)
    assert isinstance(batch["prompts"][1], BoxPrompts)
    assert isinstance(batch["prompts"][2], TextPrompts)


def test_collate_keeps_instances_as_list_of_lists() -> None:
    a = _ex("a")
    b = Example(
        image=torch.zeros((3, 64, 64)),
        image_id="b",
        prompts=TextPrompts(classes=["a"]),
        instances=[],
    )
    batch = collate_batch([a, b])
    assert isinstance(batch["instances"], list)
    assert len(batch["instances"]) == 2
    assert len(batch["instances"][0]) == 1
    assert len(batch["instances"][1]) == 0


def test_collate_image_id_order_preserved() -> None:
    batch = collate_batch([_ex("z"), _ex("y"), _ex("x")])
    assert batch["image_ids"] == ["z", "y", "x"]


def test_collate_empty_batch_raises() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_batch([])


def test_collate_image_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError) as exc:
        collate_batch([_ex("a"), _ex("b", shape=(3, 32, 32))])
    msg = str(exc.value)
    assert "(3, 64, 64)" in msg
    assert "(3, 32, 32)" in msg
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_collate.py -v
```

Expected: 6 failures; current `collate_batch` raises `NotImplementedError`.

- [ ] **Step 3: Replace `src/esam3/data/collate.py`**

```python
"""Variable-shape batch collator for the data subsystem."""

from __future__ import annotations

from typing import Any

import torch

from esam3.data.base import Example


def collate_batch(examples: list[Example]) -> dict[str, Any]:
    """Stack images, keep ragged prompts/instances as Python lists.

    Returns a dict with keys: "images" (B,3,H,W), "image_ids" (list[str]),
    "prompts" (list[Prompts]), "instances" (list[list[Instance]]).

    Raises:
        ValueError: empty input or mismatched image shapes across the batch.
    """
    if not examples:
        raise ValueError("collate_batch received empty batch")
    ref_shape = tuple(examples[0].image.shape)
    for ex in examples[1:]:
        shp = tuple(ex.image.shape)
        if shp != ref_shape:
            raise ValueError(
                f"collate_batch: image shape mismatch: {ref_shape} vs {shp}"
            )
    images = torch.stack([ex.image for ex in examples], dim=0)
    return {
        "images": images,
        "image_ids": [ex.image_id for ex in examples],
        "prompts": [ex.prompts for ex in examples],
        "instances": [list(ex.instances) for ex in examples],
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_data_collate.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/data/collate.py tests/unit/test_data_collate.py
git commit -m "feat(data): collate_batch with image stacking and ragged lists"
```

---

### Task 12: COCO module-private helpers (difficulty: M)

**Files:**
- Modify: `src/esam3/data/coco.py` (replace stub — helpers only this task)
- Test: `tests/unit/test_data_coco.py` (created)

- [ ] **Step 1: Write failing helper tests**

Create `tests/unit/test_data_coco.py`:

```python
"""Tests for data/coco.py — helpers + dataset + builder."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

from esam3.config.schema import TextPromptConfig
from esam3.data.coco import (
    _build_category_remap,
    _build_text_prompts,
    _decode_segmentation,
    _drop_crowd_only_images,
    _load_coco_index,
)


def test_load_coco_index(tiny_coco_dir: Path) -> None:
    coco = _load_coco_index(tiny_coco_dir / "annotations.json")
    assert isinstance(coco, COCO)
    assert sorted(coco.getImgIds()) == [1, 2]


def test_build_category_remap(tiny_coco_dir: Path) -> None:
    coco = _load_coco_index(tiny_coco_dir / "annotations.json")
    sparse, mapping, names = _build_category_remap(coco)
    assert sparse == [1, 2]
    assert mapping == {1: 0, 2: 1}
    assert names == ["thing_a", "thing_b"]


def test_build_category_remap_handles_sparse_ids(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "x.png", "width": 8, "height": 8}],
                "categories": [
                    {"id": 7, "name": "ginger"},
                    {"id": 3, "name": "apple"},
                ],
                "annotations": [],
            }
        )
    )
    coco = _load_coco_index(p)
    sparse, mapping, names = _build_category_remap(coco)
    assert sparse == [3, 7]
    assert mapping == {3: 0, 7: 1}
    assert names == ["apple", "ginger"]


def test_drop_crowd_only_images(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "width": 8, "height": 8},
                    {"id": 2, "file_name": "b.png", "width": 8, "height": 8},
                ],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                    },
                    {
                        "id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 1,
                    },
                ],
            }
        )
    )
    coco = _load_coco_index(p)
    kept, ann_index, dropped = _drop_crowd_only_images(coco)
    assert kept == [1]
    assert 2 not in ann_index
    assert dropped == 1


def test_decode_segmentation_polygon(tiny_coco_dir: Path) -> None:
    coco = _load_coco_index(tiny_coco_dir / "annotations.json")
    ann = coco.loadAnns([1])[0]
    mask = _decode_segmentation(ann, 32, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


def test_decode_segmentation_rle() -> None:
    """A synthetic RLE: a 4x4 mask with all ones."""
    from pycocotools import mask as mu

    rle = mu.encode(np.asfortranarray(np.ones((4, 4), dtype=np.uint8)))
    ann = {"segmentation": rle}
    out = _decode_segmentation(ann, 4, 4)
    assert out.dtype == np.bool_
    assert out.all()


def test_build_text_prompts_present() -> None:
    out = _build_text_prompts(
        present_dense_ids=[1, 0],
        class_names=["zero", "one", "two"],
        cfg=TextPromptConfig(mode="present"),
        rng=random.Random(0),
        image_id=42,
    )
    # Dense-id ascending.
    assert out == ["zero", "one"]


def test_build_text_prompts_all() -> None:
    out = _build_text_prompts(
        present_dense_ids=[1],
        class_names=["a", "b", "c"],
        cfg=TextPromptConfig(mode="all"),
        rng=random.Random(0),
        image_id=7,
    )
    assert out == ["a", "b", "c"]


def test_build_text_prompts_present_plus_negatives() -> None:
    out = _build_text_prompts(
        present_dense_ids=[0],
        class_names=["a", "b", "c", "d", "e"],
        cfg=TextPromptConfig(mode="present_plus_negatives", negatives_per_image=2),
        rng=random.Random(123),
        image_id=1,
    )
    assert out[0] == "a"  # positive first
    assert len(out) == 3
    assert len(set(out)) == 3  # no dupes


def test_build_text_prompts_sampled_fixed_k_truncates_positives() -> None:
    out = _build_text_prompts(
        present_dense_ids=list(range(10)),
        class_names=[f"c{i}" for i in range(20)],
        cfg=TextPromptConfig(mode="sampled_fixed_k", k=3),
        rng=random.Random(0),
        image_id=1,
    )
    assert len(out) == 3
    assert out == ["c0", "c1", "c2"]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_coco.py -v
```

Expected: every test fails with `ImportError` (helpers not exported yet).

- [ ] **Step 3: Replace `src/esam3/data/coco.py` with helpers + class skeleton**

This task implements only the module-private helpers (and a placeholder class). The full `COCODataset` body comes in Task 13.

```python
"""COCO instance-JSON dataset adapter.

Backed by `pycocotools.coco.COCO` for index lookups and `pycocotools.mask` for
polygon/RLE decode. Sparse COCO category ids are remapped to a dense 0..C-1
namespace; the original sparse ids are preserved on `coco_category_ids` for
eval-time round-tripping.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from esam3._registry import register
from esam3.config.schema import TextPromptConfig
from esam3.data.base import Dataset, Example

_LOG = logging.getLogger(__name__)


def _load_coco_index(ann_path: str | Path) -> COCO:
    """Load a COCO annotations JSON via pycocotools (suppresses pycocotools prints)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return COCO(str(ann_path))


def _build_category_remap(coco: COCO) -> tuple[list[int], dict[int, int], list[str]]:
    """Return `(sparse_ids_sorted, sparse_to_dense, class_names_in_dense_order)`."""
    cats = sorted(coco.dataset["categories"], key=lambda c: c["id"])
    sparse_ids = [int(c["id"]) for c in cats]
    names = [str(c["name"]) for c in cats]
    mapping = {sid: dense for dense, sid in enumerate(sparse_ids)}
    return sparse_ids, mapping, names


def _drop_crowd_only_images(
    coco: COCO,
) -> tuple[list[int], dict[int, list[dict[str, Any]]], int]:
    """Drop images that have zero non-crowd annotations.

    Returns `(image_ids_kept_sorted, ann_index_no_crowd, dropped_count)`.
    """
    kept: list[int] = []
    ann_index: dict[int, list[dict[str, Any]]] = {}
    dropped = 0
    for img_id in sorted(coco.getImgIds()):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        non_crowd = [a for a in anns if int(a.get("iscrowd", 0)) == 0]
        if not non_crowd:
            dropped += 1
            continue
        kept.append(int(img_id))
        ann_index[int(img_id)] = non_crowd
    return kept, ann_index, dropped


def _decode_segmentation(ann: dict[str, Any], h: int, w: int) -> np.ndarray:
    """Polygon or RLE -> (H, W) bool ndarray."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = coco_mask.frPyObjects(seg, h, w)
        decoded = coco_mask.decode(rles)
    elif isinstance(seg, dict):
        decoded = coco_mask.decode(seg)
    else:
        raise TypeError(f"unsupported segmentation type: {type(seg).__name__}")
    if decoded.ndim == 3:
        decoded = decoded.sum(axis=2)
    return decoded.astype(bool)


def _build_text_prompts(
    present_dense_ids: list[int],
    class_names: list[str],
    cfg: TextPromptConfig,
    rng: random.Random,
    image_id: int,
) -> list[str]:
    """Apply the configured TextPromptMode. Output order:
    positives in ascending dense-id, then negatives in deterministic order.
    """
    present_sorted = sorted(set(present_dense_ids))
    positives = [class_names[i] for i in present_sorted]
    n = len(class_names)
    if cfg.mode == "present":
        return positives
    if cfg.mode == "all":
        return list(class_names)
    if cfg.mode == "present_plus_negatives":
        pool = [i for i in range(n) if i not in set(present_sorted)]
        negatives = rng.sample(pool, k=min(cfg.negatives_per_image, len(pool)))
        return positives + [class_names[i] for i in sorted(negatives)]
    if cfg.mode == "sampled_fixed_k":
        if len(positives) >= cfg.k:
            return positives[: cfg.k]
        pool = [i for i in range(n) if i not in set(present_sorted)]
        need = cfg.k - len(positives)
        negatives = rng.sample(pool, k=min(need, len(pool)))
        return positives + [class_names[i] for i in sorted(negatives)]
    raise ValueError(f"unknown text-prompt mode: {cfg.mode}")


class COCODataset:
    """Placeholder — full impl in Task 13."""

    def __init__(self, annotations: str, images: str, prompt_mode: str) -> None:
        self.annotations = annotations
        self.images = images
        self.prompt_mode = prompt_mode

    def __len__(self) -> int:
        raise NotImplementedError("filled in by Task 13")

    def __getitem__(self, i: int) -> Example:
        raise NotImplementedError("filled in by Task 13")

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError("filled in by Task 13")


@register("dataset", "coco")
def build_coco(cfg: dict[str, Any]) -> Dataset:
    """Placeholder — full impl in Task 14."""
    return COCODataset(
        annotations=cfg["annotations"],
        images=cfg["images"],
        prompt_mode=cfg["prompt_mode"],
    )


# Suppress unused-import warnings until later tasks consume these.
_ = (Literal, np)
```

Note: the `_ = (Literal, np)` line keeps ruff F401 quiet for now; remove in Task 13. The `Dataset` import is still used by `build_coco`'s return type, retained.

- [ ] **Step 4: Run helper tests**

```bash
uv run pytest tests/unit/test_data_coco.py -v
```

Expected: 10 helper tests passed; the COCODataset stub tests don't exist yet (they're added in Task 13).

- [ ] **Step 5: Confirm the existing `test_stubs_raise::test_data_stubs` still passes**

```bash
uv run pytest tests/unit/test_stubs_raise.py::test_data_stubs -v
```

Expected: PASS — the placeholder still raises `NotImplementedError("filled in by Task 13")` which matches `"filled in by spec:"` regex? **NO**, the regex is `"filled in by spec:"`. Change the placeholder message to keep the test green during the transition. Re-edit the three `raise NotImplementedError(...)` lines in `COCODataset` to: `raise NotImplementedError("filled in by spec: spec/data-loading (Task 13)")`. Re-run.

```bash
uv run pytest tests/unit/test_stubs_raise.py::test_data_stubs -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/esam3/data/coco.py tests/unit/test_data_coco.py
git commit -m "feat(data): coco module-private helpers (index, remap, decode, prompts)"
```

---

### Task 13: `COCODataset.__init__`, `__len__`, `__getitem__`, `class_names` (difficulty: H)

**Files:**
- Modify: `src/esam3/data/coco.py` (replace placeholder class with full impl)
- Modify: `tests/conftest.py:16-32` (update `tiny_coco_dataset` fixture signature)
- Test: `tests/unit/test_data_coco.py` (append cases — the full 17-case suite)

- [ ] **Step 1: Update the `tiny_coco_dataset` conftest fixture**

Replace `tests/conftest.py:16-32` with:

```python
from esam3.config.schema import NormalizeConfig, TextPromptConfig


@pytest.fixture
def tiny_coco_dir() -> Path:
    return FIXTURES / "tiny_coco"


@pytest.fixture
def tiny_coco_dataset(tiny_coco_dir: Path) -> COCODataset:
    """A COCODataset pointing at the tiny_coco fixture (bbox prompt mode)."""
    from esam3.data.transforms import build_eval_transforms

    transforms = build_eval_transforms(
        32, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )
    return COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=transforms,
        text_prompt=TextPromptConfig(),
    )
```

Tests that exercised the old fixture will now see a real working dataset; only `test_data_stubs` previously consumed it indirectly and that's removed in Task 19.

- [ ] **Step 2: Append failing tests**

Append to `tests/unit/test_data_coco.py`:

```python
import logging
import re
from typing import Any

import torch

from esam3.config.schema import NormalizeConfig
from esam3.data.base import BoxPrompts, TextPrompts
from esam3.data.coco import COCODataset


def _patch_imagenet(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoImageProcessor

    monkeypatch.setattr(
        AutoImageProcessor,
        "from_pretrained",
        lambda name, **kwargs: (_ for _ in ()).throw(OSError("no cache")),
    )


def _build_eval(image_size: int = 32) -> Any:
    from esam3.data.transforms import build_eval_transforms

    return build_eval_transforms(
        image_size, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )


def test_class_names_dense_and_ordered(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=_build_eval(),
        text_prompt=TextPromptConfig(),
    )
    assert ds.class_names == ["thing_a", "thing_b"]
    assert ds.coco_category_ids == [1, 2]


def test_len_drops_empty_after_iscrowd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_imagenet(monkeypatch)
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "width": 8, "height": 8},
                    {"id": 2, "file_name": "b.png", "width": 8, "height": 8},
                ],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 1,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    Image.new("RGB", (8, 8)).save(images_dir / "b.png")
    caplog.set_level(logging.INFO, logger="esam3.data.coco")
    ds = COCODataset(
        annotations=str(p),
        images=str(images_dir),
        prompt_mode="bbox",
        transforms=_build_eval(8),
        text_prompt=TextPromptConfig(),
    )
    assert len(ds) == 1
    assert any(re.search(r"dropped.*1.*iscrowd", rec.message) for rec in caplog.records)


def test_getitem_text_mode_present(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text",
        transforms=_build_eval(),
        text_prompt=TextPromptConfig(mode="present"),
    )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["thing_a", "thing_b"]


def test_getitem_text_mode_all(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="text",
        transforms=_build_eval(),
        text_prompt=TextPromptConfig(mode="all"),
    )
    ex = ds[1]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["thing_a", "thing_b"]


def test_getitem_text_mode_present_plus_negatives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_imagenet(monkeypatch)
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [
                    {"id": 1, "name": "a"},
                    {"id": 2, "name": "b"},
                    {"id": 3, "name": "c"},
                    {"id": 4, "name": "d"},
                ],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    }
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    ds = COCODataset(
        annotations=str(p),
        images=str(images_dir),
        prompt_mode="text",
        transforms=_build_eval(8),
        text_prompt=TextPromptConfig(mode="present_plus_negatives", negatives_per_image=2),
        seed=42,
    )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes[0] == "a"
    assert len(ex.prompts.classes) == 3
    assert len(set(ex.prompts.classes)) == 3


def test_getitem_text_mode_sampled_fixed_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_imagenet(monkeypatch)
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [{"id": i, "name": f"c{i}"} for i in range(1, 6)],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    }
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    ds = COCODataset(
        annotations=str(p),
        images=str(images_dir),
        prompt_mode="text",
        transforms=_build_eval(8),
        text_prompt=TextPromptConfig(mode="sampled_fixed_k", k=3),
        seed=7,
    )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert len(ex.prompts.classes) == 3
    assert ex.prompts.classes[0] == "c1"


def _synth_many_cats(tmp_path: Path, n_cats: int) -> tuple[Path, Path]:
    """Build a 1-image COCO with n_cats categories, one annotation each."""
    from PIL import Image

    p = tmp_path / "ann.json"
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (32, 32)).save(images_dir / "a.png")
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
                "categories": [{"id": i + 1, "name": f"c{i}"} for i in range(n_cats)],
                "annotations": [
                    {
                        "id": i + 1, "image_id": 1, "category_id": i + 1,
                        "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    }
                    for i in range(n_cats)
                ],
            }
        )
    )
    return p, images_dir


def test_multiplex_truncation_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_imagenet(monkeypatch)
    ann, imgs = _synth_many_cats(tmp_path, 20)
    caplog.set_level(logging.WARNING, logger="esam3.data.coco")
    ds = COCODataset(
        annotations=str(ann),
        images=str(imgs),
        prompt_mode="text",
        transforms=_build_eval(32),
        text_prompt=TextPromptConfig(mode="all"),
    )
    ex = ds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert len(ex.prompts.classes) == 16
    assert any(re.search(r"truncating to 16", rec.message) for rec in caplog.records)


def test_multiplex_truncation_box(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_imagenet(monkeypatch)
    ann, imgs = _synth_many_cats(tmp_path, 20)
    caplog.set_level(logging.WARNING, logger="esam3.data.coco")
    ds = COCODataset(
        annotations=str(ann),
        images=str(imgs),
        prompt_mode="bbox",
        transforms=_build_eval(32),
        text_prompt=TextPromptConfig(),
    )
    ex = ds[0]
    assert isinstance(ex.prompts, BoxPrompts)
    assert ex.prompts.boxes.shape == (16, 4)
    assert ex.prompts.class_ids.shape == (16,)
    assert len(ex.instances) == 16


def test_getitem_bbox_mode_returns_BoxPrompts(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=_build_eval(32),
        text_prompt=TextPromptConfig(),
    )
    ex = ds[0]
    assert isinstance(ex.prompts, BoxPrompts)
    assert ex.prompts.boxes.dtype == torch.float32
    assert ex.prompts.class_ids.dtype == torch.int64
    coords = ex.prompts.boxes.reshape(-1)
    assert (coords >= 0).all() and (coords <= 32).all()


def test_polygon_segmentation_decoded(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=_build_eval(32),
        text_prompt=TextPromptConfig(),
    )
    ex = ds[0]
    assert ex.instances[0].mask.shape == (32, 32)
    assert ex.instances[0].mask.dtype == torch.bool
    assert int(ex.instances[0].mask.sum()) > 0


def test_rle_segmentation_decoded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_imagenet(monkeypatch)
    from pycocotools import mask as mu
    from PIL import Image

    rle = mu.encode(np.asfortranarray(np.ones((8, 8), dtype=np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8],
                        "area": 64, "iscrowd": 0, "segmentation": rle,
                    }
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    ds = COCODataset(
        annotations=str(p),
        images=str(images_dir),
        prompt_mode="bbox",
        transforms=_build_eval(8),
        text_prompt=TextPromptConfig(),
    )
    ex = ds[0]
    assert int(ex.instances[0].mask.sum()) > 0


def test_iscrowd_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_imagenet(monkeypatch)
    from PIL import Image

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2, "image_id": 1, "category_id": 1, "bbox": [4, 4, 4, 4],
                        "area": 16, "iscrowd": 1,
                        "segmentation": [[4, 4, 8, 4, 8, 8, 4, 8]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    ds = COCODataset(
        annotations=str(p),
        images=str(images_dir),
        prompt_mode="bbox",
        transforms=_build_eval(8),
        text_prompt=TextPromptConfig(),
    )
    ex = ds[0]
    assert len(ex.instances) == 1


def test_dropped_empty_image_logged_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_imagenet(monkeypatch)
    from PIL import Image

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "width": 8, "height": 8},
                    {"id": 2, "file_name": "b.png", "width": 8, "height": 8},
                ],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 1,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    Image.new("RGB", (8, 8)).save(images_dir / "b.png")
    caplog.set_level(logging.INFO, logger="esam3.data.coco")
    COCODataset(
        annotations=str(p),
        images=str(images_dir),
        prompt_mode="bbox",
        transforms=_build_eval(8),
        text_prompt=TextPromptConfig(),
    )
    drop_lines = [
        r for r in caplog.records if re.search(r"dropped.*iscrowd", r.message)
    ]
    assert len(drop_lines) == 1


def test_image_resize_geometry(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    ds = COCODataset(
        annotations=str(tiny_coco_dir / "annotations.json"),
        images=str(tiny_coco_dir / "images"),
        prompt_mode="bbox",
        transforms=_build_eval(64),
        text_prompt=TextPromptConfig(),
    )
    ex = ds[0]
    assert ex.image.shape == (3, 64, 64)
    assert ex.instances[0].mask.shape == (64, 64)
    coords = ex.prompts.boxes.reshape(-1)
    assert (coords >= 0).all() and (coords <= 64).all()


def test_sparse_to_dense_remap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_imagenet(monkeypatch)
    from PIL import Image

    p = tmp_path / "ann.json"
    p.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 8, "height": 8}],
                "categories": [
                    {"id": 7, "name": "g"},
                    {"id": 3, "name": "a"},
                ],
                "annotations": [
                    {
                        "id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                        "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
                    },
                    {
                        "id": 2, "image_id": 1, "category_id": 7, "bbox": [4, 0, 4, 4],
                        "area": 16, "iscrowd": 0,
                        "segmentation": [[4, 0, 8, 0, 8, 4, 4, 4]],
                    },
                ],
            }
        )
    )
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(images_dir / "a.png")
    ds = COCODataset(
        annotations=str(p),
        images=str(images_dir),
        prompt_mode="bbox",
        transforms=_build_eval(8),
        text_prompt=TextPromptConfig(),
    )
    assert len(ds.class_names) == 2
    assert ds.coco_category_ids == [3, 7]
    assert {int(inst.class_id) for inst in ds[0].instances} == {0, 1}


def test_deterministic_text_sampling_under_fixed_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_imagenet(monkeypatch)
    ann, imgs = _synth_many_cats(tmp_path, 5)

    def build() -> COCODataset:
        return COCODataset(
            annotations=str(ann),
            images=str(imgs),
            prompt_mode="text",
            transforms=_build_eval(32),
            text_prompt=TextPromptConfig(mode="sampled_fixed_k", k=4),
            seed=42,
        )

    a = build()[0]
    b = build()[0]
    assert isinstance(a.prompts, TextPrompts)
    assert isinstance(b.prompts, TextPrompts)
    assert a.prompts.classes == b.prompts.classes
```

- [ ] **Step 3: Run tests to verify failure**

```bash
uv run pytest tests/unit/test_data_coco.py -v
```

Expected: helpers pass; 17 new tests fail with `NotImplementedError`.

- [ ] **Step 4: Replace `COCODataset` in `src/esam3/data/coco.py`**

Replace the placeholder `COCODataset` class and the `_ = (Literal, np)` filler line. Remove the temporary stub messages. Add full impl:

```python
class COCODataset:
    """COCO instance-JSON dataset.

    Sparse COCO category ids -> dense 0..C-1; images with only iscrowd=1
    annotations are dropped at construction; per-image multiplex capped at 16.
    """

    coco_category_ids: list[int]

    def __init__(
        self,
        annotations: str,
        images: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        seed: int = 0,
    ) -> None:
        if prompt_mode not in ("text", "bbox"):
            raise ValueError(f"prompt_mode must be 'text' or 'bbox'; got {prompt_mode!r}")
        self._image_root = Path(images)
        self._prompt_mode: Literal["text", "bbox"] = prompt_mode
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._seed = seed
        self._multiplex_cap = 16
        self._warned_truncation = False

        self._coco = _load_coco_index(annotations)
        sparse_ids, mapping, class_names = _build_category_remap(self._coco)
        self._coco_category_ids = sparse_ids
        self.coco_category_ids = sparse_ids
        self._cat_id_to_dense = mapping
        self._class_names = class_names

        kept, ann_index, dropped = _drop_crowd_only_images(self._coco)
        self._image_ids = kept
        self._ann_index = ann_index
        if dropped:
            _LOG.info(
                "esam3.data.coco: dropped %d images (iscrowd-only) from %s",
                dropped, annotations,
            )
        _LOG.info(
            "esam3.data.coco: loaded %d images, %d dense classes from %s",
            len(self._image_ids), len(self._class_names), annotations,
        )

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, i: int) -> Example:
        import torch
        from PIL import Image

        image_id = self._image_ids[i]
        rec = self._coco.loadImgs([image_id])[0]
        img_path = self._image_root / rec["file_name"]
        np_img = np.asarray(Image.open(img_path).convert("RGB"))
        h, w = int(rec["height"]), int(rec["width"])

        anns = self._ann_index[image_id]
        bboxes_xyxy: list[list[float]] = []
        masks: list[np.ndarray] = []
        class_labels: list[int] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            bboxes_xyxy.append([float(x), float(y), float(x + bw), float(y + bh)])
            masks.append(_decode_segmentation(ann, h, w).astype(np.uint8))
            class_labels.append(self._cat_id_to_dense[int(ann["category_id"])])

        out = self._transforms(
            image=np_img,
            bboxes=bboxes_xyxy,
            masks=masks,
            class_labels=class_labels,
        )
        image_tensor: torch.Tensor = out["image"]
        out_bboxes: list[tuple[float, float, float, float]] = list(out["bboxes"])
        out_masks: list[np.ndarray] = list(out["masks"])
        out_classes: list[int] = list(out["class_labels"])

        from esam3.data.base import BoxPrompts, Example, Instance, TextPrompts

        instances: list[Instance] = []
        for box, mask_np, cls in zip(out_bboxes, out_masks, out_classes, strict=True):
            instances.append(
                Instance(
                    mask=torch.from_numpy(np.asarray(mask_np).astype(bool)),
                    class_id=int(cls),
                    box=torch.tensor(box, dtype=torch.float32),
                )
            )

        if self._prompt_mode == "text":
            present = sorted(set(out_classes))
            rng = random.Random((self._seed, int(image_id)))
            prompts_list = _build_text_prompts(
                present_dense_ids=present,
                class_names=self._class_names,
                cfg=self._text_prompt_cfg,
                rng=rng,
                image_id=int(image_id),
            )
            if len(prompts_list) > self._multiplex_cap:
                if not self._warned_truncation:
                    _LOG.warning(
                        "esam3.data.coco: image_id=%s requested %d text prompts; "
                        "truncating to %d. Suppressing further warnings for this dataset.",
                        image_id, len(prompts_list), self._multiplex_cap,
                    )
                    self._warned_truncation = True
                prompts_list = prompts_list[: self._multiplex_cap]
            return Example(
                image=image_tensor,
                image_id=str(image_id),
                prompts=TextPrompts(classes=prompts_list),
                instances=instances,
            )

        # bbox mode
        order = sorted(
            range(len(instances)),
            key=lambda k: (instances[k].class_id, float(instances[k].box[0]), float(instances[k].box[1])),
        )
        if len(order) > self._multiplex_cap:
            if not self._warned_truncation:
                _LOG.warning(
                    "esam3.data.coco: image_id=%s requested %d box prompts; "
                    "truncating to %d. Suppressing further warnings for this dataset.",
                    image_id, len(order), self._multiplex_cap,
                )
                self._warned_truncation = True
            order = order[: self._multiplex_cap]
        kept_instances = [instances[k] for k in order]
        boxes_t = torch.stack([inst.box for inst in kept_instances]) if kept_instances else torch.zeros((0, 4))
        class_ids_t = torch.tensor([inst.class_id for inst in kept_instances], dtype=torch.int64)
        return Example(
            image=image_tensor,
            image_id=str(image_id),
            prompts=BoxPrompts(boxes=boxes_t.to(torch.float32), class_ids=class_ids_t),
            instances=kept_instances,
        )

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)
```

Remove the temporary `_ = (Literal, np)` filler line.

- [ ] **Step 5: Run COCO tests**

```bash
uv run pytest tests/unit/test_data_coco.py -v
```

Expected: 27 passed (10 helpers + 17 dataset).

- [ ] **Step 6: Run the full unit suite**

```bash
uv run pytest tests/unit -v
```

Expected: every test passes except the existing `test_stubs_raise::test_data_stubs` (the data stubs are gone; we'll handle that in Task 19). Confirm only that single failure.

- [ ] **Step 7: Commit**

```bash
git add src/esam3/data/coco.py tests/conftest.py tests/unit/test_data_coco.py
git commit -m "feat(data): COCODataset with prompt modes, iscrowd filter, multiplex cap"
```

---

### Task 14: `build_coco` builder with `pipeline` + `model_name` (difficulty: M)

**Files:**
- Modify: `src/esam3/data/coco.py` (replace `build_coco`)
- Test: `tests/unit/test_data_coco.py` (append cases)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_data_coco.py`:

```python
from esam3._registry import lookup


def test_register_coco_lookup(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    builder = lookup("dataset", "coco")
    cfg: dict[str, Any] = {
        "format": "coco",
        "train": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "val": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "prompt_mode": "bbox",
        "image_size": 32,
        "augmentations": {"hflip": True, "color_jitter": 0.1},
        "text_prompt": {"mode": "present"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    ds = builder(cfg, model_name="facebook/sam3.1", pipeline="eval")
    assert len(ds) == 2
    assert ds.class_names == ["thing_a", "thing_b"]


def test_build_coco_train_pipeline_uses_train_transforms(
    monkeypatch: pytest.MonkeyPatch, tiny_coco_dir: Path
) -> None:
    _patch_imagenet(monkeypatch)
    builder = lookup("dataset", "coco")
    cfg: dict[str, Any] = {
        "format": "coco",
        "train": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "val": {
            "annotations": str(tiny_coco_dir / "annotations.json"),
            "images": str(tiny_coco_dir / "images"),
        },
        "prompt_mode": "bbox",
        "image_size": 32,
        "augmentations": {"hflip": False, "color_jitter": 0.0},
        "text_prompt": {"mode": "present"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    ds = builder(cfg, model_name="facebook/sam3.1", pipeline="train")
    assert len(ds) == 2
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_coco.py::test_register_coco_lookup -v
```

Expected: failure — current `build_coco` doesn't accept `model_name` or `pipeline`.

- [ ] **Step 3: Replace `build_coco` in `src/esam3/data/coco.py`**

Replace the existing `@register("dataset", "coco") def build_coco(...)` with:

```python
@register("dataset", "coco")
def build_coco(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pipeline: Literal["train", "eval"],
) -> Dataset:
    """Build a `COCODataset` from a validated DataConfig dict.

    The caller (trainer) chooses the split by passing the matching `train` or
    `val` sub-dict in `cfg["train"]` / `cfg["val"]`. Here `pipeline` selects the
    transform variant.
    """
    from esam3.config.schema import AugmentationsConfig, NormalizeConfig, TextPromptConfig
    from esam3.data.transforms import build_eval_transforms, build_train_transforms

    if pipeline not in ("train", "eval"):
        raise ValueError(f"pipeline must be 'train' or 'eval'; got {pipeline!r}")
    split_key = "train" if pipeline == "train" else "val"
    split = cfg[split_key]
    image_size = int(cfg["image_size"])
    normalize = NormalizeConfig.model_validate(cfg.get("normalize", {}))
    text_prompt = TextPromptConfig.model_validate(cfg.get("text_prompt", {}))
    if pipeline == "train":
        aug = AugmentationsConfig.model_validate(cfg.get("augmentations", {}))
        transforms = build_train_transforms(
            aug, image_size, model_name=model_name, normalize=normalize
        )
    else:
        transforms = build_eval_transforms(
            image_size, model_name=model_name, normalize=normalize
        )
    return COCODataset(
        annotations=split["annotations"],
        images=split["images"],
        prompt_mode=cfg["prompt_mode"],
        transforms=transforms,
        text_prompt=text_prompt,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_data_coco.py -v
```

Expected: 29 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/data/coco.py tests/unit/test_data_coco.py
git commit -m "feat(data): build_coco builder with pipeline + model_name kwargs"
```

---

### Task 15: HF module-private helpers (difficulty: M)

**Files:**
- Modify: `src/esam3/data/hf.py` (replace stub — helpers only this task)
- Test: `tests/unit/test_data_hf.py` (created)

- [ ] **Step 1: Write failing helper tests**

Create `tests/unit/test_data_hf.py`:

```python
"""Tests for data/hf.py — helpers + dataset + builder."""

from __future__ import annotations

import logging
import re
from typing import Any

import datasets as hf_datasets
import numpy as np
import pytest
import torch
from PIL import Image

from esam3.config.schema import HFFieldMap, NormalizeConfig, TextPromptConfig
from esam3.data.hf import (
    HFFieldError,
    _normalize_bbox,
    _resolve_class_names,
    _resolve_field,
    _validate_required_fields,
)


def test_resolve_field_top_level() -> None:
    row = {"image": "x"}
    assert _resolve_field(row, "image") == "x"


def test_resolve_field_dotted_path() -> None:
    row = {"objects": {"bbox": [[0, 0, 1, 1]]}}
    assert _resolve_field(row, "objects.bbox") == [[0, 0, 1, 1]]


def test_resolve_field_missing_raises_keyerror() -> None:
    with pytest.raises(KeyError, match=r"objects\.bbox"):
        _resolve_field({"objects": {}}, "objects.bbox")


def test_normalize_bbox_xywh_to_xyxy() -> None:
    assert _normalize_bbox([10.0, 20.0, 5.0, 7.0], "xywh") == (10.0, 20.0, 15.0, 27.0)


def test_normalize_bbox_xyxy_passthrough() -> None:
    assert _normalize_bbox([1.0, 2.0, 3.0, 4.0], "xyxy") == (1.0, 2.0, 3.0, 4.0)


def _build_hf_dataset(
    n: int = 2,
    *,
    include_segmentation: bool = False,
    use_class_label: bool = True,
) -> hf_datasets.Dataset:
    images = [Image.new("RGB", (8, 8)) for _ in range(n)]
    bboxes = [[[0.0, 0.0, 4.0, 4.0]] for _ in range(n)]
    categories = [[0] for _ in range(n)]
    cols: dict[str, Any] = {
        "image": images,
        "objects": [
            {"bbox": bboxes[i], "category": categories[i]} for i in range(n)
        ],
        "categories": [["thing"]] * n if not use_class_label else [["thing"]] * n,
    }
    if include_segmentation:
        for o in cols["objects"]:
            o["segmentation"] = [[[0, 0, 4, 0, 4, 4, 0, 4]]]
    features = None
    if use_class_label:
        features = hf_datasets.Features(
            {
                "image": hf_datasets.Image(),
                "objects": hf_datasets.Sequence(
                    {
                        "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                        "category": hf_datasets.ClassLabel(names=["thing"]),
                    }
                ),
                "categories": hf_datasets.Sequence(hf_datasets.Value("string")),
            }
        )
    return hf_datasets.Dataset.from_dict(cols, features=features)


def test_validate_required_fields_passes_on_default_schema() -> None:
    ds = _build_hf_dataset(use_class_label=False)
    _validate_required_fields(ds, HFFieldMap(segmentation=None))


def test_validate_required_fields_raises_on_missing_bbox() -> None:
    ds = hf_datasets.Dataset.from_dict(
        {"image": [Image.new("RGB", (8, 8))], "objects": [{"category": [0]}]}
    )
    with pytest.raises(HFFieldError) as exc:
        _validate_required_fields(ds, HFFieldMap(segmentation=None))
    msg = str(exc.value)
    assert "objects.bbox" in msg
    assert "data.hf.field_map.bbox" in msg


def test_resolve_class_names_from_classlabel_in_objects() -> None:
    ds = _build_hf_dataset(use_class_label=True)
    names = _resolve_class_names(ds, HFFieldMap())
    assert names == ["thing"]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_hf.py -v
```

Expected: import failures.

- [ ] **Step 3: Replace `src/esam3/data/hf.py` with helpers + placeholder class**

```python
"""HuggingFace `datasets` dataset adapter.

Uses a hybrid input contract: conventional dotted field paths with full
override via `HFFieldMap`. Class names come from a top-level `categories`
feature, or fall back to a `ClassLabel` inside the per-box category field.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from esam3._registry import register
from esam3.config.schema import HFFieldMap, TextPromptConfig
from esam3.data.base import Dataset, Example

_LOG = logging.getLogger(__name__)


class HFFieldError(KeyError):
    """Raised when the HF dataset does not contain a required field."""


def _resolve_field(row: dict[str, Any], dotted: str) -> Any:
    """Walk a dotted path against a row dict; raise `KeyError(dotted)` on miss."""
    node: Any = row
    parts = dotted.split(".")
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _normalize_bbox(
    b: list[float] | tuple[float, ...], fmt: Literal["xywh", "xyxy"]
) -> tuple[float, float, float, float]:
    """Return `(x0, y0, x1, y1)`."""
    a, b1, c, d = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    if fmt == "xywh":
        return (a, b1, a + c, b1 + d)
    return (a, b1, c, d)


def _validate_required_fields(ds: Any, field_map: HFFieldMap) -> None:
    """Read one row and ensure every required path resolves.

    Raises:
        HFFieldError: if any required path is missing; message contains the
            dotted path and the override key (`data.hf.field_map.<key>`).
    """
    if len(ds) == 0:
        return
    row = ds[0]
    required: list[tuple[str, str]] = [
        (field_map.image, "image"),
        (field_map.bbox, "bbox"),
        (field_map.category, "category"),
    ]
    for path, override_key in required:
        try:
            _resolve_field(row, path)
        except KeyError as e:
            raise HFFieldError(
                f"HF dataset is missing required field '{path}'. "
                f"Set data.hf.field_map.{override_key} to the correct dotted path."
            ) from e


def _resolve_class_names(ds: Any, field_map: HFFieldMap) -> list[str]:
    """Resolve dataset class names.

    Order of attempts:
      1. Top-level feature named `field_map.categories_feature` whose value is
         a `Sequence(ClassLabel)` or a `list[str]` per row.
      2. If absent, look for a `ClassLabel` feature at `<field_map.category>`
         inside `ds.features` and return its `names`.
    """
    feats = getattr(ds, "features", None)
    if feats is not None and field_map.categories_feature in feats:
        feat = feats[field_map.categories_feature]
        # Sequence(ClassLabel(names=[...])) | Sequence(Value('string'))
        inner = getattr(feat, "feature", None)
        names = getattr(inner, "names", None)
        if names:
            return list(names)
        if len(ds) > 0:
            row_val = ds[0].get(field_map.categories_feature)
            if isinstance(row_val, list) and all(isinstance(x, str) for x in row_val):
                return list(row_val)
    if feats is not None:
        node: Any = feats
        for part in field_map.category.split("."):
            inner = getattr(node, "feature", None) if not isinstance(node, dict) else None
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif inner is not None and getattr(inner, part, None) is not None:
                node = getattr(inner, part)
            else:
                node = None
                break
        names = getattr(node, "names", None) if node is not None else None
        if names:
            return list(names)
    raise HFFieldError(
        f"Cannot resolve class names. Set data.hf.field_map.categories_feature "
        f"to a top-level Sequence(ClassLabel) feature, or use a ClassLabel-typed "
        f"category field."
    )


class HFDataset:
    """Placeholder — full impl in Task 16."""

    def __init__(self, name: str, split: str, prompt_mode: str) -> None:
        self.name = name
        self.split = split
        self.prompt_mode = prompt_mode

    def __len__(self) -> int:
        raise NotImplementedError("filled in by spec: spec/data-loading (Task 16)")

    def __getitem__(self, i: int) -> Example:
        raise NotImplementedError("filled in by spec: spec/data-loading (Task 16)")

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError("filled in by spec: spec/data-loading (Task 16)")


@register("dataset", "hf")
def build_hf(cfg: dict[str, Any]) -> Dataset:
    """Placeholder — full impl in Task 16."""
    return HFDataset(
        name=cfg["name"],
        split=cfg.get("split", "train"),
        prompt_mode=cfg["prompt_mode"],
    )


_ = np  # silences F401 until Task 16
```

- [ ] **Step 4: Run helper tests**

```bash
uv run pytest tests/unit/test_data_hf.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/esam3/data/hf.py tests/unit/test_data_hf.py
git commit -m "feat(data): hf module-private helpers (resolve, validate, bbox, names)"
```

---

### Task 16: `HFDataset` + `build_hf` builder (difficulty: H)

**Files:**
- Modify: `src/esam3/data/hf.py` (replace placeholder class + builder)
- Test: `tests/unit/test_data_hf.py` (append the 8 spec-§6.2 cases)

- [ ] **Step 1: Append failing tests**

Append to `tests/unit/test_data_hf.py`:

```python
from esam3._registry import lookup
from esam3.data.base import BoxPrompts, TextPrompts
from esam3.data.hf import HFDataset


def _patch_imagenet(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoImageProcessor

    monkeypatch.setattr(
        AutoImageProcessor,
        "from_pretrained",
        lambda name, **kwargs: (_ for _ in ()).throw(OSError("no cache")),
    )


def _build_eval(image_size: int = 8) -> Any:
    from esam3.data.transforms import build_eval_transforms

    return build_eval_transforms(
        image_size, model_name="facebook/sam3.1", normalize=NormalizeConfig()
    )


def _patch_load_dataset(
    monkeypatch: pytest.MonkeyPatch, ds: hf_datasets.Dataset
) -> None:
    def fake(name: str, split: str, **kwargs: object) -> hf_datasets.Dataset:
        return ds

    monkeypatch.setattr("esam3.data.hf.hf_load_dataset", fake)


def test_required_fields_validation_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_imagenet(monkeypatch)
    bad = hf_datasets.Dataset.from_dict(
        {"image": [Image.new("RGB", (8, 8))], "objects": [{"category": [0]}]}
    )
    _patch_load_dataset(monkeypatch, bad)
    with pytest.raises(HFFieldError) as exc:
        HFDataset(
            name="x",
            split="train",
            prompt_mode="bbox",
            transforms=_build_eval(),
            text_prompt=TextPromptConfig(),
            field_map=HFFieldMap(segmentation=None),
        )
    msg = str(exc.value)
    assert "objects.bbox" in msg
    assert "data.hf.field_map.bbox" in msg


def test_field_map_override_picks_alternate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_imagenet(monkeypatch)
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "annotations": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "label": hf_datasets.ClassLabel(names=["thing"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "annotations": [{"bbox": [[0.0, 0.0, 4.0, 4.0]], "label": [0]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    hfds = HFDataset(
        name="x",
        split="train",
        prompt_mode="bbox",
        transforms=_build_eval(),
        text_prompt=TextPromptConfig(),
        field_map=HFFieldMap(
            bbox="annotations.bbox",
            category="annotations.label",
            segmentation=None,
        ),
    )
    assert len(hfds) == 1
    assert hfds.class_names == ["thing"]


def test_class_names_from_categories_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_imagenet(monkeypatch)
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "objects": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "category": hf_datasets.ClassLabel(names=["a", "b"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "objects": [{"bbox": [[0.0, 0.0, 4.0, 4.0]], "category": [0]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    hfds = HFDataset(
        name="x",
        split="train",
        prompt_mode="text",
        transforms=_build_eval(),
        text_prompt=TextPromptConfig(),
        field_map=HFFieldMap(segmentation=None),
    )
    assert hfds.class_names == ["a", "b"]


def test_getitem_text_mode_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_imagenet(monkeypatch)
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "objects": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "category": hf_datasets.ClassLabel(names=["a", "b"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "objects": [{"bbox": [[0.0, 0.0, 4.0, 4.0]], "category": [1]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    hfds = HFDataset(
        name="x", split="train", prompt_mode="text",
        transforms=_build_eval(), text_prompt=TextPromptConfig(mode="present"),
        field_map=HFFieldMap(segmentation=None),
    )
    ex = hfds[0]
    assert isinstance(ex.prompts, TextPrompts)
    assert ex.prompts.classes == ["b"]


def test_getitem_bbox_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_imagenet(monkeypatch)
    ds = _build_hf_dataset(use_class_label=True)
    _patch_load_dataset(monkeypatch, ds)
    hfds = HFDataset(
        name="x", split="train", prompt_mode="bbox",
        transforms=_build_eval(), text_prompt=TextPromptConfig(),
        field_map=HFFieldMap(segmentation=None),
    )
    ex = hfds[0]
    assert isinstance(ex.prompts, BoxPrompts)
    assert ex.prompts.boxes.dtype == torch.float32
    assert ex.prompts.class_ids.dtype == torch.int64


def test_bbox_format_xywh_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_imagenet(monkeypatch)
    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "objects": hf_datasets.Sequence(
                {
                    "bbox": hf_datasets.Sequence(hf_datasets.Value("float32"), length=4),
                    "category": hf_datasets.ClassLabel(names=["thing"]),
                }
            ),
        }
    )
    ds = hf_datasets.Dataset.from_dict(
        {
            "image": [Image.new("RGB", (8, 8))],
            "objects": [{"bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [0]}],
        },
        features=features,
    )
    _patch_load_dataset(monkeypatch, ds)
    hfds = HFDataset(
        name="x", split="train", prompt_mode="bbox",
        transforms=_build_eval(), text_prompt=TextPromptConfig(),
        field_map=HFFieldMap(segmentation=None, bbox_format="xywh"),
    )
    ex = hfds[0]
    # xywh [1,2,3,4] -> xyxy [1,2,4,6] (within an 8x8 image kept square after resize).
    box = ex.prompts.boxes[0]
    assert abs(float(box[0]) - 1.0) < 0.5
    assert abs(float(box[2]) - 4.0) < 0.5


def test_masks_from_boxes_when_segmentation_absent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_imagenet(monkeypatch)
    ds = _build_hf_dataset(use_class_label=True, include_segmentation=False)
    _patch_load_dataset(monkeypatch, ds)
    caplog.set_level(logging.WARNING, logger="esam3.data.hf")
    hfds = HFDataset(
        name="x", split="train", prompt_mode="bbox",
        transforms=_build_eval(), text_prompt=TextPromptConfig(),
        field_map=HFFieldMap(segmentation=None),
    )
    ex = hfds[0]
    assert ex.instances[0].mask.dtype == torch.bool
    assert int(ex.instances[0].mask.sum()) > 0
    assert any(re.search(r"masks-from-boxes", rec.message) for rec in caplog.records)


def test_register_hf_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_imagenet(monkeypatch)
    ds = _build_hf_dataset(use_class_label=True)
    _patch_load_dataset(monkeypatch, ds)
    builder = lookup("dataset", "hf")
    cfg: dict[str, Any] = {
        "format": "hf",
        "train": {"annotations": "unused", "images": "unused"},
        "val": {"annotations": "unused", "images": "unused"},
        "hf": {
            "name": "x",
            "split_train": "train",
            "split_val": "val",
            "field_map": {
                "image": "image",
                "bbox": "objects.bbox",
                "category": "objects.category",
                "segmentation": None,
                "categories_feature": "categories",
                "bbox_format": "xyxy",
            },
        },
        "prompt_mode": "bbox",
        "image_size": 8,
        "augmentations": {"hflip": False, "color_jitter": 0.0},
        "text_prompt": {"mode": "present"},
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    hfds = builder(cfg, model_name="facebook/sam3.1", pipeline="eval")
    assert len(hfds) > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_data_hf.py -v
```

Expected: helpers pass; 8 dataset tests fail.

- [ ] **Step 3: Replace `HFDataset` + builder in `src/esam3/data/hf.py`**

Replace the placeholder class + builder at the bottom of the file. Also add the `hf_load_dataset` alias so tests can monkeypatch it cleanly. Replace the `_ = np` filler with real use.

```python
from datasets import load_dataset as hf_load_dataset  # noqa: E402  (after helpers)


class HFDataset:
    """HuggingFace `datasets` adapter."""

    def __init__(
        self,
        name: str,
        split: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Any,
        text_prompt: TextPromptConfig,
        field_map: HFFieldMap,
        seed: int = 0,
    ) -> None:
        if prompt_mode not in ("text", "bbox"):
            raise ValueError(f"prompt_mode must be 'text' or 'bbox'; got {prompt_mode!r}")
        self._name = name
        self._split = split
        self._prompt_mode: Literal["text", "bbox"] = prompt_mode
        self._transforms = transforms
        self._text_prompt_cfg = text_prompt
        self._field_map = field_map
        self._seed = seed
        self._multiplex_cap = 16
        self._warned_truncation = False
        self._warned_masks_from_boxes = False

        self._ds = hf_load_dataset(name, split=split)
        _validate_required_fields(self._ds, field_map)
        self._class_names = _resolve_class_names(self._ds, field_map)

    def __len__(self) -> int:
        return int(len(self._ds))

    def __getitem__(self, i: int) -> Example:
        import random as _random

        import torch
        from PIL import Image as PILImage

        from esam3.data.base import BoxPrompts, Example, Instance, TextPrompts
        from esam3.data.coco import _build_text_prompts

        row = self._ds[i]
        img_obj = _resolve_field(row, self._field_map.image)
        if isinstance(img_obj, PILImage.Image):
            np_img = np.asarray(img_obj.convert("RGB"))
        else:
            np_img = np.asarray(img_obj)
            if np_img.ndim == 2:
                np_img = np.stack([np_img] * 3, axis=-1)
        h, w = int(np_img.shape[0]), int(np_img.shape[1])

        bboxes_raw = _resolve_field(row, self._field_map.bbox)
        classes = list(_resolve_field(row, self._field_map.category))
        bboxes_xyxy = [_normalize_bbox(list(b), self._field_map.bbox_format) for b in bboxes_raw]

        masks: list[np.ndarray] = []
        seg_path = self._field_map.segmentation
        seg_resolved: Any = None
        if seg_path:
            try:
                seg_resolved = _resolve_field(row, seg_path)
            except KeyError:
                seg_resolved = None
        if seg_resolved is None:
            if not self._warned_masks_from_boxes:
                _LOG.warning(
                    "esam3.data.hf: masks-from-boxes fallback used for dataset %r "
                    "(field_map.segmentation absent or None). Suppressing further warnings.",
                    self._name,
                )
                self._warned_masks_from_boxes = True
            for x0, y0, x1, y1 in bboxes_xyxy:
                m = np.zeros((h, w), dtype=np.uint8)
                xi0, yi0 = max(0, int(x0)), max(0, int(y0))
                xi1, yi1 = min(w, int(x1)), min(h, int(y1))
                if xi1 > xi0 and yi1 > yi0:
                    m[yi0:yi1, xi0:xi1] = 1
                masks.append(m)
        else:
            from esam3.data.coco import _decode_segmentation

            for ann in seg_resolved:
                if isinstance(ann, dict):
                    masks.append(_decode_segmentation({"segmentation": ann}, h, w).astype(np.uint8))
                else:
                    masks.append(_decode_segmentation({"segmentation": ann}, h, w).astype(np.uint8))

        out = self._transforms(
            image=np_img,
            bboxes=[list(b) for b in bboxes_xyxy],
            masks=masks,
            class_labels=classes,
        )
        image_tensor: torch.Tensor = out["image"]
        out_bboxes = list(out["bboxes"])
        out_masks = list(out["masks"])
        out_classes = list(out["class_labels"])

        instances: list[Instance] = []
        for box, mask_np, cls in zip(out_bboxes, out_masks, out_classes, strict=True):
            instances.append(
                Instance(
                    mask=torch.from_numpy(np.asarray(mask_np).astype(bool)),
                    class_id=int(cls),
                    box=torch.tensor(box, dtype=torch.float32),
                )
            )

        image_id = str(i)
        if self._prompt_mode == "text":
            present = sorted(set(out_classes))
            rng = _random.Random((self._seed, i))
            prompts_list = _build_text_prompts(
                present_dense_ids=present,
                class_names=self._class_names,
                cfg=self._text_prompt_cfg,
                rng=rng,
                image_id=i,
            )
            if len(prompts_list) > self._multiplex_cap:
                if not self._warned_truncation:
                    _LOG.warning(
                        "esam3.data.hf: image_id=%s requested %d text prompts; "
                        "truncating to %d. Suppressing further warnings.",
                        image_id, len(prompts_list), self._multiplex_cap,
                    )
                    self._warned_truncation = True
                prompts_list = prompts_list[: self._multiplex_cap]
            return Example(
                image=image_tensor,
                image_id=image_id,
                prompts=TextPrompts(classes=prompts_list),
                instances=instances,
            )

        order = sorted(
            range(len(instances)),
            key=lambda k: (instances[k].class_id, float(instances[k].box[0]), float(instances[k].box[1])),
        )
        if len(order) > self._multiplex_cap:
            if not self._warned_truncation:
                _LOG.warning(
                    "esam3.data.hf: image_id=%s requested %d box prompts; "
                    "truncating to %d. Suppressing further warnings.",
                    image_id, len(order), self._multiplex_cap,
                )
                self._warned_truncation = True
            order = order[: self._multiplex_cap]
        kept_instances = [instances[k] for k in order]
        boxes_t = torch.stack([inst.box for inst in kept_instances]) if kept_instances else torch.zeros((0, 4))
        class_ids_t = torch.tensor([inst.class_id for inst in kept_instances], dtype=torch.int64)
        return Example(
            image=image_tensor,
            image_id=image_id,
            prompts=BoxPrompts(boxes=boxes_t.to(torch.float32), class_ids=class_ids_t),
            instances=kept_instances,
        )

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)


@register("dataset", "hf")
def build_hf(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pipeline: Literal["train", "eval"],
) -> Dataset:
    """Build an `HFDataset` from a validated DataConfig dict."""
    from esam3.config.schema import AugmentationsConfig, NormalizeConfig
    from esam3.data.transforms import build_eval_transforms, build_train_transforms

    if pipeline not in ("train", "eval"):
        raise ValueError(f"pipeline must be 'train' or 'eval'; got {pipeline!r}")
    hf_cfg = cfg["hf"]
    split = hf_cfg["split_train"] if pipeline == "train" else hf_cfg["split_val"]
    image_size = int(cfg["image_size"])
    normalize = NormalizeConfig.model_validate(cfg.get("normalize", {}))
    text_prompt = TextPromptConfig.model_validate(cfg.get("text_prompt", {}))
    field_map = HFFieldMap.model_validate(hf_cfg.get("field_map", {}))
    if pipeline == "train":
        aug = AugmentationsConfig.model_validate(cfg.get("augmentations", {}))
        transforms = build_train_transforms(
            aug, image_size, model_name=model_name, normalize=normalize
        )
    else:
        transforms = build_eval_transforms(
            image_size, model_name=model_name, normalize=normalize
        )
    return HFDataset(
        name=hf_cfg["name"],
        split=split,
        prompt_mode=cfg["prompt_mode"],
        transforms=transforms,
        text_prompt=text_prompt,
        field_map=field_map,
    )
```

Remove the trailing `_ = np` line. The `Dataset` import is still used as the builder return type.

- [ ] **Step 4: Run HF tests**

```bash
uv run pytest tests/unit/test_data_hf.py -v
```

Expected: 17 passed (9 helpers + 8 dataset+builder).

- [ ] **Step 5: Commit**

```bash
git add src/esam3/data/hf.py tests/unit/test_data_hf.py
git commit -m "feat(data): HFDataset + build_hf with field-map + masks-from-boxes"
```

---

### Task 17: Augment example YAMLs + ARCHITECTURE.md (difficulty: L)

**Files:**
- Modify: `configs/examples/coco_text_lora.yaml`, `configs/examples/coco_bbox_qlora.yaml` (append blocks)
- Modify: `ARCHITECTURE.md` (one-line edit on the data adapters line)

- [ ] **Step 1: Confirm the existing example tests still pass before editing**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py::test_existing_example_yaml_still_validates -v
```

Expected: PASS (defaults kick in even without the new blocks).

- [ ] **Step 2: Append blocks to `configs/examples/coco_text_lora.yaml`**

Insert these two blocks at lines 23 (right after `augmentations:`), staying inside the `data:` block. Final `data:` section reads:

```yaml
data:
  format: coco
  train:
    annotations: data/coco/instances_train2017.json
    images: data/coco/train2017
  val:
    annotations: data/coco/instances_val2017.json
    images: data/coco/val2017
  prompt_mode: text
  image_size: 1024
  augmentations:
    hflip: true
    color_jitter: 0.1
  text_prompt:
    mode: present_plus_negatives
    negatives_per_image: 4
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]
```

- [ ] **Step 3: Append blocks to `configs/examples/coco_bbox_qlora.yaml`**

Final `data:` section reads:

```yaml
data:
  format: coco
  train:
    annotations: data/coco/instances_train2017.json
    images: data/coco/train2017
  val:
    annotations: data/coco/instances_val2017.json
    images: data/coco/val2017
  prompt_mode: bbox
  image_size: 1024
  augmentations:
    hflip: true
    color_jitter: 0.1
  text_prompt:
    mode: present
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]
```

- [ ] **Step 4: Edit `ARCHITECTURE.md` data line**

In `ARCHITECTURE.md`, find the line `    coco.py / hf.py    @register("dataset", ...) adapters` and replace with:

```
    coco.py / hf.py    @register("dataset", ...) adapters (call with pipeline + model_name kwargs)
```

- [ ] **Step 5: Re-run example-yaml test**

```bash
uv run pytest tests/unit/test_data_schema_extensions.py::test_existing_example_yaml_still_validates -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/examples/coco_text_lora.yaml configs/examples/coco_bbox_qlora.yaml ARCHITECTURE.md
git commit -m "docs(data): add text_prompt + normalize to example configs"
```

---

### Task 18: Boundary test — data layer must not import `TrainConfig` (difficulty: L)

**Files:**
- Test: `tests/unit/test_data_import_boundary.py` (created)

- [ ] **Step 1: Write the test**

Create `tests/unit/test_data_import_boundary.py`:

```python
"""Architectural guard: the data layer must not import TrainConfig.

The data layer accepts a `dict[str, Any]` plus `model_name: str` and
`pipeline: Literal['train','eval']` — not the full TrainConfig. Verified
via static AST walk over `src/esam3/data/`.
"""

from __future__ import annotations

import ast
from pathlib import Path

_DATA_FILES = ("coco.py", "hf.py", "transforms.py", "collate.py", "base.py")
_FORBIDDEN_NAMES = frozenset({"TrainConfig"})


def _find_imports(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "esam3.config.schema":
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "esam3.config.schema":
                    found.append("<module>")
    return found


def test_data_layer_does_not_import_train_config() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "src" / "esam3" / "data"
    offenders: dict[str, list[str]] = {}
    for fname in _DATA_FILES:
        fp = data_dir / fname
        if not fp.is_file():
            continue
        tree = ast.parse(fp.read_text(encoding="utf-8"))
        names = _find_imports(tree)
        bad = sorted(set(names) & _FORBIDDEN_NAMES)
        if bad:
            offenders[fname] = bad
    assert offenders == {}, f"data layer imports forbidden names: {offenders}"
```

- [ ] **Step 2: Run to verify pass**

```bash
uv run pytest tests/unit/test_data_import_boundary.py -v
```

Expected: PASS (none of the four data files import `TrainConfig`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_data_import_boundary.py
git commit -m "test(data): guard against TrainConfig import in data layer"
```

---

### Task 19: Drop `test_data_stubs` from `test_stubs_raise.py` (difficulty: L)

**Files:**
- Modify: `tests/unit/test_stubs_raise.py:33-44` (delete `test_data_stubs`)
- Modify: `tests/unit/test_stubs_raise.py:14-17` (drop now-unused data imports)

- [ ] **Step 1: Edit `tests/unit/test_stubs_raise.py`**

Remove the data imports at lines 14-17:

```python
from esam3.data.coco import COCODataset
from esam3.data.collate import collate_batch
from esam3.data.hf import HFDataset
from esam3.data.transforms import build_eval_transforms, build_train_transforms
```

Also remove from the schema import block at lines 8-13 the `AugmentationsConfig` name (no longer referenced after dropping `test_data_stubs`).

Delete `test_data_stubs` entirely (lines 33-44).

- [ ] **Step 2: Run the modified file**

```bash
uv run pytest tests/unit/test_stubs_raise.py -v
```

Expected: 4 tests pass (`test_model_stubs`, `test_peft_stubs`, `test_eval_stubs`, `test_train_stubs`, `test_trainer_fit_stub`).

- [ ] **Step 3: Run the full unit suite**

```bash
uv run pytest tests/unit -v
```

Expected: every test passes.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_stubs_raise.py
git commit -m "refactor(tests): drop data-stub assertions; data layer is implemented"
```

---

### Task 20: Type-check, lint, format clean sweep (difficulty: L)

**Files:**
- None (verification + any auto-fixes)

- [ ] **Step 1: Run ruff format + lint**

```bash
uv run ruff format src tests
uv run ruff check src tests --fix
```

Expected: clean working tree after format; any auto-fixable lints addressed in-place. Re-run `uv run ruff check src tests` and expect zero diagnostics. If there are non-auto-fixable findings, address each in the affected file (do not blanket-noqa).

- [ ] **Step 2: Run mypy**

```bash
uv run mypy
```

Expected: `Success: no issues found in N source files`. If a diagnostic surfaces, fix the root cause (do not `# type: ignore` unless required by a third-party stub gap; in that case keep the comment narrow with a code, e.g. `# type: ignore[attr-defined]`).

- [ ] **Step 3: Commit any formatting / lint fixes**

```bash
git status
git add -p   # interactive review; stage only formatting/lint diffs
git commit -m "chore(data): ruff + mypy clean sweep"
```

If `git status` is clean, skip the commit.

---

### Task 21: Coverage + full-suite verification (difficulty: L)

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full pytest suite with coverage**

```bash
uv run pytest -v
```

Expected: every test passes. Coverage line at the end reports `TOTAL ... NN% ...` with `NN >= 80`. (`--cov-fail-under=80` is already in `pyproject.toml:76`.)

- [ ] **Step 2: Inspect coverage of the four data modules**

```bash
uv run pytest --cov=esam3.data --cov-report=term-missing tests/unit/test_data_coco.py tests/unit/test_data_hf.py tests/unit/test_data_transforms.py tests/unit/test_data_collate.py tests/unit/test_data_schema_extensions.py
```

Expected: each of `esam3/data/coco.py`, `esam3/data/hf.py`, `esam3/data/transforms.py`, `esam3/data/collate.py` shows `Missing` lines limited to defensive `raise` clauses or unreachable branches (`Cover >= 90%` per file is the target). If a module dips below 80%, add one more test rather than `# pragma: no cover` until you've inspected what's missing.

- [ ] **Step 3: Confirm `data/base.py` is unmodified**

```bash
git diff main -- src/esam3/data/base.py
```

Expected: empty diff. If anything appears, revert it.

- [ ] **Step 4: Final commit (only if Step 2 forced a new test)**

```bash
git add -A
git commit -m "test(data): bring per-module coverage to >=90%"
```

If no test was added, skip.

---

### Task 22: Append the planner / implementer log entries (difficulty: L)

**Files:**
- Modify: `logs/log.md` (append)

- [ ] **Step 1: Append one final implementer log entry**

Append to `logs/log.md` (timestamp ISO 8601, role `implementer`):

```
[2026-05-16T00:00:00Z] [implementer] feat(data): data-loading subsystem complete — coco + hf + transforms + collate; 48 unit tests pass; ruff/mypy/pytest green; coverage >=80%
```

- [ ] **Step 2: Commit**

```bash
git add logs/log.md
git commit -m "docs(data): log data-loading subsystem completion"
```

---

## Spec coverage map

| Spec DoD section | Implemented by task(s) |
|---|---|
| §8.1 Schema (5 models + DataConfig validator) | Task 3 (`TextPromptConfig`), Task 4 (`NormalizeConfig`), Task 5 (`HFFieldMap`), Task 6 (`HFDatasetConfig`), Task 7 (`DataConfig` extension + validator) |
| §8.1 — 10 cases in §6.5 pass | Tasks 3-7 (cumulative 17 tests, exceeding the 10 in spec) |
| §8.1 — mypy strict | Tasks 7, 20 |
| §8.1 — existing `test_config_schema.py` still passes | Task 7 Step 4 |
| §8.2 Transforms (`build_eval_transforms`, `build_train_transforms`, `resolve_normalization`) | Task 8 (`resolve_normalization`), Task 9 (eval), Task 10 (train) |
| §8.2 — 7 cases in §6.3 | Tasks 8 (3), 9 (2), 10 (3) — 8 total |
| §8.2 — one INFO line per call, both branches | Task 8 Step 1 (two `caplog` cases) |
| §8.3 Collate | Task 11 |
| §8.3 — 6 cases in §6.4 | Task 11 |
| §8.3 — `ValueError` with both shapes | Task 11 (`test_collate_image_shape_mismatch_raises`) |
| §8.4 COCO (incl. iscrowd, polygon+RLE, multiplex, builder) | Tasks 12 (helpers), 13 (dataset), 14 (builder) |
| §8.4 — 17 cases in §6.1 | Task 13 (17) + 2 from Task 14 builder cases |
| §8.5 HF (field-map override, missing-field error, masks-from-boxes, builder) | Tasks 15 (helpers), 16 (dataset + builder) |
| §8.5 — 8 cases in §6.2 | Task 16 |
| §8.6 Dependencies (`albumentations`, `opencv-python-headless`, promote pillow) | Task 1 |
| §8.6 — `uv lock` regenerates | Task 1 Step 5 |
| §8.7 Docs & Logs — two `[DEFERRED]` lines in `logs/TODO.md` | Task 2 |
| §8.7 — `ARCHITECTURE.md` data sentence updated | Task 17 |
| §8.7 — two example YAMLs updated | Task 17 |
| §8.7 — `test_existing_example_yaml_still_validates` | Task 7 (test added), Task 17 (config files updated) |
| §8.8 — `ruff`, `mypy`, `pytest` clean | Task 20 |
| §8.8 — coverage `>=80%` on `src/esam3` | Task 21 |
| §8.8 — `data/base.py` unmodified | Task 21 Step 3 |
| §8.8 — no `TrainConfig` import from `src/esam3/data/` | Task 18 |
| §6.6 Removal from `test_stubs_raise.py` | Task 19 |

## Self-review checklist (executed)

1. **Spec coverage:** All eight DoD subsections (§8.1-§8.8) are mapped above. Every numbered test case in §6.1-§6.5 is named in a task. The two `[DEFERRED]` lines are written in Task 2. The example-YAML test is in Task 7 and the configs are updated in Task 17 (order matters: the schema accepts the defaults, so the test passes even before Task 17; Task 17 keeps the YAMLs aligned with the spec example).
2. **Placeholder scan:** Searched the plan for "TBD", "TODO", "FIXME", "fill in", "similar to" — the only matches are spec-quoted `[DEFERRED]` log lines (Task 2) and the literal `"filled in by"` regex in stub messages (the existing project pattern, mirrored only during the temporary in-flight state in Tasks 12-13 before the placeholders are deleted in Task 13 Step 4). No remaining placeholders direct the implementer to "fill in details" or "add error handling".
3. **Type consistency:** Type names defined and used across tasks:
   - `TextPromptConfig`, `TextPromptMode` — defined Task 3, consumed Tasks 7, 12, 13, 14, 15, 16.
   - `NormalizeConfig` — defined Task 4, consumed Tasks 7, 8, 9, 10, 14, 16.
   - `HFFieldMap` — defined Task 5, consumed Tasks 7, 15, 16.
   - `HFDatasetConfig` — defined Task 6, consumed Tasks 7, 16.
   - `DataConfig` (extension) — Task 7, consumed by Tasks 14, 16 via dict access (no class import in data layer).
   - `HFFieldError` — defined Task 15, asserted in Tasks 15, 16.
   - `resolve_normalization` — defined Task 8, called Tasks 9, 10.
   - `build_eval_transforms`, `build_train_transforms` — defined Tasks 9, 10; called by `tiny_coco_dataset` fixture (Task 13 Step 1), `build_coco` (Task 14), `build_hf` (Task 16), all tests using `_build_eval`/`_patch_imagenet`.
   - `COCODataset` class signature: `(annotations, images, prompt_mode, *, transforms, text_prompt, seed=0)` — fixed Task 13, used identically by `build_coco` (Task 14), by all 17 Task 13 tests, and by the conftest fixture (Task 13 Step 1).
   - `HFDataset` class signature: `(name, split, prompt_mode, *, transforms, text_prompt, field_map, seed=0)` — fixed Task 16, used identically by all 8 Task 16 tests and by `build_hf` (Task 16).
   - Builder signatures: `build_coco(cfg, *, model_name, pipeline)` and `build_hf(cfg, *, model_name, pipeline)` — both registered under `("dataset", "coco")` and `("dataset", "hf")` and asserted via `lookup("dataset", ...)` in Tasks 14, 16.
   - `collate_batch(examples) -> dict[str, Any]` — Task 11. Output keys `images`, `image_ids`, `prompts`, `instances` match spec §2.4.
   - All types resolve. No drift.

End of plan.
