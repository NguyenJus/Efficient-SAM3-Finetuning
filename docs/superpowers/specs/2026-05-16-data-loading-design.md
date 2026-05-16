# esam3 data-loading subsystem — Design Spec

**Status:** Approved (2026-05-16)
**Scope:** Implements step 1 of the architecture roadmap (§11 of `2026-05-15-esam3-architecture-design.md`). Fills in the four data-layer stubs (`data/coco.py`, `data/hf.py`, `data/transforms.py`, `data/collate.py`) and extends `config/schema.py` with the new data subsections. The protocols in `data/base.py` are stable and are NOT modified.

---

## 1. Goals & Non-Goals

Turn a configured dataset description (COCO instance JSON or HuggingFace `datasets` row) into `Example` objects (per `data/base.py`), apply augmentation + normalization, and collate variable-shape examples into the dict the model wrapper consumes. CPU-only; framework-agnostic with respect to the model except for normalization statistics.

**In scope (v0):**

- COCO instance-JSON adapter with polygon + RLE segmentation decode via `pycocotools.mask`.
- HuggingFace `datasets` adapter with a hybrid conventional-fields + override field-map input contract.
- `text` and `bbox` prompt modes, with `TextPrompts` configurable per `data.text_prompt.mode`.
- Albumentations-based train and eval transform pipelines, resize-longest-edge + zero-pad-to-square geometry, ImageNet-fallback normalization, `AutoImageProcessor` first-try when local cache is available.
- A `collate_batch` returning `{"images": (B,3,H,W) float, "image_ids": list[str], "prompts": list[Prompts], "instances": list[list[Instance]]}`.
- Pydantic v2 schema additions (`TextPromptConfig`, `NormalizeConfig`, `HFFieldMap`, `HFDatasetConfig`) wired into `DataConfig`.
- Dense `0..C-1` class-id remapping; sparse-vs-dense COCO id reconciliation preserved on the dataset as `coco_category_ids`.
- Unit tests using `tests/fixtures/tiny_coco/`; `AutoImageProcessor` mocked.

**Deferred (explicitly out of scope):**

- RLE-mask augmentation tuning (decode-then-augment is the v0 path; geometry on `bool` masks via nearest-neighbor is acceptable but not perf-tuned).
- Custom point and mask prompts (v0 supports text and bbox only; protocol already excludes these).
- Video / temporal frames.
- Random-crop augmentations (resize+pad only in v0).
- Named transform suites / preset menus (deferred TODO entry below).
- `DataLoader` instantiation, worker seeding, sampler choice — owned by the trainer spec.
- Model wrapper that consumes `Example`/collated dict — owned by the model spec.
- Integration and GPU tests — owned by later specs.

---

## 2. File-by-File Design

### 2.1 `src/esam3/data/coco.py`

Replaces the current `COCODataset` stub. Backed by `pycocotools.coco.COCO` for index lookups and `pycocotools.mask` for segmentation decode. Image read uses `PIL.Image.open(...).convert("RGB")` then `numpy.asarray(...)` to a `(H, W, 3)` uint8 array (Albumentations input format).

**Public class:**

```python
class COCODataset:
    """COCO instance-JSON dataset.

    State held:
      _coco:                pycocotools.coco.COCO       # lazy-loaded, see __init__
      _image_ids:           list[int]                   # image ids retained after iscrowd-only filter
      _ann_index:           dict[int, list[dict]]       # image_id -> list of non-crowd annotation records
      _coco_category_ids:   list[int]                   # sparse COCO ids, ordered by ascending value
      _cat_id_to_dense:     dict[int, int]              # sparse -> 0..C-1
      _class_names:         list[str]                   # length C, dense order
      _image_root:          Path
      _prompt_mode:         Literal["text", "bbox"]
      _text_prompt_cfg:     TextPromptConfig
      _transforms:          Callable                    # built once from build_*_transforms()
      _multiplex_cap:       int = 16                    # SAM3.1 hard cap
      _warned_truncation:   bool = False                # one-line warn-once flag
    """

    coco_category_ids: list[int]   # public mirror of _coco_category_ids for eval emission

    def __init__(
        self,
        annotations: str,
        images: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Callable[..., dict[str, Any]],
        text_prompt: TextPromptConfig,
        seed: int = 0,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Example: ...

    @property
    def class_names(self) -> list[str]: ...
```

Notes:

- `transforms` is an Albumentations `Compose` (or any callable accepting `image=`, `bboxes=`, `masks=`, `class_labels=` and returning a dict with the same keys, plus the image as a `(3,H,W) float` tensor when `ToTensorV2` is the last step).
- `seed` is recorded for the `sampled_fixed_k` text-prompt mode so eval datasets are deterministic; train pipelines may pass `cfg.run.seed` for reproducibility. The dataset does NOT seed global RNGs; that is the trainer's job.
- `__init__` performs all heavy work once: load COCO JSON, build category remap, drop iscrowd-only images, build `_ann_index`. Subsequent `__getitem__` calls are O(1) lookups plus disk read.
- One-time log lines emitted from `__init__` at INFO level: number of images dropped (iscrowd-only), count of dense classes, the chosen normalization path (delegated to `transforms.py`, which logs it on its own).

**Internal helpers (module-private):**

```python
def _load_coco_index(ann_path: Path) -> COCO: ...
def _build_category_remap(coco: COCO) -> tuple[list[int], dict[int, int], list[str]]:
    """Return (sparse_ids_sorted, sparse_to_dense, class_names_in_dense_order)."""

def _drop_crowd_only_images(coco: COCO) -> tuple[list[int], dict[int, list[dict]], int]:
    """Return (image_ids_kept, ann_index_no_crowd, dropped_count)."""

def _decode_segmentation(ann: dict, h: int, w: int) -> np.ndarray:
    """Polygon or RLE -> (H, W) bool ndarray. Uses pycocotools.mask.frPyObjects + decode."""

def _build_text_prompts(
    present_dense_ids: list[int],
    class_names: list[str],
    cfg: TextPromptConfig,
    rng: random.Random,
    image_id: int,
) -> list[str]:
    """Apply the configured TextPrompt mode and the 16-prompt multiplex cap.

    Deterministic order: sorted by dense class_id ascending; positives appear before
    negatives when relevant. RNG is used only for sampled_fixed_k.
    """
```

**Registered builder (replaces existing stub):**

```python
@register("dataset", "coco")
def build_coco(cfg: dict[str, Any], *, model_name: str) -> Dataset: ...
```

The builder reads from `cfg` (the validated `DataConfig` flattened to dict by the trainer) the keys: `train|val` (split selection lives in the trainer; the builder receives one split's dict via the trainer's call site), `images`, `annotations`, `prompt_mode`, `image_size`, `augmentations`, `text_prompt`, `normalize`. It constructs the appropriate transform pipeline (`build_train_transforms` or `build_eval_transforms`) using `model_name` for the normalization-stats lookup, then instantiates `COCODataset`. The trainer is responsible for passing `train_or_eval: Literal["train","eval"]` via `cfg["_pipeline"]` or an explicit parameter — see §3 for the contract.

**Locked builder signature (final):**

```python
@register("dataset", "coco")
def build_coco(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pipeline: Literal["train", "eval"],
) -> Dataset: ...
```

`pipeline="train"` selects `build_train_transforms`; `"eval"` selects `build_eval_transforms`. The trainer calls the builder twice (once per split) with the matching `pipeline`.

### 2.2 `src/esam3/data/hf.py`

Replaces the current `HFDataset` stub. Backed by `datasets.load_dataset(name, split=...)` with a hybrid input contract.

**Public class:**

```python
class HFDataset:
    """HuggingFace `datasets` adapter.

    State held:
      _ds:               datasets.Dataset
      _field_map:        HFFieldMap                # resolved (defaults + overrides applied)
      _class_names:      list[str]                 # from `categories` feature
      _prompt_mode:      Literal["text", "bbox"]
      _text_prompt_cfg:  TextPromptConfig
      _transforms:       Callable
      _multiplex_cap:    int = 16
      _warned_truncation: bool = False
    """

    def __init__(
        self,
        name: str,
        split: str,
        prompt_mode: Literal["text", "bbox"],
        *,
        transforms: Callable[..., dict[str, Any]],
        text_prompt: TextPromptConfig,
        field_map: HFFieldMap,
        seed: int = 0,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Example: ...

    @property
    def class_names(self) -> list[str]: ...
```

**Input contract (conventional field names, all overridable):**

| Purpose                | Default field path             | Type                                  |
|------------------------|--------------------------------|---------------------------------------|
| Image                  | `image`                        | PIL.Image or `(H,W,3) uint8` array    |
| Per-box bbox           | `objects.bbox`                 | `list[list[float]]`, xywh or xyxy     |
| Per-box class index    | `objects.category`             | `list[int]` (dense indices)           |
| Per-box segmentation   | `objects.segmentation` (opt)   | `list[<polygon-or-RLE>]`              |
| Dataset class names    | top-level feature `categories` | `Sequence[ClassLabel]` or `list[str]` |

The bbox convention (xywh vs xyxy) is configurable on `HFFieldMap.bbox_format` (default `xyxy`). HuggingFace's `datasets` convention is xywh for `imagefolder`-like configs; users set this explicitly.

If `objects.segmentation` is absent and `prompt_mode == "text"` or any mask-based loss is enabled downstream, masks are derived from boxes (filled rectangle inside the box). This is a v0 convenience; logged once. If `prompt_mode == "bbox"`, masks-from-boxes is the same behavior. This avoids hard-failing on box-only HF datasets at the data layer; the model spec decides whether mask supervision is required.

**Registered builder:**

```python
@register("dataset", "hf")
def build_hf(
    cfg: dict[str, Any],
    *,
    model_name: str,
    pipeline: Literal["train", "eval"],
) -> Dataset: ...
```

**Internal helpers:**

```python
def _resolve_field(row: dict, dotted: str) -> Any:
    """Walk a dotted path against a row dict; raise KeyError with full path on miss."""

def _validate_required_fields(ds: datasets.Dataset, field_map: HFFieldMap) -> None:
    """Read one row and check every required path; if missing raise with the
    field path AND the override key name (e.g. data.hf.field_map.bbox)."""

def _normalize_bbox(b: list[float], fmt: Literal["xywh", "xyxy"]) -> tuple[float, float, float, float]:
    """Return xyxy."""

def _resolve_class_names(ds: datasets.Dataset, field_map: HFFieldMap) -> list[str]: ...
```

### 2.3 `src/esam3/data/transforms.py`

Replaces the current `build_train_transforms` / `build_eval_transforms` stubs. Uses `albumentations` for image/bbox/mask geometry and color jitter, `cv2` is implicit through albumentations (we depend on `opencv-python-headless` rather than the GUI build).

**Public callables:**

```python
def build_eval_transforms(
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> A.Compose: ...

def build_train_transforms(
    aug_cfg: AugmentationsConfig,
    image_size: int,
    *,
    model_name: str,
    normalize: NormalizeConfig,
) -> A.Compose: ...

def resolve_normalization(
    model_name: str,
    fallback: NormalizeConfig,
) -> tuple[list[float], list[float]]:
    """Try transformers.AutoImageProcessor.from_pretrained(model_name, local_files_only=True);
    on success read image_mean/image_std. On (OSError, AttributeError, ValueError),
    return (fallback.mean, fallback.std). Logs exactly one INFO line stating which
    path was taken. Pure function w.r.t. its inputs — no globals."""
```

**Pipeline composition (locked):**

Eval:

```
A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR)
A.PadIfNeeded(min_height=image_size, min_width=image_size,
              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
              position="top_left")     # deterministic
A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
ToTensorV2()                            # -> image: (3,H,W) float32 in normalized range; masks stay torch.bool via custom step (see below)
```

Train (additional steps inserted before `A.Normalize`):

```
A.HorizontalFlip(p=0.5) if aug_cfg.hflip else nothing
A.ColorJitter(
    brightness=aug_cfg.color_jitter,
    contrast=aug_cfg.color_jitter,
    saturation=aug_cfg.color_jitter,
    hue=aug_cfg.color_jitter * 0.5,
    p=0.5,
)
```

`A.Compose(..., bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.0, min_area=0))` for the bbox track; masks are passed via the `masks=` kwarg as a `list[np.ndarray]` of `(H,W) uint8`/`bool`. After `ToTensorV2`, the dataset converts the returned `masks` list back to `(H,W) torch.bool` (Albumentations + `ToTensorV2` does NOT tensorize the `masks` list automatically when supplied via the `masks` kwarg as ndarrays — confirmed by Albumentations docs). The dataset handles this conversion in `_to_instances()` after the transform call.

Determinism: eval is deterministic (no `p<1.0` randomized steps, `LongestMaxSize` + `PadIfNeeded(position="top_left")` are deterministic). Train inherits its randomness from the caller-set RNGs (`random`, `numpy`, `albumentations`'s internal state seeded by `numpy`). This spec does NOT call `albumentations.set_seed` or any `seed_everything` helper — the trainer owns global seeding.

**Padded-region augmentation:** color jitter applies to the entire post-padded image, including the zero-padded border. This is acceptable because (a) ColorJitter's brightness/contrast multipliers leave `0` mapped near `0`, (b) the model wrapper is responsible for any pad-aware masking. A future revisit can add a pad-mask channel; not in v0.

### 2.4 `src/esam3/data/collate.py`

Replaces the current `collate_batch` stub.

**Public callable:**

```python
def collate_batch(examples: list[Example]) -> dict[str, Any]: ...
```

**Output shape (locked):**

```python
{
  "images":    torch.Tensor,                # (B, 3, H, W) float, dtype matches dataset output
  "image_ids": list[str],                   # length B
  "prompts":   list[Prompts],               # length B, TextPrompts | BoxPrompts
  "instances": list[list[Instance]],        # length B; inner lists may be empty? NO — dataset drops empty images
}
```

**Behavior:**

- `images` is stacked with `torch.stack`. All examples must share `(3, H, W)`; mismatch raises `ValueError` with both shapes printed.
- `image_ids`, `prompts`, `instances` are unstacked Python lists (ragged data is not vectorized; SAM3.1 multiplex prediction is per-image).
- The collator does NOT enforce same prompt type across the batch; the model wrapper either supports a mixed batch or the trainer's `DataLoader` is configured per-mode. v0 trainer creates one Dataset whose `prompt_mode` is uniform, so this is moot but the collator is non-restrictive.
- Empty input list raises `ValueError("collate_batch received empty batch")`.

No internal helpers.

---

## 3. Config Schema Additions

All additions live in `src/esam3/config/schema.py`. All models inherit `_Strict` (so `extra="forbid"` applies). Pydantic v2 syntax.

### 3.1 `TextPromptMode`

```python
TextPromptMode = Literal["present", "all", "present_plus_negatives", "sampled_fixed_k"]
```

### 3.2 `TextPromptConfig`

```python
class TextPromptConfig(_Strict):
    """How TextPrompts.classes is populated for each image when prompt_mode='text'.

    - present:                Use exactly the categories present in the image's
                              annotations (post-iscrowd filter). Default.
    - all:                    Use the full dataset class vocabulary every time.
    - present_plus_negatives: Use the present categories plus N randomly-sampled
                              negative class names per image (from the full
                              vocabulary minus the present set).
    - sampled_fixed_k:        Use exactly k class names: all positives, plus
                              negatives sampled to reach k. If positives exceed
                              k, positives are truncated (kept in dense-id
                              ascending order). All choices are deterministic
                              given (dataset_seed, image_id).
    """

    mode: TextPromptMode = "present"
    negatives_per_image: int = Field(default=0, ge=0)   # used only when mode='present_plus_negatives'
    k: int = Field(default=16, ge=1, le=16)              # used only when mode='sampled_fixed_k'
```

The SAM3.1 multiplex cap of 16 prompts/image is enforced inside the dataset (`_build_text_prompts`) regardless of mode. `k <= 16` is enforced by the schema; the runtime truncation handles `mode="all"` and the negatives-overflow cases.

### 3.3 `NormalizeConfig`

```python
class NormalizeConfig(_Strict):
    """Normalization stats used when AutoImageProcessor cannot be loaded.

    Resolution order at dataset construction:
      1. AutoImageProcessor.from_pretrained(model.name, local_files_only=True)
         and read image_mean/image_std.
      2. On OSError/AttributeError, fall back to (mean, std) here.
    """

    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406], min_length=3, max_length=3)
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225], min_length=3, max_length=3)
```

Per-element constraints: `mean[i] in [0, 1]`, `std[i] > 0`. Validated via a `@model_validator(mode="after")` on `NormalizeConfig`.

### 3.4 `HFFieldMap`

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
    segmentation: str | None = "objects.segmentation"  # None disables seg lookup
    categories_feature: str = "categories"
    bbox_format: Literal["xywh", "xyxy"] = "xyxy"
```

### 3.5 `HFDatasetConfig`

```python
class HFDatasetConfig(_Strict):
    """HuggingFace dataset specification (used when DataConfig.format == 'hf')."""

    name: str = Field(min_length=1)
    split_train: str = "train"
    split_val: str = "validation"
    field_map: HFFieldMap = Field(default_factory=HFFieldMap)
```

### 3.6 `DataConfig` extension

```python
class DataConfig(_Strict):
    format: DataFormat
    train: DataSplit                                 # used when format == 'coco'
    val: DataSplit                                   # used when format == 'coco'
    hf: HFDatasetConfig | None = None                # required when format == 'hf'
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

Note: `train` and `val` remain required by the schema for backward compatibility with the existing `coco_*_*.yaml` examples; when `format == "hf"`, the COCO `train`/`val` paths are present but ignored at runtime. (Alternative: make them optional; rejected because the config-loader resolves them as paths and we don't want a behavior change.)

**Assumption noted:** The `data.train` / `data.val` `DataSplit` blocks remain required in the schema even under `format: hf` to avoid an invasive change to `config/loader.py`'s `_PATH_KEYS`. The HF builder ignores them. Documented in the example YAML.

### 3.7 Sample YAML

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

```yaml
data:
  format: hf
  train: {annotations: unused.json, images: unused/}     # ignored under format=hf
  val:   {annotations: unused.json, images: unused/}
  hf:
    name: cppe-5
    split_train: train
    split_val: test
    field_map:
      bbox: objects.bbox
      category: objects.category
      bbox_format: xywh
  prompt_mode: bbox
  image_size: 1024
```

---

## 4. Data Flow

### 4.1 COCO + `prompt_mode: text`

`COCODataset(annotations, images, prompt_mode="text", transforms=<train_compose>, text_prompt=cfg.data.text_prompt, seed=cfg.run.seed)` is built once.

`__getitem__(i)`:

1. `image_id = self._image_ids[i]`.
2. `img_record = self._coco.loadImgs([image_id])[0]`; resolve `path = self._image_root / img_record["file_name"]`.
3. `np_img = np.asarray(PIL.Image.open(path).convert("RGB"))` -> `(H, W, 3) uint8`.
4. `anns = self._ann_index[image_id]` (non-crowd records only).
5. Decode each annotation:
   - `box = _xywh_to_xyxy(ann["bbox"])` -> `[x0, y0, x1, y1]`.
   - `mask = _decode_segmentation(ann, H, W)` -> `(H, W) bool` ndarray.
   - `dense_cls = self._cat_id_to_dense[ann["category_id"]]`.
6. Build the Albumentations input dict: `image=np_img, bboxes=[<list of xyxy>], masks=[<list of (H,W) uint8>], class_labels=[<dense ids>]`.
7. Call `out = self._transforms(**inputs)`. After the Compose, `out["image"]` is `(3, H', W') float`; `out["bboxes"]` is `list[tuple[float, float, float, float]]`; `out["masks"]` is `list[np.ndarray]` of `(H', W') uint8`; `out["class_labels"]` is the surviving subset (Albumentations drops boxes whose remaining geometry violates `min_area`/`min_visibility`; v0 sets both to 0 so no drops occur, but we still align the lists).
8. Convert each transformed mask back to `torch.from_numpy(m.astype(bool))`.
9. Build the TextPrompts list:
   - `present_dense_ids = sorted(set(out["class_labels"]))`.
   - `prompt_strings = _build_text_prompts(present_dense_ids, self._class_names, self._text_prompt_cfg, rng=random.Random((self._seed, image_id)), image_id=image_id)`.
   - If `len(prompt_strings) > 16`, truncate to 16 (positives kept first, deterministic order) and emit one warn-once line.
10. Build `Instance(mask, class_id=dense_cls, box=torch.tensor(xyxy))` for each surviving annotation.
11. Return `Example(image=out["image"], image_id=str(image_id), prompts=TextPrompts(classes=prompt_strings), instances=[...])`.

### 4.2 COCO + `prompt_mode: bbox`

Same as 4.1 except step 9 is replaced:

9'. Build `BoxPrompts(boxes=torch.tensor(bbox_xyxy_after_transform, dtype=torch.float32), class_ids=torch.tensor(class_labels_after_transform, dtype=torch.int64))`. If `len(boxes) > 16`, truncate to 16 with the same deterministic order (sort by dense `class_id`, then by box top-left for tie-break) and warn-once. The matching `Instance` list is truncated in parallel.

### 4.3 HF dataset (either prompt mode)

`HFDataset.__init__`:

1. `self._ds = datasets.load_dataset(name, split=split_for_pipeline)`.
2. `_validate_required_fields(self._ds, field_map)` raises with the exact dotted path + `data.hf.field_map.<key>` override hint on missing fields.
3. `self._class_names = _resolve_class_names(self._ds, field_map)`.
4. Build transforms once (passed in via constructor).

`__getitem__(i)`:

1. `row = self._ds[i]`.
2. `np_img = _to_np_uint8(_resolve_field(row, field_map.image))` (handles PIL or ndarray inputs).
3. `bboxes_raw = _resolve_field(row, field_map.bbox)` -> list of 4-tuples.
4. `classes = _resolve_field(row, field_map.category)` -> list of dense ints.
5. If `field_map.segmentation` is not None and `row` has the path: decode each segmentation entry; otherwise generate filled-rectangle masks from `bboxes_raw` and warn-once.
6. Convert bboxes to xyxy using `field_map.bbox_format`.
7. Steps 6–11 of §4.1 with `class_labels = classes`.

### 4.4 Trainer integration (for context only, not implemented here)

The trainer (later spec) does:

```python
# pseudo
ds_train = lookup("dataset", cfg.data.format)(
    cfg.data.model_dump(),       # dict serialization
    model_name=cfg.model.name,
    pipeline="train",
)
ds_val = lookup("dataset", cfg.data.format)(cfg.data.model_dump(), model_name=cfg.model.name, pipeline="eval")
loader = DataLoader(ds_train, batch_size=cfg.train.batch_size, collate_fn=collate_batch, ...)
```

Data layer does NOT import `TrainConfig`. It receives `cfg: dict[str, Any]` plus `model_name: str` and `pipeline: Literal["train","eval"]`.

---

## 5. Edge Cases

### 5.1 Multiplex overflow (>16 prompts)

- Text mode `present` overflow: image has >16 categories present. Sort by dense class_id ascending; keep first 16. Warn-once per dataset construction: `WARNING esam3.data.coco: image_id=N requested 23 text prompts; truncating to 16. Suppressing further warnings for this dataset.`
- Text mode `all` with `len(class_names) > 16`: every image overflows; the warn-once fires on the first call. Order: dense class_id ascending; positives first relative to that image's annotations.
- Text mode `sampled_fixed_k`: `k` is schema-bounded to `<=16`, so no overflow; if positives exceed `k`, truncate positives (deterministic order) — counted as a separate warn-once event noting "positives > k".
- Box mode: same approach; truncate by sorting on `(class_id, x0, y0)` for stable behavior.

### 5.2 Empty-after-iscrowd

If an image has zero non-crowd annotations after filtering, it is dropped from `_image_ids` at construction time. `__len__` reflects the post-drop count. `__getitem__` never sees an empty image. Dropped count is logged once at INFO.

Additionally a TODO entry is appended to `logs/TODO.md`:

```
[2026-05-16] [planner] [DEFERRED] revisit iscrowd handling after first real eval pass — v0 drops iscrowd=1 annotations entirely
[2026-05-16] [planner] [DEFERRED] named transform suites — let users pick "default" / "augmentation_heavy" / "geometric_only" from a menu instead of editing aug params
```

The implementation appends both lines to `logs/TODO.md` at install/first-run time of the data layer (specifically: the implementation plan's first task includes a one-time append to `logs/TODO.md`; the data layer code itself does not write to logs).

### 5.3 Missing HF field

When `_validate_required_fields` cannot resolve a path, raise:

```
KeyError("HF dataset is missing required field 'objects.bbox'. "
         "Set data.hf.field_map.bbox to the correct dotted path.")
```

(Use a custom `HFFieldError(KeyError)` to make tests precise; subclass of `KeyError` for ergonomic handling.)

### 5.4 Missing local image processor (normalize fallback)

`AutoImageProcessor.from_pretrained(model_name, local_files_only=True)` raises `OSError` when no cached model is present. `resolve_normalization` catches `(OSError, AttributeError, ValueError)`, logs:

```
INFO esam3.data.transforms: AutoImageProcessor cache miss for 'facebook/sam3.1'; falling back to NormalizeConfig (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
```

and returns `(fallback.mean, fallback.std)`. On success it logs:

```
INFO esam3.data.transforms: Using image_mean/image_std from AutoImageProcessor for 'facebook/sam3.1'.
```

Exactly one of these two lines is emitted per call.

### 5.5 Polygon vs RLE

`pycocotools.mask` handles both. For polygon, `segmentation` is `list[list[float]]`; convert via `frPyObjects(seg, H, W)` then `decode`. For RLE, `segmentation` is `dict{"counts": str|list, "size": [H, W]}`; pass directly to `decode`. Output is `(H, W, k) uint8`; sum over the last axis and cast to bool. The decode helper handles both branches transparently.

Augmentation of the decoded `(H, W) bool` mask is via nearest-neighbor under `A.LongestMaxSize` and `A.PadIfNeeded`; horizontal flip is exact. ColorJitter does not affect masks. Albumentations' `masks=` API supports a list of arrays out of the box.

### 5.6 Sparse vs dense class IDs

COCO often uses non-contiguous category ids (e.g. 1, 3, 7, ...). The category remap is built once in `_build_category_remap`:

- `sparse_ids_sorted = sorted([cat["id"] for cat in coco.dataset["categories"]])`.
- `class_names = [cat["name"] for cat in sorted(coco.dataset["categories"], key=lambda c: c["id"])]`.
- `cat_id_to_dense = {sparse_id: dense_idx for dense_idx, sparse_id in enumerate(sparse_ids_sorted)}`.

The dataset exposes:
- `class_names: list[str]` (length C, dense order) — required by the protocol.
- `coco_category_ids: list[int]` (length C, same order) — extra attribute, used by eval to emit predictions in original COCO category id space. Not part of the `Dataset` protocol.

HF datasets are already dense (typed by `datasets.ClassLabel`). No remap occurs; `class_names` comes directly from the feature. The dataset has no `coco_category_ids` attribute (or it's `None`).

### 5.7 Padded-region augmentation behavior

- `LongestMaxSize` followed by `PadIfNeeded(position="top_left", value=0, mask_value=0)` produces a `(image_size, image_size, 3)` uint8 image with zeros in the bottom/right padding strip.
- `HorizontalFlip` flips the entire square including padding (the padding moves from right to left for that example). Boxes and masks are flipped consistently by Albumentations.
- `ColorJitter` modifies all pixels including padding. As noted in §2.3, this is acceptable in v0; the model wrapper is responsible for pad-aware behavior if needed.
- `Normalize` then `ToTensorV2` produce the `(3, H, W) float32` tensor whose padded region has the normalized value of `0 - mean / std` (a slightly negative number per channel). Documented; no special-case code.

---

## 6. Testing Strategy

Unit tests only. All CPU. Mock `AutoImageProcessor.from_pretrained`. Use the existing `tests/fixtures/tiny_coco/` fixture. The model name `"facebook/sam3.1"` is referenced in fixtures but never loaded.

Test files to add (all under `tests/unit/`):

### 6.1 `tests/unit/test_data_coco.py`

Covering `data/coco.py`. Uses `tiny_coco_dir` and a small `build_train_transforms` / `build_eval_transforms` invocation with patched `AutoImageProcessor`.

Assertions per case:

1. `test_class_names_dense_and_ordered`: `ds.class_names == ["thing_a", "thing_b"]`. `ds.coco_category_ids == [1, 2]`.
2. `test_len_drops_empty_after_iscrowd`: When all anns on image 2 are iscrowd=1, `len(ds) == 1` and the dropped count is logged.
3. `test_getitem_text_mode_present`: For `tiny_coco/img_000001.png` which has both classes, `ex.prompts.classes == ["thing_a", "thing_b"]` (dense ascending).
4. `test_getitem_text_mode_all`: With `mode="all"`, every image returns the full vocabulary, sorted by dense id.
5. `test_getitem_text_mode_present_plus_negatives`: Patch the dataset's RNG; assert exactly `len(present) + negatives_per_image` strings; assert positives appear first; assert no duplicates.
6. `test_getitem_text_mode_sampled_fixed_k`: `k=3`; assert exactly 3 strings; positives included; rest sampled.
7. `test_multiplex_truncation_text`: Synthesize a COCO with 20 categories on one image; assert returned classes length == 16; assert one warning.
8. `test_multiplex_truncation_box`: Same with 20 boxes; assert `boxes.shape[0] == 16` and `class_ids.shape[0] == 16`; assert instances list length == 16.
9. `test_getitem_bbox_mode_returns_BoxPrompts`: `isinstance(ex.prompts, BoxPrompts)`; `boxes.dtype == torch.float32`; `class_ids.dtype == torch.int64`; xyxy coordinates fall inside `[0, image_size]`.
10. `test_polygon_segmentation_decoded`: One annotation; `ex.instances[0].mask.shape == (H, W)`; `ex.instances[0].mask.dtype == torch.bool`; sum of mask > 0.
11. `test_rle_segmentation_decoded`: Synthesize a COCO annotation with an RLE `segmentation` dict; assert decode produces a non-empty bool mask.
12. `test_iscrowd_skipped`: Add an iscrowd=1 annotation to one of the images; assert that annotation does not appear in `ex.instances`.
13. `test_dropped_empty_image_logged_once`: `caplog` captures exactly one INFO line containing "dropped" when no annotations remain on any image after filtering.
14. `test_image_resize_geometry`: Set `image_size=64`; assert returned image shape is `(3, 64, 64)`; assert any bbox xyxy is inside `[0, 64]`; assert masks have shape `(64, 64)`.
15. `test_sparse_to_dense_remap`: Synthesize a COCO with `category_id` values `[3, 7]`; assert `class_names` has length 2, `coco_category_ids == [3, 7]`, and dense ids on instances are `0` or `1`.
16. `test_register_coco_lookup`: `lookup("dataset", "coco")` returns the builder; calling it with the tiny_coco dict + `pipeline="eval"` returns a working dataset.
17. `test_deterministic_text_sampling_under_fixed_seed`: Two `COCODataset` instances built with `seed=42, mode="sampled_fixed_k"` return identical prompt lists for the same `image_id`.

### 6.2 `tests/unit/test_data_hf.py`

Mock `datasets.load_dataset` with an in-memory `datasets.Dataset.from_dict(...)`. Two cases per shape concern.

1. `test_required_fields_validation_default_paths`: Missing `objects.bbox` raises `HFFieldError` whose message mentions `objects.bbox` and `data.hf.field_map.bbox`.
2. `test_field_map_override_picks_alternate_path`: Provide a dataset with `annotations.bbox` instead of `objects.bbox`; with `field_map.bbox = "annotations.bbox"` the dataset loads.
3. `test_class_names_from_categories_feature`: Top-level `categories` is a `Sequence[ClassLabel]`-equivalent; `ds.class_names` matches.
4. `test_getitem_text_mode_present`: Same shape assertions as coco text test.
5. `test_getitem_bbox_mode`: Same shape assertions as coco bbox test.
6. `test_bbox_format_xywh_conversion`: When `bbox_format=xywh`, bboxes are converted to xyxy before transforms see them; verify a known input -> known output.
7. `test_masks_from_boxes_when_segmentation_absent`: `field_map.segmentation = None`; assert masks are filled rectangles inside the original boxes; one warn-once line.
8. `test_register_hf_lookup`: `lookup("dataset", "hf")` returns the builder; builder works with mocked `datasets.load_dataset`.

### 6.3 `tests/unit/test_data_transforms.py`

1. `test_resolve_normalization_uses_image_processor_when_available`: Monkeypatch `transformers.AutoImageProcessor.from_pretrained` to return an object with `image_mean=[0.1, 0.2, 0.3]`, `image_std=[0.4, 0.5, 0.6]`. Assert returned tuple matches; assert log line names `AutoImageProcessor`.
2. `test_resolve_normalization_falls_back_on_oserror`: Monkeypatch to raise `OSError`. Assert returned tuple equals `NormalizeConfig()` defaults; assert log line says `cache miss`.
3. `test_eval_transforms_shape`: Input `(40, 80, 3) uint8`; eval transforms with `image_size=64`; assert output image tensor is `(3, 64, 64) float32`; assert bbox `[0,0,80,40]` -> roughly `[0, 0, 64, 32]` after resize+pad; mask shape `(64, 64)`.
4. `test_train_transforms_deterministic_with_seeded_global_rng`: Set `random.seed(0); np.random.seed(0); torch.manual_seed(0)`; run twice; assert identical output. (Confirms our pipeline does not introduce extra RNGs.)
5. `test_train_transforms_hflip_disabled`: With `hflip=False`, run 100 iterations; assert no flip occurred (use a non-symmetric synthetic image).
6. `test_color_jitter_zero_preserves_color`: `color_jitter=0.0`; assert output equals normalize-only path.
7. `test_padding_position_top_left`: Confirm padding region is at the bottom-right of the output (top-left position preserves original content at top-left).

### 6.4 `tests/unit/test_data_collate.py`

1. `test_collate_stacks_images`: 3 examples of shape `(3, 64, 64)`; output `images.shape == (3, 3, 64, 64)`.
2. `test_collate_keeps_prompts_as_list`: Mixed `TextPrompts` and `BoxPrompts` examples; output `prompts` is a Python list of len 3, types preserved.
3. `test_collate_keeps_instances_as_list_of_lists`: Each inner list length matches the example's instance count.
4. `test_collate_image_id_order_preserved`: Output `image_ids` matches input order.
5. `test_collate_empty_batch_raises`: `collate_batch([])` raises `ValueError`.
6. `test_collate_image_shape_mismatch_raises`: Examples with `(3, 64, 64)` and `(3, 32, 32)` raise `ValueError` whose message includes both shapes.

### 6.5 `tests/unit/test_data_schema_extensions.py`

1. `test_text_prompt_config_defaults`: `TextPromptConfig().mode == "present"`, `negatives_per_image == 0`, `k == 16`.
2. `test_text_prompt_config_k_bounded`: `k=17` raises `ValidationError`; `k=0` raises.
3. `test_normalize_config_defaults`: ImageNet stats; lengths are 3.
4. `test_normalize_config_validation_rejects_wrong_length`: `mean=[0.1, 0.2]` raises.
5. `test_normalize_config_validation_rejects_nonpositive_std`: `std=[0.0, 0.1, 0.1]` raises.
6. `test_hf_field_map_defaults`: Match the table in §2.2.
7. `test_hf_dataset_config_required_name`: Missing `name` raises.
8. `test_data_config_requires_hf_when_format_hf`: `format="hf"` and `hf=None` raises with a message mentioning `data.hf`.
9. `test_data_config_accepts_coco_without_hf`: `format="coco"` and `hf=None` is valid.
10. `test_existing_example_yaml_still_validates`: `configs/examples/coco_text_lora.yaml` and `coco_bbox_qlora.yaml` still validate (defaults for new fields kick in).

### 6.6 Removal from `test_stubs_raise.py`

The data-layer assertions in `test_data_stubs` are removed (or, equivalently, the file is updated to drop the data-layer calls). The implementation plan does this when the stubs are replaced; the spec just records the contract.

### 6.7 Coverage

The data layer must keep the package's >=80% coverage gate. Adding ~17+8+7+6+10 = 48 unit tests against ~600 LOC of new code is comfortably above 80%.

---

## 7. Dependencies

Two new core dependencies, added to `pyproject.toml` under `[project].dependencies`:

- `albumentations>=1.4` — MIT License. Compatible with Apache-2.0 (MIT is permissive; redistribution within an Apache-2.0 work is allowed provided the MIT notice is preserved in any source distribution that bundles it; we don't bundle, so the runtime dep poses no obligation beyond pip-time resolution).
- `opencv-python-headless>=4.10` — Apache License 2.0. Same license as this package; trivially compatible. We choose `-headless` rather than `opencv-python` to avoid GUI/X11 dependencies on training servers.

License compatibility statement: this package is Apache-2.0. Albumentations (MIT) and opencv-python-headless (Apache-2.0) are both permissive and compatible. Neither is statically linked nor vendored. Both appear in the user-visible `pyproject.toml`.

No new optional dependencies. `pillow>=10` is already in `[dependency-groups].dev`; it must also be a core dep now (for `PIL.Image.open` in the COCO reader path) — add `"pillow>=10"` to `[project].dependencies` and remove the duplicate from `dev`.

---

## 8. Definition of Done

A work item is "done" only when ALL of the following are true:

### 8.1 Schema (D, difficulty: M)

- `TextPromptMode`, `TextPromptConfig`, `NormalizeConfig`, `HFFieldMap`, `HFDatasetConfig` exist in `src/esam3/config/schema.py` exactly as specified in §3.
- `DataConfig` is extended with `hf`, `text_prompt`, `normalize` fields plus the `_check_format_specific` validator.
- All 10 cases in §6.5 pass.
- `mypy --strict` passes on `src/esam3`.
- Existing `tests/unit/test_config_schema.py` still passes unchanged.

### 8.2 Transforms (D, difficulty: M)

- `build_eval_transforms`, `build_train_transforms`, `resolve_normalization` implemented per §2.3.
- All 7 cases in §6.3 pass. The `AutoImageProcessor` success and fallback branches are both exercised.
- The function emits exactly one INFO log line per call (asserted in tests via `caplog`).
- `mypy --strict` passes.

### 8.3 Collate (D, difficulty: L)

- `collate_batch` implemented per §2.4.
- All 6 cases in §6.4 pass.
- `ValueError` is raised with informative messages on empty batch and shape mismatch.

### 8.4 COCO dataset (D, difficulty: H)

- `COCODataset` implemented per §2.1 and §4.1–4.2.
- Sparse-to-dense remap; `class_names` and `coco_category_ids` populated.
- iscrowd filter drops crowd anns; images empty after filter are dropped from `__len__`.
- Polygon and RLE decode both supported.
- Multiplex cap of 16 enforced; warn-once observed in tests.
- `@register("dataset", "coco")` builder respects `pipeline` and `model_name`.
- All 17 cases in §6.1 pass.

### 8.5 HF dataset (D, difficulty: H)

- `HFDataset` implemented per §2.2 and §4.3.
- Field-map default + override path resolution.
- Missing-required-field error names both the path and the override key.
- Filled-rectangle mask fallback when segmentation is absent; warn-once.
- `@register("dataset", "hf")` builder respects `pipeline` and `model_name`.
- All 8 cases in §6.2 pass.

### 8.6 Dependencies (D, difficulty: L)

- `pyproject.toml` lists `albumentations>=1.4`, `opencv-python-headless>=4.10`, and promotes `pillow>=10` to core deps.
- `uv lock` regenerates `uv.lock` cleanly.
- License compatibility statement appears in this spec.

### 8.7 Docs & Logs (D, difficulty: L)

- Two `[DEFERRED]` lines appended to `logs/TODO.md` per §5.2.
- `ARCHITECTURE.md` data-section sentence "@register('dataset', ...) adapters" is updated to mention `pipeline` and `model_name` keyword args. (One-line edit.)
- The two `configs/examples/*.yaml` files are augmented with new `text_prompt` and `normalize` blocks (using defaults so behavior is unchanged) and validated by `test_existing_example_yaml_still_validates`.

### 8.8 Cross-cutting

- `uv sync --all-extras --dev && ruff check && mypy && pytest` all pass.
- Coverage on `src/esam3` is `>=80%`.
- `data/base.py` is unmodified (assert via `git diff` in CI step or implementer self-check).
- No imports of `esam3.config.schema.TrainConfig` from inside `src/esam3/data/`. The only typed config the data layer imports is `AugmentationsConfig`, `TextPromptConfig`, `NormalizeConfig`, `HFFieldMap`, `HFDatasetConfig`. Verified by a `tests/unit/test_data_import_boundary.py` that imports `ast` and grep-walks the four data files.

---

## 9. What This Spec Does NOT Cover

- `torch.utils.data.DataLoader` instantiation, `num_workers`, `pin_memory`, `persistent_workers` — owned by the training-loop spec.
- Seed propagation across `random`, `numpy`, `torch`, `albumentations`, and DataLoader workers — owned by the training-loop spec. This spec only documents that the dataset accepts a `seed` parameter used for `sampled_fixed_k` text-prompt determinism.
- The model wrapper that consumes the `{"images", "image_ids", "prompts", "instances"}` dict — owned by the model spec.
- GPU and integration tests — those are categorized under `@pytest.mark.gpu` / `@pytest.mark.integration` and built in later specs (steps 5, 9 of the roadmap).
- COCO mAP scoring and per-class AP — owned by the eval spec.
- Tensorboard / W&B image logging from data samples — owned by the tracking spec; the data layer surfaces no logging callbacks.

---

## 10. Assumptions

Listed so the implementer does not need to ask:

1. `data.train` / `data.val` `DataSplit` blocks remain required at the schema level even under `format: hf`, to avoid changes to `config/loader.py._PATH_KEYS`. The HF builder ignores them. (See §3.6.)
2. `pillow` is promoted from a dev dep to a core dep; no version bump needed.
3. `opencv-python-headless` is preferred over `opencv-python`; this is a deliberate choice for headless training environments.
4. `Albumentations.ToTensorV2` returns the image as `torch.float32 (3, H, W)` after `A.Normalize`, but does NOT tensorize the `masks` list passed via the `masks=` kwarg. The dataset converts each mask to `torch.bool` after the Compose call.
5. The 16-prompt multiplex cap is a SAM3.1 hard constraint, applied uniformly to text and box prompts even though box-prompt limits may not be exactly the same in the real model — the cap is conservative and harmless.
6. `pycocotools.coco.COCO` is acceptable to import at the top of `data/coco.py` (already in the project's runtime deps).
7. `datasets.load_dataset` may be invoked at `HFDataset.__init__` time. For huge datasets this is materially slow; v0 accepts this. A streaming path is deferred.
8. The `seed` parameter on `COCODataset.__init__` is local to text-prompt sampling and does NOT seed `random`/`numpy`/`torch`. Trainer owns those.
9. The data layer's logger name is `esam3.data.<module>` (standard `logging.getLogger(__name__)`).
10. `iscrowd=1` annotations are dropped entirely in v0 (no mask, no box, no class). Revisit logged as TODO.

End of spec.
