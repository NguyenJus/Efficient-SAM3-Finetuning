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
    return decoded.astype(bool)  # type: ignore[no-any-return]


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
        raise NotImplementedError("filled in by spec: spec/data-loading (Task 13)")

    def __getitem__(self, i: int) -> Example:
        raise NotImplementedError("filled in by spec: spec/data-loading (Task 13)")

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError("filled in by spec: spec/data-loading (Task 13)")


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
