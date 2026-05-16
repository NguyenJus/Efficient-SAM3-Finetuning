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


_ = (np, TextPromptConfig)  # silences F401 until Task 16
