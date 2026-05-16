"""Tests for data/hf.py — helpers + dataset + builder."""

from __future__ import annotations

from typing import Any

import datasets as hf_datasets
import pytest
from PIL import Image

from esam3.config.schema import HFFieldMap
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
        "categories": [["thing"]] * n,
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
