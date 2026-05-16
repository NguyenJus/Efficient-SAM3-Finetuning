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
    assert out[0] == "a"
    assert len(out) == 3
    assert len(set(out)) == 3


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
