"""Tests for the interactive setup wizard (CPU-only; prompt primitives monkeypatched)."""

from __future__ import annotations

import json
from pathlib import Path

from custom_sam_peft.cli import setup_wizard as sw


def test_deep_merge_nested_dicts() -> None:
    dst = {"data": {"format": "coco"}}
    sw._deep_merge(dst, {"data": {"val_split": {"fraction": 0.1}}})
    assert dst == {"data": {"format": "coco", "val_split": {"fraction": 0.1}}}


def test_deep_merge_scalar_overwrites() -> None:
    dst = {"peft": {"method": "lora"}}
    sw._deep_merge(dst, {"peft": {"method": "qlora"}})
    assert dst["peft"]["method"] == "qlora"


def test_ctx_constructs_with_cuda_flag_and_run_mode() -> None:
    ctx = sw.Ctx(answers={}, cuda_available=False)
    assert ctx.answers == {}
    assert ctx.cuda_available is False
    assert ctx.run_mode == "train"  # default
    assert ctx.categories is None


def _write_coco(path: Path, per_cat_counts: dict[int, int], *, iscrowd_extra: int = 0) -> None:
    categories = [{"id": cid, "name": f"c{cid}"} for cid in per_cat_counts]
    images, annotations = [], []
    img_id, ann_id = 0, 0
    for cid, count in per_cat_counts.items():
        for _ in range(count):
            images.append({"id": img_id, "file_name": f"{img_id}.jpg", "height": 4, "width": 4})
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [0, 0, 2, 2],
                    "area": 4,
                    "iscrowd": 0,
                }
            )
            img_id += 1
            ann_id += 1
    for _ in range(iscrowd_extra):
        images.append({"id": img_id, "file_name": f"{img_id}.jpg", "height": 4, "width": 4})
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": next(iter(per_cat_counts)),
                "bbox": [0, 0, 2, 2],
                "area": 4,
                "iscrowd": 1,
            }
        )
        img_id += 1
        ann_id += 1
    path.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )


def test_infer_balanced_below_3x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 10, 3: 12})  # R≈1.2
    assert sw.infer_class_imbalance(str(p)) == "balanced"


def test_infer_moderate_3x_to_10x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 40})  # R=4
    assert sw.infer_class_imbalance(str(p)) == "moderate"


def test_infer_severe_at_or_above_10x(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 5, 2: 100})  # R=20
    assert sw.infer_class_imbalance(str(p)) == "severe"


def test_infer_thresholds_boundary_exact(tmp_path: Path) -> None:
    p3 = tmp_path / "r3.json"
    _write_coco(p3, {1: 10, 2: 30})  # R=3.0 → moderate
    assert sw.infer_class_imbalance(str(p3)) == "moderate"
    p10 = tmp_path / "r10.json"
    _write_coco(p10, {1: 10, 2: 100})  # R=10.0 → severe
    assert sw.infer_class_imbalance(str(p10)) == "severe"


def test_infer_unreadable_defaults_balanced(tmp_path: Path) -> None:
    assert sw.infer_class_imbalance(str(tmp_path / "missing.json")) == "balanced"


def test_infer_iscrowd_excluded(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    _write_coco(p, {1: 10, 2: 10}, iscrowd_extra=50)
    assert sw.infer_class_imbalance(str(p)) == "balanced"


# ---------------------------------------------------------------------------
# Task 12: STEPS registry + run_wizard
# ---------------------------------------------------------------------------


def _patch_prompts(monkeypatch, *, texts=None, choices=None, confirms=None):
    """Feed scripted answers to the three primitives in call order."""
    t = iter(texts or [])
    c = iter(choices or [])
    cf = iter(confirms or [])
    monkeypatch.setattr(sw, "ask_text", lambda *a, **k: next(t))
    monkeypatch.setattr(sw, "ask_choice", lambda *a, **k: next(c))
    monkeypatch.setattr(sw, "ask_confirm", lambda *a, **k: next(cf))


def test_step_fragment_shapes_are_nested_dicts(monkeypatch) -> None:
    _patch_prompts(
        monkeypatch,
        texts=["my-run", "ann.json", "imgs/", "5", ""],
        choices=["train", "coco", "none", "natural", "medium", "lora"],
    )
    monkeypatch.setattr(sw, "infer_class_imbalance", lambda *a, **k: "balanced")
    ctx = sw.Ctx(answers={}, cuda_available=False)
    answers = sw.run_wizard(ctx)
    assert answers["run"]["name"] == "my-run"
    assert answers["data"]["format"] == "coco"
    assert answers["data"]["train"]["annotations"] == "ann.json"
    assert answers["peft"]["method"] == "lora"
    assert answers["train"]["epochs"] == 5
    assert answers["train"]["loss"]["class_imbalance"] == "balanced"
    assert ctx.run_mode == "train"


def test_when_gating_skips_class_imbalance_in_eval_mode() -> None:
    step = next(s for s in sw.STEPS if s.id == "class_imbalance")
    ctx = sw.Ctx(answers={"data": {"format": "coco"}}, cuda_available=False, run_mode="eval")
    assert step.when(ctx) is False


def test_when_gating_skips_vram_autosize_without_cuda(monkeypatch) -> None:
    _patch_prompts(monkeypatch, choices=["lora"])
    step = next(s for s in sw.STEPS if s.id == "peft_sizing")
    ctx = sw.Ctx(answers={}, cuda_available=False)
    assert step.ask(ctx) == {"peft": {"method": "lora"}}
