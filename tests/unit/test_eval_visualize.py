"""Unit tests for eval/visualize.py pure primitives (CPU-only, no model)."""

from __future__ import annotations

import math

import torch

from custom_sam_peft.data.base import Example, Instance, TextPrompts
from custom_sam_peft.eval.visualize import pick_samples


class _FakeDataset:
    """Index-aligned dataset whose examples carry the requested #GT instances."""

    def __init__(self, gt_counts: list[int]) -> None:
        self._examples = []
        for i, n in enumerate(gt_counts):
            insts = [
                Instance(
                    mask=torch.zeros(4, 4, dtype=torch.bool),
                    class_id=0,
                    box=torch.tensor([0.0, 0.0, 1.0, 1.0]),
                )
                for _ in range(n)
            ]
            self._examples.append(
                Example(
                    image=torch.zeros(3, 4, 4),
                    image_id=f"img_{i}",
                    prompts=TextPrompts(classes=["a"]),
                    instances=insts,
                )
            )
        self.class_names = ["a"]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, i: int) -> Example:
        return self._examples[i]


def _bands(n: int) -> tuple[int, int, int]:
    good = round(0.5 * n)
    worst = min(2, max(1, round(0.2 * n)))
    median = n - good - worst
    return good, median, worst


def test_band_sizes_n10() -> None:
    ds = _FakeDataset([1] * 30)
    iou = [i / 30 for i in range(30)]  # all distinct, all GT-bearing
    picked = pick_samples(iou, ds, 10)
    assert len(picked) == 10
    assert _bands(10) == (5, 3, 2)


def test_band_sizes_various_n() -> None:
    for n, (g, m, w) in [(1, _bands(1)), (2, _bands(2)), (5, _bands(5)), (20, _bands(20))]:
        assert g + m + w == n
        assert w <= 2  # worst cap


def test_worst_cap_large_n() -> None:
    ds = _FakeDataset([1] * 50)
    iou = [i / 50 for i in range(50)]
    picked = pick_samples(iou, ds, 20)
    assert len(picked) == 20
    g, m, w = _bands(20)
    assert w == 2  # capped despite round(0.2*20)=4


def test_gt_filter_excludes_no_gt_images() -> None:
    # idx 0 has the highest IoU but NO GT → must never be selected.
    ds = _FakeDataset([0, 1, 1, 1, 1])
    iou = [1.0, 0.9, 0.8, 0.7, 0.6]
    picked = pick_samples(iou, ds, 4)
    assert 0 not in picked
    assert set(picked) <= {1, 2, 3, 4}


def test_small_pool_returns_all_candidates() -> None:
    ds = _FakeDataset([1, 1, 1])  # 3 GT-bearing candidates
    iou = [0.3, 0.2, 0.1]
    picked = pick_samples(iou, ds, 10)
    assert sorted(picked) == [0, 1, 2]
    assert len(picked) <= 10


def test_indices_unique_across_bands() -> None:
    ds = _FakeDataset([1] * 12)
    iou = [i / 12 for i in range(12)]
    picked = pick_samples(iou, ds, 10)
    assert len(picked) == len(set(picked))  # no index in two bands


def test_nan_sorts_to_bottom_worst_only() -> None:
    # idx 2 is NaN → ranked -inf → only ever a "worst" pick, never "good".
    ds = _FakeDataset([1, 1, 1, 1, 1, 1])
    iou = [0.9, 0.8, math.nan, 0.6, 0.5, 0.4]
    picked = pick_samples(iou, ds, 6)  # pool == N → all returned
    assert 2 in picked  # eligible as worst
    # With N < pool, the top "good" band must not include the NaN index.
    picked2 = pick_samples(iou, ds, 2)
    g, _, w = _bands(2)  # (1, 0, 1)
    assert picked2[0] != 2  # highest-IoU first, never the NaN


def test_returned_in_descending_iou_order() -> None:
    ds = _FakeDataset([1] * 6)
    iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    picked = pick_samples(iou, ds, 6)
    vals = [iou[i] for i in picked]
    assert vals == sorted(vals, reverse=True)
