"""Eval GT-vs-Pred qualitative visualization (final/standalone eval path only).

Owns: variety-weighted image selection, config-aware denormalization, GT-instance
to render-entry conversion, the per-image matched render pair, the compositor, and
the top-level write_eval_visualizations entry point. Reuses predict/visualize.py for
the shared single-panel renderer, palette, and color map.

n-channel rule (§7.1): for inputs with more than 3 channels, only the first 3
denormalized channels are rendered as RGB (best-effort preview, not a faithful
multi-spectral visualization).

Spec: docs/superpowers/specs/2026-05-29-eval-visualize-design.md.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from custom_sam_peft.data.base import Dataset

_LOG = logging.getLogger(__name__)


def _spread_indices(sorted_indices: list[int], k: int) -> list[int]:
    """Pick k evenly spaced elements from sorted_indices (preserving order)."""
    if k <= 0 or not sorted_indices:
        return []
    if k >= len(sorted_indices):
        return list(sorted_indices)
    # Evenly spaced positions across [0, len-1].
    positions = [round(j * (len(sorted_indices) - 1) / (k - 1)) for j in range(k)] if k > 1 else [0]
    seen: set[int] = set()
    out: list[int] = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            out.append(sorted_indices[p])
    # Back-fill if rounding collided (keep k distinct positions when possible).
    j = 0
    while len(out) < k and j < len(sorted_indices):
        if sorted_indices[j] not in out:
            out.append(sorted_indices[j])
        j += 1
    return out


def pick_samples(
    per_example_iou: Sequence[float],
    dataset: Dataset,
    count: int,
) -> list[int]:
    """Return up to `count` dataset indices, variety-weighted toward high IoU.

    Filters to candidates with >=1 GT instance (excludes no-GT images), ranks by
    per_example_iou (NaN -> -inf, eligible only as 'worst'), and picks a
    good/median/worst spread per spec §5.3. Returns <= count indices when the
    candidate pool is smaller than count. Indices are returned in descending-IoU
    order so the written composites are filename-stable and roughly best-to-worst.
    """
    # Candidate filter: >=1 GT instance. per_example_iou is index-aligned to the
    # dataset slice the metrics pass evaluated (full or lite).
    candidates = [
        i for i in range(len(per_example_iou)) if len(dataset[i].instances) > 0
    ]
    if not candidates:
        return []

    def rank_key(i: int) -> float:
        v = per_example_iou[i]
        return -math.inf if (v is None or math.isnan(v)) else float(v)

    ranked = sorted(candidates, key=rank_key, reverse=True)  # descending IoU

    if len(ranked) <= count:
        return ranked  # small-pool rule: take all, already descending

    good = round(0.5 * count)
    worst = min(2, max(1, round(0.2 * count)))
    median = count - good - worst

    n = len(ranked)
    good_slice = ranked[:good] if good > 0 else []
    worst_slice = ranked[n - worst :] if worst > 0 else []
    # Median band: the middle region between the good and worst slices.
    mid_lo = good
    mid_hi = n - worst
    median_pool = ranked[mid_lo:mid_hi]

    picked_good = _spread_indices(good_slice, good)
    picked_median = _spread_indices(median_pool, median)
    picked_worst = _spread_indices(worst_slice, worst)

    # Disjoint by construction (slices don't overlap). De-dup defensively and
    # back-fill from the next band if a band came up short.
    chosen: list[int] = []
    for idx in [*picked_good, *picked_median, *picked_worst]:
        if idx not in chosen:
            chosen.append(idx)
    if len(chosen) < count:
        for idx in ranked:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == count:
                break

    # Return in descending-IoU order.
    chosen.sort(key=rank_key, reverse=True)
    return chosen[:count]
