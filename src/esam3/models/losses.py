"""SAM 3.1 training losses (per-class, open-vocab head)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, interpolate

from esam3.config.schema import LossConfig  # noqa: F401
from esam3.data.base import Instance  # noqa: F401
from esam3.models.matching import (  # noqa: F401
    CanonicalOutputs,
    HungarianMatcher,
    meta_to_canonical,
)


def _dice_loss(pred_logits: Tensor, target: Tensor) -> Tensor:
    p = pred_logits.sigmoid().flatten(1)
    t = target.flatten(1).float()
    num = 2 * (p * t).sum(-1) + 1.0
    den = p.sum(-1) + t.sum(-1) + 1.0
    return (1.0 - num / den).mean()


def mask_loss(pred: Tensor, target: Tensor) -> Tensor:
    """0.5 · Dice + 0.5 · BCE on matched mask pairs.

    `pred` and `target` are (N, H_p, W_p) and (N, H_t, W_t). If the spatial
    shapes differ, `pred` is bilinear-upsampled to the target resolution.
    """
    if pred.shape[-2:] != target.shape[-2:]:
        pred = interpolate(
            pred[:, None], size=target.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
    bce = binary_cross_entropy_with_logits(pred, target.float())
    dice = _dice_loss(pred, target)
    return 0.5 * dice + 0.5 * bce


def _box_cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _giou_pairwise(b1: Tensor, b2: Tensor) -> Tensor:
    """Element-wise GIoU between two (N, 4) tensors in xyxy."""
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, :2], b2[:, :2])
    rb = torch.min(b1[:, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(b1[:, :2], b2[:, :2])
    rb_c = torch.max(b1[:, 2:], b2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def box_loss(pred: Tensor, target: Tensor) -> Tensor:
    """smoothL1 + (1 - GIoU) on matched box pairs. Boxes are normalized cxcywh."""
    smooth_l1 = torch.nn.functional.smooth_l1_loss(pred, target, reduction="mean")
    giou = _giou_pairwise(_box_cxcywh_to_xyxy(pred), _box_cxcywh_to_xyxy(target))
    return smooth_l1 + (1.0 - giou).mean()
