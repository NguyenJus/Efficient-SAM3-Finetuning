"""Adapter + Hungarian matcher for SAM 3.1 training.

`meta_to_canonical` is the SINGLE point in the codebase that knows Meta's
native output dict key names. If Meta renames a field, only this function
breaks. Filled in by Task 5 once the actual key names are inspected against
a real `Sam3Wrapper` forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CanonicalOutputs:
    """Output of `meta_to_canonical`. Used by the matcher and losses.

    Shapes:
      class_logits: (B, Q, C+1)   # last index = "no-object"
      pred_boxes:   (B, Q, 4)     # normalized cx,cy,w,h in [0, 1]
      pred_masks:   (B, Q, 288, 288)
      presence:     (B, Q)        # objectness logit
    """

    class_logits: Tensor
    pred_boxes: Tensor
    pred_masks: Tensor
    presence: Tensor


def meta_to_canonical(outputs: dict) -> CanonicalOutputs:
    """Convert Meta sam3's native output dict to CanonicalOutputs.

    Implementation deferred to Task 5 (requires inspection of real Meta output).
    """
    raise NotImplementedError("filled in by Task 5 of spec/model-loading")


from scipy.optimize import linear_sum_assignment  # noqa: E402
from torch.nn.functional import interpolate  # noqa: E402

from esam3.data.base import Instance  # noqa: E402


def _box_cxcywh_to_xyxy(box: Tensor) -> Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1, y1 = cx - 0.5 * w, cy - 0.5 * h
    x2, y2 = cx + 0.5 * w, cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU between every pair in boxes1 (N,4) and boxes2 (M,4), xyxy."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def _dice_cost(pred_masks: Tensor, tgt_masks: Tensor) -> Tensor:
    """Dice cost between every pred (Q, H, W) and target (N, H, W) mask. Returns (Q, N)."""
    p = pred_masks.sigmoid().flatten(1)  # (Q, H*W)
    t = tgt_masks.flatten(1).float()      # (N, H*W)
    num = 2 * p @ t.t()
    den = p.sum(-1)[:, None] + t.sum(-1)[None, :]
    return 1.0 - (num + 1.0) / (den + 1.0)


class HungarianMatcher:
    """DETR-style bipartite matcher. Non-differentiable; called under no_grad."""

    def __init__(
        self,
        lambda_cls: float,
        lambda_l1: float,
        lambda_giou: float,
        lambda_mask: float,
    ) -> None:
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_mask = lambda_mask

    @torch.no_grad()
    def __call__(
        self,
        outputs: CanonicalOutputs,
        targets: list[list[Instance]],
    ) -> list[tuple[Tensor, Tensor]]:
        b, q, _ = outputs.class_logits.shape
        mask_h, mask_w = outputs.pred_masks.shape[-2:]
        results: list[tuple[Tensor, Tensor]] = []
        for i in range(b):
            tgts = targets[i]
            if len(tgts) == 0:
                results.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                ))
                continue
            probs = outputs.class_logits[i].softmax(-1)  # (Q, C+1)
            tgt_class = torch.tensor(
                [t.class_id for t in tgts], dtype=torch.long, device=probs.device
            )
            cost_cls = -probs[:, tgt_class]  # (Q, N)

            tgt_boxes = torch.stack([t.box for t in tgts]).to(outputs.pred_boxes.device)
            cost_l1 = torch.cdist(outputs.pred_boxes[i], tgt_boxes, p=1)  # (Q, N)
            cost_giou = -_giou(
                _box_cxcywh_to_xyxy(outputs.pred_boxes[i]),
                _box_cxcywh_to_xyxy(tgt_boxes),
            )

            tgt_masks = torch.stack([t.mask for t in tgts]).to(outputs.pred_masks.device)
            tgt_masks_low = interpolate(
                tgt_masks[None].float(),
                size=(mask_h, mask_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            cost_mask = _dice_cost(outputs.pred_masks[i], tgt_masks_low)

            cost = (
                self.lambda_cls * cost_cls
                + self.lambda_l1 * cost_l1
                + self.lambda_giou * cost_giou
                + self.lambda_mask * cost_mask
            )
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            results.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))
        return results
