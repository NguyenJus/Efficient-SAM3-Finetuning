"""Unit tests for per-component losses + total_loss in models/losses.py."""

from __future__ import annotations

import torch

from esam3.models.losses import box_loss, mask_loss, objectness_loss


def test_mask_loss_zero_on_perfect_match() -> None:
    pred = torch.full((2, 32, 32), -10.0)
    pred[:, :16, :] = 10.0
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_mask_loss_positive_when_wrong() -> None:
    pred = torch.zeros(2, 32, 32)
    target = torch.zeros(2, 32, 32)
    target[:, :16, :] = 1.0
    loss = mask_loss(pred, target)
    assert loss.item() > 0.0


def test_mask_loss_upsamples_pred_to_target_resolution() -> None:
    pred = torch.zeros(2, 16, 16)
    target = torch.zeros(2, 32, 32)
    loss = mask_loss(pred, target)
    assert torch.isfinite(loss)


def test_box_loss_zero_on_perfect_match() -> None:
    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    loss = box_loss(pred, target)
    assert loss.item() < 1e-4


def test_box_loss_positive_when_offset() -> None:
    pred = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    loss = box_loss(pred, target)
    assert loss.item() > 0.5


def test_objectness_loss_zero_when_predictions_agree() -> None:
    obj_logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(obj_logits, matched)
    assert loss.dim() == 0
    assert loss.item() < 0.05


def test_objectness_loss_high_when_predictions_invert() -> None:
    obj_logits = torch.tensor([[-10.0, 10.0, -10.0, 10.0]])
    matched = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    loss = objectness_loss(obj_logits, matched)
    assert loss.item() > 1.0
