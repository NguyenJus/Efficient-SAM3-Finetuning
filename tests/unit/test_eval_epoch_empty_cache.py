"""Test that _eval_epoch calls torch.cuda.empty_cache() before evaluating.

Patches torch.cuda.is_available -> True and torch.cuda.empty_cache to a Mock,
then stubs the Evaluator so no real forward runs.  CPU-only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from custom_sam_peft.eval.metrics import MetricsReport


def _make_trainer() -> object:
    """Build a minimal Trainer via its public __init__ with all-Mock dependencies."""
    from custom_sam_peft.train.trainer import Trainer

    model = MagicMock(spec=torch.nn.Module)
    model.parameters.return_value = iter([])  # no CUDA params; runtime -> cpu

    val_ds = MagicMock()
    val_ds.__len__ = lambda self: 2
    val_ds.class_names = ["cat"]

    cfg = MagicMock()
    cfg.peft.method = "lora"
    cfg.train.optimizer = "adamw"
    cfg.model.dtype = "float32"
    cfg.eval.batch_size = 1
    cfg.eval.model_copy.return_value = cfg.eval

    tracker = MagicMock()

    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.val_ds = val_ds
    trainer.cfg = cfg
    trainer.tracker = tracker
    # _eval_epoch calls _maybe_save_best, which reads these; an empty `overall`
    # report makes it return early (metric is None) so no save is attempted.
    trainer._best_metric_key = "mAP"
    trainer._best_metric_value = float("-inf")
    return trainer


def test_eval_epoch_calls_empty_cache_when_cuda_available(tmp_path) -> None:
    """_eval_epoch must call torch.cuda.empty_cache() before running the Evaluator."""
    trainer = _make_trainer()

    stub_report = MetricsReport(overall={}, per_class={}, n_images=0, n_predictions=0)
    stub_evaluator = MagicMock()
    stub_evaluator.evaluate.return_value = stub_report

    with (
        patch("custom_sam_peft.train.trainer.torch.cuda.is_available", return_value=True),
        patch("custom_sam_peft.train.trainer.torch.cuda.empty_cache") as mock_empty_cache,
        patch("custom_sam_peft.train.trainer.Evaluator", return_value=stub_evaluator),
    ):
        trainer._eval_epoch(step=1, run_dir=tmp_path, oom_state=None)  # type: ignore[attr-defined]

    mock_empty_cache.assert_called_once()
