"""SAM 3.1 loader + forward wrapper. See docs/superpowers/specs/2026-05-16-model-loading-design.md.

Revised by docs/superpowers/plans/2026-05-16-model-loading-revised.md to match
Meta's open-vocab head: one prompt class per forward call. Trainer loops over
the fixed class vocabulary externally.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from esam3.config.schema import ModelConfig
from esam3.data.base import Prompts, TextPrompts


class Sam3Wrapper(nn.Module):
    """Thin wrapper around Meta's SAM 3.1 model.

    Contract:
      - `forward(images, prompts)` accepts a batch of B images and a list of
        B `Prompts` objects, one per image.
      - All prompts in a batch MUST be the same variant (TextPrompts XOR
        BoxPrompts); the wrapper raises on mixed batches.
      - For TextPrompts, each image's prompt MUST contain exactly one class
        name; the trainer is responsible for looping over the fixed class
        vocabulary and accumulating losses across classes.
      - Returns Meta's native output dict unchanged.
    """

    def __init__(self, model: nn.Module, image_size: int = 1008, mask_size: int = 288) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.mask_size = mask_size

    def forward(self, images: Tensor, prompts: list[Prompts]) -> dict[str, Any]:
        self._validate_prompts(images, prompts)
        return self.model(images, prompts)

    @staticmethod
    def _validate_prompts(images: Tensor, prompts: list[Prompts]) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be (B, 3, H, W); got shape {tuple(images.shape)}")
        b = images.shape[0]
        if len(prompts) != b:
            raise ValueError(
                f"len(prompts)={len(prompts)} must equal batch size {b}"
            )
        if not prompts:
            return
        first = type(prompts[0])
        for p in prompts:
            if type(p) is not first:
                raise ValueError(
                    "All prompts in a batch must be the same prompt variant "
                    "(TextPrompts or BoxPrompts), not mixed."
                )
            if isinstance(p, TextPrompts) and len(p.classes) != 1:
                raise ValueError(
                    f"TextPrompts must contain exactly one class per forward "
                    f"call (got {len(p.classes)}). Loop over the class vocabulary "
                    f"externally."
                )


def load_sam31(cfg: ModelConfig) -> Sam3Wrapper:
    """Load SAM 3.1 via Meta's sam3 package. Implementation lands in Task 17."""
    raise NotImplementedError("filled in by Task 17 of spec/model-loading-revised")
