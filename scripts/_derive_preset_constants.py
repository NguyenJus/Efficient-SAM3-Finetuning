"""Re-derive presets.py seed constants from a measured probe on the local GPU.

Maintainer-only. Run on the 16 GB card:

    uv run python scripts/_derive_preset_constants.py --r 16 --k 16 --batch 1

Prints values to paste into presets.py (MODEL_PARAMS, BASE_ACTIVATION_AT_1024,
and the per-K_eff activation coefficient). Validates against the new
config-aware probe (Component 2). Not imported by the package or the test suite.

Spec: design §3.3.
"""

from __future__ import annotations

import argparse

import torch

from custom_sam_peft.config.schema import ModelConfig, PEFTConfig
from custom_sam_peft.data.base import TextPrompts
from custom_sam_peft.presets import (
    WORKSPACE_BYTES,
    _adapter_bytes,
    _attention_bytes_per_example,
    _model_bytes,
    _optimizer_bytes,
)

_GB = 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--method", choices=["lora", "qlora"], default="lora")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    from custom_sam_peft.models.sam3 import MULTIPLEX_CAP, SAM3_IMAGE_SIZE, load_sam31
    from custom_sam_peft.peft_adapters.lora import apply_lora

    image_size = SAM3_IMAGE_SIZE
    k_eff = min(args.k, MULTIPLEX_CAP)

    wrapper = load_sam31(ModelConfig(), channels=3, channel_semantics="rgb")
    apply_lora(wrapper, PEFTConfig(method=args.method, r=args.r))
    device = next(wrapper.parameters()).device
    images = torch.zeros(args.batch, 3, image_size, image_size, dtype=torch.bfloat16, device=device)
    prompts = [TextPrompts(classes=[f"class_{j}" for j in range(k_eff)]) for _ in range(args.batch)]

    torch.cuda.reset_peak_memory_stats()
    out = wrapper(images, prompts, support=None)
    loss = torch.zeros((), device=device, dtype=torch.float32)
    for t in out.values():
        if isinstance(t, torch.Tensor):
            loss = loss + t.float().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    peak = int(torch.cuda.max_memory_allocated())

    overhead = (
        _model_bytes(args.method)
        + _adapter_bytes(args.r)
        + _optimizer_bytes(args.r)
        + WORKSPACE_BYTES
        + _attention_bytes_per_example(image_size) * args.batch
    )
    activation_total = max(0, peak - overhead)
    per_k_per_example = activation_total / max(1, args.batch * k_eff)

    print(f"measured peak:            {peak / _GB:.2f} GiB")  # noqa: T201
    print(f"modeled overhead:         {overhead / _GB:.2f} GiB")  # noqa: T201
    print(f"residual activation:      {activation_total / _GB:.2f} GiB")  # noqa: T201
    print(  # noqa: T201
        f"-> per-K_eff activation:  {int(per_k_per_example)} bytes "
        f"({per_k_per_example / _GB:.3f} GiB)"
    )
    print(  # noqa: T201
        f"-> BASE_ACTIVATION_AT_1024 candidate (scale to 1024px): "
        f"{int(per_k_per_example * (1024 / image_size) ** 2)} bytes"
    )


if __name__ == "__main__":
    main()
