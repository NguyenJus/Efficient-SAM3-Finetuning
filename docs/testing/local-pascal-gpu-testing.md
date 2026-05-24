# Local Pascal (GTX 1080) GPU Testing

The dev box holds a GTX 1080 (compute capability 6.1 / sm_61, 8 GB VRAM, ~7 GB
effective after WSL/Xwayland overhead). It is a **GPU-test target**, not a
training/inference platform: it exercises real GPU code paths (sm_61 kernels,
bitsandbytes 4-bit, the gradient-checkpointing fix, float16 dtype handling).

## Provision

The default `uv sync` installs cu130 torch (no sm_61 cubin). To reach the 1080:

    uv sync --extra gpu-pascal   # cu118 torch (sm_60..sm_90 + PTX) + bitsandbytes

This extra is isolated via a uv explicit index + extra-scoped source routing, so
the bare `uv sync` and `uv sync --extra dev` paths are unchanged (still cu130).

## Run the gpu_local tier

    bash scripts/run_gpu_tests.sh local

Or directly: `uv run pytest -m gpu_local tests/gpu/ tests/integration/ tests/predict/`.

## float16 caveat (Pascal has no fast bf16)

bf16 is **emulated** below compute capability 8.0, so the 1080 trains/runs in
**float16**. A `bfloat16` request is coerced to `float16` with a one-time
warning (see `coerce_dtype_for_capability`). This means numerics validated on
the 1080 do NOT certify the bf16 T4 release path — that confirmation is a
follow-up (gpu_t4 tier).

## Milestone evidence

<!-- Filled in by the A-2 hard-gate task: the sm_61-kernel + bnb-Linear4bit proofs. -->
