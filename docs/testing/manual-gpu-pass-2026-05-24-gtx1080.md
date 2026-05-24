# Manual GPU Test Pass — 2026-05-24 (GTX 1080, §4.3 hard-gate milestone)

Operational tracker for the §4.3 Pascal hard-gate milestone and the local
`gpu_local` tier. Companion to
[`local-pascal-gpu-testing.md`](local-pascal-gpu-testing.md) and
[`gpu-test-policy.md`](gpu-test-policy.md).

Hardware: **NVIDIA GeForce GTX 1080**, compute capability 6.1 (sm_61), 8 GB VRAM,
driver 582.28, WSL2.

## How to run

Provision the Pascal env first (cu118 wheel + bitsandbytes):

    uv sync --extra gpu-pascal

Then run the gpu_local test tier:

    bash scripts/run_gpu_tests.sh local

Or directly:

    uv run --extra gpu-pascal pytest -m gpu_local tests/gpu/ tests/integration/ tests/predict/

Single proofs (as used in this milestone):

    uv run --extra gpu-pascal python <proof-script.py>

Restore the dev env when done:

    uv sync --extra dev

## Test checklist

### §4.3 milestone proofs (2026-05-24)

- [x] **Step 1** — sm_61 CUDA matmul via PTX JIT (cu118 torch 2.7.1)
- [x] **Step 2** — bnb `Linear4bit` NF4 forward on sm_61, float16

### gpu_local tier (populated by later tasks)

- [ ] *(populated by C-2 / Phase-0 trace tasks)*

## Session log

### §4.3 milestone — 2026-05-24 — PASS

#### Context

Task A-1 landed (commit eb523ab) adding an opt-in `gpu-pascal` uv extra that
resolves `torch 2.7.1+cu118` + `bitsandbytes 0.49.2`, isolated from the default
`2.12.0+cu130`. Resolution facts:

| env | torch | bnb |
|---|---|---|
| bare `uv sync` / `uv sync --extra dev` | 2.12.0+cu130 (NO sm_61 cubin) | — |
| `uv sync --extra gpu-pascal` | **2.7.1+cu118** (sm_60..sm_90 + PTX) | **0.49.2** |

The cu130 default torch ships no sm_61 cubin; cu118 covers sm_61 via PTX JIT
from `compute_60`.

#### Step 1 — sm_61 CUDA matmul kernel (cu118 PTX JIT)

Command:

```bash
uv run --extra gpu-pascal python - <<'PY'
import torch
print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = a @ b                      # forces a CUDA matmul kernel launch -> PTX JIT compute_60 -> sm_61
torch.cuda.synchronize()
ref = (a.cpu() @ b.cpu())
err = (c.cpu() - ref).abs().max().item()
print("matmul max abs err", err)
assert err < 1e-2, "sm_61 matmul produced wrong results"
print("SM_61 KERNEL OK")
PY
```

Verbatim output:

```
torch 2.7.1+cu118
device NVIDIA GeForce GTX 1080 cc (6, 1)
matmul max abs err 0.00011444091796875
SM_61 KERNEL OK
```

Result: **PASS** — no `no kernel image is available` error; PTX JIT compiled
`compute_60` → sm_61 successfully; matmul error 1.14e-4 < 1e-2.

#### Step 2 — bnb Linear4bit NF4 forward on sm_61, float16

Command:

```bash
uv run --extra gpu-pascal python - <<'PY'
import torch, bitsandbytes as bnb
print("bnb", bnb.__version__)
lin = bnb.nn.Linear4bit(256, 128, bias=False, quant_type="nf4", compute_dtype=torch.float16)
lin = lin.to("cuda")           # quantization fires on .to(cuda)
x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
y = lin(x)                     # NF4 4-bit kernel forward on sm_61
torch.cuda.synchronize()
print("out", tuple(y.shape), y.dtype, "finite", bool(torch.isfinite(y).all()))
assert y.shape == (4, 128) and torch.isfinite(y).all()
print("BNB LINEAR4BIT OK")
PY
```

Verbatim output:

```
bnb 0.49.2
out (4, 128) torch.float16 finite True
BNB LINEAR4BIT OK
```

Result: **PASS** — NF4 4-bit kernel launched on sm_61; output shape (4, 128),
dtype float16, all values finite.

#### Gate decision

Both §4.3 proofs passed. The Pascal track is **unblocked**. Downstream tasks
B/C and D's `gpu_local` calibration may proceed.

---

### Phase-0 trace + fix classification

<!-- filled by C-2 -->

*(Populated when task C-2 runs the full gpu_local tier with pytest verbose output
and classifies any failures: genuine sm_61 issues vs. dtype/env issues vs.
test-fixture gaps.)*

---

### Phase-3 calibration numbers

<!-- filled by C-4 -->

*(Populated when task C-4 records peak VRAM, throughput, and loss curves from
the gpu_local training smoke tests on the GTX 1080.)*
