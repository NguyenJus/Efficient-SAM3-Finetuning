"""Single device-move helper. The data collator is the ONLY caller."""

from __future__ import annotations

from typing import Any

import torch

from custom_sam_peft.runtime._runtime import Runtime


def to_device(obj: Any, runtime: Runtime) -> Any:
    """Recursively move tensors in `obj` onto `runtime.device`.

    The §9.2 static guard test enforces that this is the only place
    `.to(device)` runs outside the runtime/ module itself.
    """
    if torch.is_tensor(obj):
        return obj.to(runtime.device)
    if isinstance(obj, dict):
        return {k: to_device(v, runtime) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, runtime) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj


def require_cuda() -> None:
    """Raise EnvironmentError if CUDA is not available.

    Call this at every runtime entry point (train, eval, predict, calibrate,
    setup wizard) before attempting any GPU work.
    """
    if not torch.cuda.is_available():
        from custom_sam_peft.errors import EnvironmentError as _EnvError

        raise _EnvError(
            "CUDA GPU is required but no CUDA device was detected.",
            precondition="cuda_available",
            expected="torch.cuda.is_available() == True",
            found="torch.cuda.is_available() == False",
            fix="Run on a machine with a CUDA-capable GPU.",
        )
