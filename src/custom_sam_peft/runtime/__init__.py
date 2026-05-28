"""Runtime API. Single seam for device + dtype + rank-awareness."""

from custom_sam_peft.runtime._device import require_cuda, to_device
from custom_sam_peft.runtime._patches import Sam3Patches
from custom_sam_peft.runtime._runtime import Runtime

__all__ = ["Runtime", "Sam3Patches", "require_cuda", "to_device"]
