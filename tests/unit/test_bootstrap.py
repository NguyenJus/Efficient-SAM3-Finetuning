"""esam3._bootstrap imports every registrant so the registry is populated."""

from __future__ import annotations

import sys

from esam3._registry import list_registered, reset_registry


def test_bootstrap_populates_all_kinds() -> None:
    reset_registry()
    # Force a clean reimport so module-level @register decorators run again.

    for mod in (
        "esam3.data.coco",
        "esam3.data.hf",
        "esam3.data",
        "esam3.peft_adapters.lora",
        "esam3.peft_adapters.qlora",
        "esam3.peft_adapters",
        "esam3.tracking.noop",
        "esam3.tracking.tensorboard",
        "esam3.tracking.wandb",
        "esam3.tracking",
        "esam3._bootstrap",
    ):
        if mod in sys.modules:
            del sys.modules[mod]

    import esam3._bootstrap  # noqa: F401

    assert set(list_registered("dataset")) >= {"coco", "hf"}
    assert set(list_registered("peft")) >= {"lora", "qlora"}
    assert set(list_registered("tracker")) >= {"tensorboard", "wandb", "none"}
