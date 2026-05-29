"""Frozen dataclasses shared across the training subsystem.

`OomEvent` records one rung of the trainer's per-step OOM-retry ladder.
The runner accumulates these into a flat list returned in the run result;
the bundler renders the count + final state into summary.md's `## Edge cases`.

Spec: docs/superpowers/specs/2026-05-22-algo-vram-preset-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OomEvent:
    """One step where the trainer caught OOM and adapted before retrying.

    `action` records the adaptive rung:
      - "microbatch_halved": `state.micro_batch_size //= 2`, retry same step.
      - "multiplex_halved": inner B-ladder exhausted at micro_batch=1; the
        trainer zero_grad'd, halved `effective_K`, re-chunked ALL classes into
        more/smaller groups, and replayed the whole step. No class is dropped.
        Carries the new `effective_K`. Spec §4.

    The fields capture *post*-adaptation state so downstream rendering can
    reconstruct the run's safety-net history without re-traversing mutable state.
    """

    step: int
    action: Literal["microbatch_halved", "multiplex_halved"]
    new_micro_batch_size: int
    effective_K: int | None = None  # set only for "multiplex_halved" events
