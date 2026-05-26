"""Interactive `csp init --interactive` wizard.

Declarative WizardStep registry → answers dict → render config_full.yaml →
validate via load_config → emit. See
docs/superpowers/specs/2026-05-26-interactive-setup-wizard-design.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import typer

from custom_sam_peft.config.schema import ClassImbalance

IMBALANCE_MODERATE_RATIO = 3.0  # R < 3 → balanced
IMBALANCE_SEVERE_RATIO = 10.0  # 3 <= R < 10 → moderate; R >= 10 → severe

RunMode = Literal["train", "run", "eval"]


@dataclass
class Ctx:
    answers: dict[str, Any]
    cuda_available: bool
    run_mode: RunMode = "train"
    categories: list[str] | None = None
    category_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class WizardStep:
    id: str
    ask: Callable[[Ctx], dict[str, Any]]
    when: Callable[[Ctx], bool] = field(default=lambda ctx: True)


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively merge src into dst. Nested dicts merge; scalars/lists overwrite."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def ask_text(
    prompt: str,
    *,
    default: str | None = None,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Free-text prompt; re-asks on validate failure. validate returns an error string or None."""
    while True:
        value = (
            typer.prompt(prompt, default=default) if default is not None else typer.prompt(prompt)
        )
        value = str(value).strip()
        if validate is not None:
            err = validate(value)
            if err is not None:
                typer.echo(err)
                continue
        return value


def ask_choice(prompt: str, choices: list[str], *, default: str | None = None) -> str:
    """Membership-checked choice; re-asks on invalid."""
    rendered = f"{prompt} [{'/'.join(choices)}]"
    while True:
        value = (
            typer.prompt(rendered, default=default)
            if default is not None
            else typer.prompt(rendered)
        )
        value = str(value).strip()
        if value in choices:
            return value
        typer.echo(f"choose one of: {', '.join(choices)}")


def ask_confirm(prompt: str, *, default: bool = True) -> bool:
    return typer.confirm(prompt, default=default)


def infer_class_imbalance(annotations: str) -> ClassImbalance:
    """Detect a class-imbalance tier from per-category instance counts.

    Mirrors data/subset.py per-class frequency; uses the pycocotools-backed
    primitives in data/coco.py. On ANY failure (missing/unreadable file, zero
    present categories) returns "balanced".
    """
    try:
        from custom_sam_peft.data.coco import _build_category_remap, _load_coco_index

        coco = _load_coco_index(annotations)
        _sparse_ids, remap, _names = _build_category_remap(coco)
        counts: dict[int, int] = {}
        for img_id in coco.getImgIds():
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            for a in anns:
                if int(a.get("iscrowd", 0)) != 0:
                    continue
                dense = remap.get(int(a["category_id"]))
                if dense is None:
                    continue
                counts[dense] = counts.get(dense, 0) + 1
        present = [c for c in counts.values() if c > 0]
        if not present:
            raise ValueError("no present categories")
        ratio = max(present) / min(present)
    except Exception:
        return "balanced"

    if ratio < IMBALANCE_MODERATE_RATIO:
        return "balanced"
    if ratio < IMBALANCE_SEVERE_RATIO:
        return "moderate"
    return "severe"
