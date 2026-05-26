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


# ---------------------------------------------------------------------------
# Step ask-functions
# ---------------------------------------------------------------------------


def _ask_run_mode(ctx: Ctx) -> dict[str, Any]:
    ctx.run_mode = ask_choice("Run mode?", ["train", "run", "eval"], default="train")  # type: ignore[assignment]
    return {}


def _ask_run_name(ctx: Ctx) -> dict[str, Any]:
    name = ask_text("Run name?", default="my-run")
    return {"run": {"name": name}}


def _ask_dataset_source(ctx: Ctx) -> dict[str, Any]:
    fmt = ask_choice("Dataset format?", ["coco", "hf"], default="coco")
    if fmt == "coco":
        ann = ask_text("Path to COCO train annotations (.json)?")
        imgs = ask_text("Path to COCO train images dir?")
        return {"data": {"format": "coco", "train": {"annotations": ann, "images": imgs}}}
    name = ask_text("HuggingFace dataset name (org/dataset)?")
    return {"data": {"format": "hf", "hf": {"name": name}}}


def _ask_validation(ctx: Ctx) -> dict[str, Any]:
    fmt = ctx.answers.get("data", {}).get("format", "coco")
    mode = ask_choice("Validation?", ["explicit", "auto-split", "none"], default="auto-split")
    if mode == "none":
        if ctx.run_mode in {"eval", "run"}:
            typer.echo(
                "note: eval/run needs a validation set to score against; "
                "selecting none means eval will have nothing to evaluate."
            )
        return {}
    if mode == "auto-split":
        frac = ask_text("Auto-split fraction (0<f<=0.5)?", default="0.1")
        return {"data": {"val_split": {"fraction": float(frac)}}}
    if fmt == "hf":
        split = ask_text("HF validation split name?", default="validation")
        return {"data": {"hf": {"split_val": split}}}
    ann = ask_text("Path to COCO val annotations (.json)?")
    imgs = ask_text("Path to COCO val images dir?")
    return {"data": {"val": {"annotations": ann, "images": imgs}}}


def _ask_domain(ctx: Ctx) -> dict[str, Any]:
    domain = ask_choice(
        "Domain?",
        ["natural", "medical", "satellite", "microscopy", "none"],
        default="natural",
    )
    intensity = ask_choice(
        "Augmentation intensity?", ["safe", "medium", "aggressive"], default="medium"
    )
    return {
        "data": {"augmentations": {"preset": domain, "intensity": intensity}},
        "train": {"loss": {"preset": domain}},
    }


def _coco_train_annotations(ctx: Ctx) -> str | None:
    data = ctx.answers.get("data", {})
    if data.get("format") != "coco":
        return None
    ann = data.get("train", {}).get("annotations")
    return str(ann) if ann is not None else None


def _ask_class_imbalance(ctx: Ctx) -> dict[str, Any]:
    ann = _coco_train_annotations(ctx)
    if ann is None:
        typer.echo(
            "could not auto-detect class imbalance (non-COCO/no annotations); "
            "defaulting to balanced"
        )
        tier: ClassImbalance = "balanced"
    else:
        tier = infer_class_imbalance(ann)
        typer.echo(f"detected class imbalance: {tier}")
    return {"train": {"loss": {"class_imbalance": tier}}}


def _ask_peft_sizing(ctx: Ctx) -> dict[str, Any]:
    from custom_sam_peft.presets import decide_preset

    if ctx.cuda_available and ask_confirm(
        "Auto-size the PEFT config to your GPU's VRAM?", default=True
    ):
        image_size = ctx.answers.get("data", {}).get("image_size", 1008)
        try:
            decision = decide_preset(image_size)
        except RuntimeError as exc:
            typer.echo(f"could not auto-size: {exc}; falling back to manual")
        else:
            typer.echo(decision.label())
            return decision.config_patch
    method = ask_choice("PEFT method?", ["lora", "qlora"], default="lora")
    return {"peft": {"method": method}}


def _ask_epochs(ctx: Ctx) -> dict[str, Any]:
    def _positive_int(s: str) -> str | None:
        try:
            return None if int(s) > 0 else "epochs must be a positive integer"
        except ValueError:
            return "epochs must be a positive integer"

    epochs = ask_text("Number of epochs?", default="10", validate=_positive_int)
    return {"train": {"epochs": int(epochs)}}


def _ask_model_weights(ctx: Ctx) -> dict[str, Any]:
    from pathlib import Path

    def _is_file_or_blank(s: str) -> str | None:
        if s == "":
            return None
        return None if Path(s).is_file() else f"no file at {s}"

    raw = ask_text(
        "Path to an existing SAM 3.1 checkpoint (.pt)? Leave blank to use "
        "`models/sam3.1` and download if missing.",
        default="",
        validate=_is_file_or_blank,
    )
    if raw:
        p = Path(raw)
        return {"model": {"local_dir": str(p.parent), "checkpoint_file": p.name}}
    hits = sorted(Path("models").glob("**/sam3.1_multiplex.pt")) if Path("models").is_dir() else []
    if hits:
        return {"model": {"local_dir": str(hits[0].parent)}}
    return {}


STEPS: list[WizardStep] = [
    WizardStep("run_mode", _ask_run_mode),
    WizardStep("run_name", _ask_run_name),
    WizardStep("dataset_source", _ask_dataset_source),
    WizardStep("validation", _ask_validation),
    WizardStep("domain", _ask_domain),
    WizardStep(
        "class_imbalance",
        _ask_class_imbalance,
        when=lambda ctx: ctx.run_mode in {"train", "run"},
    ),
    WizardStep("peft_sizing", _ask_peft_sizing),
    WizardStep("epochs", _ask_epochs, when=lambda ctx: ctx.run_mode != "eval"),
    WizardStep("model_weights", _ask_model_weights),
]


def run_wizard(ctx: Ctx) -> dict[str, Any]:
    for step in STEPS:
        if step.when(ctx):
            fragment = step.ask(ctx)
            _deep_merge(ctx.answers, fragment)
    return ctx.answers
