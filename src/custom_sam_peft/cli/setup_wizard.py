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
