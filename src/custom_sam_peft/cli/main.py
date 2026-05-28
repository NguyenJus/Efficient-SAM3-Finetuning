"""`custom-sam-peft` CLI entry point — wires subcommands into a Typer app."""

from __future__ import annotations

import sys

import click
import typer

from custom_sam_peft._bootstrap import bootstrap

bootstrap()  # populate plugin registry + configure logging before subcommand imports

from custom_sam_peft.cli import (  # noqa: E402
    calibrate_cmd,
    doctor_cmd,
    eval_cmd,
    export_cmd,
    init_cmd,
    predict_cmd,
    run_cmd,
    train_cmd,
)
from custom_sam_peft.cli._progress import _silence_third_party_progress  # noqa: E402
from custom_sam_peft.errors import CustomSamPeftError  # noqa: E402

# Suppress HF / datasets progress bars once at app entry, unconditionally.
# progress_session also calls this defensively on entry — the double-call is safe.
_silence_third_party_progress()

# ---------------------------------------------------------------------------
# Flag-value override for --resume
#
# typer.Option does not support "flag_value" (option that can be used with or
# without a value) in a way that survives the typer → click translation.
# We work around this by injecting raw click commands for `train` and `run`
# via a custom TyperGroup subclass.  The click commands use click's native
# is_flag=False / flag_value mechanism and delegate back to the underlying
# cmd function, which already expects `resume: str | None` and the sentinel.
# ---------------------------------------------------------------------------

_LATEST_SENTINEL = train_cmd._LATEST_SENTINEL  # shared sentinel value


def _build_train_click_cmd() -> click.Command:
    """Return a raw click command for `train` with flag-value --resume support."""

    @click.command("train", help="Run a finetune.")
    @click.option(
        "--config", required=True, type=click.Path(), help="Path to training config YAML."
    )
    @click.option(
        "--override",
        multiple=True,
        default=(),
        help="Override config keys: dotted.key=value.",
    )
    @click.option(
        "--resume",
        is_flag=False,
        flag_value=_LATEST_SENTINEL,
        default=None,
        type=str,
        help=(
            "Resume checkpoint. Pass a path, or omit value for the latest "
            "checkpoint matching cfg.run.name."
        ),
    )
    @click.option(
        "--eval/--no-eval",
        "do_eval",
        default=False,
        help="After training, run evaluation.",
    )
    @click.option(
        "--export/--no-export",
        "do_export",
        default=False,
        help="After training, export a run bundle.",
    )
    @click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
    @click.option(
        "--progress",
        "progress_flag",
        default="auto",
        metavar="MODE",
        help="Progress display mode: auto|on|off|plain.",
    )
    def _train_cmd(config, override, resume, do_eval, do_export, verbose, progress_flag):  # type: ignore[misc]
        from pathlib import Path

        train_cmd.train(  # type: ignore[call-arg]
            config=Path(config),
            override=list(override),
            resume=resume,
            do_eval=do_eval,
            do_export=do_export,
            verbose=verbose,
            progress_flag=progress_flag,
        )

    return _train_cmd


def _build_run_click_cmd() -> click.Command:
    """Return a raw click command for `run` with flag-value --resume support."""

    @click.command(
        "run",
        help="Train + eval + (optional) export + bundle. Alias for train --eval --export.",
    )
    @click.option("--config", required=True, type=click.Path(), help="Path to config YAML.")
    @click.option(
        "--resume",
        is_flag=False,
        flag_value=_LATEST_SENTINEL,
        default=None,
        type=str,
        help=(
            "Resume checkpoint. Pass a path, or omit value for the latest "
            "checkpoint matching cfg.run.name."
        ),
    )
    @click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
    @click.option(
        "--progress",
        "progress_flag",
        default="auto",
        metavar="MODE",
        help="Progress display mode: auto|on|off|plain.",
    )
    def _run_cmd(config, resume, verbose, progress_flag):  # type: ignore[misc]
        from pathlib import Path

        run_cmd.run(  # type: ignore[call-arg]
            config=Path(config),
            resume=resume,
            verbose=verbose,
            progress_flag=progress_flag,
        )

    return _run_cmd


_CLICK_OVERRIDES: dict[str, click.Command] = {
    "train": _build_train_click_cmd(),
    "run": _build_run_click_cmd(),
}


class _ResumeAwareGroup(typer.core.TyperGroup):
    """TyperGroup that replaces 'train'/'run' with click commands supporting flag-value --resume."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        base = set(super().list_commands(ctx))
        return sorted(base | set(_CLICK_OVERRIDES.keys()))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in _CLICK_OVERRIDES:
            return _CLICK_OVERRIDES[cmd_name]
        return super().get_command(ctx, cmd_name)


app = typer.Typer(
    name="custom-sam-peft",
    help="Closed-vocab finetuning of SAM-family models with LoRA / QLoRA.",
    no_args_is_help=True,
    add_completion=False,
    cls=_ResumeAwareGroup,
)

# train and run are registered via _CLICK_OVERRIDES above; the typer registrations
# below ensure help, completion, and introspection still see them.
app.command("train", help="Run a finetune.")(train_cmd.train)
app.command("eval", help="Evaluate a checkpoint.")(eval_cmd.evaluate)
app.command("predict", help="Run inference on images with optional adapter.")(predict_cmd.predict)
app.command("export", help="Export adapter or merged model.")(export_cmd.export)
app.command("init", help="Write a starter config.")(init_cmd.init)
app.command("doctor", help="Report environment + dependency status.")(doctor_cmd.doctor)
app.command("calibrate", help="Probe peak VRAM and cache for tighter preset packing.")(
    calibrate_cmd.calibrate
)
app.command(
    "run", help="Train + eval + (optional) export + bundle. Alias for train --eval --export."
)(run_cmd.run)

# Module-level flag: set to True by main() when -v / --verbose appears in sys.argv
# so that the CustomSamPeftError handler can decide whether to render or re-raise.
_verbose: bool = False


def _render_error(e: CustomSamPeftError) -> str:
    """Format a CustomSamPeftError into the four-part user-facing message."""
    parts = [str(e)]
    if e.expected:
        parts.append(f"Expected: {e.expected}")
    if e.found:
        parts.append(f"Found: {e.found}")
    if e.fix:
        parts.append(f"Fix: {e.fix}")
    parts.append("Rerun with -v for full traceback.")
    return "\n".join(parts)


def main() -> None:
    """Entry point that wraps app() with CustomSamPeftError handling."""
    global _verbose
    _verbose = "-v" in sys.argv or "--verbose" in sys.argv
    try:
        app()
    except CustomSamPeftError as e:
        if _verbose:
            raise
        typer.secho(_render_error(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
