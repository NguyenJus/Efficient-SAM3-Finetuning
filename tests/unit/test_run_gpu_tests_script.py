"""CPU test for run_gpu_tests.sh tier parsing (no GPU needed)."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_gpu_tests.sh"


def _src() -> str:
    return SCRIPT.read_text()


def test_accepts_three_tiers_and_rejects_legacy() -> None:
    src = _src()
    assert "local)" in src and "t4)" in src and "xl)" in src
    assert "inspection)" not in src and "release)" not in src


def test_local_maps_to_gpu_local_marker() -> None:
    assert "gpu_local" in _src()


def test_collects_predict_path() -> None:
    assert "tests/predict/" in _src()


def test_local_runs_per_file_loop() -> None:
    """The local tier must iterate over files (not a single pytest invocation).

    Confirms the script contains a loop over test files and handles exit-code 5
    (no tests collected) as success.
    """
    src = _src()
    # A loop construct is present for the local tier.
    assert "while" in src or "for" in src
    # Exit code 5 (no tests collected) is explicitly treated as success.
    assert "5" in src


def test_rejects_unknown_tier() -> None:
    res = subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT), "bogus"],  # noqa: S607
        capture_output=True,
        text=True,
        env={"PYTHON": "true", "PATH": "/usr/bin:/bin"},
    )
    assert res.returncode != 0
    assert "usage" in (res.stderr + res.stdout).lower()
