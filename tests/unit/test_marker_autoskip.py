"""CPU tests for conftest GPU-compat floor and tier autoskip logic."""

from __future__ import annotations

import importlib

import pytest


def _conftest():
    return importlib.import_module("tests.conftest")


@pytest.mark.parametrize(
    ("cap", "expected"),
    [((6, 0), True), ((6, 1), True), ((7, 5), True), ((8, 0), True), ((5, 0), False)],
)
def test_has_compatible_gpu_floor_is_cc60(monkeypatch, cap, expected) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: cap)
    monkeypatch.setattr(_conftest(), "_torch_can_launch_kernel", lambda: True)
    assert _conftest()._has_compatible_gpu() is expected


def test_compatible_gpu_false_when_kernel_unsupported(monkeypatch) -> None:
    """CC >= 6.0 but the installed torch build cannot launch a kernel (e.g. cu130
    on a GTX 1080: sm_61 not in the cubin set) -> not a usable GPU."""
    import torch

    conftest = _conftest()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (6, 1))
    monkeypatch.setattr(conftest, "_torch_can_launch_kernel", lambda: False)
    assert conftest._has_compatible_gpu() is False


class _FakeItem:
    def __init__(self, *keywords: str) -> None:
        self.keywords = set(keywords)
        self.markers: list[object] = []

    def add_marker(self, marker: object) -> None:
        self.markers.append(marker)


def test_gpu_t4_skipped_on_local_runner(monkeypatch) -> None:
    """A gpu_t4 test is skipped when the runner can only satisfy the local tier."""
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_current_tier", lambda: "gpu_local")
    item = _FakeItem("gpu_t4", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert item.markers, "gpu_t4 test was not skipped on the local tier"


def test_gpu_local_runs_on_local_runner(monkeypatch) -> None:
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_current_tier", lambda: "gpu_local")
    item = _FakeItem("gpu_local", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert not item.markers, "gpu_local test should not be skipped on the local runner"


def test_gpu_xl_skip_reason_names_124(monkeypatch) -> None:
    import tests.conftest as conftest

    monkeypatch.setattr(conftest, "_has_compatible_gpu", lambda: True)
    monkeypatch.setattr(conftest, "_current_tier", lambda: "gpu_local")
    item = _FakeItem("gpu_xl", "requires_compatible_gpu")
    conftest.pytest_collection_modifyitems(config=None, items=[item])  # type: ignore[arg-type]
    assert item.markers, "gpu_xl test not skipped"
    reason = getattr(item.markers[0], "kwargs", {}).get("reason", "")
    assert "#124" in reason, reason
