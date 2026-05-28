"""Shared fixtures for tests/integration/."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _pretend_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch torch.cuda.is_available to True so require_cuda() passes.

    Stub models in tests/integration/ never call real CUDA APIs, so this is safe.
    """
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
