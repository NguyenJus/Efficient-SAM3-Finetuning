"""Direct tests for esam3.utils.huggingface — resolve_hf_token + download_model.

All HuggingFace Hub calls are mocked. No network access in any test.
"""

from __future__ import annotations

import pytest

from esam3.utils.huggingface import resolve_hf_token


def test_resolve_hf_token_explicit_arg_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: "cache-token",
    )
    assert resolve_hf_token("explicit") == "explicit"


def test_resolve_hf_token_env_wins_over_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: "cache-token",
    )
    assert resolve_hf_token() == "env-token"


def test_resolve_hf_token_cache_used_when_no_arg_or_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: "cache-token",
    )
    assert resolve_hf_token() == "cache-token"


def test_resolve_hf_token_returns_none_when_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.get_token",
        lambda: None,
    )
    assert resolve_hf_token() is None


# ---------------------------------------------------------------------------
# download_model — happy paths
# ---------------------------------------------------------------------------


from pathlib import Path  # noqa: E402

from esam3.utils.huggingface import download_model  # noqa: E402


def _fake_snapshot_factory(calls: list[dict[str, object]]):
    """Return a stub for snapshot_download that records its kwargs."""

    def _fake(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(kwargs.get("local_dir", ""))

    return _fake


def test_download_model_skips_when_local_dir_non_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "models"
    local_dir.mkdir()
    (local_dir / "sentinel.txt").write_text("present")

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.snapshot_download",
        _fake_snapshot_factory(calls),
    )

    out = download_model("facebook/sam3.1", local_dir)
    assert out == local_dir
    assert calls == []  # MUST NOT contact the Hub


def test_download_model_calls_snapshot_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "models"  # does not exist yet

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.snapshot_download",
        _fake_snapshot_factory(calls),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("esam3.utils.huggingface.huggingface_hub.get_token", lambda: None)

    download_model("facebook/sam3.1", local_dir)
    assert local_dir.exists()  # mkdir(parents=True, exist_ok=True)
    assert len(calls) == 1
    kw = calls[0]
    assert kw["repo_id"] == "facebook/sam3.1"
    assert kw["local_dir"] == str(local_dir)
    assert kw["revision"] is None
    assert kw["token"] is None
    # In huggingface_hub==1.15.0 there is no `local_dir_use_symlinks` kwarg;
    # real files are the default when `local_dir` is supplied. Asserting its
    # ABSENCE locks the planner verification into the test suite.
    assert "local_dir_use_symlinks" not in kw


def test_download_model_honors_force_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_dir = tmp_path / "models"
    local_dir.mkdir()
    (local_dir / "sentinel.txt").write_text("present")

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.snapshot_download",
        _fake_snapshot_factory(calls),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("esam3.utils.huggingface.huggingface_hub.get_token", lambda: None)

    download_model("facebook/sam3.1", local_dir, force=True)
    assert len(calls) == 1


def test_download_model_passes_revision_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "models"

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.snapshot_download",
        _fake_snapshot_factory(calls),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("esam3.utils.huggingface.huggingface_hub.get_token", lambda: None)

    download_model("repo", local_dir, revision="v1.0")
    assert calls[0]["revision"] == "v1.0"


def test_download_model_passes_resolved_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_dir = tmp_path / "models"

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.snapshot_download",
        _fake_snapshot_factory(calls),
    )
    monkeypatch.setenv("HF_TOKEN", "env-tok")

    download_model("repo", local_dir)
    assert calls[0]["token"] == "env-tok"


def test_download_model_logs_fetch_line_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_dir = tmp_path / "models"

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "esam3.utils.huggingface.huggingface_hub.snapshot_download",
        _fake_snapshot_factory(calls),
    )
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    import logging

    with caplog.at_level(logging.INFO, logger="esam3.utils.huggingface"):
        download_model("repo", local_dir)

    messages = [r.getMessage() for r in caplog.records]
    assert any("fetching repo" in m and str(local_dir) in m for m in messages), messages
    assert all("secret-token" not in m for m in messages)
