"""HuggingFace Hub helpers for token resolution and model download.

This module is a thin wrapper around ``huggingface_hub``: it never calls
``login()`` (no token persistence), and it never logs the resolved token.

Verified against ``huggingface_hub==1.15.0``: real-file materialization is
the default when ``local_dir=`` is supplied to ``snapshot_download``; the
older ``local_dir_use_symlinks`` kwarg has been removed and is not needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import huggingface_hub

logger = logging.getLogger(__name__)


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve an HF token from explicit arg → ``HF_TOKEN`` env → cached creds.

    Returns the token string, or ``None`` if none is available. Never persists
    the token; never logs its value; never calls ``huggingface_hub.login()``.
    """
    if token:
        return token
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    return huggingface_hub.get_token() or None


def download_model(
    repo_id: str,
    local_dir: Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Snapshot-download ``repo_id`` into ``local_dir`` if not already present.

    Idempotent unless ``force=True``: when ``local_dir`` exists and is
    non-empty, returns immediately without contacting the Hub.

    The consumer who knows the expected filename (e.g. ``_resolve_checkpoint_path``)
    should re-check file-level presence after this returns — the "non-empty"
    skip condition is intentionally weak.

    Returns ``local_dir`` on success.
    """
    if not force and local_dir.exists() and any(local_dir.iterdir()):
        return local_dir

    resolved = resolve_hf_token(token)
    logger.info("fetching %s → %s", repo_id, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        revision=revision,
        token=resolved,
    )
    return local_dir
