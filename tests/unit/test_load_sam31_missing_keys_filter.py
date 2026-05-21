"""Tests for the convs.3 missing-keys filter logic in esam3.models.sam3.

This test suite exercises ``_classify_missing_keys`` in isolation — no sam3,
torch, or GPU required.  The function is a pure helper that decides whether a
(missing_keys, unexpected_keys) pair from load_state_dict should be silently
suppressed ("ok") or cause a loud RuntimeError ("fail").

Background: the released sam3.1_multiplex.pt is built from a 3-scale neck;
our load_sam31 instantiates a 4-scale neck.  convs[3] is dropped by the
scalp=1 trim in vl_combiner so its random init never participates in training.
The filter detects exactly that pattern and suppresses the noise.
"""

from __future__ import annotations

from esam3.models.sam3 import _KNOWN_MISSING_KEYS, _classify_missing_keys

# ---------------------------------------------------------------------------
# Happy-path: exactly the known set → "ok"
# ---------------------------------------------------------------------------


def test_classify_exactly_known_set_is_ok() -> None:
    """The four convs.3 keys and no unexpected keys → "ok" (harmless noise)."""
    result = _classify_missing_keys(
        missing=set(_KNOWN_MISSING_KEYS),
        unexpected=set(),
    )
    assert result == "ok"


def test_classify_empty_missing_is_ok() -> None:
    """No missing keys at all (e.g. a future sam3 ships convs.3) → "ok".

    If the released checkpoint starts shipping convs[3] weights the missing set
    shrinks to empty.  That is strictly safer — the neck is now fully
    initialised — so we accept it.
    """
    result = _classify_missing_keys(missing=set(), unexpected=set())
    assert result == "ok"


def test_classify_subset_of_known_is_ok() -> None:
    """Proper subset of known missing keys and no unexpected → "ok".

    A new sam3 release might ship some (but not all) of the convs.3 keys.
    Fewer missing keys can only be safer, so a subset is accepted.
    """
    subset = {
        "backbone.vision_backbone.convs.3.conv_1x1.weight",
        "backbone.vision_backbone.convs.3.conv_1x1.bias",
    }
    assert subset < _KNOWN_MISSING_KEYS  # sanity: really a proper subset
    result = _classify_missing_keys(missing=subset, unexpected=set())
    assert result == "ok"


# ---------------------------------------------------------------------------
# Failure: known set PLUS extra keys → "fail"
# ---------------------------------------------------------------------------


def test_classify_known_set_plus_extra_key_is_fail() -> None:
    """Known set + one extra missing key → "fail" (could be checkpoint regression)."""
    extended = set(_KNOWN_MISSING_KEYS) | {"backbone.some_new_layer.weight"}
    result = _classify_missing_keys(missing=extended, unexpected=set())
    assert result == "fail"


def test_classify_entirely_unknown_missing_key_is_fail() -> None:
    """A single missing key not in the known set → "fail"."""
    result = _classify_missing_keys(
        missing={"backbone.transformer.encoder.layers.0.weight"},
        unexpected=set(),
    )
    assert result == "fail"


# ---------------------------------------------------------------------------
# Failure: any unexpected key → "fail"
# ---------------------------------------------------------------------------


def test_classify_unexpected_key_alone_is_fail() -> None:
    """Any unexpected key → "fail", even if missing keys look fine."""
    result = _classify_missing_keys(
        missing=set(_KNOWN_MISSING_KEYS),
        unexpected={"backbone.some_extra_layer.weight"},
    )
    assert result == "fail"


def test_classify_empty_missing_with_unexpected_is_fail() -> None:
    """No missing keys but unexpected keys present → "fail"."""
    result = _classify_missing_keys(
        missing=set(),
        unexpected={"backbone.unexpected.weight"},
    )
    assert result == "fail"


# ---------------------------------------------------------------------------
# Interaction: both missing-outside-known and unexpected → "fail"
# ---------------------------------------------------------------------------


def test_classify_both_extra_missing_and_unexpected_is_fail() -> None:
    """Both extra missing and unexpected keys → "fail"."""
    result = _classify_missing_keys(
        missing=set(_KNOWN_MISSING_KEYS) | {"backbone.new.weight"},
        unexpected={"backbone.extra.weight"},
    )
    assert result == "fail"
