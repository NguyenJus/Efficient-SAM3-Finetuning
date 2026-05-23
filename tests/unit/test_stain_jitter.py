"""Tests for custom_sam_peft.data.transforms.StainJitter — HED-space stain jitter."""

from __future__ import annotations

import numpy as np
import pytest

from custom_sam_peft.data.transforms import StainJitter


def _random_uint8_image(shape: tuple[int, int, int] = (32, 32, 3)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_identity_at_sigma_zero() -> None:
    img = _random_uint8_image()
    out = StainJitter(sigma=0.0, p=1.0).apply(img)
    assert np.array_equal(out, img)


def test_dtype_and_shape_preserved() -> None:
    img = _random_uint8_image()
    out = StainJitter(sigma=0.1, p=1.0).apply(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_range_preserved() -> None:
    img = _random_uint8_image()
    out = StainJitter(sigma=0.1, p=1.0).apply(img)
    assert out.min() >= 0
    assert out.max() <= 255


def test_mask_untouched_through_compose() -> None:
    import albumentations as A

    img = _random_uint8_image()
    mask = np.ones((32, 32), dtype=np.uint8) * 7
    compose = A.Compose(
        [StainJitter(sigma=0.1, p=1.0)],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )
    out = compose(image=img, mask=mask, bboxes=[], class_labels=[])
    assert np.array_equal(out["mask"], mask)


def test_determinism_with_numpy_seed() -> None:
    img = _random_uint8_image()
    np.random.seed(0)
    out1 = StainJitter(sigma=0.1, p=1.0).apply(img)
    np.random.seed(0)
    out2 = StainJitter(sigma=0.1, p=1.0).apply(img)
    assert np.array_equal(out1, out2)


def test_sigma_negative_rejected() -> None:
    with pytest.raises(ValueError, match="sigma must be >= 0"):
        StainJitter(sigma=-0.1)


def test_p_zero_passes_through() -> None:
    import albumentations as A

    img = _random_uint8_image()
    compose = A.Compose(
        [StainJitter(sigma=0.5, p=0.0)],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
            min_area=0.0,
        ),
    )
    out = compose(image=img, bboxes=[], class_labels=[])
    assert np.array_equal(out["image"], img)
