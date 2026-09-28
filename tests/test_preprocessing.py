"""Tests for the preprocessing pipeline using synthetic fundus-like images."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.data import preprocessing as pp
from src.utils.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def fake_fundus():
    """A reddish disc with a few dark dots on a black 300x260 background."""
    img = np.zeros((260, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 130), 100, (180, 80, 40), -1)
    for x, y in [(120, 110), (170, 150), (150, 90)]:
        cv2.circle(img, (x, y), 3, (90, 30, 20), -1)
    noise = np.random.default_rng(0).integers(0, 15, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def test_crop_removes_black_border(fake_fundus):
    cropped = pp.crop_black_border(fake_fundus, threshold=30)
    assert abs(cropped.shape[0] - 201) <= 2
    assert abs(cropped.shape[1] - 201) <= 2


def test_crop_keeps_dark_image_unchanged():
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    assert pp.crop_black_border(dark).shape == dark.shape


def test_pipeline_output_shape_and_type(cfg, fake_fundus):
    out = pp.preprocess(fake_fundus, cfg)
    size = cfg["image"]["size"]
    assert out.shape == (size, size, 3)
    assert out.dtype == np.uint8


def test_steps_are_ordered_and_named(cfg, fake_fundus):
    steps = pp.preprocess_steps(fake_fundus, cfg)
    names = list(steps)
    assert names[0] == "Original" and names[-1] == "Masked (final)"
    assert "CLAHE" in names and "Ben Graham" in names


def test_steps_can_be_disabled(cfg, fake_fundus):
    for key in ("crop_black_border", "clahe", "ben_graham"):
        cfg["preprocessing"][key] = False
    cfg["preprocessing"]["denoise"] = "none"
    steps = pp.preprocess_steps(fake_fundus, cfg)
    assert list(steps) == ["Original", "Resized"]


def test_ben_graham_centres_flat_image_on_128():
    flat = np.full((64, 64, 3), 90, dtype=np.uint8)
    out = pp.ben_graham(flat, sigma=5)
    assert np.allclose(out, 128, atol=1)


def test_retina_mask_is_inside_retina(fake_fundus):
    mask = pp.retina_mask(fake_fundus, threshold=30)
    assert mask[0, 0] == 0 and mask[130, 150] == 1
    # the eroded mask must be smaller than the raw retina disc (radius 100)
    assert mask.sum() < np.pi * 100**2


def test_crop_ignores_isolated_noise(fake_fundus):
    noisy = fake_fundus.copy()
    noisy[2, 2] = 255
    assert pp.crop_black_border(noisy, 30).shape == pp.crop_black_border(fake_fundus, 30).shape


def test_circular_mask_blacks_out_corners():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    out = pp.circular_mask(img)
    assert out[0, 0].sum() == 0 and out[50, 50].sum() > 0


def test_clahe_increases_contrast():
    rng = np.random.default_rng(1)
    low = (rng.normal(120, 5, (128, 128, 3))).clip(0, 255).astype(np.uint8)
    before = pp.image_quality_metrics(low)["rms_contrast"]
    after = pp.image_quality_metrics(pp.apply_clahe(low, 3.0))["rms_contrast"]
    assert after > before


def test_normalize_matches_imagenet_stats():
    img = np.full((4, 4, 3), 255, dtype=np.uint8)
    out = pp.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert out.dtype == np.float32 and np.allclose(out, 1.0)


def test_crop_works_on_very_dark_retina():
    """Retina darker than 10 (a very dark photo) is still found and cropped."""
    img = np.zeros((200, 260, 3), dtype=np.uint8)
    cv2.circle(img, (130, 100), 80, (9, 7, 6), -1)
    cropped = pp.crop_black_border(img)
    assert abs(cropped.shape[1] - 161) <= 3


def test_mask_excludes_glow_around_overexposed_retina():
    """A faint glow around a white retina must not be treated as retina."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 95, (25, 25, 25), -1)   # glow
    cv2.circle(img, (100, 100), 70, (255, 255, 255), -1)  # overexposed retina
    mask = pp.retina_mask(img)
    assert mask[100, 100 + 75] == 0  # glow ring excluded
    assert mask[100, 100] == 1
    # the retina edge is excluded, so sharpness inside the mask is ~0
    assert pp.image_quality_metrics(img)["sharpness"] < 1
