"""Tests for the dataset, cache, augmentation and class balancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.augmentation import apply_augmentation, get_train_transforms
from src.data.dataset import DRDataset, build_cache, cache_dir_for, class_weights, make_sampler
from src.utils.config import load_config


def _setup(fake_project):
    cfg = load_config(fake_project)
    df = pd.read_csv(cfg["paths"]["splits_dir"] / "train.csv", dtype={"patient_id": str})
    return cfg, df


def test_dataset_returns_normalised_tensor(fake_project):
    cfg, df = _setup(fake_project)
    x, y = DRDataset(df, cfg)[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert 0 <= y <= 4
    assert -3 < x.mean() < 3  # roughly standardised


def test_cache_matches_on_the_fly(fake_project):
    cfg, df = _setup(fake_project)
    cache = build_cache(df.head(5), cfg, workers=1)
    assert len(list(cache.glob("*.png"))) == 5
    a = DRDataset(df.head(5), cfg, cache_dir=cache)[0][0]
    b = DRDataset(df.head(5), cfg)[0][0]
    assert torch.allclose(a, b, atol=1e-5)  # PNG is lossless


def test_cache_dir_changes_with_preprocessing(fake_project):
    cfg, _ = _setup(fake_project)
    before = cache_dir_for(cfg)
    cfg["preprocessing"]["clahe"] = False
    assert cache_dir_for(cfg) != before


def test_augmentation_keeps_shape_and_changes_image(fake_project):
    cfg, df = _setup(fake_project)
    img = DRDataset(df, cfg).load_preprocessed(0)
    aug = get_train_transforms(cfg)(image=img)["image"]
    assert aug.shape == img.shape and aug.dtype == img.dtype
    assert not np.array_equal(aug, img)


def test_augmentation_keeps_background_black(fake_project):
    cfg, df = _setup(fake_project)
    img = DRDataset(df, cfg).load_preprocessed(0)
    cfg["augmentation"].update(p_color=1.0, brightness=0.5, rotate_limit=0)
    t = get_train_transforms(cfg)
    for _ in range(5):
        aug = apply_augmentation(t, img.copy())
        assert aug[0, 0].sum() == 0 and aug[-1, -1].sum() == 0


def test_class_weights_favour_rare_classes():
    labels = np.array([0] * 90 + [4] * 10)
    w = class_weights(labels, 5, power=1.0)
    assert w[4] > w[0]
    assert np.isclose(w[labels].mean(), 1.0)
    w_sqrt = class_weights(labels, 5, power=0.5)
    assert w_sqrt[4] / w_sqrt[0] < w[4] / w[0]  # square root is gentler


def test_sampler_balances_classes():
    labels = np.array([0] * 900 + [1] * 100)
    torch.manual_seed(0)
    drawn = labels[list(make_sampler(labels, 2, power=1.0))]
    assert 0.4 < drawn.mean() < 0.6  # about half from the rare class
