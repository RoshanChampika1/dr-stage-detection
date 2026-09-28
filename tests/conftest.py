"""Shared test fixtures: a tiny fake dataset with the Kaggle folder layout."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
import yaml

from src.data.dataset_index import build_index
from src.data.split import patient_level_split
from src.utils.config import DEFAULT_CONFIG

FOLDERS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


@pytest.fixture(scope="session")
def fake_project(tmp_path_factory):
    """Create fake fundus images, split CSVs and a config pointing at them.

    Returns the path of the temporary config file.
    """
    root = tmp_path_factory.mktemp("fake_dr")
    data = root / "raw"
    rng, nrng = random.Random(0), np.random.default_rng(0)
    for pid in range(1, 61):
        for eye in ("left", "right"):
            label = rng.choices(range(5), [0.5, 0.15, 0.15, 0.1, 0.1])[0]
            d = data / "colored_images" / FOLDERS[label]
            d.mkdir(parents=True, exist_ok=True)
            img = np.zeros((96, 96, 3), np.uint8)
            cv2.circle(img, (48, 48), 42, (40, 80, 170), -1)
            for _ in range(label * 4):
                cv2.circle(img, (rng.randint(20, 76), rng.randint(20, 76)), 2, (20, 20, 90), -1)
            img = cv2.add(img, nrng.integers(0, 6, img.shape, dtype=np.uint8))
            cv2.imwrite(str(d / f"{pid}_{eye}.png"), img)

    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {
        "data_root": str(data),
        "splits_dir": str(root / "splits"),
        "processed_dir": str(root / "processed"),
        "models_dir": str(root / "models"),
        "figures_dir": str(root / "figures"),
        "logs_dir": str(root / "logs"),
        "metrics_dir": str(root / "metrics"),
    }
    cfg["image"]["size"] = 64

    df = build_index(data, {n: i for i, n in enumerate(FOLDERS)}, [".png"])
    (root / "splits").mkdir()
    for name, part in patient_level_split(df, seed=0).items():
        part.to_csv(root / "splits" / f"{name}.csv", index=False)

    cfg_path = root / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path
