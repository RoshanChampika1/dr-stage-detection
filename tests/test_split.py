"""Tests for the dataset index and the patient-level split.

A small fake dataset with the same folder layout as the Kaggle dataset is
created in a temporary folder, so the tests run without downloading data.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from src.data.dataset_index import build_index
from src.data.split import patient_level_split, split_counts

FOLDERS = {"No_DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferate_DR": 4}
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


@pytest.fixture
def fake_dataset(tmp_path):
    """Create 200 patients x 2 eyes of tiny images in class folders."""
    rng = random.Random(0)
    weights = [0.6, 0.1, 0.15, 0.08, 0.07]
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    for pid in range(1, 201):
        for eye in ("left", "right"):
            label = rng.choices(range(5), weights)[0]
            folder = [k for k, v in FOLDERS.items() if v == label][0]
            out = tmp_path / "colored_images" / folder
            out.mkdir(parents=True, exist_ok=True)
            img.save(out / f"{pid}_{eye}.png")
    return tmp_path


def test_index_reads_all_images(fake_dataset):
    df = build_index(fake_dataset, FOLDERS, [".png"])
    assert len(df) == 400
    assert df["patient_id"].nunique() == 200
    assert set(df["eye"]) == {"left", "right"}
    assert df["label"].between(0, 4).all()


def test_index_raises_on_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path, FOLDERS, [".png"])


def test_split_has_no_patient_leakage(fake_dataset):
    df = build_index(fake_dataset, FOLDERS, [".png"])
    splits = patient_level_split(df, seed=42)
    tr, va, te = (set(splits[k]["patient_id"]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 200


def test_split_fractions_and_reproducibility(fake_dataset):
    df = build_index(fake_dataset, FOLDERS, [".png"])
    a = patient_level_split(df, seed=42)
    b = patient_level_split(df, seed=42)
    assert a["test"]["image_id"].tolist() == b["test"]["image_id"].tolist()
    n = len(df)
    assert abs(len(a["train"]) / n - 0.70) < 0.05
    assert abs(len(a["val"]) / n - 0.15) < 0.05


def test_split_counts_table(fake_dataset):
    df = build_index(fake_dataset, FOLDERS, [".png"])
    table = split_counts(patient_level_split(df), CLASS_NAMES)
    assert table.loc["Total"].sum() == 400
    assert list(table.columns) == ["train", "val", "test"]
