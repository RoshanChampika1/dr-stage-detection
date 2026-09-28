"""Patient-level stratified train / validation / test split.

Each patient usually has two images (left and right eye). If one eye ended up
in training and the other in testing, the model would be evaluated on a
patient it has already seen, which inflates the test score (data leakage).
To prevent this, the split is done on patients rather than on images.

Stratification uses each patient's most severe stage across both eyes, so
the rare Severe and Proliferative stages appear in every split in roughly the
same proportion as the full dataset.

Usage:
    python -m src.data.split
    python -m src.data.split --config configs/config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset_index import build_index
from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.seed import set_seed


def patient_level_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split an image index into train/val/test without sharing patients.

    Args:
        df: Image index from :func:`build_index`.
        train_frac, val_frac, test_frac: Split fractions (must sum to 1).
        seed: Random seed for a reproducible split.

    Returns:
        Dict with keys "train", "val", "test", each an image-level DataFrame.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.")

    # One row per patient, labelled with their worst stage.
    patients = df.groupby("patient_id")["label"].max().reset_index()

    # First cut off the training patients, then divide the rest into val/test.
    train_p, rest_p = train_test_split(
        patients,
        train_size=train_frac,
        stratify=patients["label"],
        random_state=seed,
    )
    val_share = val_frac / (val_frac + test_frac)
    val_p, test_p = train_test_split(
        rest_p,
        train_size=val_share,
        stratify=rest_p["label"],
        random_state=seed,
    )

    splits = {}
    for name, part in (("train", train_p), ("val", val_p), ("test", test_p)):
        ids = set(part["patient_id"])
        splits[name] = df[df["patient_id"].isin(ids)].reset_index(drop=True)
    return splits


def split_counts(splits: dict[str, pd.DataFrame], class_names: list[str]) -> pd.DataFrame:
    """Return a table of image counts per class for each split (for the report)."""
    table = pd.DataFrame(
        {name: part["label"].value_counts().sort_index() for name, part in splits.items()}
    ).reindex(range(len(class_names)), fill_value=0)
    table.index = [f"{i} {n}" for i, n in enumerate(class_names)]
    table.loc["Total"] = table.sum()
    return table.fillna(0).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    paths, data_cfg, split_cfg = cfg["paths"], cfg["data"], cfg["split"]

    df = build_index(
        paths["data_root"],
        data_cfg["folder_to_label"],
        data_cfg["image_extensions"],
        data_cfg.get("labels_csv"),
    )
    print(f"Indexed {len(df)} images from {df['patient_id'].nunique()} patients")

    splits = patient_level_split(
        df, split_cfg["train"], split_cfg["val"], split_cfg["test"], cfg["seed"]
    )

    paths["splits_dir"].mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(paths["splits_dir"] / f"{name}.csv", index=False)

    # Sanity check: no patient may appear in more than one split.
    sets = {n: set(p["patient_id"]) for n, p in splits.items()}
    assert not (sets["train"] & sets["val"]), "patient leakage train/val"
    assert not (sets["train"] & sets["test"]), "patient leakage train/test"
    assert not (sets["val"] & sets["test"]), "patient leakage val/test"

    counts = split_counts(splits, data_cfg["class_names"])
    paths["metrics_dir"].mkdir(parents=True, exist_ok=True)
    counts.to_csv(paths["metrics_dir"] / "split_counts.csv")
    print(counts.to_string())
    print(f"Saved split CSVs to {paths['splits_dir']}")


if __name__ == "__main__":
    main()
