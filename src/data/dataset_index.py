"""Build an index of all images in the Kaggle dataset.

The dataset stores images in one folder per class (``No_DR``, ``Mild``, ...)
and may also ship a ``trainLabels.csv`` file. Image names follow the EyePACS
convention ``<patient_id>_<eye>`` (for example ``10_left``), which is what
lets us split the data by patient later.

The resulting DataFrame has one row per image with the columns:

    image_id    file name without extension, e.g. "10_left"
    patient_id  number before the underscore, e.g. "10"
    eye         "left" or "right"
    label       integer stage 0 to 4
    path        path relative to the data root (portable across machines)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_labels_csv(data_root: Path, name: str | None) -> Path | None:
    """Return the first CSV named ``name`` anywhere under ``data_root``."""
    if not name:
        return None
    matches = sorted(data_root.rglob(name))
    return matches[0] if matches else None


def build_index(
    data_root: str | Path,
    folder_to_label: dict[str, int],
    image_extensions: list[str],
    labels_csv: str | None = None,
) -> pd.DataFrame:
    """Scan ``data_root`` and return a DataFrame describing every image.

    Labels come from the class folder name. If a labels CSV is present, it is
    used to cross-check the folder labels, and any image missing a label is
    reported.

    Args:
        data_root: Folder containing the extracted dataset.
        folder_to_label: Mapping from class folder name to stage label.
        image_extensions: File extensions treated as images.
        labels_csv: Optional labels CSV file name (columns ``image``, ``level``).

    Returns:
        DataFrame with columns image_id, patient_id, eye, label, path.

    Raises:
        FileNotFoundError: If no images are found under ``data_root``.
    """
    data_root = Path(data_root)
    exts = {e.lower() for e in image_extensions}

    rows = []
    for img_path in data_root.rglob("*"):
        if img_path.suffix.lower() not in exts:
            continue
        folder = img_path.parent.name
        if folder not in folder_to_label:
            continue  # skip images outside the class folders
        image_id = img_path.stem
        patient_id, _, eye = image_id.partition("_")
        rows.append(
            {
                "image_id": image_id,
                "patient_id": patient_id,
                "eye": eye,
                "label": folder_to_label[folder],
                "path": img_path.relative_to(data_root).as_posix(),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No images found under {data_root}. Expected class folders "
            f"{list(folder_to_label)}. Check paths.data_root in the config "
            "or set the DR_DATA_ROOT environment variable."
        )

    df = pd.DataFrame(rows)

    # Some dataset versions contain the same image twice (for example inside a
    # nested copy of the folder). Keep one copy per image id.
    df = df.sort_values("path").drop_duplicates("image_id").reset_index(drop=True)

    csv_path = _find_labels_csv(data_root, labels_csv)
    if csv_path is not None:
        labels = pd.read_csv(csv_path)
        merged = df.merge(labels, left_on="image_id", right_on="image", how="left")
        mismatch = merged["level"].notna() & (merged["level"] != merged["label"])
        if mismatch.any():
            print(f"Warning: {mismatch.sum()} folder labels disagree with {csv_path.name}")

    return df.sort_values(["patient_id", "eye"]).reset_index(drop=True)
