"""PyTorch dataset, preprocessed-image cache and class balancing.

Loading flow for one image:
    original PNG -> preprocessing (src/data/preprocessing.py)
    -> augmentation (training only) -> ImageNet normalisation -> tensor

Preprocessing is deterministic, so each image is preprocessed once and
cached as a PNG. Every later epoch reads the cached file, which removes the
preprocessing cost from training time.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.augmentation import apply_augmentation
from src.data.preprocessing import load_image, normalize, preprocess


def cache_dir_for(cfg: dict[str, Any]) -> Path:
    """Cache folder specific to the current preprocessing settings.

    The folder name contains a hash of the preprocessing config, so changing
    any setting (e.g. switching CLAHE off for the ablation) uses a new cache
    instead of silently reusing old images.
    """
    key = json.dumps({"p": cfg["preprocessing"], "size": cfg["image"]["size"]}, sort_keys=True)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return Path(cfg["paths"]["processed_dir"]) / f"prep_{digest}"


def _cache_one(args: tuple) -> None:
    """Preprocess one image and write it to the cache (worker function)."""
    src, dst, cfg = args
    cv2.setNumThreads(0)
    img = preprocess(load_image(src), cfg)
    cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def build_cache(df: pd.DataFrame, cfg: dict[str, Any], workers: int = 4) -> Path:
    """Preprocess every image in ``df`` that is not cached yet.

    Returns:
        The cache folder.
    """
    out = cache_dir_for(cfg)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["paths"]["data_root"])
    todo = [
        (root / r.path, out / f"{r.image_id}.png", cfg)
        for r in df.itertuples()
        if not (out / f"{r.image_id}.png").exists()
    ]
    if todo:
        print(f"Preprocessing {len(todo):,} images into {out} ...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_cache_one, todo, chunksize=64))
    return out


class DRDataset(Dataset):
    """Fundus images and stage labels for one split.

    Args:
        df: Split DataFrame with columns image_id, path, label.
        cfg: Full config.
        transform: Albumentations pipeline (training) or None.
        cache_dir: Folder of preprocessed images. If None, images are
            preprocessed on the fly (slower, used for small tests).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict[str, Any],
        transform: Callable | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.cache_dir = cache_dir
        self.root = Path(cfg["paths"]["data_root"])
        self.mean, self.std = cfg["image"]["mean"], cfg["image"]["std"]

    def __len__(self) -> int:
        return len(self.df)

    def load_preprocessed(self, idx: int) -> np.ndarray:
        """Return the preprocessed RGB uint8 image for row ``idx``."""
        row = self.df.iloc[idx]
        if self.cache_dir is not None:
            return load_image(self.cache_dir / f"{row.image_id}.png")
        return preprocess(load_image(self.root / row.path), self.cfg)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.load_preprocessed(idx)
        if self.transform is not None:
            img = apply_augmentation(self.transform, img)
        x = normalize(img, self.mean, self.std)
        tensor = torch.from_numpy(x.transpose(2, 0, 1).copy())  # HWC -> CHW
        return tensor, int(self.df.iloc[idx].label)


def class_weights(labels: np.ndarray, num_classes: int, power: float = 0.5) -> np.ndarray:
    """Per-class weights, w_c = (N / (K * n_c)) ** power.

    The weights are rescaled so the average weight over all training images
    is 1, which keeps the loss on the same scale as unweighted training.
    ``power`` = 1 is full inverse-frequency weighting; 0.5 (square root) is
    gentler and avoids over-fitting the rarest classes.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    w = (len(labels) / (num_classes * counts)) ** power
    w = w / (w[labels].mean())
    return w.astype(np.float32)


def make_sampler(labels: np.ndarray, num_classes: int, power: float = 0.5) -> WeightedRandomSampler:
    """Sampler that draws rare-class images more often (with replacement).

    Each image gets its class weight, so each epoch sees a more balanced mix
    of stages while the number of images per epoch stays the same.
    """
    w = class_weights(labels, num_classes, power)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w[labels], dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
