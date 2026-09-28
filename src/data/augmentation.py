"""Data augmentation for training images (albumentations).

Augmentation is applied to the training set only. Validation and test
images are never augmented, so evaluation always sees the real images.

Why each augmentation is valid for fundus photographs:

- Rotation (any angle) and flips: a retina photo has no natural "up"; the
  camera orientation and left/right eye only change where the optic disc
  sits, not the disease stage.
- Zoom and shift: simulates different fields of view and framing.
- Brightness / contrast: simulates exposure differences between cameras.
- Small hue / saturation shifts: simulates colour differences between
  camera models. Kept small so red lesions stay red.

Transformations that would change the medical meaning (large colour
changes, heavy blur that hides microaneurysms, cut-outs that could delete
a lesion) are deliberately not used.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np


def get_train_transforms(cfg: dict[str, Any]) -> A.Compose:
    """Random augmentation pipeline for training images."""
    a = cfg["augmentation"]
    return A.Compose(
        [
            A.Affine(
                rotate=(-a["rotate_limit"], a["rotate_limit"]),
                scale=tuple(a["scale"]),
                translate_percent=(-a["translate"], a["translate"]),
                p=1.0,
            ),
            A.HorizontalFlip(p=a["hflip"]),
            A.VerticalFlip(p=a["vflip"]),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=a["brightness"], contrast_limit=a["contrast"], p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=a["hue_shift"],
                        sat_shift_limit=a["sat_shift"],
                        val_shift_limit=0,
                        p=1.0,
                    ),
                ],
                p=a["p_color"],
            ),
        ]
    )


def get_eval_transforms(cfg: dict[str, Any]) -> None:
    """Validation / test images are not augmented (returns None)."""
    return None


def apply_augmentation(transform: A.Compose, img: np.ndarray) -> np.ndarray:
    """Augment an image and keep its background black.

    Brightness / contrast changes would otherwise turn the black background
    grey, which never happens in real (preprocessed) images. The background
    mask is passed through the same geometric transforms as the image, and
    everything outside it is set back to 0 afterwards.
    """
    mask = (img.max(axis=2) > 0).astype(np.uint8)
    out = transform(image=img, mask=mask)
    aug = out["image"]
    aug[out["mask"] == 0] = 0
    return aug
