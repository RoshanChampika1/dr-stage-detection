"""Fundus image preprocessing pipeline.

The same functions are used for training, evaluation and the inference API,
so every image is processed identically everywhere.

Pipeline (each step can be switched on or off in ``configs/config.yaml``):

1. Crop the black border   remove the empty background around the retina
2. Resize                  back to the network input size (224 x 224)
3. Denoise                 median or bilateral filter to remove sensor noise
4. CLAHE                   local contrast enhancement on the LAB lightness
5. Ben Graham enhancement  subtract a Gaussian blur to highlight lesions
                           and edges and even out uneven lighting
6. Retina mask             blank everything outside the (slightly shrunk)
                           retina, where step 5 leaves a bright halo

Normalisation to ImageNet mean/std is applied last, when the image is
converted to a tensor (see :func:`normalize`).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def crop_black_border(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop the dark background so the retina fills the frame.

    Pixels with grey level above ``threshold`` are treated as retina. The
    image is cropped to the bounding box of those pixels. If the image is
    almost entirely dark (a failed photo), it is returned unchanged.

    Args:
        img: RGB image, uint8, shape (H, W, 3).
        threshold: Grey level separating background from retina.

    Returns:
        Cropped RGB image.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = grey > threshold
    if mask.sum() < 0.05 * mask.size:
        return img
    # A row/column counts as retina only if at least 1% of it is bright,
    # so isolated noisy pixels in the background do not stop the crop.
    rows = np.where(mask.sum(axis=1) > 0.01 * mask.shape[1])[0]
    cols = np.where(mask.sum(axis=0) > 0.01 * mask.shape[0])[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize to ``size`` x ``size`` using area interpolation (sharp when shrinking)."""
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def denoise(img: np.ndarray, method: str = "median", kernel: int = 3) -> np.ndarray:
    """Remove noise with a median or bilateral filter.

    The median filter removes salt-and-pepper noise while keeping edges.
    The bilateral filter smooths flat regions but preserves vessel edges.
    """
    if method == "median":
        return cv2.medianBlur(img, kernel)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=kernel * 2 + 1, sigmaColor=40, sigmaSpace=40)
    return img


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the L channel.

    Working in LAB colour space enhances brightness contrast (making
    microaneurysms, haemorrhages and exudates more visible) without shifting
    the colours. The clip limit stops noise from being over-amplified.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def ben_graham(img: np.ndarray, sigma: float = 10) -> np.ndarray:
    """Ben Graham's local average colour subtraction.

    ``4 * img - 4 * GaussianBlur(img) + 128``. Subtracting the blurred image
    removes slow changes in lighting across the retina and keeps the fine
    detail, acting as an edge / lesion enhancement filter. This method was
    used by the winner of the 2015 Kaggle DR competition.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def circular_mask(img: np.ndarray, scale: float = 0.95) -> np.ndarray:
    """Set everything outside a centred circle to black.

    After cropping, the retina is roughly a circle filling the image. The
    rim is removed because blur subtraction creates a bright halo there.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * scale), 1, thickness=-1)
    return img * mask[:, :, None]


def retina_mask(img: np.ndarray, threshold: int = 10, shrink: float = 0.06) -> np.ndarray:
    """Binary mask (0/1, uint8) of the retina, shrunk slightly at the edge.

    The retina is found by thresholding the grey image. The largest region is
    kept and filled with its convex hull (the retina is convex, so dark
    lesions or notches on the edge do not create holes). The mask is then
    eroded by ``shrink`` x image size to remove the bright halo that blur
    subtraction creates along the retina edge. Falls back to a centred circle
    if no retina is found.
    """
    h, w = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raw = (grey > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.fillConvexPoly(mask, hull, 1)
    if mask.sum() < 0.05 * mask.size:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.45), 1, thickness=-1)
        return mask
    k = max(3, int(min(h, w) * shrink) * 2 + 1)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def preprocess_steps(img: np.ndarray, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run the pipeline and keep every intermediate image.

    Used for the step-by-step figure in the report. Disabled steps are skipped.

    Args:
        img: RGB uint8 image.
        cfg: Full config dict (uses ``preprocessing`` and ``image.size``).

    Returns:
        Ordered dict mapping step name to image. The last entry is the final output.
    """
    p = cfg["preprocessing"]
    size = cfg["image"]["size"]
    steps = {"Original": img}

    if p.get("crop_black_border", True):
        img = crop_black_border(img, p.get("border_threshold", 10))
        steps["Border cropped"] = img

    img = resize(img, size)
    steps["Resized"] = img

    if p.get("denoise", "none") != "none":
        img = denoise(img, p["denoise"], p.get("denoise_kernel", 3))
        steps["Denoised"] = img

    # Find the retina before enhancement changes the background brightness.
    mask = retina_mask(img, p.get("border_threshold", 10))

    if p.get("clahe", True):
        img = apply_clahe(img, p.get("clahe_clip_limit", 2.0), p.get("clahe_tile_grid", 8))
        steps["CLAHE"] = img

    if p.get("ben_graham", True):
        img = ben_graham(img, p.get("ben_graham_sigma", 10))
        steps["Ben Graham"] = img

    if p.get("ben_graham", True):
        img = img * mask[:, :, None]
        steps["Masked (final)"] = img

    return steps


def preprocess(img: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Apply the full preprocessing pipeline and return the final RGB uint8 image."""
    return list(preprocess_steps(img, cfg).values())[-1]


def load_image(path) -> np.ndarray:
    """Read an image from disk as RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize(img: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """Scale to [0, 1] and standardise with the ImageNet mean and std.

    Pretrained backbones expect inputs in the same range as ImageNet.

    Returns:
        float32 array of shape (H, W, 3).
    """
    x = img.astype(np.float32) / 255.0
    return (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)


def image_quality_metrics(img: np.ndarray, threshold: int = 10) -> dict[str, float]:
    """Simple quality measures used to show that preprocessing helps.

    - rms_contrast: standard deviation of grey levels inside the retina
    - entropy: Shannon entropy of the grey histogram (information content)
    - sharpness: variance of the Laplacian (edge strength)
    """
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = grey > threshold
    vals = grey[mask] if mask.any() else grey.ravel()
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    prob = hist[hist > 0] / hist.sum()
    return {
        "rms_contrast": float(vals.std()),
        "entropy": float(-(prob * np.log2(prob)).sum()),
        "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
    }
