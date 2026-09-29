"""Grad-CAM heatmaps (Selvaraju et al., 2017).

Grad-CAM shows which regions of the image drove the prediction. The
gradient of the predicted class score with respect to the last
convolutional feature map says how important each channel is; the
channel-weighted sum of the feature map, passed through ReLU, is the
heatmap. It is used to check that the model looks at lesions
(haemorrhages, exudates, new vessels) and not at image artefacts.

Works with any timm CNN through ``forward_features`` / ``forward_head``.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


def gradcam(model: nn.Module, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute a Grad-CAM heatmap for one image.

    Args:
        model: timm classification model (put in eval mode by this function).
        x: Normalised input tensor, shape (1, 3, H, W).
        target: Class to explain. Defaults to the predicted class.

    Returns:
        (heatmap in [0, 1] with shape (H, W), explained class, softmax probabilities)
    """
    model.eval()
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    cls = int(probs.argmax()) if target is None else int(target)

    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)          # channel importance
    cam = torch.relu((weights * feats).sum(dim=1))[0].detach()   # (h, w)
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32), cls, probs


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap (jet colour map) onto an RGB uint8 image.

    The black background (outside the retina) is left black, so the heatmap
    is only shown where there is retina.
    """
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (img * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    out[img.max(axis=2) == 0] = 0
    return out
