"""Model loading and single-image inference used by the web API.

Uses exactly the same preprocessing code as training
(``src/data/preprocessing.py``), with the preprocessing settings stored in
the checkpoint, so a web upload is processed identically to a training
image.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.data.preprocessing import image_quality_metrics, normalize, preprocess, retina_threshold
from src.evaluation.gradcam import gradcam, overlay
from src.models.factory import build_model

# Image-quality limits, based on the EDA of the training data (report 2.4).
MIN_BRIGHTNESS = 35      # mean grey level inside the retina; darker = underexposed
MAX_BRIGHTNESS = 215     # brighter = overexposed, detail washed out
MIN_SHARPNESS = 25       # Laplacian variance inside the retina; lower = blurred
MIN_RETINA_FRACTION = 0.15  # retina should cover a good part of a fundus photo
MAX_CORNER_BRIGHTNESS = 40  # fundus photos have a black border, so dark corners
LOW_CONFIDENCE = 0.50

ADVICE = {
    0: "No signs of diabetic retinopathy detected. Continue routine annual screening.",
    1: "Mild non-proliferative DR (microaneurysms only). Re-screen in 6-12 months.",
    2: "Moderate non-proliferative DR. Refer to an ophthalmologist.",
    3: "Severe non-proliferative DR. Urgent referral to an ophthalmologist.",
    4: "Proliferative DR. Urgent referral: sight-threatening, needs treatment.",
}


def encode_png(img: np.ndarray) -> str:
    """RGB uint8 image -> base64 PNG string (for JSON responses)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image(data: bytes) -> np.ndarray:
    """Uploaded file bytes -> RGB uint8 image. Raises ValueError if not an image."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("The file is not a readable image (use JPG or PNG).")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


class Predictor:
    """Loads a trained checkpoint and predicts the DR stage of one image.

    Args:
        model_path: Checkpoint written by ``src/training/train.py``.
        card_path: Optional JSON "model card" with the run name, test
            metrics and the screening threshold chosen on the validation set.
    """

    def __init__(self, model_path: str | Path, card_path: str | Path | None = None) -> None:
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = build_model(ckpt["backbone"], ckpt["num_classes"], pretrained=False, dropout=0.0)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        cfg = ckpt["config"]
        self.cfg = {"preprocessing": cfg["preprocessing"], "image": cfg["image"]}
        self.class_names = ckpt["class_names"]
        self.backbone = ckpt["backbone"]
        self.val_metrics = ckpt.get("val_metrics", {})

        self.card: dict[str, Any] = {}
        if card_path and Path(card_path).exists():
            self.card = json.loads(Path(card_path).read_text())
        self.threshold = float(self.card.get("screening_threshold", 0.5))

    def info(self) -> dict[str, Any]:
        """Model metadata for the web page."""
        return {
            "run": self.card.get("run", "unknown"),
            "backbone": self.backbone,
            "class_names": self.class_names,
            "image_size": self.cfg["image"]["size"],
            "preprocessing": self.cfg["preprocessing"],
            "screening_threshold": self.threshold,
            "val_qwk": self.val_metrics.get("qwk"),
            "test_metrics": self.card.get("test_metrics", {}),
        }

    def quality_warnings(self, img: np.ndarray) -> list[str]:
        """Warnings for images the model is likely to misjudge (EDA findings).

        Checks, in order: is there a retina at all, does the image have the
        dark border of a fundus photograph, and is it too dark, overexposed
        or blurred.
        """
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = grey.shape
        min_thr = self.cfg["preprocessing"].get("border_threshold", 4)
        retina_fraction = float((grey > retina_threshold(grey, min_thr)).mean())
        if retina_fraction < MIN_RETINA_FRACTION:
            return ["No retina found. Is this a colour fundus photograph?"]

        k = max(2, min(h, w) // 12)
        corners = [grey[:k, :k], grey[:k, -k:], grey[-k:, :k], grey[-k:, -k:]]
        if np.median([c.mean() for c in corners]) > MAX_CORNER_BRIGHTNESS:
            return ["This does not look like a fundus photograph (no dark border "
                    "around the retina). The result is not meaningful."]

        warnings = []
        q = image_quality_metrics(img, min_thr)
        if q["brightness"] < MIN_BRIGHTNESS:
            warnings.append("Image is very dark; lesions may be hidden.")
        elif q["brightness"] > MAX_BRIGHTNESS:
            warnings.append("Image is overexposed; retinal detail may be washed out.")
        if q["sharpness"] < MIN_SHARPNESS:
            warnings.append("Image looks blurred or out of focus.")
        return warnings

    def predict(self, img: np.ndarray) -> dict[str, Any]:
        """Predict the stage of one RGB uint8 fundus image.

        Returns a JSON-serialisable dict with the stage, per-class
        probabilities, the screening decision, quality warnings and two
        base64 PNG images: the model input and the Grad-CAM overlay.
        """
        t0 = time.perf_counter()
        prep = preprocess(img, self.cfg)
        x = normalize(prep, self.cfg["image"]["mean"], self.cfg["image"]["std"])
        tensor = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)

        cam, stage, probs = gradcam(self.model, tensor)  # also gives the prediction
        elapsed = (time.perf_counter() - t0) * 1000

        p_dr = float(1.0 - probs[0])
        confidence = float(probs[stage])
        warnings = self.quality_warnings(img)
        if confidence < LOW_CONFIDENCE:
            warnings.append("Low confidence: the model is unsure between stages.")

        return {
            "stage": int(stage),
            "label": self.class_names[stage],
            "confidence": round(confidence, 4),
            "probabilities": [
                {"stage": i, "label": n, "probability": round(float(p), 4)}
                for i, (n, p) in enumerate(zip(self.class_names, probs))
            ],
            "dr_probability": round(p_dr, 4),
            "screening_threshold": round(self.threshold, 4),
            "refer": bool(p_dr >= self.threshold),
            "advice": ADVICE.get(int(stage), ""),
            "warnings": warnings,
            "model_input_png": encode_png(prep),
            "gradcam_png": encode_png(overlay(prep, cam)),
            "inference_ms": round(elapsed, 1),
        }
