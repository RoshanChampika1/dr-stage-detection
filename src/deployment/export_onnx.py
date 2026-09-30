"""Export a trained checkpoint to ONNX for in-browser inference.

The web page runs the model with ONNX Runtime Web, so the photograph never
leaves the user's device and no server is needed. Two files are written:

    web/model/dr_model.onnx      network: image tensor -> (probabilities, feature maps)
    web/model/model_meta.json    classifier weights, class names, preprocessing,
                                 normalisation, screening threshold, test metrics

Why the feature maps and classifier weights are exported (Grad-CAM without
gradients): the timm CNNs used here end in global average pooling followed by
one linear layer, logit_c = b_c + sum_k W_ck * mean_hw F_k(h, w). The gradient
of logit_c with respect to F_k(h, w) is W_ck / (H * W) at every position, so
the Grad-CAM channel weights (spatial mean of the gradient) are W_ck / (H * W)
and the Grad-CAM map ReLU(sum_k W_ck F_k) / (H * W) is, after normalisation to
[0, 1], exactly the Class Activation Map computed from the feature maps and the
classifier weights. The browser can therefore compute the same heatmap with
one multiply-add per channel and no backward pass. ``verify`` checks this
equivalence numerically.

Usage:
    python -m src.deployment.export_onnx --checkpoint models/effb0_raw_lr3e-4/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.gradcam import gradcam
from src.models.factory import build_model
from src.utils.config import PROJECT_ROOT


class ExportWrapper(nn.Module):
    """Returns softmax probabilities and the last convolutional feature maps."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.model.forward_features(x)
        logits = self.model.forward_head(feats)
        return torch.softmax(logits, dim=1), feats


def load_model(checkpoint: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["backbone"], ckpt["num_classes"], pretrained=False, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), ckpt


def cam_from_features(feats: np.ndarray, weight: np.ndarray, cls: int) -> np.ndarray:
    """Class Activation Map (7x7), normalised to [0, 1] (the browser computes the same)."""
    cam = np.maximum(np.tensordot(weight[cls], feats, axes=(0, 0)), 0)
    cam -= cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def export(checkpoint: Path, out_dir: Path, card_path: Path | None) -> Path:
    model, ckpt = load_model(checkpoint)
    size = ckpt["config"]["image"]["size"]
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "dr_model.onnx"

    torch.onnx.export(
        ExportWrapper(model), torch.randn(1, 3, size, size), str(onnx_path),
        input_names=["image"], output_names=["probabilities", "features"],
        opset_version=17, dynamo=False,
    )

    head = model.get_classifier()
    card = json.loads(card_path.read_text()) if card_path and card_path.exists() else {}
    meta = {
        "run": card.get("run", checkpoint.parent.name),
        "backbone": ckpt["backbone"],
        "class_names": ckpt["class_names"],
        "image_size": size,
        "mean": ckpt["config"]["image"]["mean"],
        "std": ckpt["config"]["image"]["std"],
        "preprocessing": ckpt["config"]["preprocessing"],
        "screening_threshold": card.get("screening_threshold", 0.5),
        "test_metrics": card.get("test_metrics", {}),
        "classifier_weight": np.round(head.weight.detach().numpy(), 6).tolist(),
        "classifier_bias": np.round(head.bias.detach().numpy(), 6).tolist(),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta))
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) and model_meta.json")
    return onnx_path


def verify(checkpoint: Path, onnx_path: Path, n: int = 4) -> dict:
    """Check ONNX vs PyTorch outputs and CAM vs Grad-CAM on random inputs."""
    import cv2
    import onnxruntime as ort

    model, ckpt = load_model(checkpoint)
    size = ckpt["config"]["image"]["size"]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    weight = model.get_classifier().weight.detach().numpy()
    max_prob_diff, max_cam_diff = 0.0, 0.0
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = torch.from_numpy(rng.normal(0, 1, (1, 3, size, size)).astype(np.float32))
        probs_onnx, feats = sess.run(None, {"image": x.numpy()})
        with torch.no_grad():
            probs_torch = torch.softmax(model(x), 1).numpy()
        max_prob_diff = max(max_prob_diff, float(np.abs(probs_onnx - probs_torch).max()))

        grad_cam, cls, _ = gradcam(model, x)
        cam = cam_from_features(feats[0], weight, cls)
        cam = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-12)
        max_cam_diff = max(max_cam_diff, float(np.abs(cam - grad_cam).max()))
    result = {"max_probability_difference": max_prob_diff, "max_cam_difference": max_cam_diff}
    print(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a checkpoint to ONNX for the web page")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "web" / "model"))
    ap.add_argument("--card", default=str(PROJECT_ROOT / "api" / "model_card.json"))
    args = ap.parse_args()
    onnx_path = export(Path(args.checkpoint), Path(args.out_dir), Path(args.card))
    verify(Path(args.checkpoint), onnx_path)


if __name__ == "__main__":
    main()
