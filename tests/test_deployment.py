"""Tests for the ONNX export and the browser (JavaScript) preprocessing."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.preprocessing import preprocess
from src.deployment.export_onnx import export, verify
from src.models.factory import build_model
from src.utils.config import PROJECT_ROOT, load_config


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    cfg = load_config()
    cfg["preprocessing"].update(denoise="none", clahe=False, ben_graham=False)
    model = build_model("efficientnet_b0", 5, pretrained=False)
    with torch.no_grad():  # non-trivial head so the checks are meaningful
        g = torch.Generator().manual_seed(0)
        model.get_classifier().weight.copy_(torch.randn(5, 1280, generator=g) * 0.05)
    path = tmp / "best.pt"
    torch.save({"model_state": model.state_dict(), "backbone": "efficientnet_b0", "num_classes": 5,
                "class_names": cfg["data"]["class_names"],
                "config": {"preprocessing": cfg["preprocessing"], "image": cfg["image"]}}, path)
    return path


def test_onnx_matches_pytorch_and_cam_equals_gradcam(checkpoint, tmp_path):
    onnx_path = export(checkpoint, tmp_path, None)
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    assert np.array(meta["classifier_weight"]).shape == (5, 1280)
    assert meta["image_size"] == 224 and len(meta["class_names"]) == 5
    r = verify(checkpoint, onnx_path, n=2)
    assert r["max_probability_difference"] < 1e-5
    assert r["max_cam_difference"] < 1e-4  # CAM from features + weights == Grad-CAM


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_preprocessing_matches_python(tmp_path):
    cfg = load_config()
    cfg["preprocessing"].update(denoise="none", clahe=False, ben_graham=False)
    rng = np.random.default_rng(0)
    cases = []
    for h, w in [(224, 224), (200, 260), (700, 560), (1100, 1300)]:  # enlarge and shrink paths
        img = np.zeros((h, w, 3), np.uint8)
        cv2.ellipse(img, (w // 2, h // 2), (int(w * 0.42), int(h * 0.45)), 0, 0, 360, (170, 80, 40), -1)
        img = cv2.add(img, rng.integers(0, 40, img.shape, dtype=np.uint8))
        cases.append(img)
    for i, img in enumerate(cases):
        h, w = img.shape[:2]
        np.dstack([img, np.full((h, w), 255, np.uint8)]).tofile(tmp_path / f"in_{i}.raw")
    script = f"""
const fs = require('fs');
eval(fs.readFileSync({json.dumps(str(PROJECT_ROOT / 'web' / 'preprocess.js'))}, 'utf8') + ';global.DRPre=DRPre;');
const sizes = {json.dumps([[c.shape[1], c.shape[0]] for c in cases])};
const meta = {{preprocessing: {json.dumps(cfg['preprocessing'])}, image_size: 224}};
sizes.forEach(([w, h], i) => {{
  const d = new Uint8ClampedArray(fs.readFileSync({json.dumps(str(tmp_path))} + '/in_' + i + '.raw'));
  const out = DRPre.preprocess({{width: w, height: h, data: d}}, meta);
  fs.writeFileSync({json.dumps(str(tmp_path))} + '/js_' + i + '.raw', Buffer.from(out.data));
}});
"""
    (tmp_path / "run.js").write_text(script)
    subprocess.run(["node", str(tmp_path / "run.js")], check=True, timeout=120)
    for i, img in enumerate(cases):
        py = preprocess(img, cfg).astype(int)
        js = np.fromfile(tmp_path / f"js_{i}.raw", np.uint8).reshape(224, 224, 4)[:, :, :3].astype(int)
        diff = np.abs(py - js)
        # OpenCV uses fixed-point weights for 8-bit images and the browser uses
        # floating point, so single grey levels may differ; nothing more.
        assert diff.max() <= 1 and diff.mean() < 0.2, (i, diff.max(), diff.mean())
