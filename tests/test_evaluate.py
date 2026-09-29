"""Tests for Grad-CAM and test-set evaluation."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from src.evaluation.gradcam import gradcam, overlay
from src.models.factory import build_model
from src.utils.config import PROJECT_ROOT, load_config


def test_gradcam_shape_and_range():
    model = build_model("efficientnet_b0", 5, pretrained=False)
    cam, cls, probs = gradcam(model, torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1
    assert 0 <= cls < 5 and np.isclose(probs.sum(), 1, atol=1e-4)
    out = overlay(np.zeros((64, 64, 3), np.uint8), cam)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_evaluate_end_to_end(fake_project):
    run = "eval_smoke"
    base = [sys.executable, "-m"]
    train = base + ["src.training.train", "--config", str(fake_project), "--run-name", run,
                    "--backbone", "mobilenetv3", "--head-epochs", "1", "--finetune-epochs", "1",
                    "--batch-size", "8", "--workers", "0", "--no-pretrained"]
    ev = base + ["src.evaluation.evaluate", "--config", str(fake_project), "--runs", run,
                 "--workers", "0"]
    for cmd in (train, ev):
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stdout + res.stderr

    p = load_config(fake_project)["paths"]
    m = json.load(open(p["metrics_dir"] / f"{run}_test.json"))
    assert 0 <= m["accuracy"] <= 1 and m["ms_per_image_cpu"] > 0
    assert 0 <= m["screening_threshold"] <= 1 and "screening_sensitivity" in m
    for fig in ("confusion_matrix", "roc", "gradcam"):
        assert (p["figures_dir"] / f"{run}_{fig}.png").exists()
    preds = pd.read_csv(p["metrics_dir"] / f"{run}_test_predictions.csv")
    assert {"pred", "confidence", "p_0", "p_4"} <= set(preds.columns)
    assert run in set(pd.read_csv(p["metrics_dir"] / "test_results.csv")["run"])


def test_tables_merge_runs_from_different_sessions(tmp_path):
    """Per-run JSON files from separate sessions combine into one table."""
    from src.evaluation.summarize import rebuild_tables

    for run, qwk in (("a", 0.5), ("b", 0.7)):
        json.dump({"run": run, "val_qwk": qwk}, open(tmp_path / f"{run}_summary.json", "w"))
        json.dump({"run": run, "qwk": qwk, "accuracy": 0.8}, open(tmp_path / f"{run}_test.json", "w"))
    exp, test = rebuild_tables(tmp_path)
    assert set(exp["run"]) == {"a", "b"} and set(test["run"]) == {"a", "b"}
    assert (tmp_path / "experiments.csv").exists() and (tmp_path / "test_results.csv").exists()
