"""Tests for the model factory, metrics and an end-to-end training run."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.metrics import compute_metrics
from src.models.factory import (
    build_model,
    count_parameters,
    keep_frozen_batchnorm_fixed,
    set_backbone_trainable,
)
from src.utils.config import PROJECT_ROOT, load_config


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet50", "mobilenetv3"])
def test_model_output_shape(backbone):
    model = build_model(backbone, 5, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 5)


def test_freezing_leaves_only_head_trainable():
    model = build_model("efficientnet_b0", 5, pretrained=False)
    set_backbone_trainable(model, False)
    total, trainable = count_parameters(model)
    head = sum(p.numel() for p in model.get_classifier().parameters())
    assert trainable == head < total
    set_backbone_trainable(model, True)
    assert count_parameters(model)[1] == total


def test_frozen_batchnorm_stats_do_not_change():
    model = build_model("efficientnet_b0", 5, pretrained=False)
    set_backbone_trainable(model, False)
    model.train()
    keep_frozen_batchnorm_fixed(model)
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    model(torch.randn(4, 3, 64, 64) * 5 + 3)
    assert torch.equal(before, bn.running_mean)


def test_metrics_perfect_and_binary():
    y = np.array([0, 0, 1, 2, 3, 4])
    probs = np.eye(5)[y]
    m = compute_metrics(y, y, probs)
    assert m["accuracy"] == 1 and m["qwk"] == 1 and m["binary_auc"] == 1
    # predicting stage 2 instead of 1 is still correct for "any DR"
    m2 = compute_metrics(y, np.array([0, 0, 2, 2, 3, 4]))
    assert m2["binary_accuracy"] == 1 and m2["accuracy"] < 1


def test_qwk_penalises_far_mistakes_more():
    y = np.array([0, 1, 2, 3, 4] * 4)
    near = np.clip(y + 1, 0, 4)
    far = np.where(y < 2, 4, 0)
    assert compute_metrics(y, near)["qwk"] > compute_metrics(y, far)["qwk"]


@pytest.mark.parametrize("balancing", ["class_weights", "sampler"])
def test_training_end_to_end(fake_project, balancing):
    run = f"smoke_{balancing}"
    cmd = [sys.executable, "-m", "src.training.train", "--config", str(fake_project),
           "--run-name", run, "--backbone", "mobilenetv3", "--balancing", balancing,
           "--head-epochs", "1", "--finetune-epochs", "2", "--batch-size", "8",
           "--workers", "0", "--no-pretrained"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr

    cfg = load_config(fake_project)
    p = cfg["paths"]
    assert (p["models_dir"] / run / "best.pt").exists()
    assert (p["figures_dir"] / f"{run}_curves.png").exists()
    hist = pd.read_csv(p["logs_dir"] / run / "history.csv")
    assert list(hist["stage"]) == ["head", "finetune", "finetune"]
    best = json.load(open(p["metrics_dir"] / f"{run}_best_val.json"))
    assert 0 <= best["accuracy"] <= 1
    exp = pd.read_csv(p["metrics_dir"] / "experiments.csv")
    assert run in set(exp["run"])

    ckpt = torch.load(p["models_dir"] / run / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["backbone"] == "mobilenetv3" and ckpt["num_classes"] == 5
