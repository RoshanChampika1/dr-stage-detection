"""Evaluate trained models on the held-out TEST split.

The test split is used only here, after all training and model selection
(which used the validation split) is finished.

For each run it writes:
    outputs/metrics/RUN_test.json               all metrics
    outputs/metrics/RUN_test_per_class.csv      precision / recall / F1 / support per stage
    outputs/metrics/RUN_test_predictions.csv    prediction and probabilities per image
    outputs/figures/RUN_confusion_matrix.png    counts and row-normalised
    outputs/figures/RUN_roc.png                 binary DR ROC + one-vs-rest ROC per stage
    outputs/figures/RUN_misclassified.png       most confident mistakes
    outputs/figures/RUN_gradcam.png             Grad-CAM per stage + for mistakes
and outputs/metrics/test_results.csv is rebuilt from all RUN_test.json files
(model comparison table).

Usage:
    python -m src.evaluation.evaluate --runs effb0_cw
    python -m src.evaluation.evaluate --runs effb0_cw effb0_sampler resnet50_cw
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset, build_cache
from src.data.preprocessing import load_image
from src.evaluation.gradcam import gradcam, overlay
from src.evaluation.metrics import compute_metrics
from src.evaluation.summarize import rebuild_tables
from src.models.factory import build_model
from src.utils.config import DEFAULT_CONFIG, load_config


def load_checkpoint(path, device):
    """Load a checkpoint saved by train.py and rebuild its model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], ckpt["num_classes"], pretrained=False, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict(model, loader, device):
    """Return (true labels, softmax probabilities) for a loader."""
    ys, probs = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def time_inference(model, ds, device, n=50) -> float:
    """Average milliseconds per single image (batch size 1), excluding loading."""
    xs = [ds[i][0].unsqueeze(0).to(device) for i in range(min(n, len(ds)))]
    with torch.no_grad():
        for x in xs[:5]:
            model(x)  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for x in xs:
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / len(xs)


def plot_confusion(y, pred, names, path, title):
    cm = confusion_matrix(y, pred, labels=range(len(names)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=names, yticklabels=names)
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], vmin=0, vmax=1,
                xticklabels=names, yticklabels=names)
    axes[0].set_title("Counts"); axes[1].set_title("Normalised by true stage (recall on diagonal)")
    for ax in axes:
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(fig)


def plot_roc(y, probs, names, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bt = (y > 0).astype(int)
    if len(np.unique(bt)) == 2:
        fpr, tpr, _ = roc_curve(bt, 1 - probs[:, 0])
        axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0].set_title("Binary: any DR (stage 1–4) vs No DR")
    for k, name in enumerate(names):
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            fpr, tpr, _ = roc_curve(yk, probs[:, k])
            axes[1].plot(fpr, tpr, lw=1.5, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    axes[1].set_title("One-vs-rest per stage")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(fig)


def plot_misclassified(df, ds, root, names, path, title, n=12):
    """Most confident mistakes: original image with true vs predicted stage."""
    wrong = df[df.pred != df.label].sort_values("confidence", ascending=False).head(n)
    if wrong.empty:
        return
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, wrong.iterrows()):
        ax.imshow(load_image(root / r.path))
        ax.set_title(f"true {r.label} → pred {r.pred}\nconf {r.confidence:.2f}", fontsize=9,
                     color="darkred" if abs(r.label - r.pred) > 1 else "black")
    fig.suptitle(f"{title}\nred title = error of more than one stage")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(fig)


def plot_gradcam(model, df, ds, root, names, path, title, device):
    """Row 1-2: a correct example per stage (original / Grad-CAM).
    Row 3-4: confident mistakes (original / Grad-CAM for the predicted class)."""
    correct = df[df.pred == df.label].sort_values("confidence", ascending=False).groupby("label").head(1)
    wrong = df[df.pred != df.label].sort_values("confidence", ascending=False).head(len(names))
    sets = [("Correct", correct.sort_values("label")), ("Mistake", wrong)]
    cols = len(names)
    fig, axes = plt.subplots(4, cols, figsize=(cols * 2.5, 4 * 2.6))
    for ax in axes.ravel():
        ax.axis("off")
    for block, (kind, rows) in enumerate(sets):
        for j, (_, r) in enumerate(rows.iterrows()):
            idx = int(r.name)
            x = ds[idx][0].unsqueeze(0).to(device)
            with torch.enable_grad():
                cam, cls, _ = gradcam(model, x)
            prep = ds.load_preprocessed(idx)
            axes[2 * block, j].imshow(load_image(root / r.path))
            axes[2 * block, j].set_title(f"{kind}: true {r.label}, pred {r.pred}", fontsize=9)
            axes[2 * block + 1, j].imshow(overlay(prep, cam))
            axes[2 * block + 1, j].set_title(f"Grad-CAM for '{names[cls]}'", fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(fig)


def evaluate_run(run: str, base_cfg: dict, device: torch.device, workers: int) -> dict:
    paths = base_cfg["paths"]
    model, ckpt = load_checkpoint(paths["models_dir"] / run / "best.pt", device)

    # Use this run's own preprocessing / image settings (e.g. the ablation run
    # without enhancement), with this machine's paths.
    cfg = copy.deepcopy(base_cfg)
    cfg["preprocessing"] = ckpt["config"]["preprocessing"]
    cfg["image"] = ckpt["config"]["image"]
    names = ckpt["class_names"]
    k = ckpt["num_classes"]

    test_df = pd.read_csv(paths["splits_dir"] / "test.csv", dtype={"patient_id": str})
    cache = build_cache(test_df, cfg, workers=max(workers, 1)) if cfg["training"].get("cache_images", True) else None
    ds = DRDataset(test_df, cfg, None, cache)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")

    y, probs = predict(model, loader, device)
    pred = probs.argmax(1)
    m = compute_metrics(y, pred, probs, k)
    m["ms_per_image_" + device.type] = time_inference(model, ds, device)
    if device.type == "cuda":  # also report CPU speed, relevant for deployment
        m["ms_per_image_cpu"] = time_inference(model.cpu(), ds, torch.device("cpu"), n=20)
        model.to(device)
    m.update(run=run, backbone=ckpt["backbone"], best_val_epoch=ckpt["epoch"],
             n_test=int(len(y)), params=int(sum(p.numel() for p in model.parameters())))

    out_m, out_f = paths["metrics_dir"], paths["figures_dir"]
    out_m.mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)
    with open(out_m / f"{run}_test.json", "w") as f:
        json.dump(m, f, indent=2)
    pd.DataFrame({"stage": names, "precision": m["per_class_precision"], "recall": m["per_class_recall"],
                  "f1": m["per_class_f1"], "support": m["support"]}).round(4).to_csv(
        out_m / f"{run}_test_per_class.csv", index=False)

    df = test_df.copy()
    df["pred"], df["confidence"] = pred, probs.max(1)
    for i, n in enumerate(names):
        df[f"p_{i}"] = probs[:, i].round(4)
    df.to_csv(out_m / f"{run}_test_predictions.csv", index=False)

    title = f"{run} ({ckpt['backbone']}) on the test set"
    root = paths["data_root"]
    plot_confusion(y, pred, names, out_f / f"{run}_confusion_matrix.png", title)
    plot_roc(y, probs, names, out_f / f"{run}_roc.png", title)
    plot_misclassified(df, ds, root, names, out_f / f"{run}_misclassified.png", title)
    plot_gradcam(model, df, ds, root, names, out_f / f"{run}_gradcam.png", title, device)

    print(f"{run}: acc {m['accuracy']:.3f} | macro F1 {m['macro_f1']:.3f} | QWK {m['qwk']:.3f} | "
          f"binary AUC {m.get('binary_auc', float('nan')):.3f} sens {m['binary_sensitivity']:.3f} "
          f"spec {m['binary_specificity']:.3f}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained runs on the test split")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    workers = args.workers if args.workers is not None else cfg["training"]["num_workers"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run in args.runs:
        if not (cfg["paths"]["models_dir"] / run / "best.pt").exists():
            print(f"Skipping {run}: no checkpoint")
            continue
        evaluate_run(run, cfg, device, workers)

    _, table = rebuild_tables(cfg["paths"]["metrics_dir"])
    if len(table):
        print("\nTest results (all evaluated runs):")
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
