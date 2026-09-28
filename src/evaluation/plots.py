"""Plots for training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history: pd.DataFrame, out_path: Path, title: str = "") -> None:
    """Accuracy, loss and QWK curves (train vs validation) per epoch.

    A dashed vertical line marks where the backbone was unfrozen
    (end of the head-only stage).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = history["epoch"]
    for ax, key, name in [(axes[0], "acc", "Accuracy"), (axes[1], "loss", "Loss")]:
        ax.plot(ep, history[f"train_{key}"], marker="o", ms=3, label="Train")
        ax.plot(ep, history[f"val_{key}"], marker="o", ms=3, label="Validation")
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    axes[2].plot(ep, history["val_qwk"], marker="o", ms=3, color="tab:green", label="Val QWK")
    axes[2].plot(ep, history["val_macro_f1"], marker="o", ms=3, color="tab:purple", label="Val macro F1")
    axes[2].set_title("Validation QWK and macro F1"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    stage1_end = history.loc[history["stage"] == "head", "epoch"].max()
    if pd.notna(stage1_end):
        for ax in axes:
            ax.axvline(stage1_end + 0.5, ls="--", color="grey", lw=1)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
