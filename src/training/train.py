"""Train a DR stage classifier with two-stage transfer learning.

Stage 1 ("head"):     backbone frozen, train the new classification head.
Stage 2 ("finetune"): all layers trainable, lower learning rate, LR schedule
                      and early stopping.

Every epoch is validated. The checkpoint with the best validation QWK is
kept. Outputs for a run named RUN:

    models/RUN/best.pt                    best checkpoint (weights + config)
    outputs/logs/RUN/history.csv          per-epoch metrics
    outputs/logs/RUN/config.yaml          exact settings used
    outputs/metrics/RUN_best_val.json     validation metrics of the best epoch
    outputs/metrics/experiments.csv       one summary row per run (results table)
    outputs/figures/RUN_curves.png        accuracy / loss / QWK curves

Examples:
    python -m src.training.train --run-name effb0_cw
    python -m src.training.train --run-name effb0_sampler --balancing sampler
    python -m src.training.train --run-name resnet50_cw --backbone resnet50
    python -m src.training.train --run-name effb0_raw --no-enhancement
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.augmentation import get_train_transforms
from src.data.dataset import DRDataset, build_cache, class_weights, make_sampler
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_history
from src.models.factory import (
    build_model,
    count_parameters,
    keep_frozen_batchnorm_fixed,
    set_backbone_trainable,
)
from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a DR stage classifier")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--run-name", required=True, help="Name for this experiment's outputs")
    ap.add_argument("--backbone", help="efficientnet_b0 | resnet50 | mobilenetv3")
    ap.add_argument("--balancing", choices=["class_weights", "sampler", "none"])
    ap.add_argument("--balance-power", type=float)
    ap.add_argument("--head-epochs", type=int)
    ap.add_argument("--finetune-epochs", type=int)
    ap.add_argument("--head-lr", type=float)
    ap.add_argument("--finetune-lr", type=float)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"])
    ap.add_argument("--scheduler", choices=["cosine", "plateau"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--subset", type=float, default=1.0,
                    help="Fraction of the training set to use (quick tuning runs)")
    ap.add_argument("--no-enhancement", action="store_true",
                    help="Ablation: switch off denoising, CLAHE and Ben Graham (crop + resize only)")
    ap.add_argument("--no-pretrained", action="store_true", help="Random init (tests / ablation)")
    return ap.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Copy command-line overrides into the config."""
    t = cfg["training"]
    mapping = {
        "backbone": "backbone", "balancing": "balancing", "balance_power": "balance_power",
        "head_epochs": "head_epochs", "finetune_epochs": "finetune_epochs",
        "head_lr": "head_lr", "finetune_lr": "finetune_lr", "dropout": "dropout",
        "batch_size": "batch_size", "optimizer": "optimizer", "scheduler": "scheduler",
        "workers": "num_workers",
    }
    for arg, key in mapping.items():
        value = getattr(args, arg)
        if value is not None:
            t[key] = value
    if args.no_pretrained:
        t["pretrained"] = False
    if args.no_enhancement:
        p = cfg["preprocessing"]
        p["denoise"], p["clahe"], p["ben_graham"] = "none", False, False
    return cfg


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    """One pass over ``loader``. Trains if an optimizer is given, else evaluates.

    Returns:
        (mean loss, true labels, predicted labels, softmax probabilities)
    """
    training = optimizer is not None
    model.train(training)
    if training:
        keep_frozen_batchnorm_fixed(model)
    total_loss, n = 0.0, 0
    ys, preds, probs = [], [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            n += len(y)
            p = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(1))
            ys.append(y.cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    t, paths = cfg["training"], cfg["paths"]
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t["amp"]) and device.type == "cuda"
    run = args.run_name
    print(f"Run '{run}' on {device} | backbone={t['backbone']} balancing={t['balancing']} amp={amp}")

    # ---- Data -----------------------------------------------------------
    read = lambda s: pd.read_csv(paths["splits_dir"] / f"{s}.csv", dtype={"patient_id": str})
    train_df, val_df = read("train"), read("val")
    if args.subset < 1.0:
        train_df = train_df.groupby("label", group_keys=False).sample(
            frac=args.subset, random_state=cfg["seed"])
        print(f"Using a {args.subset:.0%} stratified subset: {len(train_df)} training images")

    cache = None
    if t.get("cache_images", True):
        cache = build_cache(pd.concat([train_df, val_df]), cfg, workers=max(t["num_workers"], 1))

    train_ds = DRDataset(train_df, cfg, get_train_transforms(cfg), cache)
    val_ds = DRDataset(val_df, cfg, None, cache)
    labels = train_df["label"].to_numpy()
    k = cfg["data"]["num_classes"]

    sampler = make_sampler(labels, k, t["balance_power"]) if t["balancing"] == "sampler" else None
    loader_kw = dict(batch_size=t["batch_size"], num_workers=t["num_workers"],
                     pin_memory=device.type == "cuda", persistent_workers=t["num_workers"] > 0)
    train_dl = DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    weight = None
    if t["balancing"] == "class_weights":
        w = class_weights(labels, k, t["balance_power"])
        print("Class weights:", np.round(w, 3).tolist())
        weight = torch.tensor(w, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t["label_smoothing"])

    # ---- Model ----------------------------------------------------------
    model = build_model(t["backbone"], k, t["pretrained"], t["dropout"]).to(device)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
    except (AttributeError, TypeError):  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ---- Output locations -----------------------------------------------
    model_dir = paths["models_dir"] / run
    log_dir = paths["logs_dir"] / run
    for d in (model_dir, log_dir, paths["metrics_dir"], paths["figures_dir"]):
        d.mkdir(parents=True, exist_ok=True)
    cfg_dump = copy.deepcopy(cfg)
    cfg_dump["paths"] = {k_: str(v) for k_, v in cfg["paths"].items()}
    with open(log_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dump, f, sort_keys=False)

    history, best, best_metrics, stale = [], -np.inf, None, 0
    monitor = t["monitor"].replace("val_", "")
    start = time.time()

    stages = [("head", t["head_epochs"], t["head_lr"], False),
              ("finetune", t["finetune_epochs"], t["finetune_lr"], True)]
    epoch = 0
    for stage, n_epochs, lr, unfreeze in stages:
        if n_epochs <= 0:
            continue
        set_backbone_trainable(model, unfreeze)
        total, trainable = count_parameters(model)
        print(f"\n== Stage '{stage}': {n_epochs} epochs, lr={lr}, "
              f"trainable params {trainable:,} / {total:,}")
        opt = make_optimizer(model, t["optimizer"], lr, t["weight_decay"])
        if t["scheduler"] == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
        stale = 0

        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tr_loss, ty, tp, _ = run_epoch(model, train_dl, criterion, device, opt, scaler, amp)
            va_loss, vy, vp, vprob = run_epoch(model, val_dl, criterion, device, amp=amp)
            m = compute_metrics(vy, vp, vprob, k)
            score = m[monitor]
            current_lr = opt.param_groups[0]["lr"]
            if t["scheduler"] == "plateau":
                sched.step(score)
            else:
                sched.step()

            row = {"epoch": epoch, "stage": stage, "lr": current_lr,
                   "train_loss": tr_loss, "train_acc": float((ty == tp).mean()),
                   "val_loss": va_loss, "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
                   "val_qwk": m["qwk"], "val_binary_auc": m.get("binary_auc", np.nan),
                   "seconds": time.time() - t0}
            history.append(row)
            pd.DataFrame(history).to_csv(log_dir / "history.csv", index=False)

            improved = score > best
            if improved:
                best, best_metrics, stale = score, {**m, "epoch": epoch, "stage": stage}, 0
                torch.save({"model_state": model.state_dict(), "backbone": t["backbone"],
                            "num_classes": k, "class_names": cfg["data"]["class_names"],
                            "config": cfg_dump, "epoch": epoch, "val_metrics": best_metrics},
                           model_dir / "best.pt")
            else:
                stale += 1
            print(f"ep {epoch:2d} [{stage}] lr {current_lr:.1e} | train loss {tr_loss:.4f} "
                  f"acc {row['train_acc']:.3f} | val loss {va_loss:.4f} acc {m['accuracy']:.3f} "
                  f"F1 {m['macro_f1']:.3f} QWK {m['qwk']:.3f} | {row['seconds']:.0f}s"
                  + ("  * best" if improved else ""))

            # Early stopping only in the fine-tuning stage.
            if unfreeze and stale >= t["early_stopping_patience"]:
                print(f"Early stopping: no {t['monitor']} improvement for {stale} epochs")
                break

    minutes = (time.time() - start) / 60
    hist = pd.DataFrame(history)
    plot_history(hist, paths["figures_dir"] / f"{run}_curves.png",
                 f"{run}: {t['backbone']}, balancing={t['balancing']}")
    with open(paths["metrics_dir"] / f"{run}_best_val.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    summary = {
        "run": run, "backbone": t["backbone"], "balancing": t["balancing"],
        "balance_power": t["balance_power"], "enhancement": not args.no_enhancement,
        "pretrained": t["pretrained"], "finetune_lr": t["finetune_lr"], "dropout": t["dropout"],
        "batch_size": t["batch_size"], "subset": args.subset, "epochs_run": epoch,
        "best_epoch": best_metrics["epoch"], "val_acc": round(best_metrics["accuracy"], 4),
        "val_macro_f1": round(best_metrics["macro_f1"], 4), "val_qwk": round(best_metrics["qwk"], 4),
        "val_binary_auc": round(best_metrics.get("binary_auc", float("nan")), 4),
        "minutes": round(minutes, 1),
    }
    exp_file = paths["metrics_dir"] / "experiments.csv"
    exp = pd.read_csv(exp_file) if exp_file.exists() else pd.DataFrame()
    exp = pd.concat([exp[exp.get("run", pd.Series(dtype=str)) != run] if len(exp) else exp,
                     pd.DataFrame([summary])], ignore_index=True)
    exp.to_csv(exp_file, index=False)

    print(f"\nDone in {minutes:.1f} min. Best epoch {best_metrics['epoch']}: "
          f"val acc {best_metrics['accuracy']:.3f}, macro F1 {best_metrics['macro_f1']:.3f}, "
          f"QWK {best_metrics['qwk']:.3f}")
    print(f"Checkpoint: {model_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
