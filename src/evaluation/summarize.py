"""Rebuild the results tables from the per-run result files.

Every training run saves ``outputs/metrics/RUN_summary.json`` and every
evaluation saves ``outputs/metrics/RUN_test.json``. This module combines
them into:

    outputs/metrics/experiments.csv   validation results, one row per run
    outputs/metrics/test_results.csv  test results, one row per run

Because the tables are rebuilt from the per-run files, results from runs
done in different sessions or machines merge correctly.

Usage:
    python -m src.evaluation.summarize
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.config import load_config

TEST_COLUMNS = [
    "run", "backbone", "params", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "qwk", "binary_accuracy", "binary_sensitivity", "binary_specificity",
    "binary_auc", "ms_per_image_cuda", "ms_per_image_cpu",
]


def _collect(metrics_dir: Path, suffix: str) -> list[dict]:
    rows = []
    for f in sorted(metrics_dir.glob(f"*{suffix}")):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def rebuild_tables(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write experiments.csv and test_results.csv; return both tables."""
    metrics_dir = Path(metrics_dir)
    exp = pd.DataFrame(_collect(metrics_dir, "_summary.json"))
    if len(exp):
        exp.to_csv(metrics_dir / "experiments.csv", index=False)

    test = pd.DataFrame(_collect(metrics_dir, "_test.json"))
    if len(test):
        test = test[[c for c in TEST_COLUMNS if c in test.columns]]
        test.round(4).to_csv(metrics_dir / "test_results.csv", index=False)
    return exp, test


def main() -> None:
    cfg = load_config()
    exp, test = rebuild_tables(cfg["paths"]["metrics_dir"])
    pd.set_option("display.width", 200)
    print("Validation (experiments.csv):")
    print(exp.to_string(index=False) if len(exp) else "  none")
    print("\nTest (test_results.csv):")
    print(test.round(3).to_string(index=False) if len(test) else "  none")


if __name__ == "__main__":
    main()
