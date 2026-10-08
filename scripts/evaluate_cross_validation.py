import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_NAMES = [
    "Accuracy",
    "Precision",
    "Sensitivity (Recall)",
    "Specificity",
    "F1 Score",
    "Balanced Accuracy",
    "ROC-AUC",
    "Average Precision (AP)",
]

COUNT_NAMES = ["TN", "FP", "FN", "TP"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all cross-validation folds."
    )

    parser.add_argument(
        "--experiment_prefix",
        default="cv_v1_normalized",
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
    )

    return parser.parse_args()


def read_metrics(metrics_path):
    content = metrics_path.read_text(encoding="utf-8")

    results = {}

    for name in METRIC_NAMES:
        pattern = rf"^{re.escape(name)}:\s*([0-9.]+)$"
        match = re.search(pattern, content, re.MULTILINE)

        if match is None:
            raise ValueError(
                f"Metric '{name}' not found in {metrics_path}"
            )

        results[name] = float(match.group(1))

    for name in COUNT_NAMES:
        pattern = rf"^{name}:\s*(\d+)$"
        match = re.search(pattern, content, re.MULTILINE)

        if match is None:
            raise ValueError(
                f"Count '{name}' not found in {metrics_path}"
            )

        results[name] = int(match.group(1))

    return results


def main():
    args = parse_args()

    output_root = (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / args.experiment_prefix
    )

    output_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for fold in range(1, args.n_splits + 1):
        experiment_name = f"{args.experiment_prefix}_fold_{fold}"

        validation_file = (
            PROJECT_ROOT
            / "data"
            / "folds"
            / f"fold_{fold}_validation.csv"
        )

        checkpoint_file = (
            PROJECT_ROOT
            / "checkpoints"
            / experiment_name
            / "best_checkpoint.pt"
        )

        fold_output = output_root / f"fold_{fold}"

        if not validation_file.is_file():
            raise FileNotFoundError(validation_file)

        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)

        print(f"\nEvaluating Fold {fold}/{args.n_splits}", flush=True)

        command = [
            sys.executable,
            "src/evaluate.py",
            "--validation_file",
            str(validation_file),
            "--model_file",
            str(checkpoint_file),
            "--output_dir",
            str(fold_output),
        ]

        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        metrics = read_metrics(fold_output / "metrics.txt")

        rows.append({
            "Fold": fold,
            **metrics,
        })

    # --------------------------------------------------
    # Save individual fold results
    # --------------------------------------------------

    csv_path = output_root / "cv_results.csv"

    fieldnames = ["Fold", *METRIC_NAMES, *COUNT_NAMES]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------

    summary_path = output_root / "cv_summary.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("5-Fold Slide-Independent Cross-Validation\n")
        file.write("=" * 50 + "\n\n")

        for name in METRIC_NAMES:
            values = np.array(
                [row[name] for row in rows],
                dtype=float,
            )

            mean = np.mean(values)
            std = np.std(values, ddof=1)

            line = f"{name:<25}: {mean:.4f} ± {std:.4f}"

            print(line)
            file.write(line + "\n")

        file.write("\nAggregated Confusion Matrix\n")

        for name in COUNT_NAMES:
            total = sum(row[name] for row in rows)
            file.write(f"{name}: {total}\n")

    print(f"\nResults saved to: {output_root}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()