import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run slide-independent cross-validation."
    )

    parser.add_argument(
        "--experiment_prefix",
        default="cv_v1_normalized",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--use_scheduler",
        action="store_true",
    )

    return parser.parse_args()


def run_command(command):
    print("\n" + "=" * 60, flush=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    print("=" * 60, flush=True)

    subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        check=True,
    )


def validate_rgb_stats(stats_path, train_csv):
    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    expected_count = len(pd.read_csv(train_csv))

    if stats["image_count"] != expected_count:
        raise ValueError(
            f"RGB statistics image count mismatch: {stats_path}"
        )

    recorded_csv = Path(stats["train_csv"])

    if not recorded_csv.is_absolute():
        recorded_csv = PROJECT_ROOT / recorded_csv

    if recorded_csv.resolve() != train_csv.resolve():
        raise ValueError(
            f"RGB statistics were calculated from another CSV: {stats_path}"
        )


def main():
    args = parse_args()

    folds_dir = PROJECT_ROOT / "data" / "folds"

    for fold in range(1, args.n_splits + 1):
        experiment_name = f"{args.experiment_prefix}_fold_{fold}"

        train_csv = folds_dir / f"fold_{fold}_train.csv"
        valid_csv = folds_dir / f"fold_{fold}_validation.csv"
        stats_json = folds_dir / f"fold_{fold}_rgb_stats.json"

        checkpoint_dir = (
            PROJECT_ROOT / "checkpoints" / experiment_name
        )

        best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
        last_checkpoint = checkpoint_dir / "last_checkpoint.pt"

        print(f"\n{'#' * 60}", flush=True)
        print(f"FOLD {fold}/{args.n_splits}", flush=True)
        print(f"Experiment: {experiment_name}", flush=True)
        print(f"{'#' * 60}", flush=True)

        if not train_csv.is_file() or not valid_csv.is_file():
            raise FileNotFoundError(
                f"Missing training or validation CSV for fold {fold}"
            )

        # Calculate RGB statistics using training images only.
        if not stats_json.is_file():
            run_command(
                [
                    sys.executable,
                    "scripts/calculate_rgb_stats.py",
                    "--train_csv",
                    train_csv,
                    "--output_json",
                    stats_json,
                ]
            )

        validate_rgb_stats(stats_json, train_csv)

        # Do not silently overwrite an existing experiment.
        if best_checkpoint.exists() or last_checkpoint.exists():
            print(
                f"Skipping fold {fold}: checkpoints already exist. "
                "This does not verify that training completed.",
                flush=True,
            )
            continue

        command = [
            sys.executable,
            "src/train.py",
            "--experiment_name",
            experiment_name,
            "--train_file",
            train_csv,
            "--valid_file",
            valid_csv,
            "--rgb_stats",
            stats_json,
            "--num_epochs",
            args.num_epochs,
            "--batch_size",
            args.batch_size,
            "--learning_rate",
            args.learning_rate,
            "--momentum",
            args.momentum,
            "--weight_decay",
            args.weight_decay,
            "--seed",
            args.seed,
        ]

        if args.use_scheduler:
            command.append("--use_scheduler")

        run_command(command)

    print("\nCross-validation training loop finished.", flush=True)


if __name__ == "__main__":
    main()