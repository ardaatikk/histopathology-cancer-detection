import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create slide-independent training and validation splits."
    )

    parser.add_argument(
        "--data_file",
        required=True,
        help="CSV containing img_id and label columns.",
    )

    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "data"),
        help="Directory for output CSV files.",
    )

    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.25,
        help="Target validation fraction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def extract_slide_ids(data):
    """Extract slide IDs from image identifiers."""

    return data["img_id"].str.split("__img_").str[0]


def create_grouped_split(data, validation_size, seed):
    """Create a slide-independent split with approximate stratification."""

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    groups = extract_slide_ids(data)

    n_splits = round(1 / validation_size)

    if n_splits < 2:
        raise ValueError("At least two folds are required.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    overall_positive_rate = data["label"].mean()

    best_split = None
    best_score = float("inf")

    for train_idx, val_idx in splitter.split(
        data,
        y=data["label"],
        groups=groups,
    ):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        size_difference = abs(
            len(val_data) / len(data) - validation_size
        )

        class_difference = abs(
            val_data["label"].mean() - overall_positive_rate
        )

        score = size_difference + class_difference

        if score < best_score:
            best_score = score
            best_split = (train_data.copy(), val_data.copy())

    return best_split


def validate_split(train_data, val_data):
    """Verify that images and slides do not overlap."""

    train_images = set(train_data["img_id"])
    val_images = set(val_data["img_id"])

    train_slides = set(extract_slide_ids(train_data))
    val_slides = set(extract_slide_ids(val_data))

    if train_images & val_images:
        raise ValueError("Image-level overlap detected.")

    if train_slides & val_slides:
        raise ValueError("Slide-level overlap detected.")

    print("\nSplit validation: PASSED")
    print("Image overlap: 0")
    print("Slide overlap: 0")


def main():
    args = parse_args()

    data = pd.read_csv(args.data_file)

    required_columns = {"img_id", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Input CSV must contain: {required_columns}"
        )

    if data["img_id"].isna().any() or data["label"].isna().any():
        raise ValueError("Missing image IDs or labels detected.")

    if data["img_id"].duplicated().any():
        raise ValueError("Duplicate image IDs detected.")

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    if not data["img_id"].str.contains(
        r"^slide_.+__img_.+$", regex=True
    ).all():
        raise ValueError("Unexpected image ID format.")

    train_data, val_data = create_grouped_split(
        data=data,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    validate_split(train_data, val_data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)

    print("\nDATASET SPLIT SUMMARY")
    print("-" * 40)

    for name, subset in [
        ("Train", train_data),
        ("Validation", val_data),
    ]:
        print(f"\n{name}:")
        print(f"  Images: {len(subset)}")
        print(f"  Slides: {extract_slide_ids(subset).nunique()}")
        print(f"  Positive rate: {subset['label'].mean():.4f}")
        print(f"  Class distribution: {subset['label'].value_counts().to_dict()}")

    print(f"\nTraining CSV: {train_path}")
    print(f"Validation CSV: {val_path}")


if __name__ == "__main__":
    main()