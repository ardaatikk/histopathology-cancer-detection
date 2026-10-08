import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_slide_id(image_id: str) -> str:
    image_id = str(image_id)

    if "__img_" not in image_id:
        raise ValueError(f"Invalid image ID format: {image_id}")

    return image_id.split("__img_", maxsplit=1)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Create slide-independent cross-validation folds."
    )

    parser.add_argument(
        "--input_file",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "folds",
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    required_columns = {"img_id", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required_columns - set(df.columns)}"
        )

    if df["img_id"].isna().any() or df["label"].isna().any():
        raise ValueError("Image IDs and labels cannot contain missing values.")

    if df["img_id"].duplicated().any():
        raise ValueError("Duplicate image IDs detected.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    df["slide_id"] = df["img_id"].apply(get_slide_id)

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(
            X=df,
            y=df["label"],
            groups=df["slide_id"],
        ),
        start=1,
    ):
        train_df = df.iloc[train_idx].copy()
        valid_df = df.iloc[valid_idx].copy()

        train_slides = set(train_df["slide_id"])
        valid_slides = set(valid_df["slide_id"])

        if train_slides.intersection(valid_slides):
            raise ValueError(
                f"Slide leakage detected in fold {fold_number}."
            )

        train_output = args.output_dir / f"fold_{fold_number}_train.csv"
        valid_output = args.output_dir / f"fold_{fold_number}_validation.csv"

        train_df.drop(columns=["slide_id"]).to_csv(
            train_output,
            index=False,
        )

        valid_df.drop(columns=["slide_id"]).to_csv(
            valid_output,
            index=False,
        )

        print(f"\nFold {fold_number}")
        print("-" * 40)

        print(
            f"Train: {len(train_df)} images, "
            f"{len(train_slides)} slides"
        )

        print(
            f"Validation: {len(valid_df)} images, "
            f"{len(valid_slides)} slides"
        )

        print(
            "Train class distribution:",
            train_df["label"].value_counts().sort_index().to_dict(),
        )

        print(
            "Validation class distribution:",
            valid_df["label"].value_counts().sort_index().to_dict(),
        )

        print("Slide overlap: 0")

    print("\nAll cross-validation folds created successfully.")


if __name__ == "__main__":
    main()