import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a labeled dataset into training and validation CSV files."
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="Input labeled CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        default="./data",
        help="Directory for train and validation CSV files.",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.25,
        help="Fraction of samples assigned to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data = pd.read_csv(args.data_file)

    if "label" not in data.columns:
        raise ValueError(
            "Input CSV must contain a 'label' column."
        )

    train_data, validation_data = train_test_split(
        data,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
        stratify=data["label"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    validation_path = output_dir / "validation.csv"

    train_data.to_csv(train_path, index=False)
    validation_data.to_csv(validation_path, index=False)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Training CSV saved to: {train_path}")
    print(f"Validation CSV saved to: {validation_path}")


if __name__ == "__main__":
    main()