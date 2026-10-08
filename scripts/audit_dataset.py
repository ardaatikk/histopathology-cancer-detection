from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_datasets():
    """Load the existing dataset splits."""

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    return train, val, test


def extract_slide_id(df):
    """Extract slide identifiers from image IDs."""

    df = df.copy()

    df["slide_id"] = df["img_id"].str.split("__img_").str[0]

    return df


def print_dataset_summary(name, df):
    """Print basic information about a dataset split."""

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 50}")

    print(f"Total images: {len(df)}")
    print(f"Unique images: {df['img_id'].nunique()}")
    print(f"Unique slides: {df['slide_id'].nunique()}")

    if "label" in df.columns:
        print("\nClass distribution:")
        print(df["label"].value_counts().sort_index())


def check_image_overlap(train, val, test):
    """Check whether image IDs appear in multiple splits."""

    splits = {
        "Train": set(train["img_id"]),
        "Validation": set(val["img_id"]),
        "Test": set(test["img_id"]),
    }

    print("\nIMAGE-LEVEL OVERLAP")

    names = list(splits)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first = names[i]
            second = names[j]

            overlap = splits[first] & splits[second]

            print(f"{first} vs {second}: {len(overlap)} images")


def check_slide_overlap(train, val, test):
    """Check whether slides appear in multiple splits."""

    splits = {
        "Train": set(train["slide_id"]),
        "Validation": set(val["slide_id"]),
        "Test": set(test["slide_id"]),
    }

    print("\nSLIDE-LEVEL OVERLAP")

    names = list(splits)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first = names[i]
            second = names[j]

            overlap = splits[first] & splits[second]

            print(f"{first} vs {second}: {len(overlap)} slides")


def main():

    train, val, test = load_datasets()

    train = extract_slide_id(train)
    val = extract_slide_id(val)
    test = extract_slide_id(test)

    print_dataset_summary("Train", train)
    print_dataset_summary("Validation", val)
    print_dataset_summary("Test", test)

    check_image_overlap(train, val, test)
    check_slide_overlap(train, val, test)


if __name__ == "__main__":
    main()