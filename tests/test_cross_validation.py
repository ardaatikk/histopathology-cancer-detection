import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FOLDS_DIR = DATA_DIR / "folds"

N_FOLDS = 5
REQUIRED_COLUMNS = {"img_id", "label"}


def load_csv(path: Path) -> pd.DataFrame:
    assert path.exists(), f"Missing file: {path}"

    df = pd.read_csv(path)

    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert not df[list(REQUIRED_COLUMNS)].isna().any().any()

    return df


def get_slide_ids(df: pd.DataFrame) -> set[str]:
    assert df["img_id"].str.match(
        r"^slide_[^_]+__img_.+$"
    ).all(), "Invalid image ID format"

    return set(
        df["img_id"].str.split("__img_").str[0]
    )


def load_fold(fold: int):
    train = load_csv(
        FOLDS_DIR / f"fold_{fold}_train.csv"
    )

    validation = load_csv(
        FOLDS_DIR / f"fold_{fold}_validation.csv"
    )

    return train, validation


@pytest.fixture(scope="module")
def labeled_data():
    return load_csv(DATA_DIR / "labeled.csv")


@pytest.mark.parametrize("fold", range(1, N_FOLDS + 1))
def test_no_slide_leakage(fold):
    train, validation = load_fold(fold)

    train_slides = get_slide_ids(train)
    validation_slides = get_slide_ids(validation)

    overlap = train_slides & validation_slides

    assert not overlap, (
        f"Fold {fold}: slide leakage detected: {overlap}"
    )


@pytest.mark.parametrize("fold", range(1, N_FOLDS + 1))
def test_no_duplicate_images(fold):
    train, validation = load_fold(fold)

    assert train["img_id"].is_unique
    assert validation["img_id"].is_unique

    overlap = set(train["img_id"]) & set(validation["img_id"])

    assert not overlap, (
        f"Fold {fold}: overlapping images detected"
    )


@pytest.mark.parametrize("fold", range(1, N_FOLDS + 1))
def test_fold_covers_labeled_dataset(fold, labeled_data):
    train, validation = load_fold(fold)

    combined = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    expected = labeled_data.set_index("img_id")["label"]
    actual = combined.set_index("img_id")["label"]

    assert expected.index.is_unique
    assert actual.index.is_unique

    pd.testing.assert_series_equal(
        actual.sort_index(),
        expected.sort_index(),
        check_names=False,
    )


def test_validation_coverage_across_folds(labeled_data):
    validation_ids = []

    for fold in range(1, N_FOLDS + 1):
        _, validation = load_fold(fold)

        validation_ids.extend(
            validation["img_id"].tolist()
        )

    assert len(validation_ids) == len(labeled_data)

    assert len(validation_ids) == len(set(validation_ids)), (
        "Some images appear in multiple validation folds"
    )

    assert set(validation_ids) == set(labeled_data["img_id"])


@pytest.mark.parametrize("fold", range(1, N_FOLDS + 1))
def test_fold_rgb_statistics(fold):
    path = FOLDS_DIR / f"fold_{fold}_rgb_stats.json"

    assert path.exists(), f"Missing RGB statistics: {path}"

    with path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    # Supports either naming convention.
    mean = stats.get("mean", stats.get("rgb_mean"))
    std = stats.get("std", stats.get("rgb_std"))

    assert mean is not None, "Missing RGB mean"
    assert std is not None, "Missing RGB std"

    assert len(mean) == 3
    assert len(std) == 3

    assert all(0 <= value <= 1 for value in mean)
    assert all(0 < value <= 1 for value in std)