from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "cv_v1_normalized"
)

N_FOLDS = 5


def main():
    predictions = []

    for fold in range(1, N_FOLDS + 1):
        file_path = (
            EVALUATION_DIR
            / f"fold_{fold}"
            / "predictions.csv"
        )

        df = pd.read_csv(file_path)

        df["fold"] = fold

        predictions.append(df)

    oof = pd.concat(
        predictions,
        ignore_index=True,
    )

    if oof["img_id"].duplicated().any():
        raise ValueError(
            "Duplicate images detected across validation folds."
        )

    expected_ids = pd.read_csv(
        PROJECT_ROOT / "data" / "labeled.csv"
    )["img_id"]

    if set(oof["img_id"]) != set(expected_ids):
        raise ValueError(
            "OOF predictions do not cover the complete labeled dataset."
        )

    print(f"Total images: {len(oof)}")
    print(f"Correct predictions: {(oof['true_label'] == oof['predicted_label']).sum()}")
    print(f"Incorrect predictions: {(oof['true_label'] != oof['predicted_label']).sum()}")

    print("\nError distribution:")
    print(oof["error_type"].value_counts())

    # --------------------------------------------------
    # Slide-level error analysis
    # --------------------------------------------------

    oof["is_error"] = (
        oof["true_label"] != oof["predicted_label"]
    )

    oof["is_false_negative"] = (
        oof["error_type"] == "FN"
    )

    oof["is_false_positive"] = (
        oof["error_type"] == "FP"
    )
    
    output_file = EVALUATION_DIR / "oof_predictions.csv"

    oof.to_csv(
        output_file,
        index=False,
    )

    slide_summary = (
        oof.groupby("slide_id")
        .agg(
            total_images=("img_id", "count"),
            total_errors=("is_error", "sum"),
            false_negatives=("is_false_negative", "sum"),
            false_positives=("is_false_positive", "sum"),
        )
        .reset_index()
    )

    slide_summary["error_rate"] = (
        slide_summary["total_errors"]
        / slide_summary["total_images"]
    )

    slide_summary = slide_summary.sort_values(
        ["total_errors", "error_rate"],
        ascending=False,
    )

    slide_output = (
        EVALUATION_DIR / "slide_error_analysis.csv"
    )

    slide_summary.to_csv(
        slide_output,
        index=False,
    )

    print("\nTop 10 slides with most errors:")
    print(slide_summary.head(10).to_string(index=False))

    print(f"\nSlide analysis saved to: {slide_output}")

    # --------------------------------------------------
    # High-confidence errors
    # --------------------------------------------------

    errors = oof[oof["is_error"]].copy()

    errors["predicted_class_confidence"] = (
        errors["cancer_probability"].where(
            errors["predicted_label"] == 1,
            1 - errors["cancer_probability"],
        )
    )

    errors = errors.sort_values(
        "predicted_class_confidence",
        ascending=False,
    )

    errors_output = (
        EVALUATION_DIR / "high_confidence_errors.csv"
    )

    errors.to_csv(
        errors_output,
        index=False,
    )

    print("\nTop 10 high-confidence errors:")

    print(
        errors[
            [
                "img_id",
                "fold",
                "true_label",
                "predicted_label",
                "cancer_probability",
                "predicted_class_confidence",
                "error_type",
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nHigh-confidence errors saved to: {errors_output}")

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main() 