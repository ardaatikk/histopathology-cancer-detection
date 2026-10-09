import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.gradcam import (
    GradCAM,
    build_transform,
    get_image_path,
    load_model,
    predict,
    prepare_image,
)

from src.integrated_gradients import compute_integrated_gradients


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OOF_FILE = (
    PROJECT_ROOT
    / "outputs/evaluation/cv_v1_normalized/oof_predictions.csv"
)

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/xai_comparison"

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images"


EXAMPLE_IDS = [
    "slide_ZHW1__img_54SW",
    "slide_8IXU__img_LRXP",
    "slide_M9QC__img_OMD0",
    "slide_Z7EQ__img_Q75L",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Grad-CAM and Integrated Gradients."
    )

    parser.add_argument(
        "--oof_file",
        type=Path,
        default=DEFAULT_OOF_FILE,
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--image_dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--all_errors",
        action="store_true",
        help="Analyze all false positive and false negative cases.",
    )

    return parser.parse_args()


def attribution_to_map(attributions):
    return (
        attributions
        .detach()
        .cpu()
        .squeeze(0)
        .sum(dim=0)
        .numpy()
    )


def save_comparison(
    original_image,
    cams,
    attributions,
    true_label,
    prediction,
    probability,
    fold,
    error_type,
    output_path,
):
    original = np.asarray(original_image)

    ig_0 = attribution_to_map(attributions[0])
    ig_1 = attribution_to_map(attributions[1])

    ig_scale = np.percentile(
        np.abs(
            np.concatenate([ig_0.ravel(), ig_1.ravel()])
        ),
        99,
    )

    ig_scale = max(float(ig_scale), 1e-12)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(original)
    axes[0, 1].imshow(
        cams[0],
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[0, 1].set_title("Grad-CAM — Class 0")

    axes[0, 2].imshow(original)
    axes[0, 2].imshow(
        cams[1],
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[0, 2].set_title("Grad-CAM — Class 1")

    axes[1, 0].imshow(
        ig_0,
        cmap="bwr",
        vmin=-ig_scale,
        vmax=ig_scale,
    )
    axes[1, 0].set_title("Integrated Gradients — Class 0")

    axes[1, 1].imshow(
        ig_1,
        cmap="bwr",
        vmin=-ig_scale,
        vmax=ig_scale,
    )
    axes[1, 1].set_title("Integrated Gradients — Class 1")

    axes[1, 2].imshow(original)
    axes[1, 2].imshow(
        cams[prediction],
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[1, 2].set_title("Predicted Class — Grad-CAM")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"{error_type} | Fold {fold} | "
        f"True: {true_label} | Predicted: {prediction} | "
        f"Cancer probability: {probability:.4f}",
        fontsize=13,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def analyze_image(
    row,
    checkpoint_dir,
    image_dir,
    output_dir,
    n_steps,
    device,
):
    image_id = row["img_id"]
    fold = int(row["fold"])

    checkpoint_path = (
        checkpoint_dir
        / f"cv_v1_normalized_fold_{fold}"
        / "best_checkpoint.pt"
    )

    model, rgb_mean, rgb_std = load_model(
        model_file=checkpoint_path,
        device=device,
    )

    transform = build_transform(
        mean=rgb_mean,
        std=rgb_std,
    )

    image_path = get_image_path(
        img_dir=image_dir,
        image_id=image_id,
    )

    original_image, input_tensor = prepare_image(
        image_path=image_path,
        transform=transform,
        device=device,
    )

    prediction, probability = predict(
        model=model,
        tensor=input_tensor,
    )

    expected_prediction = int(row["predicted_label"])

    if prediction != expected_prediction:
        raise ValueError(
            f"OOF prediction mismatch for {image_id}: "
            f"expected {expected_prediction}, got {prediction}"
        )

    cam = GradCAM(
        model=model,
        target_layer=model.resnet18[7][-1]
    )

    try:
        cams = {
            class_id: cam.generate(
                image_tensor=input_tensor,
                target_class=class_id,
            )
            for class_id in [0, 1]
        }
    finally:
        cam.close()

    attributions = {}

    for class_id in [0, 1]:
        attr, delta = compute_integrated_gradients(
            model=model,
            input_tensor=input_tensor,
            target_class=class_id,
            n_steps=n_steps,
        )

        attributions[class_id] = attr

        print(
            f"{image_id} | Class {class_id} | "
            f"IG delta: {delta.item():.6f}"
        )

    error_type = row["error_type"]

    output_path = (
        output_dir
        / error_type
        / f"xai_{image_id}.png"
    )

    save_comparison(
        original_image=original_image,
        cams=cams,
        attributions=attributions,
        true_label=int(row["true_label"]),
        prediction=prediction,
        probability=probability,
        fold=fold,
        error_type=error_type,
        output_path=output_path,
    )

    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    if args.n_steps < 1:
        raise ValueError("--n_steps must be at least 1.")

    oof = pd.read_csv(args.oof_file)

    if args.all_errors:
        selected = oof[
            oof["error_type"].isin(["FP", "FN"])
        ].copy()
    else:
        selected = oof[
            oof["img_id"].isin(EXAMPLE_IDS)
        ].copy()

        missing = set(EXAMPLE_IDS) - set(selected["img_id"])

        if missing:
            raise ValueError(
                f"Missing example images: {sorted(missing)}"
            )

    device = torch.device("cpu")

    for _, row in selected.iterrows():
        analyze_image(
            row=row,
            checkpoint_dir=args.checkpoint_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            n_steps=args.n_steps,
            device=device,
        )

    print(f"\nAnalyzed {len(selected)} images.")


if __name__ == "__main__":
    main()