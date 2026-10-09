import argparse
from pathlib import Path

import torch
from captum.attr import IntegratedGradients

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.gradcam import (
        build_transform,
        load_model,
        predict,
        prepare_image,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"src", "src.gradcam"}:
        raise

    from gradcam import (
        build_transform,
        load_model,
        predict,
        prepare_image,
    )


def compute_integrated_gradients(
    model,
    input_tensor,
    target_class,
    n_steps=64,
):
    ig = IntegratedGradients(model)

    baseline = torch.zeros_like(input_tensor)

    attributions, convergence_delta = ig.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    return attributions, convergence_delta

def save_ig_visualization(
    original_image,
    attributions,
    output_path,
):
    attribution_map = (
        attributions
        .detach()
        .cpu()
        .squeeze(0)
        .sum(dim=0)
        .numpy()
    )

    positive = np.maximum(attribution_map, 0)
    negative = np.maximum(-attribution_map, 0)

    scale = np.percentile(
        np.abs(attribution_map),
        99,
    )

    scale = max(float(scale), 1e-12)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")

    axes[1].imshow(
        positive,
        cmap="Reds",
        vmin=0,
        vmax=scale,
    )
    axes[1].set_title("Positive Attribution")

    axes[2].imshow(
        negative,
        cmap="Blues",
        vmin=0,
        vmax=scale,
    )
    axes[2].set_title("Negative Attribution")

    for ax in axes:
        ax.axis("off")

    output_path = Path(output_path)
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

    print(f"Visualization saved: {output_path}")

def save_ig_comparison(
    original_image,
    class_0_attributions,
    class_1_attributions,
    output_path,
):
    def to_attribution_map(attributions):
        return (
            attributions
            .detach()
            .cpu()
            .squeeze(0)
            .sum(dim=0)
            .numpy()
        )

    class_0_map = to_attribution_map(class_0_attributions)
    class_1_map = to_attribution_map(class_1_attributions)

    scale = np.percentile(
        np.abs(
            np.concatenate([
                class_0_map.ravel(),
                class_1_map.ravel(),
            ])
        ),
        99,
    )

    scale = max(float(scale), 1e-12)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")

    axes[1].imshow(
        class_0_map,
        cmap="bwr",
        vmin=-scale,
        vmax=scale,
    )
    axes[1].set_title("Class 0 - Non-cancer")

    im = axes[2].imshow(
        class_1_map,
        cmap="bwr",
        vmin=-scale,
        vmax=scale,
    )
    axes[2].set_title("Class 1 - Cancer")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(
        im,
        ax=axes[1:],
        shrink=0.75,
        label="Integrated Gradients Attribution",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Comparison saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_file", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--n_steps", type=int, default=64)
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/integrated_gradients"),
        help="Directory for Integrated Gradients visualizations.",
    )

    parser.add_argument(
        "--target_class",
        type=int,
        choices=[0, 1],
        default=None,
        help="Class to explain. Defaults to the predicted class.",
    )
    
    parser.add_argument(
        "--compare_classes",
        action="store_true",
        help="Generate a side-by-side comparison of both classes.",
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    model, rgb_mean, rgb_std = load_model(
        model_file=args.model_file,
        device=device,
    )

    transform = build_transform(
        mean=rgb_mean,
        std=rgb_std,
    )

    original_image, input_tensor = prepare_image(
        image_path=args.image,
        transform=transform,
        device=device,
    )

    prediction, cancer_probability = predict(
        model=model,
        tensor=input_tensor,
    )

    if args.compare_classes:
        class_0_attributions, delta_0 = compute_integrated_gradients(
            model=model,
            input_tensor=input_tensor,
            target_class=0,
            n_steps=args.n_steps,
        )

        class_1_attributions, delta_1 = compute_integrated_gradients(
            model=model,
            input_tensor=input_tensor,
            target_class=1,
            n_steps=args.n_steps,
        )

        output_path = (
            args.output_dir
            / f"ig_comparison_{args.image.stem}.png"
        )

        save_ig_comparison(
            original_image=original_image,
            class_0_attributions=class_0_attributions,
            class_1_attributions=class_1_attributions,
            output_path=output_path,
        )

        print(f"Prediction: {prediction}")
        print(f"Cancer probability: {cancer_probability:.6f}")
        print(f"Class 0 convergence delta: {delta_0.item():.6f}")
        print(f"Class 1 convergence delta: {delta_1.item():.6f}")

    else:
        target_class = (
            prediction
            if args.target_class is None
            else args.target_class
        )

        attributions, delta = compute_integrated_gradients(
            model=model,
            input_tensor=input_tensor,
            target_class=target_class,
            n_steps=args.n_steps,
        )

        output_path = (
            args.output_dir
            / f"ig_class_{target_class}_{args.image.stem}.png"
        )

        save_ig_visualization(
            original_image=original_image,
            attributions=attributions,
            output_path=output_path,
        )

        print(f"Prediction: {prediction}")
        print(f"Cancer probability: {cancer_probability:.6f}")
        print(f"Attribution shape: {tuple(attributions.shape)}")
        print(f"Convergence delta: {delta.item():.6f}")
        print(f"Explained class: {target_class}")


if __name__ == "__main__":
    main()