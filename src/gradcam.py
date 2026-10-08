import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from model import CancerDetectionModel
from preprocessing import IMAGE_SIZE, get_preprocessing_transform


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for the cancer detection model"
    )

    parser.add_argument("--img_dir", default="./images")
    parser.add_argument("--validation_file", default="./data/validation.csv")
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--output_dir", default="./assets/gradcam")
    parser.add_argument("--image", default=None)
    parser.add_argument("--single_output_dir", default="./outputs/gradcam")

    return parser.parse_args()


# ============================================================
# Image preprocessing
# ============================================================

def build_transform(mean=None, std=None):
    return get_preprocessing_transform(
        training=False,
        mean=mean,
        std=std,
    )


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class, signed=False):
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        output = self.model(image_tensor)
        score = output[:, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations or gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (weights * self.activations).sum(dim=1)

        # Standard Grad-CAM removes negative contributions.
        # Signed Grad-CAM preserves them.
        if not signed:
            cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        if signed:
            # Symmetric normalization preserves positive and negative signs.
            scale = float(np.max(np.abs(cam)))

            if scale > 0:
                cam = cam / scale
        else:
            # Preserve existing standard Grad-CAM normalization.
            cam -= cam.min()

            if cam.max() > 0:
                cam /= cam.max()

        return cam

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# Model loading
# ============================================================

def load_model(model_file, device):
    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=True,
    )

    rgb_mean = checkpoint.get("rgb_mean")
    rgb_std = checkpoint.get("rgb_std")

    if rgb_mean is None or rgb_std is None:
        raise ValueError(
            "Checkpoint does not contain RGB normalization statistics."
        )

    model = CancerDetectionModel(
        num_classes=2,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, rgb_mean, rgb_std


# ============================================================
# Image helpers
# ============================================================

def get_image_path(img_dir, image_id):
    img_dir = Path(img_dir)

    candidates = [
        img_dir / image_id,
        img_dir / f"{image_id}.jpeg",
        img_dir / f"{image_id}.jpg",
        img_dir / f"{image_id}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find image for ID: {image_id}"
    )


def prepare_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = TF.to_tensor(image)

    tensor = (
        transform(image_tensor)
        .unsqueeze(0)
        .to(device)
    )

    display_image = TF.to_pil_image(
        TF.resize(image_tensor, IMAGE_SIZE)
    )

    return display_image, tensor


# ============================================================
# Prediction
# ============================================================

def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)

        prediction = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        cancer_probability = probabilities[0, 1].item()

    return prediction, cancer_probability


# ============================================================
# Standard Grad-CAM visualization
# ============================================================

def save_visualization(
    original_image,
    cam,
    label,
    prediction,
    probability,
    output_path,
):
    original = np.asarray(original_image).astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(original)
    axes[2].imshow(
        cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"True label: {label} | "
        f"Prediction: {prediction} | "
        f"Cancer probability: {probability:.3f}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_single_visualization(
    original_image,
    cam,
    prediction,
    probability,
    output_path,
):
    original = np.asarray(original_image).astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(original)
    axes[2].imshow(
        cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    class_name = "Cancer" if prediction == 1 else "Non-cancer"

    fig.suptitle(
        f"Prediction: {class_name} | "
        f"Cancer probability: {probability:.3f}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Extended Grad-CAM visualizations
# ============================================================

def save_extended_visualizations(
    original_image,
    cancer_cam,
    non_cancer_cam,
    signed_cancer_cam,
    signed_non_cancer_cam,
    prediction,
    probability,
    output_dir,
    image_stem,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = np.asarray(original_image)

    # ====================================================
    # Cancer vs Non-cancer standard Grad-CAM
    # ====================================================

    class_path = output_dir / f"gradcam_classes_{image_stem}.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(original)
    axes[1].imshow(
        cancer_cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[1].set_title("Cancer Grad-CAM (class 1)")

    axes[2].imshow(original)
    axes[2].imshow(
        non_cancer_cam,
        cmap="jet",
        vmin=0,
        vmax=1,
        alpha=0.45,
    )
    axes[2].set_title("Non-cancer Grad-CAM (class 0)")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Prediction: {prediction} | "
        f"Cancer probability: {probability:.4f}"
    )

    fig.tight_layout()
    fig.savefig(class_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ====================================================
    # Signed Grad-CAM for both classes
    # ====================================================

    signed_path = output_dir / f"gradcam_signed_{image_stem}.png"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    class_maps = [
        ("Cancer (class 1)", signed_cancer_cam),
        ("Non-cancer (class 0)", signed_non_cancer_cam),
    ]

    for row, (class_name, signed_cam) in enumerate(class_maps):
        positive = np.maximum(signed_cam, 0)
        negative = np.maximum(-signed_cam, 0)

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Original — {class_name}")

        axes[row, 1].imshow(
            positive,
            cmap="Reds",
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title(f"Positive — {class_name}")

        axes[row, 2].imshow(
            negative,
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title(f"Negative — {class_name}")

        for axis in axes[row]:
            axis.axis("off")

    fig.suptitle(
        f"Signed Grad-CAM | Prediction: {prediction} | "
        f"Cancer probability: {probability:.4f}"
    )

    fig.tight_layout()
    fig.savefig(signed_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return class_path, signed_path


# ============================================================
# Single-image analysis
# ============================================================

def analyze_single_image(
    image_path,
    model,
    gradcam,
    transform,
    device,
    output_dir,
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original_image, tensor = prepare_image(
        image_path=image_path,
        transform=transform,
        device=device,
    )

    prediction, probability = predict(
        model,
        tensor,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Standard Grad-CAM (backward-compatible output)
    # --------------------------------------------------------

    cam = gradcam.generate(
        tensor,
        target_class=prediction,
    )

    output_path = (
        output_dir / f"gradcam_{image_path.stem}.png"
    )

    save_single_visualization(
        original_image=original_image,
        cam=cam,
        prediction=prediction,
        probability=probability,
        output_path=output_path,
    )

    # --------------------------------------------------------
    # Class-specific Grad-CAM
    # --------------------------------------------------------

    cancer_cam = gradcam.generate(
        tensor,
        target_class=1,
    )

    non_cancer_cam = gradcam.generate(
        tensor,
        target_class=0,
    )
    
    # --------------------------------------------------------
    # Signed Grad-CAM for both classes
    # --------------------------------------------------------

    signed_cancer_cam = gradcam.generate(
        tensor,
        target_class=1,
        signed=True,
    )

    signed_non_cancer_cam = gradcam.generate(
        tensor,
        target_class=0,
        signed=True,
    )
    
    class_path, signed_path = save_extended_visualizations(
        original_image=original_image,
        cancer_cam=cancer_cam,
        non_cancer_cam=non_cancer_cam,
        signed_cancer_cam=signed_cancer_cam,
        signed_non_cancer_cam=signed_non_cancer_cam,
        prediction=prediction,
        probability=probability,
        output_dir=output_dir,
        image_stem=image_path.stem,
    )

    class_name = "Cancer" if prediction == 1 else "Non-cancer"

    print(f"Image: {image_path}")
    print(f"Prediction: {class_name}")
    print(f"Cancer probability: {probability:.4f}")
    print(f"Grad-CAM saved to: {output_path}")
    print(f"Class comparison saved to: {class_path}")
    print(f"Signed Grad-CAM saved to: {signed_path}")

    return output_path


# ============================================================
# Validation example discovery
# ============================================================

def find_validation_examples(
    validation_df,
    img_dir,
    model,
    transform,
    device,
):
    examples = {
        "true_positive": None,
        "true_negative": None,
        "false_positive": None,
        "false_negative": None,
    }

    for _, row in validation_df.iterrows():
        image_id = str(row["img_id"])
        label = int(row["label"])

        image_path = get_image_path(
            img_dir,
            image_id,
        )

        original_image, tensor = prepare_image(
            image_path=image_path,
            transform=transform,
            device=device,
        )

        prediction, probability = predict(
            model,
            tensor,
        )

        if label == 1 and prediction == 1:
            category = "true_positive"

        elif label == 0 and prediction == 0:
            category = "true_negative"

        elif label == 0 and prediction == 1:
            category = "false_positive"

        else:
            category = "false_negative"

        if examples[category] is None:
            examples[category] = {
                "image_id": image_id,
                "label": label,
                "prediction": prediction,
                "probability": probability,
                "image": original_image,
                "tensor": tensor,
            }

        if all(value is not None for value in examples.values()):
            break

    return examples


# ============================================================
# Validation visualization
# ============================================================

def generate_validation_visualizations(
    validation_file,
    img_dir,
    output_dir,
    model,
    gradcam,
    transform,
    device,
):
    validation_df = pd.read_csv(validation_file)

    examples = find_validation_examples(
        validation_df=validation_df,
        img_dir=img_dir,
        model=model,
        transform=transform,
        device=device,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, example in examples.items():
        if example is None:
            print(
                f"No {category.replace('_', ' ')} example found."
            )
            continue

        cam = gradcam.generate(
            example["tensor"],
            example["prediction"],
        )

        output_path = output_dir / f"{category}.png"

        save_visualization(
            original_image=example["image"],
            cam=cam,
            label=example["label"],
            prediction=example["prediction"],
            probability=example["probability"],
            output_path=output_path,
        )

        print(
            f"{category}: {example['image_id']} -> {output_path}"
        )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model, rgb_mean, rgb_std = load_model(
        model_file=args.model_file,
        device=device,
    )

    transform = build_transform(
        mean=rgb_mean,
        std=rgb_std,
    )

    print(f"RGB Mean: {rgb_mean}")
    print(f"RGB Std:  {rgb_std}")

    # ResNet-18 layer4: final convolutional block.
    target_layer = model.resnet18[7]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:
        if args.image is not None:
            analyze_single_image(
                image_path=args.image,
                model=model,
                gradcam=gradcam,
                transform=transform,
                device=device,
                output_dir=args.single_output_dir,
            )
            return

        generate_validation_visualizations(
            validation_file=args.validation_file,
            img_dir=args.img_dir,
            output_dir=args.output_dir,
            model=model,
            gradcam=gradcam,
            transform=transform,
            device=device,
        )

        print(
            "\nGrad-CAM visualizations generated successfully."
        )

    finally:
        gradcam.close()


if __name__ == "__main__":
    main()