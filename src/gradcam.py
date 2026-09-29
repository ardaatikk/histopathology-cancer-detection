import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CancerDetectionModel


parser = argparse.ArgumentParser(
    description="Generate Grad-CAM visualizations for the cancer detection model"
)

parser.add_argument(
    "--img_dir",
    default="./images",
    help="Images directory",
)

parser.add_argument(
    "--validation_file",
    default="./data/validation.csv",
    help="Validation annotations file",
)

parser.add_argument(
    "--model_file",
    default="./checkpoints/weights_epoch_100.pt",
    help="Model checkpoint",
)

parser.add_argument(
    "--output_dir",
    default="./assets/gradcam",
    help="Directory for Grad-CAM visualizations",
)

parser.add_argument(
    "--image",
    default=None,
    help="Optional path to a single image for Grad-CAM analysis",
)

parser.add_argument(
    "--single_output_dir",
    default="./outputs/gradcam",
    help="Directory for single-image Grad-CAM outputs",
)

FLAGS = parser.parse_args()


MEAN = (0.62376275, 0.43274997, 0.64434578)
STD = (0.2201862, 0.23024299, 0.19410873)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


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

    def generate(self, image_tensor, target_class):
        self.model.zero_grad()

        output = self.model(image_tensor)

        score = output[:, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


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


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")

    display_image = image.resize((224, 224))

    tensor = transform(image).unsqueeze(0).to(device)

    return display_image, tensor


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


def save_visualization(
    original_image,
    cam,
    label,
    prediction,
    probability,
    output_path,
):
    original = np.asarray(original_image).astype(np.float32) / 255.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(original)
    axes[2].imshow(
        cam,
        cmap="jet",
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

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


output_dir = Path(FLAGS.output_dir)
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


model = CancerDetectionModel(
    num_classes=2,
    pretrained=False,
).to(device)

model.load_state_dict(
    torch.load(
        FLAGS.model_file,
        map_location=device,
    )
)

model.eval()


# CancerDetectionModel stores the ResNet backbone as nn.Sequential.
# Index 7 corresponds to ResNet-18's final convolutional block (layer4).
target_layer = model.resnet18[7]

gradcam = GradCAM(
    model,
    target_layer,
)

def analyze_single_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original_image, tensor = prepare_image(image_path)

    prediction, probability = predict(
        model,
        tensor,
    )

    cam = gradcam.generate(
        tensor,
        prediction,
    )

    original = (
        np.asarray(original_image)
        .astype(np.float32)
        / 255.0
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(
        cam,
        cmap="jet",
    )
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(original)
    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
    )
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    class_name = (
        "Cancer"
        if prediction == 1
        else "Non-cancer"
    )

    fig.suptitle(
        f"Prediction: {class_name} | "
        f"Cancer probability: {probability:.3f}"
    )

    plt.tight_layout()

    single_output_dir = Path(FLAGS.single_output_dir)
    single_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        single_output_dir
        / f"gradcam_{image_path.stem}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Image: {image_path}")
    print(f"Prediction: {class_name}")
    print(
        f"Cancer probability: "
        f"{probability:.4f}"
    )
    print(
        f"Grad-CAM saved to: "
        f"{output_path}"
    )

if FLAGS.image is not None:
    analyze_single_image(FLAGS.image)
    gradcam.close()
    raise SystemExit(0)

validation_df = pd.read_csv(
    FLAGS.validation_file
)


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
        FLAGS.img_dir,
        image_id,
    )

    original_image, tensor = prepare_image(
        image_path
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

    if all(
        value is not None
        for value in examples.values()
    ):
        break


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

    output_path = (
        output_dir /
        f"{category}.png"
    )

    save_visualization(
        example["image"],
        cam,
        example["label"],
        example["prediction"],
        example["probability"],
        output_path,
    )

    print(
        f"{category}: "
        f"{example['image_id']} -> "
        f"{output_path}"
    )


gradcam.close()

print("\nGrad-CAM visualizations generated successfully.")