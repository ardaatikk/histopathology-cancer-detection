import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from torch.utils.data import DataLoader

from dataset import HistopathologyDataset
from model import CancerDetectionModel
from preprocessing import get_preprocessing_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the histopathology cancer detection model."
    )

    parser.add_argument(
        "--img_dir",
        default="./images",
        help="Directory containing histopathology images.",
    )

    parser.add_argument(
        "--validation_file",
        default="./data/validation.csv",
        help="Validation annotations CSV.",
    )

    parser.add_argument(
        "--model_file",
        required=True,
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )

    parser.add_argument(
        "--output_dir",
        default="./outputs/evaluation/baseline_slide_independent",
        help="Directory for evaluation outputs.",
    )

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load checkpoint and preprocessing configuration
    # --------------------------------------------------

    checkpoint = torch.load(
        args.model_file,
        map_location=device,
        weights_only=True,
    )

    rgb_mean = checkpoint.get("rgb_mean")
    rgb_std = checkpoint.get("rgb_std")

    if rgb_mean is None or rgb_std is None:
        raise ValueError(
            "Checkpoint does not contain RGB normalization statistics."
        )

    print(f"RGB Mean: {rgb_mean}")
    print(f"RGB Std:  {rgb_std}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    transform = get_preprocessing_transform(
        training=False,
        mean=rgb_mean,
        std=rgb_std,
    )

    dataset = HistopathologyDataset(
        annotations_file=args.validation_file,
        img_dir=args.img_dir,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Validation set has {len(dataset)} instances.")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = CancerDetectionModel(
        num_classes=2,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)

            probabilities = F.softmax(
                outputs,
                dim=1,
            )

            predictions = probabilities.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    all_labels = np.asarray(all_labels)
    all_predictions = np.asarray(all_predictions)
    all_probabilities = np.asarray(all_probabilities)

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------
    # Classification metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    sensitivity = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions,
    )

    roc_auc = roc_auc_score(
        all_labels,
        all_probabilities,
    )

    pr_auc = average_precision_score(
        all_labels,
        all_probabilities,
    )

    # --------------------------------------------------
    # Print metrics
    # --------------------------------------------------

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1,
        "Balanced Accuracy": balanced_accuracy,
        "ROC-AUC": roc_auc,
        "Average Precision (AP)": pr_auc,
    }

    print("\nEvaluation Results")
    print("-" * 40)

    for name, value in metrics.items():
        print(f"{name:<25}: {value:.4f}")

    print("\nConfusion Matrix")
    print("-" * 40)

    print(f"True Negatives  (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives  (TP): {tp}")

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics_path = output_dir / "metrics.txt"

    with open(metrics_path, "w") as file:
        file.write(
            "Histopathology Cancer Detection\n"
        )

        file.write(
            "Slide-Independent Validation Results\n"
        )

        file.write("=" * 45 + "\n\n")

        file.write(
            f"Checkpoint: {args.model_file}\n"
        )

        file.write(
            f"Checkpoint Epoch: {checkpoint['epoch']}\n"
        )

        file.write(
            f"Validation Samples: {len(dataset)}\n\n"
        )

        for name, value in metrics.items():
            file.write(
                f"{name}: {value:.4f}\n"
            )

        file.write("\nConfusion Matrix\n")

        file.write(f"TN: {tn}\n")
        file.write(f"FP: {fp}\n")
        file.write(f"FN: {fn}\n")
        file.write(f"TP: {tp}\n")

    # --------------------------------------------------
    # Confusion matrix visualization
    # --------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-cancer",
            "Cancer",
        ],
    )

    display.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------
    # ROC curve
    # --------------------------------------------------

    fpr, tpr, _ = roc_curve(
        all_labels,
        all_probabilities,
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------
    # Precision-Recall curve
    # --------------------------------------------------

    pr_precision, pr_recall, _ = precision_recall_curve(
        all_labels,
        all_probabilities,
    )

    plt.figure()

    plt.plot(
        pr_recall,
        pr_precision,
        label=f"AP = {pr_auc:.3f}",
    )

    positive_prevalence = np.mean(all_labels)

    plt.axhline(
        y=positive_prevalence,
        linestyle="--",
        color="gray",
        label=f"Baseline = {positive_prevalence:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=300,
    )

    plt.close()

    print(
        f"\nEvaluation outputs saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()