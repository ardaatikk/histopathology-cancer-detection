import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HistopathologyDataset
from model import CancerDetectionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained histopathology cancer detection model."
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
        default="./checkpoints/weights_epoch_100.pt",
        help="Model checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--output_dir",
        default="./assets",
        help="Directory for evaluation outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(
                (0.62376275, 0.43274997, 0.64434578),
                (0.2201862, 0.23024299, 0.19410873),
            ),
        ]
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

    model = CancerDetectionModel(
        num_classes=2,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.model_file,
            map_location=device,
        )
    )

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    all_labels = np.asarray(all_labels)
    all_predictions = np.asarray(all_predictions)
    all_probabilities = np.asarray(all_probabilities)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )
    precision = precision_score(
        all_labels,
        all_predictions,
    )
    recall = recall_score(
        all_labels,
        all_predictions,
    )
    f1 = f1_score(
        all_labels,
        all_predictions,
    )
    roc_auc = roc_auc_score(
        all_labels,
        all_probabilities,
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    metrics_path = output_dir / "metrics.txt"

    with open(metrics_path, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"ROC-AUC: {roc_auc:.4f}\n")

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-cancer", "Cancer"],
    )

    display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
    )
    plt.close()

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

    print(f"Evaluation outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()