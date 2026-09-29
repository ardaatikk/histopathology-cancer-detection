import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_inference import InferenceDataset
from model import CancerDetectionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on histopathology images."
    )

    parser.add_argument(
        "--img_dir",
        default="./images",
        help="Directory containing histopathology images.",
    )
    parser.add_argument(
        "--test_file",
        default="./data/test.csv",
        help="Test annotations CSV.",
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
        default="./outputs",
        help="Directory for prediction outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(
                (0.62376275, 0.43274997, 0.64434578),
                (0.2201862, 0.23024299, 0.19410873),
            ),
        ]
    )

    test_dataset = InferenceDataset(
        annotations_file=args.test_file,
        img_dir=args.img_dir,
        transform=transform,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Test set has {len(test_dataset)} instances.")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model_file).stem
    output_file = (
        output_dir / f"predictions__{model_name}.csv"
    )

    with open(output_file, "w") as log_file:
        log_file.write("img_id,cancer_score\n")

        with torch.no_grad():
            for images, img_ids in test_dataloader:
                images = images.to(device)

                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)

                cancer_scores = (
                    probabilities[:, 1]
                    .detach()
                    .cpu()
                    .tolist()
                )

                for img_id, score in zip(
                    img_ids,
                    cancer_scores,
                ):
                    log_file.write(
                        f"{img_id},{score}\n"
                    )

    print(f"Predictions saved to: {output_file}")


if __name__ == "__main__":
    main()