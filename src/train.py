import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HistopathologyDataset
from model import CancerDetectionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the histopathology cancer detection model."
    )

    parser.add_argument(
        "--img_dir",
        default="./images",
        help="Directory containing histopathology images.",
    )
    parser.add_argument(
        "--train_file",
        default="./data/train.csv",
        help="Training annotations CSV.",
    )
    parser.add_argument(
        "--valid_file",
        default="./data/validation.csv",
        help="Validation annotations CSV.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--log_dir",
        default="./outputs/training",
        help="Directory for training logs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(
                (0.62376275, 0.43274997, 0.64434578),
                (0.2201862, 0.23024299, 0.19410873),
            ),
        ]
    )

    train_dataset = HistopathologyDataset(
        annotations_file=args.train_file,
        img_dir=args.img_dir,
        transform=transform,
    )

    valid_dataset = HistopathologyDataset(
        annotations_file=args.valid_file,
        img_dir=args.img_dir,
        transform=transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Training set has {len(train_dataset)} instances.")
    print(f"Validation set has {len(valid_dataset)} instances.")

    model = CancerDetectionModel(num_classes=2).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file_path = log_dir / f"training_{timestamp}.csv"

    best_valid_loss = float("inf")

    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss,train_accuracy,"
            "valid_loss,valid_accuracy\n"
        )

        for epoch in range(args.num_epochs):
            model.train()

            train_epoch_loss = 0.0
            train_correct = 0
            train_total = 0

            for train_images, train_labels in train_dataloader:
                train_images = train_images.to(device)
                train_labels = train_labels.to(device)

                optimizer.zero_grad()

                train_outputs = model(train_images)
                train_loss = loss_fn(
                    train_outputs,
                    train_labels,
                )

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()

                train_predicted = train_outputs.argmax(dim=1)

                train_total += train_labels.size(0)
                train_correct += (
                    train_predicted == train_labels
                ).sum().item()

            train_loss_value = (
                train_epoch_loss / len(train_dataloader)
            )

            train_accuracy = (
                100 * train_correct / train_total
            )

            model.eval()

            valid_epoch_loss = 0.0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for valid_images, valid_labels in valid_dataloader:
                    valid_images = valid_images.to(device)
                    valid_labels = valid_labels.to(device)

                    valid_outputs = model(valid_images)

                    valid_loss = loss_fn(
                        valid_outputs,
                        valid_labels,
                    )

                    valid_epoch_loss += valid_loss.item()

                    valid_predicted = valid_outputs.argmax(dim=1)

                    valid_total += valid_labels.size(0)
                    valid_correct += (
                        valid_predicted == valid_labels
                    ).sum().item()

            valid_loss_value = (
                valid_epoch_loss / len(valid_dataloader)
            )

            valid_accuracy = (
                100 * valid_correct / valid_total
            )

            print(
                f"Epoch [{epoch + 1}/{args.num_epochs}], "
                f"Train Loss: {train_loss_value:.4f}, "
                f"Train Accuracy: {train_accuracy:.2f}%, "
                f"Valid Loss: {valid_loss_value:.4f}, "
                f"Valid Accuracy: {valid_accuracy:.2f}%"
            )

            log_file.write(
                f"{epoch + 1},"
                f"{train_loss_value:.4f},"
                f"{train_accuracy:.2f},"
                f"{valid_loss_value:.4f},"
                f"{valid_accuracy:.2f}\n"
            )
            log_file.flush()

            if (epoch + 1) % 10 == 0:
                checkpoint_path = (
                    checkpoint_dir
                    / f"weights_{timestamp}_epoch_{epoch + 1}.pt"
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path,
                )

            if valid_loss_value < best_valid_loss:
                best_valid_loss = valid_loss_value

                best_checkpoint_path = (
                    checkpoint_dir
                    / f"best_weights_{timestamp}.pt"
                )

                torch.save(
                    model.state_dict(),
                    best_checkpoint_path,
                )

    print(f"Training log saved to: {log_file_path}")
    print(
        "Best checkpoint saved to: "
        f"{checkpoint_dir / f'best_weights_{timestamp}.pt'}"
    )


if __name__ == "__main__":
    main()