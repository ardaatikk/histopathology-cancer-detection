import argparse
import datetime
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import HistopathologyDataset
from model import CancerDetectionModel
from preprocessing import (
    IMAGE_SIZE,
    RGB_MEAN,
    RGB_STD,
    get_preprocessing_transform,
)


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
        help="Total number of training epochs.",
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint.",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline_slide_independent",
        help="Name of the training experiment.",
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="SGD weight decay coefficient.",
    )
    
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="Enable ReduceLROnPlateau learning rate scheduler.",
    )

    parser.add_argument(
        "--rgb_stats",
        type=Path,
        default=None,
        help="Optional JSON file containing fold-specific RGB statistics.",
    )

    return parser.parse_args()

def load_rgb_stats(stats_path):
    if stats_path is None:
        return RGB_MEAN, RGB_STD

    with Path(stats_path).open("r", encoding="utf-8") as file:
        stats = json.load(file)

    mean = tuple(stats["rgb_mean"])
    std = tuple(stats["rgb_std"])

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("RGB mean and std must contain exactly 3 values.")

    if not all(np.isfinite(value) for value in (*mean, *std)):
        raise ValueError("RGB statistics must contain finite values.")

    if not all(value > 0 for value in std):
        raise ValueError("RGB standard deviations must be positive.")

    return mean, std

def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def set_seed(seed):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Select the available computation device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def validate_checkpoint(
    checkpoint,
    args,
    train_file_hash,
    validation_file_hash,
    rgb_mean,
    rgb_std,
):
    """Validate checkpoint compatibility before resuming."""

    if checkpoint.get("experiment_name") != args.experiment_name:
        raise ValueError(
            "Checkpoint experiment name does not match "
            "the current experiment."
        )

    if checkpoint.get("train_file_hash") is not None:
        if checkpoint["train_file_hash"] != train_file_hash:
            raise ValueError(
                "Training dataset has changed since the checkpoint."
            )

    if checkpoint.get("validation_file_hash") is not None:
        if checkpoint["validation_file_hash"] != validation_file_hash:
            raise ValueError(
                "Validation dataset has changed since the checkpoint."
            )

    if checkpoint.get("seed") != args.seed:
        raise ValueError(
            "Checkpoint seed does not match."
        )

    if checkpoint.get("batch_size") != args.batch_size:
        raise ValueError(
            "Checkpoint batch size does not match."
        )

    if checkpoint.get("learning_rate") != args.learning_rate:
        raise ValueError(
            "Checkpoint learning rate does not match."
        )

    if checkpoint.get("momentum", args.momentum) != args.momentum:
        raise ValueError(
            "Checkpoint momentum does not match."
        )
    
    if checkpoint.get("weight_decay", 0.0) != args.weight_decay:
        raise ValueError(
            "Checkpoint weight decay does not match."
        )

    if checkpoint.get("image_size") is not None:
        if tuple(checkpoint["image_size"]) != tuple(IMAGE_SIZE):
            raise ValueError(
                "Checkpoint image size does not match."
            )

    if checkpoint.get("rgb_mean") is not None:
        if tuple(checkpoint["rgb_mean"]) != tuple(rgb_mean):
            raise ValueError(
                "Checkpoint RGB mean does not match."
            )

    if checkpoint.get("rgb_std") is not None:
        if tuple(checkpoint["rgb_std"]) != tuple(rgb_std):
            raise ValueError(
                "Checkpoint RGB standard deviation does not match."
            )


def main():
    args = parse_args()

    rgb_mean, rgb_std = load_rgb_stats(args.rgb_stats)

    print(f"RGB Mean: {rgb_mean}")
    print(f"RGB Std:  {rgb_std}")

    # --------------------------------------------------
    # Reproducibility and device
    # --------------------------------------------------

    set_seed(args.seed)

    device = get_device()

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Experiment directories
    # --------------------------------------------------

    checkpoint_dir = (
        Path(args.checkpoint_dir) / args.experiment_name
    )

    log_dir = (
        Path(args.log_dir) / args.experiment_name
    )

    best_checkpoint_path = (
        checkpoint_dir / "best_checkpoint.pt"
    )

    last_checkpoint_path = (
        checkpoint_dir / "last_checkpoint.pt"
    )

    if args.resume is None:
        if (
            best_checkpoint_path.exists()
            or last_checkpoint_path.exists()
        ):
            raise FileExistsError(
                f"Checkpoints already exist in {checkpoint_dir}. "
                "Choose another experiment name or use --resume."
            )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Dataset hashes
    # --------------------------------------------------

    train_file_hash = calculate_file_hash(
        args.train_file
    )

    validation_file_hash = calculate_file_hash(
        args.valid_file
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    train_transform = get_preprocessing_transform(
        training=True,
        mean=rgb_mean,
        std=rgb_std,
    )

    valid_transform = get_preprocessing_transform(
        training=False,
        mean=rgb_mean,
        std=rgb_std,
    )
    
    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = HistopathologyDataset(
        annotations_file=args.train_file,
        img_dir=args.img_dir,
        transform=train_transform,
    )

    valid_dataset = HistopathologyDataset(
        annotations_file=args.valid_file,
        img_dir=args.img_dir,
        transform=valid_transform,
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

    print(
        f"Training set has {len(train_dataset)} instances."
    )

    print(
        f"Validation set has {len(valid_dataset)} instances."
    )

    # --------------------------------------------------
    # Model, loss and optimizer
    # --------------------------------------------------

    model = CancerDetectionModel(
        num_classes=2
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    scheduler = None

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

    # --------------------------------------------------
    # Training state
    # --------------------------------------------------

    start_epoch = 0

    best_valid_loss = float("inf")

    # --------------------------------------------------
    # Resume training
    # --------------------------------------------------

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=True,
        )

        validate_checkpoint(
            checkpoint=checkpoint,
            args=args,
            train_file_hash=train_file_hash,
            validation_file_hash=validation_file_hash,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
        )
        
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        checkpoint_uses_scheduler = checkpoint.get(
            "scheduler_name"
        ) == "ReduceLROnPlateau"

        if checkpoint_uses_scheduler != args.use_scheduler:
            raise ValueError(
                "Checkpoint scheduler configuration does not match."
            )

        if scheduler is not None:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )
        start_epoch = checkpoint["epoch"]

        best_valid_loss = checkpoint["best_valid_loss"]

        print(
            f"Resuming training from epoch {start_epoch}"
        )

    # --------------------------------------------------
    # Training log
    # --------------------------------------------------

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_file_path = (
        log_dir / f"training_{timestamp}.csv"
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss,train_accuracy,"
            "valid_loss,valid_accuracy,learning_rate\n"
        )

        for epoch in range(
            start_epoch,
            args.num_epochs,
        ):

            # ------------------------------------------
            # Training phase
            # ------------------------------------------

            model.train()

            train_epoch_loss = 0.0
            train_correct = 0
            train_total = 0

            train_progress = tqdm(
                train_dataloader,
                desc=(
                    f"Epoch {epoch + 1}/"
                    f"{args.num_epochs} [Train]"
                ),
                leave=False,
            )

            for train_images, train_labels in train_progress:
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

                batch_size = train_labels.size(0)

                train_epoch_loss += (
                    train_loss.item() * batch_size
                )

                train_predicted = (
                    train_outputs.argmax(dim=1)
                )

                train_total += batch_size

                train_correct += (
                    train_predicted == train_labels
                ).sum().item()

                train_progress.set_postfix(
                    loss=f"{train_loss.item():.4f}",
                    acc=(
                        f"{100 * train_correct / train_total:.2f}%"
                    ),
                )

            train_loss_value = (
                train_epoch_loss / train_total
            )

            train_accuracy = (
                100 * train_correct / train_total
            )

            # ------------------------------------------
            # Validation phase
            # ------------------------------------------

            model.eval()

            valid_epoch_loss = 0.0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():

                valid_progress = tqdm(
                    valid_dataloader,
                    desc=(
                        f"Epoch {epoch + 1}/"
                        f"{args.num_epochs} [Valid]"
                    ),
                    leave=False,
                )

                for valid_images, valid_labels in valid_progress:
                    valid_images = valid_images.to(device)
                    valid_labels = valid_labels.to(device)

                    valid_outputs = model(valid_images)

                    valid_loss = loss_fn(
                        valid_outputs,
                        valid_labels,
                    )

                    batch_size = valid_labels.size(0)

                    valid_epoch_loss += (
                        valid_loss.item() * batch_size
                    )

                    valid_predicted = (
                        valid_outputs.argmax(dim=1)
                    )

                    valid_total += batch_size

                    valid_correct += (
                        valid_predicted == valid_labels
                    ).sum().item()

                    valid_progress.set_postfix(
                        loss=f"{valid_loss.item():.4f}",
                        acc=(
                            f"{100 * valid_correct / valid_total:.2f}%"
                        ),
                    )

            valid_loss_value = (
                valid_epoch_loss / valid_total
            )

            valid_accuracy = (
                100 * valid_correct / valid_total
            )
            
            if scheduler is not None:
                scheduler.step(valid_loss_value)

            current_lr = optimizer.param_groups[0]["lr"]

            # ------------------------------------------
            # Epoch summary
            # ------------------------------------------

            print(
                f"Epoch [{epoch + 1}/{args.num_epochs}], "
                f"Train Loss: {train_loss_value:.4f}, "
                f"Train Accuracy: {train_accuracy:.2f}%, "
                f"Valid Loss: {valid_loss_value:.4f}, "
                f"Valid Accuracy: {valid_accuracy:.2f}%, "
                f"LR: {current_lr:.6f}"
            )

            # ------------------------------------------
            # CSV logging
            # ------------------------------------------

            log_file.write(
                f"{epoch + 1},"
                f"{train_loss_value:.4f},"
                f"{train_accuracy:.2f},"
                f"{valid_loss_value:.4f},"
                f"{valid_accuracy:.2f},"
                f"{current_lr:.8f}\n"
            )

            log_file.flush()

            # ------------------------------------------
            # Best model tracking
            # ------------------------------------------

            is_best = (
                valid_loss_value < best_valid_loss
            )

            if is_best:
                best_valid_loss = valid_loss_value

            # ------------------------------------------
            # Checkpoint metadata
            # ------------------------------------------

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
                "validation_loss": valid_loss_value,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "experiment_name": args.experiment_name,
                "model_name": "ResNet18",
                "optimizer_name": "SGD",
                "momentum": args.momentum,
                "image_size": IMAGE_SIZE,
                "rgb_mean": rgb_mean,
                "rgb_std": rgb_std,
                "train_file_hash": train_file_hash,
                "validation_file_hash": validation_file_hash,
                "weight_decay": args.weight_decay,
                "scheduler_state_dict": (
                    scheduler.state_dict()
                    if scheduler is not None
                    else None
                ),
                "scheduler_name": (
                    "ReduceLROnPlateau"
                    if scheduler is not None
                    else None
                ),
                "scheduler_factor": 0.5 if scheduler is not None else None,
                "scheduler_patience": 3 if scheduler is not None else None,
            }

            # ------------------------------------------
            # Save latest checkpoint
            # ------------------------------------------

            torch.save(
                checkpoint,
                last_checkpoint_path,
            )

            # ------------------------------------------
            # Save best checkpoint
            # ------------------------------------------

            if is_best:
                torch.save(
                    checkpoint,
                    best_checkpoint_path,
                )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print(
        f"Training log saved to: {log_file_path}"
    )

    print(
        f"Best checkpoint saved to: "
        f"{best_checkpoint_path}"
    )


if __name__ == "__main__":
    main()