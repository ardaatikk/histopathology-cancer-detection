import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate RGB statistics using training images only."
    )

    parser.add_argument(
        "--image_folder",
        type=Path,
        default=PROJECT_ROOT / "images",
        help="Directory containing the images.",
    )

    parser.add_argument(
        "--train_csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "train.csv",
        help="CSV containing training image IDs.",
    )

    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path for saving RGB statistics as JSON.",
    )

    return parser.parse_args()


def calculate_rgb_stats(image_folder, train_csv):
    """Calculate pixel-weighted RGB mean and standard deviation."""

    train_data = pd.read_csv(train_csv)

    if "img_id" not in train_data.columns:
        raise ValueError("Training CSV must contain an 'img_id' column.")

    if train_data["img_id"].isna().any():
        raise ValueError("Missing image IDs detected.")

    if train_data["img_id"].duplicated().any():
        raise ValueError("Duplicate training image IDs detected.")

    rgb_sum = np.zeros(3, dtype=np.float64)
    squared_sum = np.zeros(3, dtype=np.float64)

    pixel_count = 0
    image_count = 0

    for image_id in train_data["img_id"]:
        image_path = image_folder / f"{image_id}.jpeg"

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Training image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixels = np.asarray(image, dtype=np.float64) / 255.0

        rgb_sum += pixels.sum(axis=(0, 1))
        squared_sum += np.square(pixels).sum(axis=(0, 1))

        pixel_count += pixels.shape[0] * pixels.shape[1]
        image_count += 1

    if image_count == 0:
        raise RuntimeError("No training images found.")

    rgb_mean = rgb_sum / pixel_count

    variance = squared_sum / pixel_count - np.square(rgb_mean)
    variance = np.maximum(variance, 0.0)

    rgb_std = np.sqrt(variance)

    return image_count, rgb_mean, rgb_std


def main():
    args = parse_args()

    image_count, rgb_mean, rgb_std = calculate_rgb_stats(
        image_folder=args.image_folder,
        train_csv=args.train_csv,
    )

    print("\nTRAINING DATASET RGB STATISTICS")
    print("-" * 40)

    print(f"Processed images: {image_count}")

    print("\nRGB Mean:")
    print(rgb_mean.tolist())

    print("\nRGB Standard Deviation:")
    print(rgb_std.tolist())

    if args.output_json is not None:
        output_path = args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)

        statistics = {
            "image_count": image_count,
            "train_csv": str(args.train_csv),
            "rgb_mean": rgb_mean.tolist(),
            "rgb_std": rgb_std.tolist(),
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(statistics, file, indent=4)

        print(f"\nRGB statistics saved to: {output_path}")


if __name__ == "__main__":
    main()