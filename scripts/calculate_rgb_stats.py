import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate RGB mean and standard deviation for the image dataset."
    )
    parser.add_argument(
        "--image_folder",
        default="./images",
        help="Directory containing the images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_folder = Path(args.image_folder)

    rgb_sum = np.zeros(3, dtype=np.float64)
    squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    image_count = 0

    image_extensions = {".jpg", ".jpeg", ".png"}

    for image_path in image_folder.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixels = np.asarray(image, dtype=np.float64) / 255.0

        rgb_sum += pixels.sum(axis=(0, 1))
        squared_sum += (pixels ** 2).sum(axis=(0, 1))

        pixel_count += pixels.shape[0] * pixels.shape[1]
        image_count += 1

    if image_count == 0:
        raise RuntimeError(
            f"No supported images found in {image_folder}"
        )

    rgb_mean = rgb_sum / pixel_count
    variance = squared_sum / pixel_count - rgb_mean ** 2
    rgb_std = np.sqrt(variance)

    print(f"Processed {image_count} images.")
    print("RGB Mean:", rgb_mean)
    print("RGB STD: ", rgb_std)


if __name__ == "__main__":
    main()