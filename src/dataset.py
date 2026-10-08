from pathlib import Path

import pandas as pd
import torch
import numpy as np
from PIL import Image


class HistopathologyDataset(torch.utils.data.Dataset):
    """Dataset for labeled histopathology image classification."""

    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        self.annotations = pd.read_csv(annotations_file)

        required_columns = {"img_id", "label"}

        if not required_columns.issubset(self.annotations.columns):
            raise ValueError(
                f"Annotations must contain: {required_columns}"
            )

        if self.annotations[["img_id", "label"]].isna().any().any():
            raise ValueError("Missing image IDs or labels detected.")

        if not self.annotations["label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary: 0 or 1.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        image_id = str(row["img_id"])
        label = int(row["label"])

        image_path = self.img_dir / f"{image_id}.jpeg"

        with Image.open(image_path) as img:
            image_array = np.array(
                img.convert("RGB"),
                dtype=np.uint8,
                copy=True,
            )

        image = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label