from pathlib import Path

import pandas as pd
import torch
from torchvision.io import read_image


class InferenceDataset(torch.utils.data.Dataset):
    """Dataset for unlabeled histopathology images."""

    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.image_ids = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = str(self.image_ids.iloc[idx, 0])
        image_path = self.img_dir / f"{image_id}.jpeg"

        image = read_image(str(image_path)).float() / 255.0

        if self.transform:
            image = self.transform(image)

        return image, image_id