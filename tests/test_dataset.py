import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import HistopathologyDataset
from dataset_inference import InferenceDataset


def create_test_image(path):
    image = Image.new(
        "RGB",
        (32, 32),
        color=(120, 80, 160),
    )
    image.save(path)


def test_histopathology_dataset_length(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    annotations = pd.DataFrame(
        {
            "img_id": ["sample_1", "sample_2"],
            "label": [0, 1],
        }
    )

    csv_path = tmp_path / "train.csv"
    annotations.to_csv(csv_path, index=False)

    dataset = HistopathologyDataset(
        annotations_file=csv_path,
        img_dir=image_dir,
    )

    assert len(dataset) == 2


def test_histopathology_dataset_returns_image_and_label(
    tmp_path,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    create_test_image(
        image_dir / "sample_1.jpeg"
    )

    annotations = pd.DataFrame(
        {
            "img_id": ["sample_1"],
            "label": [1],
        }
    )

    csv_path = tmp_path / "train.csv"
    annotations.to_csv(csv_path, index=False)

    dataset = HistopathologyDataset(
        annotations_file=csv_path,
        img_dir=image_dir,
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 1

    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_histopathology_dataset_applies_transform(
    tmp_path,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    create_test_image(
        image_dir / "sample_1.jpeg"
    )

    annotations = pd.DataFrame(
        {
            "img_id": ["sample_1"],
            "label": [0],
        }
    )

    csv_path = tmp_path / "train.csv"
    annotations.to_csv(csv_path, index=False)

    def transform(image):
        return image + 1.0

    dataset = HistopathologyDataset(
        annotations_file=csv_path,
        img_dir=image_dir,
        transform=transform,
    )

    image, label = dataset[0]

    assert label == 0
    assert image.min() >= 1.0


def test_inference_dataset_length(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    annotations = pd.DataFrame(
        {
            "img_id": [
                "sample_1",
                "sample_2",
                "sample_3",
            ]
        }
    )

    csv_path = tmp_path / "test.csv"
    annotations.to_csv(csv_path, index=False)

    dataset = InferenceDataset(
        annotations_file=csv_path,
        img_dir=image_dir,
    )

    assert len(dataset) == 3


def test_inference_dataset_returns_image_and_id(
    tmp_path,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    create_test_image(
        image_dir / "sample_1.jpeg"
    )

    annotations = pd.DataFrame(
        {
            "img_id": ["sample_1"],
        }
    )

    csv_path = tmp_path / "test.csv"
    annotations.to_csv(csv_path, index=False)

    dataset = InferenceDataset(
        annotations_file=csv_path,
        img_dir=image_dir,
    )

    image, image_id = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert image_id == "sample_1"

    assert image.min() >= 0.0
    assert image.max() <= 1.0
