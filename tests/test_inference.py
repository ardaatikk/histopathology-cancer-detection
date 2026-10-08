import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import inference
from model import CancerDetectionModel


def create_test_image(path):
    image = Image.new(
        "RGB",
        (32, 32),
        color=(120, 80, 160),
    )
    image.save(path)


def create_checkpoint(path):
    model = CancerDetectionModel(
        num_classes=2,
        pretrained=False,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "rgb_mean": (0.61495719, 0.42157306, 0.64444248),
        "rgb_std": (0.21410407, 0.22494683, 0.18994613),
    }

    torch.save(
        checkpoint,
        path,
    )


def test_checkpoint_can_be_loaded(tmp_path):
    checkpoint_path = (
        tmp_path / "model.pt"
    )

    create_checkpoint(
        checkpoint_path
    )

    model = CancerDetectionModel(
        num_classes=2,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    assert isinstance(
        model,
        CancerDetectionModel,
    )


def test_inference_main_creates_prediction_csv(
    tmp_path,
    monkeypatch,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    create_test_image(
        image_dir / "sample_1.jpeg"
    )

    test_file = tmp_path / "test.csv"

    pd.DataFrame(
        {
            "img_id": ["sample_1"],
        }
    ).to_csv(
        test_file,
        index=False,
    )

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    create_checkpoint(
        checkpoint_path
    )

    output_dir = tmp_path / "outputs"

    args = type(
        "Args",
        (),
        {
            "img_dir": str(image_dir),
            "test_file": str(test_file),
            "model_file": str(
                checkpoint_path
            ),
            "batch_size": 1,
            "output_dir": str(
                output_dir
            ),
        },
    )()

    monkeypatch.setattr(
        inference,
        "parse_args",
        lambda: args,
    )

    inference.main()

    output_file = (
        output_dir
        / "predictions__model.csv"
    )

    assert output_file.exists()

    predictions = pd.read_csv(
        output_file
    )

    assert list(
        predictions.columns
    ) == [
        "img_id",
        "cancer_score",
    ]

    assert len(predictions) == 1

    assert (
        predictions.loc[
            0,
            "img_id",
        ]
        == "sample_1"
    )

    score = float(
        predictions.loc[
            0,
            "cancer_score",
        ]
    )

    assert 0.0 <= score <= 1.0


def test_inference_preserves_multiple_image_ids(
    tmp_path,
    monkeypatch,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    image_ids = [
        "sample_1",
        "sample_2",
    ]

    for image_id in image_ids:
        create_test_image(
            image_dir
            / f"{image_id}.jpeg"
        )

    test_file = tmp_path / "test.csv"

    pd.DataFrame(
        {
            "img_id": image_ids,
        }
    ).to_csv(
        test_file,
        index=False,
    )

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    create_checkpoint(
        checkpoint_path
    )

    output_dir = tmp_path / "outputs"

    args = type(
        "Args",
        (),
        {
            "img_dir": str(image_dir),
            "test_file": str(test_file),
            "model_file": str(
                checkpoint_path
            ),
            "batch_size": 2,
            "output_dir": str(
                output_dir
            ),
        },
    )()

    monkeypatch.setattr(
        inference,
        "parse_args",
        lambda: args,
    )

    inference.main()

    output_file = (
        output_dir
        / "predictions__model.csv"
    )

    predictions = pd.read_csv(
        output_file
    )

    assert predictions[
        "img_id"
    ].tolist() == image_ids

    assert predictions[
        "cancer_score"
    ].between(
        0.0,
        1.0,
    ).all()
