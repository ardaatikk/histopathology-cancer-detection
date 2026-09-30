import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import CancerDetectionModel


def test_model_can_be_created_without_pretrained_weights():
    model = CancerDetectionModel(
        pretrained=False,
    )

    assert isinstance(
        model,
        CancerDetectionModel,
    )


def test_model_classifier_structure():
    model = CancerDetectionModel(
        pretrained=False,
    )

    assert isinstance(
        model.fc1,
        nn.Linear,
    )

    assert model.fc1.in_features == 512
    assert model.fc1.out_features == 128

    assert isinstance(
        model.relu,
        nn.ReLU,
    )

    assert isinstance(
        model.fc2,
        nn.Linear,
    )

    assert model.fc2.in_features == 128
    assert model.fc2.out_features == 2


def test_model_respects_num_classes():
    model = CancerDetectionModel(
        num_classes=4,
        pretrained=False,
    )

    assert model.fc2.out_features == 4


def test_model_forward_output_shape():
    model = CancerDetectionModel(
        pretrained=False,
    )

    model.eval()

    batch = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        output = model(
            batch
        )

    assert output.shape == (
        2,
        2,
    )


def test_model_outputs_finite_values():
    model = CancerDetectionModel(
        pretrained=False,
    )

    model.eval()

    batch = torch.randn(
        1,
        3,
        224,
        224,
    )

    with torch.no_grad():
        output = model(
            batch
        )

    assert torch.isfinite(
        output
    ).all()
