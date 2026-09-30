import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from gradcam import GradCAM
from model import CancerDetectionModel


def create_model():
    """
    Create a lightweight model for Grad-CAM tests
    without downloading pretrained weights.
    """
    model = CancerDetectionModel(
        num_classes=2,
        pretrained=False,
    )

    model.eval()

    return model


def test_gradcam_generates_heatmap():
    model = create_model()

    target_layer = model.resnet18[7]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    try:
        cam = gradcam.generate(
            image_tensor=image,
            target_class=1,
        )
    finally:
        gradcam.close()

    assert isinstance(
        cam,
        np.ndarray,
    )

    assert cam.shape == (
        224,
        224,
    )


def test_gradcam_heatmap_is_normalized():
    model = create_model()

    target_layer = model.resnet18[7]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    try:
        cam = gradcam.generate(
            image_tensor=image,
            target_class=1,
        )
    finally:
        gradcam.close()

    assert np.isfinite(
        cam
    ).all()

    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_gradcam_hooks_capture_model_data():
    model = create_model()

    target_layer = model.resnet18[7]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    image = torch.randn(
        1,
        3,
        224,
        224,
    )

    try:
        gradcam.generate(
            image_tensor=image,
            target_class=1,
        )

        assert (
            gradcam.activations
            is not None
        )

        assert (
            gradcam.gradients
            is not None
        )

        assert (
            gradcam.activations.ndim
            == 4
        )

        assert (
            gradcam.gradients.shape
            == gradcam.activations.shape
        )

    finally:
        gradcam.close()
