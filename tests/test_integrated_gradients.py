from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn

from src.integrated_gradients import (
    compute_integrated_gradients,
    save_ig_comparison,
)


class DummyModel(nn.Module):
    def forward(self, x):
        features = x.mean(dim=(2, 3))

        class_0 = features[:, 0] - features[:, 1]
        class_1 = features[:, 1] + features[:, 2]

        return torch.stack(
            [class_0, class_1],
            dim=1,
        )


def test_integrated_gradients_shape():
    model = DummyModel().eval()

    input_tensor = torch.rand(
        1, 3, 32, 32
    )

    attributions, delta = compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        target_class=1,
        n_steps=16,
    )

    assert attributions.shape == input_tensor.shape
    assert torch.isfinite(attributions).all()
    assert torch.isfinite(delta).all()


def test_integrated_gradients_both_classes():
    model = DummyModel().eval()

    input_tensor = torch.rand(
        1, 3, 32, 32
    )

    attr_0, _ = compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        target_class=0,
        n_steps=16,
    )

    attr_1, _ = compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        target_class=1,
        n_steps=16,
    )

    assert attr_0.shape == attr_1.shape
    assert not torch.allclose(attr_0, attr_1)


def test_integrated_gradients_completeness():
    model = DummyModel().eval()

    input_tensor = torch.rand(
        1, 3, 32, 32
    )

    attributions, delta = compute_integrated_gradients(
        model=model,
        input_tensor=input_tensor,
        target_class=1,
        n_steps=32,
    )

    baseline = torch.zeros_like(input_tensor)

    with torch.no_grad():
        expected_difference = (
            model(input_tensor)[0, 1]
            - model(baseline)[0, 1]
        )

    assert torch.allclose(
        attributions.sum(),
        expected_difference,
        atol=1e-5,
    )

    assert abs(delta.item()) < 1e-5


def test_ig_comparison_visualization(tmp_path):
    original_image = np.random.rand(
        32, 32, 3
    )

    class_0_attributions = torch.rand(
        1, 3, 32, 32
    )

    class_1_attributions = torch.rand(
        1, 3, 32, 32
    )

    output_path = (
        tmp_path / "ig_comparison.png"
    )

    save_ig_comparison(
        original_image=original_image,
        class_0_attributions=class_0_attributions,
        class_1_attributions=class_1_attributions,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0