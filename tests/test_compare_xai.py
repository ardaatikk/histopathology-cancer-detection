import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

from scripts.compare_xai import (
    EXAMPLE_IDS,
    attribution_to_map,
    save_comparison,
)


def test_attribution_to_map():
    attribution = torch.ones((1, 3, 224, 224))

    result = attribution_to_map(attribution)

    assert result.shape == (224, 224)
    assert np.allclose(result, 3.0)


def test_example_selection():
    df = pd.DataFrame({
        "img_id": EXAMPLE_IDS + ["other_image"],
        "error_type": ["TP", "TN", "FP", "FN", "TP"],
    })

    selected = df[df["img_id"].isin(EXAMPLE_IDS)]

    assert len(selected) == 4
    assert set(selected["error_type"]) == {
        "TP", "TN", "FP", "FN"
    }


def test_save_comparison(tmp_path):
    original_image = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    cams = {
        0: np.zeros((224, 224)),
        1: np.ones((224, 224)),
    }

    attributions = {
        0: torch.zeros((1, 3, 224, 224)),
        1: torch.ones((1, 3, 224, 224)),
    }

    output_path = tmp_path / "comparison.png"

    save_comparison(
        original_image=original_image,
        cams=cams,
        attributions=attributions,
        true_label=0,
        prediction=1,
        probability=0.95,
        fold=1,
        error_type="FP",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0