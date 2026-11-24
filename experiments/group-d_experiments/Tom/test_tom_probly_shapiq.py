"""Fast CPU tests for the vision-only probly/shapiq integration."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tom.explainer import ProblyImageExplainer
from Tom.games import ProblyImageGame
from Tom.models import ViTShapleyBackbone
from Tom.probly_adapter import ProblyModelWrapper


class TinyImageClassifier:
    def predict_proba(self, images):
        x = np.asarray(images).reshape(len(images), -1)
        p = 1 / (1 + np.exp(-x.mean(axis=1)))
        return np.stack((1 - p, p), axis=-1)


@pytest.mark.parametrize(("index", "order"), [("SV", 1), ("k-SII", 2)])
def test_efficiency_axiom(index, order):
    image = np.arange(3 * 8 * 8, dtype=float).reshape(3, 8, 8) / 200
    game = ProblyImageGame(image, TinyImageClassifier(), segmentation="grid", grid=(2, 2))
    iv = ProblyImageExplainer(
        image, TinyImageClassifier(), index=index, max_order=order, approximator="exact",
        segmentation="grid", grid=(2, 2),
    ).explain()
    attributed = sum(value for interaction, value in iv.interactions.items() if interaction)
    difference = game.value_function(np.ones((1, 4), bool))[0] - game.value_function(np.zeros((1, 4), bool))[0]
    assert np.isclose(attributed, difference, atol=1e-6)


@pytest.mark.parametrize("transformation", [None, "dropout", "ensemble", "credal_wrapper", "credal_ensembling", "laplace", "evidential"])
def test_classification_notions(transformation):
    wrapper = ProblyModelWrapper(TinyImageClassifier(), transformation=transformation, num_samples=3)
    image = np.ones((1, 3, 8, 8))
    for notion in ("total", "aleatoric", "epistemic", "target_probability", "credal_width", "credal_area"):
        assert np.asarray(wrapper.predict(image, notion, target_class=1)).shape == (1,)


@pytest.mark.parametrize("segmentation,kwargs", [
    ("patches", {"grid": (2, 2)}), ("grid", {"grid": (2, 3)}), ("slic", {"n_segments": 4}),
    ("custom", {"labels": np.tile(np.arange(4).reshape(2, 2), (4, 4))}),
])
def test_image_segmentations(segmentation, kwargs):
    game = ProblyImageGame(np.ones((3, 8, 8)), TinyImageClassifier(), segmentation=segmentation, **kwargs)
    assert game.n_players > 0
    assert game(np.zeros((1, game.n_players), dtype=bool)).shape == (1,)


def test_vit_patch_token_masking_interface():
    import torch
    image = np.ones((1, 3, 8, 8), dtype=np.float32)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    adapter = ViTShapleyBackbone(model)
    assert adapter.predict_tokens(image, np.ones((1, 4))).shape == (1, 2)


@pytest.mark.parametrize("index", ["SV", "SII", "k-SII", "FSII", "STII", "FBII"])
def test_supported_interaction_indices(index):
    image = np.ones((3, 8, 8), dtype=np.float32)
    iv = ProblyImageExplainer(image, TinyImageClassifier(), index=index, max_order=2,
        approximator="exact", segmentation="grid", grid=(2, 2)).explain()
    assert iv.n_players == 4
