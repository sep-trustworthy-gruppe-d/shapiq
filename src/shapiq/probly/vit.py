# ruff: noqa: ANN401, SLF001
"""The vision transformer of shapiq (``ViTTokenModel``) as a probly model."""

from __future__ import annotations

from math import isqrt
from typing import Any

import numpy as np

from .maskers import CallableMasker
from .plot import player_images

_METHODS = ("dropout", "dropconnect", "ensemble")

__all__ = ["ProblyViT"]


def _make_head(torch: Any, vit: Any) -> Any:
    """Return the ViT after the embeddings as a module: tokens -> class logits."""
    backbone = vit.vit

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # transformers < 5 has one ``encoder``, transformers >= 5 a list of ``layers``
            self.layers = getattr(backbone, "encoder", None) or backbone.layers
            self.norm, self.classifier = backbone.layernorm, vit.classifier

        def forward(self, tokens: Any) -> Any:
            if isinstance(self.layers, torch.nn.ModuleList):
                hidden = tokens
                for layer in self.layers:
                    hidden = layer(hidden, None)
            else:
                hidden = self.layers(tokens).last_hidden_state
            return self.classifier(self.norm(hidden)[:, 0])

    return Head()


class ProblyViT:
    """The vision transformer of shapiq (``ViTTokenModel``) as a probly model.

    The token model loads the pretrained ViT and defines the players (square groups of patches).
    This class applies a probly method to the ViT after the embeddings, so that every prediction
    is random, and removes players in token space exactly as the token model does (zero mask token,
    position embedding kept). Pass it to the :class:`ProblyExplainer`:

    Examples:
        >>> model = ProblyViT(n_players=9, method="dropout", p=0.1, n_samples=8)
        >>> explainer = ProblyExplainer(model, uncertainty="total", index="k-SII", max_order=2)
        >>> interaction_values = explainer.explain(image, budget=512)
        >>> model.plot(interaction_values, image)

    Attributes:
        n_players: The number of players (patch groups): 9, 16, 36 or 144.
        model: The probly model (the ViT after the embeddings).
        masker: Defines what an absent player means (used by the explainer automatically).
        image: The image that was embedded last (set by :meth:`prepare`).
    """

    def __init__(
        self,
        n_players: int = 9,
        *,
        method: str = "dropout",
        n_samples: int = 8,
        token_model: Any = None,
        **method_kwargs: Any,
    ) -> None:
        """Load the ViT of shapiq and transform it with a probly method.

        Args:
            n_players: The number of players: 9, 16, 36 or 144.
            method: The probly transformation: ``"dropout"``, ``"dropconnect"`` or ``"ensemble"``.
                Use ``"dropout"`` with pretrained weights: probly's DropConnect does not rescale
                the weights during training, which shifts the predictions of a pretrained model,
                and an ensemble resets its members' weights, so train them first.
            n_samples: Forward passes per prediction (ensembles use one per member instead).
            token_model: An existing ``shapiq_games.vision._vit.ViTTokenModel`` to reuse. By
                default one is loaded.
            **method_kwargs: Passed to the probly transformation, e.g. ``p=0.1`` for dropout or
                ``num_members=5`` for ensembles.
        """
        if method not in _METHODS:
            msg = f"method must be one of {_METHODS}, got {method!r}."
            raise ValueError(msg)
        try:
            import probly.transformation

            from shapiq_games.vision._regions import token_players
            from shapiq_games.vision._vit import ViTTokenModel
        except ImportError as error:
            msg = "ProblyViT needs probly, torch and transformers: pip install probly torch transformers"
            raise ImportError(msg) from error

        if token_model is None:  # the image is only a placeholder: explain() brings its own
            token_model = ViTTokenModel(np.zeros((384, 384, 3), np.uint8), n_players)
        torch, vit = token_model._torch, token_model._model
        self._torch, self._vit, self._processor = torch, vit, token_model._processor
        self._token_model, self.n_players = token_model, token_model.n_players
        self.image = token_model.image
        side = vit.config.image_size // vit.config.patch_size
        grid = isqrt(self.n_players)

        # The token model already set a zero mask token: masked patches keep their position embedding.
        blank = torch.zeros(
            1, vit.config.num_channels, vit.config.image_size, vit.config.image_size
        )
        with torch.no_grad():
            all_masked = torch.ones(1, side * side, dtype=torch.bool)
            self._empty = vit.vit.embeddings(blank, bool_masked_pos=all_masked)[0].numpy()

        transform = getattr(probly.transformation, method)
        self.model = transform(_make_head(torch, vit), **method_kwargs)
        # an ensemble is a list of members, any other probly model is sampled repeatedly
        is_ensemble = isinstance(self.model, torch.nn.ModuleList)
        self._members = list(self.model) if is_ensemble else [self.model] * n_samples
        for member in self._members:
            member.train()  # keep random layers such as dropout active

        self._player_of_token = token_players(side, grid, grid).ravel()
        self.masker = CallableMasker(self.n_players, self._mask)

    def _mask(self, tokens: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """Replace the tokens of absent players; the class token (index 0) always stays."""
        keep = np.concatenate([[True], coalition[self._player_of_token]])
        return np.where(keep[:, None], tokens, self._empty)

    def prepare(self, image: Any) -> np.ndarray:
        """Embed an image (PIL image or array) into tokens, the input the explainer works on."""
        self.image = np.asarray(image)
        pixels = self._processor(images=[image], return_tensors="pt")["pixel_values"]
        with self._torch.no_grad():
            return self._vit.vit.embeddings(pixels)[0].numpy()

    def predict_samples(self, tokens: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, n_inputs, n_classes)``."""
        torch = self._torch
        inputs = torch.as_tensor(tokens, dtype=torch.float32)
        with torch.no_grad():
            return np.stack([torch.softmax(m(inputs), dim=-1).numpy() for m in self._members])

    def predicted_class(self, image: Any) -> tuple[str, float]:
        """Return the name and mean probability of the class the model predicts for an image."""
        probs = self.predict_samples(self.prepare(image)[None]).mean(axis=0)[0]
        best = int(probs.argmax())
        return self._token_model.categories[best].split(",")[0], float(probs[best])

    def plot(self, interaction_values: Any, image: Any = None, **kwargs: Any) -> Any:
        """Plot the effects of the regions on the uncertainty with the image plot of shapiq.

        Args:
            interaction_values: The result of :meth:`ProblyExplainer.explain`.
            image: The explained image. Defaults to the last explained image.
            **kwargs: Passed to :func:`shapiq.plot.image.image_plot`, e.g. ``top_k``.

        Returns:
            The figure and the two axes of the plot.
        """
        from shapiq.plot.image import image_plot

        image = np.asarray(self.image if image is None else image)
        label, probability = self.predicted_class(image)
        return image_plot(
            interaction_values,
            image,
            grid=isqrt(self.n_players),
            title=f"Predicted class: {label} ({probability:.0%})",
            value_label="uncertainty",
            **kwargs,
        )

    def plot_network(self, interaction_values: Any, image: Any = None, **kwargs: Any) -> Any:
        """Draw the network plot of shapiq with the numbered grid of the players on the image.

        The image shows where the players are and each node shows its part of the image. It needs
        interactions (``max_order=2``).

        Args:
            interaction_values: The result of :meth:`ProblyExplainer.explain`.
            image: The explained image. Defaults to the last explained image.
            **kwargs: Passed to :meth:`shapiq.InteractionValues.plot_network`.

        Returns:
            The figure and the axis of the plot (or ``None`` if ``show=True``).
        """
        overview, crops = player_images(
            self.image if image is None else image, isqrt(self.n_players)
        )
        kwargs.setdefault("show", False)
        return interaction_values.plot_network(
            center_image=overview, feature_image_patches=crops, **kwargs
        )
