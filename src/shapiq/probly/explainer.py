# ruff: noqa: ANN401
"""The ProblyExplainer: explains the uncertainty of a model with Shapley interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import ValidApproximatorTypes, setup_approximator
from shapiq.game_theory.indices import is_empty_value_the_baseline

from .imputer import ProblyImputer, to_numpy
from .maskers import FeatureMasker, Masker, PatchMasker

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.approximator.base import Approximator
    from shapiq.explainer.custom_types import ValidApproximationExplainerIndices
    from shapiq.interaction_values import InteractionValues

    from .measures import Task, UncertaintyKind

_TABULAR_NDIM = 2  # (n_samples, n_features)

__all__ = ["ProblyExplainer"]


class ProblyExplainer(Explainer):
    """Explain where the uncertainty of a model comes from.

    The players are the features, patches, nodes, ... of one input, as defined by the masker. The
    value of a coalition is the uncertainty of the model when only these players are present. The
    Shapley values or interactions of this game show which players raise or lower the uncertainty.

    Examples:
        >>> model = ProblyViT(n_players=9, method="dropout", p=0.1, n_samples=8)
        >>> explainer = ProblyExplainer(model, uncertainty="total", index="k-SII", max_order=2)
        >>> interaction_values = explainer.explain(image, budget=512)
        >>> model.plot(interaction_values, image)

        Other models need background ``data`` (and ``patch_size`` for images) or a ``masker``:

        >>> explainer = ProblyExplainer(my_model, data=X_background, index="k-SII", max_order=2)
    """

    def __init__(
        self,
        model: Any,
        data: np.ndarray | None = None,
        *,
        masker: Masker | None = None,
        patch_size: int | None = None,
        uncertainty: UncertaintyKind | Callable[[np.ndarray], np.ndarray] = "epistemic",
        task: Task = "classification",
        from_logits: bool = False,
        predict_samples: Callable[[Any], Any] | None = None,
        n_samples: int = 8,
        approximator: Literal["auto", "exact"] | ValidApproximatorTypes | Approximator = "auto",
        index: ValidApproximationExplainerIndices = "SV",
        max_order: int = 1,
        batch_size: int = 64,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the explainer.

        Args:
            model: The model. It needs several predictions per input, in one of three ways: a
                ``predict_samples(batch)`` method returning ``(n_samples, n_inputs, n_outputs)``
                (or pass the function as ``predict_samples``), or a stochastic callable such as MC
                dropout, which is called ``n_samples`` times. A model may also provide a
                ``masker`` and a ``prepare(x)`` method (raw input -> masker input).
            data: Background data for the baseline of absent players: a table
                ``(n_samples, n_features)``, or images with ``patch_size``. Not needed with a masker.
            masker: Defines the players and what an absent player means. Chosen from ``data`` if
                not given.
            patch_size: Image mode: the side length of the patches that are the players.
            uncertainty: ``"total"``, ``"aleatoric"``, ``"epistemic"`` or a custom function.
            task: ``"classification"`` or ``"regression"``.
            from_logits: Whether classification outputs are logits.
            predict_samples: A function mapping a batch to ``(n_samples, n_inputs, n_outputs)``.
            n_samples: Forward passes per prediction for a stochastic callable.
            approximator: ``"auto"``, ``"exact"``, an approximator name or object, as in
                :class:`~shapiq.TabularExplainer`.
            index: The interaction index, e.g. ``"SV"``, ``"k-SII"``, ``"FSII"``.
            max_order: The highest interaction order.
            batch_size: Number of coalitions passed to the model at once.
            random_state: The random state.
            verbose: Whether to show a progress bar.

        Raises:
            TypeError: If the model gives no way to get several predictions.
            ValueError: If no masker is given and none can be chosen from the data.
        """
        self._model = model
        sampler = self._sampler(model, predict_samples, n_samples)
        super().__init__(sampler, None, None, index=index, max_order=max_order)
        masker = masker or getattr(model, "masker", None) or self._default_masker(data, patch_size)
        self._prepare = getattr(model, "prepare", None)
        self._imputer = ProblyImputer(
            sampler,
            masker,
            uncertainty=uncertainty,
            task=task,
            from_logits=from_logits,
            batch_size=batch_size,
            random_state=random_state,
            verbose=verbose,
        )
        self._approximator_spec = approximator
        self._random_state = random_state

    @staticmethod
    def _sampler(model: Any, predict_samples: Callable | None, n_samples: int) -> Callable:
        """Return a function mapping a batch to ``(n_samples, n_inputs, n_outputs)``."""
        if predict_samples is not None:
            return predict_samples
        if hasattr(model, "predict_samples"):
            return model.predict_samples
        if callable(model):
            return lambda batch: np.stack([to_numpy(model(batch)) for _ in range(n_samples)])
        msg = "The model needs a predict_samples method, or must be a stochastic callable."
        raise TypeError(msg)

    @staticmethod
    def _default_masker(data: np.ndarray | None, patch_size: int | None) -> Masker:
        """Choose a masker from the background data: patches for images, features for tables."""
        if data is None:
            msg = "Pass a masker, or data (with patch_size for images) to choose one."
            raise ValueError(msg)
        data = np.asarray(data)
        baseline = data.mean(axis=0)
        if patch_size is not None:
            return PatchMasker(patch_size, baseline=baseline)
        if data.ndim != _TABULAR_NDIM:
            msg = "Data that is not a table (n_samples, n_features) needs a patch_size or a masker."
            raise ValueError(msg)
        return FeatureMasker(baseline=baseline)

    def explain_function(  # type: ignore[override]
        self, x: Any, budget: int | None = None, *, random_state: int | None = None
    ) -> InteractionValues:
        """Explain the uncertainty of one input.

        Args:
            x: The input, in the form the model's ``prepare`` or the masker expects.
            budget: Number of coalitions to evaluate. Defaults to ``min(2**n_players, 2048)``.
                Ignored for ``approximator="exact"``.
            random_state: The random state.

        Returns:
            The interaction values over the players.
        """
        if self._prepare is not None:
            x = self._prepare(x)
        self.imputer.fit(x)
        n_players = self.imputer.n_players
        self._approximator = setup_approximator(
            self._approximator_spec,
            cast("ValidApproximationExplainerIndices", self.index),  # no Moebius or Fourier
            self.max_order,
            n_players,
            self._random_state if random_state is None else random_state,
        )
        self.set_random_state(random_state)
        budget = budget or min(2**n_players, 2048)
        interaction_values = self.approximator(budget=budget, game=self.imputer)
        interaction_values.baseline_value = self.baseline_value
        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        return interaction_values

    @property
    def baseline_value(self) -> float:
        """The uncertainty without any player (for the last explained input)."""
        return self.imputer.empty_prediction

    def plot(self, interaction_values: InteractionValues, image: Any = None, **kwargs: Any) -> Any:
        """Plot the effects on the uncertainty (for a :class:`ProblyViT`): regions and pairs.

        Args:
            interaction_values: The result of :meth:`explain`.
            image: The explained image. Defaults to the image of the model.
            **kwargs: Passed to :func:`shapiq.plot.image.image_plot`, e.g. ``top_k``.

        Returns:
            The figure and the two axes of the plot.

        Raises:
            TypeError: If the model has no plot.
        """
        if not hasattr(self._model, "plot"):
            msg = "plot is available for a ProblyViT."
            raise TypeError(msg)
        return self._model.plot(interaction_values, image, **kwargs)
