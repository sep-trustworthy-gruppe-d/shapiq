

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.imputer.base import Imputer

from .measures import uncertainty as compute_uncertainty

if TYPE_CHECKING:
    from collections.abc import Callable

    from .maskers import Masker
    from .measures import Task, UncertaintyKind

__all__ = ["ProblyImputer", "to_numpy"]


def to_numpy(a: Any) -> np.ndarray:
    """Convert arrays and ``torch`` tensors to a float NumPy array."""
    return np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=float)


class ProblyImputer(Imputer):
    """A game whose value is the uncertainty of a model on a partly absent input.

    For a coalition, the masker removes the absent players from the input, the model predicts
    several times, and the uncertainty of these predictions is the value. The number of players is
    read from the explained input, so inputs of different size (e.g. graphs) work.
    """

    def __init__(
        self,
        predict_samples: Callable[[Any], Any],
        masker: Masker,
        x: Any = None,
        *,
        uncertainty: UncertaintyKind | Callable[[np.ndarray], np.ndarray] = "epistemic",
        task: Task = "classification",
        from_logits: bool = False,
        empty_value: float | None = None,
        batch_size: int = 64,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the imputer.

        Args:
            predict_samples: Maps a batch of inputs to predictions ``(n_samples, n_inputs, n_out)``.
            masker: Defines the players and what an absent player means.
            x: The explained input (can be set later with :meth:`fit`).
            uncertainty: ``"total"``, ``"aleatoric"``, ``"epistemic"`` or a custom function.
            task: ``"classification"`` or ``"regression"``.
            from_logits: Whether classification outputs are logits.
            empty_value: Value of the coalition without players. ``None`` evaluates the model on the
                fully masked input, or uses ``0`` if the masker cannot produce one.
            batch_size: Number of coalitions passed to the model at once.
            random_state: The random state.
            verbose: Whether to show a progress bar.
        """
        self._predict_samples, self.masker = predict_samples, masker
        self.uncertainty, self.task, self.from_logits = uncertainty, task, from_logits
        self.batch_size = batch_size
        self._empty_value = 0.0 if empty_value is None and not masker.empty_ok else empty_value
        self._n = 1  # placeholder until an input is fitted
        super().__init__(
            predict_samples,
            np.zeros((1, 1)),  # background data of the base class, unused
            None,
            sample_size=1,
            random_state=random_state,
            verbose=verbose,
        )
        if x is not None:
            self.fit(x)

    def _n_players(self) -> int:
        return self._n

    @property
    def x(self) -> Any:
        """The explained input."""
        if self._x is None:
            msg = "The imputer has not been fitted yet."
            raise AttributeError(msg)
        return self._x

    def fit(self, x: Any) -> ProblyImputer:
        """Set the explained input, its number of players and the value of the empty coalition."""
        n_players = self.masker.n_players(x)
        if n_players < 1:
            msg = "The input has no players."
            raise ValueError(msg)
        self._x, self.n_players = x, int(n_players)
        self.empty_prediction = float(self.value_function(np.zeros((1, self.n_players), bool))[0])
        self.normalization_value = self.empty_prediction
        return self

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the uncertainty of the model for each coalition (``(n_coalitions, n_players)``)."""
        coalitions = np.asarray(coalitions, dtype=bool)
        values = np.zeros(len(coalitions))
        todo = np.arange(len(coalitions))
        if self._empty_value is not None:
            is_empty = ~coalitions.any(axis=1)
            values[is_empty] = self._empty_value
            todo = todo[~is_empty]
        for start in range(0, len(todo), self.batch_size):
            rows = todo[start : start + self.batch_size]
            batch = self.masker.collate([self.masker.mask(self.x, coalitions[i]) for i in rows])
            out = compute_uncertainty(
                to_numpy(self._predict_samples(batch)),
                self.uncertainty,
                self.task,
                from_logits=self.from_logits,
            )
            if len(out) != len(rows):
                msg = f"predict_samples returned {len(out)} inputs for a batch of {len(rows)}."
                raise ValueError(msg)
            values[rows] = out
        return values
