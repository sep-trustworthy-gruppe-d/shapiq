"""Uncertainty measures computed from repeated predictions of a model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

UncertaintyKind = Literal["total", "aleatoric", "epistemic"]
Task = Literal["classification", "regression"]

_PREDS_NDIM = 3  # (n_samples, n_inputs, n_outputs)

__all__ = ["Task", "UncertaintyKind", "uncertainty"]


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy over the last axis."""
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def uncertainty(
    preds: np.ndarray,
    kind: UncertaintyKind | Callable[[np.ndarray], np.ndarray] = "epistemic",
    task: Task = "classification",
    *,
    from_logits: bool = False,
) -> np.ndarray:
    """Compute one uncertainty value per input from repeated predictions.

    Classification uses the entropy decomposition: total is the entropy of the mean prediction,
    aleatoric the mean entropy of the single predictions, epistemic their difference. Regression
    uses the variance of the predicted means (epistemic) and the mean of the predicted variances
    (aleatoric, optional second output).

    Args:
        preds: Predictions of shape ``(n_samples, n_inputs, n_outputs)``. For classification the
            class probabilities (or logits with ``from_logits``), for regression the mean and
            optionally the variance.
        kind: ``"total"``, ``"aleatoric"``, ``"epistemic"``, or a function mapping ``preds`` to
            one value per input.
        task: ``"classification"`` or ``"regression"``.
        from_logits: Whether classification outputs are logits.

    Returns:
        The uncertainty per input, shape ``(n_inputs,)``.

    Raises:
        ValueError: If the shape, ``kind`` or ``task`` is invalid.
    """
    preds = np.asarray(preds, dtype=float)
    if preds.ndim == _PREDS_NDIM - 1:  # a single output per input
        preds = preds[..., None]
    if preds.ndim != _PREDS_NDIM:
        msg = f"preds needs the shape (n_samples, n_inputs, n_outputs), got {preds.shape}."
        raise ValueError(msg)

    if callable(kind):
        return np.asarray(kind(preds), dtype=float).reshape(-1)

    if task == "classification":
        probs = _softmax(preds) if from_logits else preds
        total = _entropy(probs.mean(axis=0))
        aleatoric = _entropy(probs).mean(axis=0)
        epistemic = total - aleatoric
    elif task == "regression":
        epistemic = preds[..., 0].var(axis=0)
        aleatoric = preds[..., 1].mean(axis=0) if preds.shape[-1] > 1 else np.zeros_like(epistemic)
        total = epistemic + aleatoric
    else:
        msg = f"task must be 'classification' or 'regression', got {task!r}."
        raise ValueError(msg)

    values = {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}
    if kind not in values:
        msg = f"kind must be one of {list(values)} or a callable, got {kind!r}."
        raise ValueError(msg)
    return values[kind]
