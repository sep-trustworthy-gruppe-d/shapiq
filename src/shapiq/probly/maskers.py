# ruff: noqa: ANN401
"""Maskers define the players of an input and what an absent player means."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_CHANNELS_NDIM = 3  # images are (C, H, W) or (H, W, C); grayscale (H, W) has one dimension less

__all__ = ["CallableMasker", "FeatureMasker", "Masker", "PatchMasker"]


class Masker(ABC):
    """Turns an input and a coalition (boolean vector of present players) into a model input."""

    empty_ok: bool = True
    """Whether the model accepts an input without any present player. If not, the value of the
    empty coalition is ``0``."""

    @abstractmethod
    def n_players(self, x: Any) -> int:
        """Return the number of players of the input."""

    @abstractmethod
    def mask(self, x: Any, coalition: np.ndarray) -> Any:
        """Return ``x`` with the players that are ``False`` in ``coalition`` removed."""

    def collate(self, items: list[Any]) -> Any:
        """Combine masked inputs into a batch: stack arrays, otherwise keep a list."""
        return np.stack(items) if all(isinstance(i, np.ndarray) for i in items) else items


class FeatureMasker(Masker):
    """Tabular rows: absent features take a baseline value (e.g. the feature means)."""

    def __init__(self, baseline: np.ndarray | float = 0.0) -> None:
        """Initialize with a scalar or a vector of shape ``(n_features,)``."""
        self.baseline = np.asarray(baseline, dtype=float)

    def n_players(self, x: np.ndarray) -> int:
        """Every feature is a player."""
        return int(np.asarray(x).size)

    def mask(self, x: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """Replace the absent features by the baseline."""
        return np.where(coalition, np.asarray(x).reshape(-1), self.baseline)


class PatchMasker(Masker):
    """Images: players are square patches in a row-major grid; absent ones take a baseline."""

    def __init__(
        self, patch_size: int, baseline: np.ndarray | float = 0.0, *, channels_first: bool = True
    ) -> None:
        """Initialize the masker.

        Args:
            patch_size: Side length of a patch in pixels (border patches may be smaller).
            baseline: A scalar or array broadcastable to the image, e.g. the mean image.
            channels_first: Whether images are ``(C, H, W)`` (else ``(H, W, C)``).
        """
        if patch_size < 1:
            msg = "patch_size must be at least 1."
            raise ValueError(msg)
        self.patch_size = patch_size
        self.baseline = np.asarray(baseline, dtype=float)
        self.channels_first = channels_first

    def _shape(self, x: np.ndarray) -> tuple[int, int, int, int]:
        """Return the image height and width, and the number of patch rows and columns."""
        if x.ndim not in (_CHANNELS_NDIM - 1, _CHANNELS_NDIM):
            msg = f"An image needs 2 or 3 dimensions, got {x.ndim}."
            raise ValueError(msg)
        hwc = x.ndim == _CHANNELS_NDIM and not self.channels_first
        h, w = x.shape[:2] if hwc else x.shape[-2:]
        return h, w, -(-h // self.patch_size), -(-w // self.patch_size)

    def n_players(self, x: np.ndarray) -> int:
        """The number of patches."""
        *_, rows, cols = self._shape(np.asarray(x))
        return rows * cols

    def mask(self, x: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """Replace the pixels of absent patches by the baseline."""
        x = np.asarray(x)
        h, w, rows, cols = self._shape(x)
        p = self.patch_size
        pixels = np.repeat(np.repeat(coalition.reshape(rows, cols), p, axis=0), p, axis=1)[:h, :w]
        if x.ndim == _CHANNELS_NDIM:
            pixels = pixels[None] if self.channels_first else pixels[..., None]
        return np.where(pixels, x, self.baseline)


class CallableMasker(Masker):
    """Any data (text, graphs, ...): define players and masking with your own functions."""

    def __init__(
        self,
        n_players: int | Callable[[Any], int],
        mask_fn: Callable[[Any, np.ndarray], Any],
        collate_fn: Callable[[list[Any]], Any] | None = None,
        *,
        empty_ok: bool = True,
    ) -> None:
        """Initialize the masker.

        Args:
            n_players: The number of players, or a function of the input returning it.
            mask_fn: ``mask_fn(x, coalition)`` returns the masked input.
            collate_fn: Combines masked inputs into a batch (default: stack arrays, else a list).
            empty_ok: Whether the model accepts an input without any present player.
        """
        self._n_players, self._mask_fn, self._collate_fn = n_players, mask_fn, collate_fn
        self.empty_ok = empty_ok

    def n_players(self, x: Any) -> int:
        """Return the number of players."""
        return int(self._n_players(x) if callable(self._n_players) else self._n_players)

    def mask(self, x: Any, coalition: np.ndarray) -> Any:
        """Apply the mask function."""
        return self._mask_fn(x, coalition)

    def collate(self, items: list[Any]) -> Any:
        """Apply the collate function."""
        return self._collate_fn(items) if self._collate_fn else super().collate(items)
