"""This module contains the image plot."""

from __future__ import annotations

from math import isqrt
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from ._config import BLUE, NEUTRAL, RED

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from shapiq.interaction_values import InteractionValues

__all__ = ["image_plot"]

_MAX_LABELED_PLAYERS = 16  # the regions are only labeled while the labels stay readable
_GRAYSCALE_NDIM = 2


def _grid_regions(height: int, width: int, rows: int, cols: int) -> np.ndarray:
    """Number the cells of a ``rows x cols`` grid over an image row by row, from 0."""
    row = np.arange(height) * rows // height
    col = np.arange(width) * cols // width
    return row[:, None] * cols + col[None, :]


def _get_regions(
    image: np.ndarray,
    n_players: int,
    grid: int | tuple[int, int] | None,
    regions: np.ndarray | None,
) -> np.ndarray:
    """Return the player of every pixel, either given or as a grid over the image."""
    height, width = image.shape[:2]
    if regions is None:
        if grid is None:  # a square grid is the default
            grid = isqrt(n_players)
        rows, cols = (grid, grid) if isinstance(grid, int) else grid
        if rows * cols != n_players:
            msg = f"A {rows} x {cols} grid has {rows * cols} regions, but there are {n_players} players."
            raise ValueError(msg)
        return _grid_regions(height, width, rows, cols)
    if grid is not None:
        msg = "Pass either grid or regions, not both."
        raise ValueError(msg)
    regions = np.asarray(regions)
    if regions.shape != (height, width):
        msg = f"regions must have the shape of the image {(height, width)}, got {regions.shape}."
        raise ValueError(msg)
    if regions.min() < 0 or regions.max() >= n_players:
        msg = f"regions must contain player indices from 0 to {n_players - 1}."
        raise ValueError(msg)
    return regions


def _draw_regions(
    fig: Figure,
    ax: Axes,
    image: np.ndarray,
    *,
    regions: np.ndarray,
    values: np.ndarray,
    names: Sequence[str],
    limit: float,
) -> None:
    """Draw the value of every region over the image."""
    cmap = LinearSegmentedColormap.from_list("image_plot", [BLUE.hex, NEUTRAL.hex, RED.hex])
    ax.imshow(image, cmap="gray" if image.ndim == _GRAYSCALE_NDIM else None)
    heat = ax.imshow(values[regions], cmap=cmap, vmin=-limit, vmax=limit, alpha=0.55)
    if len(values) <= _MAX_LABELED_PLAYERS:  # label every region at its center
        ys, xs = np.indices(regions.shape)
        counts = np.bincount(regions.ravel(), minlength=len(values))
        centers_y = np.bincount(regions.ravel(), ys.ravel(), len(values)) / np.maximum(counts, 1)
        centers_x = np.bincount(regions.ravel(), xs.ravel(), len(values)) / np.maximum(counts, 1)
        for player in map(int, np.flatnonzero(counts)):
            ax.text(
                centers_x[player],
                centers_y[player],
                f"{names[player]}\n{values[player]:+.3g}",
                ha="center",
                va="center",
                fontsize=9,
                bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1.5},
            )
    ax.set_axis_off()
    ax.set_title("Effect of each region", loc="left")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.02)


def _draw_effects(
    ax: Axes,
    effects: dict[tuple[int, ...], float],
    names: Sequence[str],
    limit: float,
    value_label: str | None,
) -> None:
    """Draw the strongest effects of single regions and of groups of regions as bars."""
    values = list(effects.values())
    colors = [RED.hex if value >= 0 else BLUE.hex for value in values]
    bars = ax.barh(range(len(values)), values, height=0.55, color=colors)
    ax.bar_label(bars, fmt="%+.3g", padding=3, fontsize=9)
    ax.set_yticks(range(len(values)), [" & ".join(names[i] for i in key) for key in effects])
    ax.invert_yaxis()
    ax.axvline(0, color="#c2c2c2", lw=1)
    ax.set_xlim(-limit * 1.6, limit * 1.6)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e7e7e7", lw=1)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    what = "interaction value" if value_label is None else f"effect on {value_label}"
    positive = "positive" if value_label is None else f"raises {value_label}"
    negative = "negative" if value_label is None else f"lowers {value_label}"
    has_groups = any(len(key) > 1 for key in effects)
    ax.set_xlabel(what)
    ax.set_title(
        "Strongest effects" + (" (regions and groups A & B)" if has_groups else " (regions)"),
        loc="left",
    )
    ax.legend(
        handles=[Patch(color=RED.hex, label=positive), Patch(color=BLUE.hex, label=negative)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )


def image_plot(
    interaction_values: InteractionValues,
    image: ArrayLike,
    *,
    grid: int | tuple[int, int] | None = None,
    regions: np.ndarray | None = None,
    feature_names: Sequence[str] | Mapping[int, str] | None = None,
    top_k: int = 8,
    title: str | None = None,
    value_label: str | None = None,
    show: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]] | None:
    """Plot the effects of the regions of an image on the image.

    The players are regions of the image: the cells of a grid or any other partition, e.g.
    superpixels. The left plot colors every region by its first-order value over the image
    (red for positive and blue for negative values), the right plot shows the ``top_k`` strongest
    effects of single regions and, if they were computed, of groups of regions.

    Args:
        interaction_values: The explanation of the regions as an InteractionValues object.
        image: The explained image, an array of shape ``(height, width)`` or ``(height, width, 3)``
            or anything that converts to it, e.g. a ``PIL`` image.
        grid: The regions are the cells of a grid over the image, numbered row by row from 0.
            An int ``g`` is a ``g x g`` grid, a tuple is ``(rows, columns)``. Defaults to a square
            grid if ``regions`` is not given.
        regions: Alternatively, an integer array of the shape of the image with the player of every
            pixel (e.g. from superpixels). Not together with ``grid``.
        feature_names: The names of the regions (list or dict mapping the player to its name). If
            no names are provided, the player indices are used. Defaults to ``None``.
        top_k: The number of effects in the bar chart. Defaults to ``8``.
        title: A headline, e.g. the predicted class. The index and the baseline value are
            appended. Defaults to ``None``.
        value_label: What the values explain, e.g. ``"uncertainty"`` or ``"the prediction"``. It
            appears in the axis label and the legend. Defaults to ``None``.
        show: Whether to show the plot. Defaults to ``False``.

    Returns:
        A tuple of the figure and the two axes (regions, effects) if ``show`` is ``False``,
        otherwise ``None``.

    Raises:
        ValueError: If the regions do not match the image or the players.

    Example:
        >>> iv = explainer.explain(image)  # an explanation with 9 players
        >>> fig, axes = image_plot(iv, image, grid=3, title="Predicted class: crocodile")
    """
    n_players = interaction_values.n_players
    image = np.asarray(image)
    regions = _get_regions(image, n_players, grid, regions)
    names = [str(i) for i in range(n_players)]
    if isinstance(feature_names, dict):
        for player, name in feature_names.items():
            names[player] = str(name)
    elif feature_names is not None:
        names = [str(name) for name in feature_names]

    single = np.array([interaction_values[(i,)] for i in range(n_players)])
    ranked = sorted(
        (k for k in interaction_values.interaction_lookup if k),
        key=lambda k: -abs(interaction_values[k]),
    )
    effects = {key: float(interaction_values[key]) for key in ranked[:top_k]}
    limit = max(float(np.abs(single).max()), *map(abs, effects.values()), 1e-12)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    _draw_regions(fig, left, image, regions=regions, values=single, names=names, limit=limit)
    _draw_effects(right, effects, names, limit, value_label)
    headline = f"{title}   |   " if title else ""
    fig.suptitle(
        f"{headline}{interaction_values.index}, baseline value: {interaction_values.baseline_value:.4g}",
        x=0.02,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()

    if not show:
        return fig, (left, right)
    plt.show()
    return None
