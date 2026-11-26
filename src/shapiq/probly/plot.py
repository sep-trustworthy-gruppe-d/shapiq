# ruff: noqa: ANN401
"""Images of the players for the network plot of shapiq."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["player_images"]


def player_images(image: np.ndarray, grid: int) -> tuple[Any, list[Any]]:
    """Return the image with a numbered grid and the numbered crop of every player.

    The players are numbered row by row from 0, as in the plots of shapiq.
    """
    from PIL import Image, ImageDraw, ImageFont

    def label(picture: Any, text: str, xy: tuple[float, float]) -> None:
        font = ImageFont.load_default(size=max(14, picture.height // 8))
        ImageDraw.Draw(picture).text(
            xy, text, fill="white", font=font, stroke_width=2, stroke_fill="black", anchor="mm"
        )

    overview = Image.fromarray(np.asarray(image)).convert("RGB")
    width, height = overview.size
    crops = []
    for player in range(grid * grid):
        row, col = divmod(player, grid)
        box = (
            col * width // grid,
            row * height // grid,
            (col + 1) * width // grid,
            (row + 1) * height // grid,
        )
        crop = overview.crop(box)
        label(crop, str(player), (crop.width / 2, crop.height / 2))
        crops.append(crop)
    draw = ImageDraw.Draw(overview)
    for k in range(1, grid):
        draw.line([(k * width / grid, 0), (k * width / grid, height)], fill="white", width=2)
        draw.line([(0, k * height / grid), (width, k * height / grid)], fill="white", width=2)
    for player in range(grid * grid):
        row, col = divmod(player, grid)
        label(overview, str(player), ((col + 0.5) * width / grid, (row + 0.5) * height / grid))
    return overview, crops
