"""Orientation visualization helpers (pure NumPy; no napari/matplotlib dependency).

Maps a director field to an RGB image where hue encodes the axial azimuth ``φ ∈ [0, π)`` and
brightness encodes a weight (local order or director magnitude). Used by the GUI to show the
orientation field, and unit-tested without the optional GUI stack.
"""

from __future__ import annotations

import numpy as np

from collagen_shg.representations import conventions as cv

__all__ = ["director_to_rgb", "hsv_to_rgb"]


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV→RGB. Inputs broadcast to a common shape; returns ``(..., 3)`` in [0, 1]."""
    h, s, v = np.broadcast_arrays(h, s, v)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def director_to_rgb(director: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Director field ``[3, ...]`` → RGB ``[..., 3]``: hue = azimuth, brightness = weight.

    ``weight`` (e.g. ``order_S``) sets brightness; if omitted, the director magnitude is used.
    """
    director = np.asarray(director, dtype=np.float64)
    azimuth = cv.wrap_axial(np.arctan2(director[1], director[0]))  # [0, pi)
    hue = azimuth / np.pi
    if weight is None:
        weight = np.linalg.norm(director, axis=0)
    weight = np.asarray(weight, dtype=np.float64)
    wmax = weight.max()
    value = weight / wmax if wmax > 0 else weight
    sat = np.ones_like(hue)
    return hsv_to_rgb(hue, sat, value)
