"""Family G — topological defects.

In the director field, disclinations of charge ±½ (in 2D) are detected by the winding number
along a closed contour; their density (count per unit area) is a powerful descriptor of
liquid-crystal–like tissues (cornea, cartilage arcades). Because orientations are axial
(period π), edge differences are wrapped to ``(−π/2, π/2]`` and the charge of a 2×2 plaquette is
``q = (1/2π) ∮ dθ`` ∈ {0, ±½, ...}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DefectResult", "defect_density"]


@dataclass
class DefectResult:
    density: float  # defects per unit area (per pixel^2)
    n_defects: int
    defect_map: np.ndarray  # per-plaquette charge, shape (Y-1, X-1)
    total_charge: float


def _axial_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest axial difference ``b − a`` wrapped into ``(−π/2, π/2]`` (period π)."""
    d = b - a
    return (d + np.pi / 2.0) % np.pi - np.pi / 2.0


def defect_density(
    orientation_field: np.ndarray, *, threshold: float = 0.25
) -> DefectResult:
    """Topological defect density of a 2D axial orientation field via plaquette winding number.

    Returns the per-plaquette charge map, the number of defects (``|q| > threshold``) and the
    density (defects per pixel²).
    """
    theta = np.asarray(orientation_field, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError(f"orientation_field must be 2D [Y, X], got shape {theta.shape}")

    a = theta[:-1, :-1]
    b = theta[:-1, 1:]
    c = theta[1:, 1:]
    d = theta[1:, :-1]
    loop = _axial_diff(a, b) + _axial_diff(b, c) + _axial_diff(c, d) + _axial_diff(d, a)
    charge = loop / (2.0 * np.pi)

    is_defect = np.abs(charge) > threshold
    n_defects = int(is_defect.sum())
    area = charge.size  # number of plaquettes ~ image area in pixel^2
    density = n_defects / area if area > 0 else 0.0
    return DefectResult(
        density=density,
        n_defects=n_defects,
        defect_map=charge,
        total_charge=float(charge[is_defect].sum()),
    )
