"""Multi-scale orientation field (Livrable 3).

Computes the structure-tensor orientation at several integration scales ``rho`` and selects, per
voxel, the scale of strongest local structure (max fractional anisotropy). This reveals
organization across scales while staying robust in dense regions — the field-based "short-cut"
to extraction that the spec recommends for the degree of organization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from collagen_shg.metrics.structure_tensor import (
    StructureTensor2DResult,
    structure_tensor_2d,
    structure_tensor_3d,
)

__all__ = [
    "OrientationField3D",
    "multiscale_orientation_3d",
    "orientation_field_2d",
]


@dataclass
class OrientationField3D:
    director: np.ndarray  # [3, Z, Y, X] fibre axis (minor eigenvector)
    fa: np.ndarray  # [Z, Y, X] fractional anisotropy (max over scales)
    scale_index: np.ndarray  # [Z, Y, X] index of the selected rho per voxel


def multiscale_orientation_3d(
    volume: np.ndarray, *, sigma: float = 1.0, rhos: Sequence[float] = (1.0, 2.0, 4.0)
) -> OrientationField3D:
    """3D orientation field selecting, per voxel, the rho scale with maximal anisotropy."""
    results = [structure_tensor_3d(volume, sigma, rho) for rho in rhos]
    fa_stack = np.stack([r.fa for r in results], axis=0)  # (S, Z, Y, X)
    best = np.argmax(fa_stack, axis=0)  # (Z, Y, X)

    director = np.zeros_like(results[0].director)
    for s, r in enumerate(results):
        mask = best == s
        director[:, mask] = r.director[:, mask]
    fa = fa_stack.max(axis=0)
    return OrientationField3D(director=director, fa=fa, scale_index=best)


def orientation_field_2d(
    image: np.ndarray, *, sigma: float = 1.0, rho: float = 4.0
) -> StructureTensor2DResult:
    """2D orientation field (azimuth + coherence) — thin wrapper over the structure tensor."""
    return structure_tensor_2d(image, sigma, rho)
