"""Organization metrics (Livrable 1) — interface only.

Target signatures from the Livrable 1 implementation contract (Tableau 3). Axes/angle
conventions and types (``Phantom``, director fields) are those of Phase 0. Families A–G
(structure tensor, order parameters, orientation correlation ξ, Fourier, texture, per-fiber,
topological defects) are implemented in Livrable 1; here only the stable function surface is
fixed so downstream code can be written against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "structure_tensor_2d",
    "structure_tensor_3d",
    "order_parameter_2d",
    "order_tensor_3d",
    "orientation_correlation",
    "power_spectrum_orientation",
    "glcm_features",
    "lbp_histogram",
    "gabor_energy",
    "fiber_metrics",
    "persistence_length",
    "defect_density",
]

_NI = "metrics family lands in Livrable 1"


def structure_tensor_2d(image: np.ndarray, sigma: float, rho: float) -> tuple[Any, Any]:
    """Family A (2D): image -> (orientation, coherence). Fiber axis = minor eigenvector."""
    raise NotImplementedError(_NI)


def structure_tensor_3d(volume: np.ndarray, sigma: float, rho: float) -> tuple[Any, Any]:
    """Family A (3D): volume -> (director[3,Z,Y,X], fractional anisotropy)."""
    raise NotImplementedError(_NI)


def order_parameter_2d(orientation: np.ndarray, weights: np.ndarray | None = None) -> Any:
    """Family B (2D): theta field (+weights) -> (S2, theta_bar, kappa) via doubled angle."""
    raise NotImplementedError(_NI)


def order_tensor_3d(director: np.ndarray, weights: np.ndarray | None = None) -> Any:
    """Family B (3D): director field (+weights) -> (S3, mean director, Saupe Q)."""
    raise NotImplementedError(_NI)


def orientation_correlation(field: np.ndarray, max_r: int) -> Any:
    """Family C: theta or director field -> (C(r), xi, plateau S2**2)."""
    raise NotImplementedError(_NI)


def power_spectrum_orientation(image: np.ndarray) -> Any:
    """Family D: image -> (A(phi), orientation histogram, characteristic spacing)."""
    raise NotImplementedError(_NI)


def glcm_features(image: np.ndarray, distances: Any, angles: Any) -> dict:
    """Family E: image -> Haralick features dict (+ directional anisotropy)."""
    raise NotImplementedError(_NI)


def lbp_histogram(image: np.ndarray, P: int, R: float) -> Any:
    """Family E: image -> LBP histogram."""
    raise NotImplementedError(_NI)


def gabor_energy(image: np.ndarray, freqs: Any, angles: Any) -> Any:
    """Family E: image -> (E[f, phi], orientation)."""
    raise NotImplementedError(_NI)


def fiber_metrics(centerlines: Any) -> Any:
    """Family F: centerlines -> per-fiber table + network statistics."""
    raise NotImplementedError(_NI)


def persistence_length(centerline: np.ndarray) -> float:
    """Family F: centerline -> persistence length Lp."""
    raise NotImplementedError(_NI)


def defect_density(orientation_field: np.ndarray) -> Any:
    """Family G: theta field -> (density, defect map) via winding number."""
    raise NotImplementedError(_NI)