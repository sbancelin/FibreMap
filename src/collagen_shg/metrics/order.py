"""Family B — order parameters and circular statistics.

Fibres are axial (period π), so the scalar degree of order is computed on the **doubled angle**.
In 2D, ``S2`` is the mean resultant length of ``exp(i·2θ)`` (0 isotropic → 1 perfectly aligned),
``theta_bar`` the mean orientation, and ``kappa`` the von Mises (axial) concentration related to
``S2`` by the Bessel ratio ``I1/I0``. In 3D, the nematic order tensor (de Saupe) ``Q`` has
largest eigenvalue ``S3`` with the mean director as its eigenvector (axial → use Watson/Bingham,
never von Mises–Fisher).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collagen_shg.representations import conventions as cv

__all__ = [
    "OrderParameter2D",
    "OrderTensor3D",
    "order_parameter_2d",
    "order_tensor_3d",
    "vonmises_kappa_from_R",
]


@dataclass
class OrderParameter2D:
    S2: float  # mean resultant length of the doubled angle, in [0, 1]
    theta_bar: float  # mean axial orientation in [0, pi)
    kappa: float  # von Mises (axial) concentration


@dataclass
class OrderTensor3D:
    S3: float  # largest eigenvalue of the Saupe tensor (1 aligned, 0 isotropic)
    director: np.ndarray  # (3,) mean director (x, y, z), the associated eigenvector
    Q: np.ndarray  # (3, 3) Saupe nematic order tensor


def vonmises_kappa_from_R(R: float) -> float:
    """Best–Fisher approximation of the von Mises concentration κ from resultant length R."""
    R = float(np.clip(R, 0.0, 1.0 - 1e-12))
    if R < 0.53:
        return 2.0 * R + R**3 + 5.0 * R**5 / 6.0
    if R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1.0 - R)
    return 1.0 / (R**3 - 4.0 * R**2 + 3.0 * R)


def order_parameter_2d(
    orientation: np.ndarray, weights: np.ndarray | None = None
) -> OrderParameter2D:
    """Axial order from a θ field via the doubled angle → (S2, theta_bar, kappa).

    ``weights`` (e.g. structure-tensor coherence) defaults to uniform. Means are weighted.
    """
    theta = np.asarray(orientation, dtype=np.float64).ravel()
    w = np.ones_like(theta) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    wsum = w.sum()
    if wsum <= 0:
        raise ValueError("weights sum to zero")
    c = float(np.sum(w * np.cos(2.0 * theta)) / wsum)
    s = float(np.sum(w * np.sin(2.0 * theta)) / wsum)
    R = float(np.hypot(c, s))
    theta_bar = float(cv.angle_from_doubled(c, s))
    return OrderParameter2D(S2=R, theta_bar=theta_bar, kappa=vonmises_kappa_from_R(R))


def order_tensor_3d(
    director: np.ndarray, weights: np.ndarray | None = None
) -> OrderTensor3D:
    """Saupe nematic order tensor of a director field → (S3, mean director, Q).

    ``director`` is ``[3, ...]`` (e.g. ``[3, Z, Y, X]``); non-unit columns are normalized.
    ``Q = <(3 n nᵀ − I) / 2>``; ``S3`` is its largest eigenvalue.
    """
    n = np.asarray(director, dtype=np.float64)
    if n.shape[0] != 3:
        raise ValueError(f"director must have shape [3, ...], got {n.shape}")
    n = n.reshape(3, -1)
    norm = np.linalg.norm(n, axis=0)
    valid = norm > 0
    n = n[:, valid] / norm[valid]
    if weights is None:
        w = np.ones(n.shape[1], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()[valid]
    wsum = w.sum()
    if wsum <= 0:
        raise ValueError("weights sum to zero (no valid directors)")

    second_moment = np.einsum("m,im,jm->ij", w, n, n) / wsum  # <n nᵀ>
    Q = (3.0 * second_moment - np.eye(3)) / 2.0
    eigvals, eigvecs = np.linalg.eigh(Q)
    S3 = float(eigvals[-1])
    mean_director = cv._canonical_sign(eigvecs[:, -1])
    return OrderTensor3D(S3=S3, director=mean_director, Q=Q)
