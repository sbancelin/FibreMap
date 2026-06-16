"""Organization descriptors for resolved tissues (Livrable 3).

Assembles the recommended core triplet (structure-tensor order, correlation length ξ, defects)
into an ``OrganizationDescriptors`` record + a fixed-length descriptor vector for inter-tissue
comparison, with bootstrap confidence intervals on the order parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from collagen_shg.metrics.correlation import orientation_correlation
from collagen_shg.metrics.defects import defect_density
from collagen_shg.metrics.order import order_parameter_2d, order_tensor_3d
from collagen_shg.representations import conventions as cv

__all__ = [
    "OrganizationDescriptors",
    "organization_descriptors_3d",
    "bootstrap_order_ci",
    "descriptor_vector",
    "DESCRIPTOR_NAMES",
]

DESCRIPTOR_NAMES = ("S2", "S3", "xi_um", "defect_density", "fa_mean")


@dataclass
class OrganizationDescriptors:
    S2: float
    S3: float
    xi_um: float
    defect_density: float
    mean_phi: float
    fa_mean: float
    n_voxels: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _azimuth(director: np.ndarray) -> np.ndarray:
    return cv.wrap_axial(np.arctan2(director[1], director[0]))


def organization_descriptors_3d(
    director: np.ndarray,
    fa: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    max_r: int = 16,
) -> OrganizationDescriptors:
    """Compute organization descriptors from a 3D director field weighted by anisotropy."""
    director = np.asarray(director, dtype=np.float64)
    weights = np.asarray(fa, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    ot = order_tensor_3d(director, weights=weights)
    azimuth = _azimuth(director)
    op = order_parameter_2d(azimuth, weights=weights)

    corr = orientation_correlation(director, max_r=max_r)
    mean_voxel = float(np.mean(voxel_size_zyx))
    xi_um = float(corr.xi * mean_voxel) if np.isfinite(corr.xi) else float("inf")

    mid = director.shape[1] // 2
    defects = defect_density(azimuth[mid])

    return OrganizationDescriptors(
        S2=float(op.S2),
        S3=float(ot.S3),
        xi_um=xi_um,
        defect_density=float(defects.density),
        mean_phi=float(op.theta_bar),
        fa_mean=float(weights.mean()),
        n_voxels=int(director[0].size),
    )


def bootstrap_order_ci(
    director: np.ndarray,
    fa: np.ndarray,
    *,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    max_voxels: int = 20000,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals for ``S2`` and ``S3`` over informative voxels."""
    rng = rng or np.random.default_rng(0)
    flat_dir = np.asarray(director, dtype=np.float64).reshape(3, -1)
    w = np.asarray(fa, dtype=np.float64).reshape(-1)

    thresh = w.max() * 0.1 if w.max() > 0 else 0.0
    idx = np.flatnonzero(w > thresh)
    if idx.size == 0:
        idx = np.arange(w.size)
    if idx.size > max_voxels:
        idx = rng.choice(idx, size=max_voxels, replace=False)

    s2s = np.empty(n_boot)
    s3s = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(idx, size=idx.size, replace=True)
        d = flat_dir[:, sample]
        ww = w[sample]
        s3s[b] = order_tensor_3d(d, weights=ww).S3
        s2s[b] = order_parameter_2d(_azimuth(d), weights=ww).S2

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "S2": (float(np.percentile(s2s, lo)), float(np.percentile(s2s, hi))),
        "S3": (float(np.percentile(s3s, lo)), float(np.percentile(s3s, hi))),
    }


def descriptor_vector(desc: OrganizationDescriptors) -> np.ndarray:
    """Fixed-length feature vector (DESCRIPTOR_NAMES order) for PCA / classification."""
    xi = desc.xi_um if np.isfinite(desc.xi_um) else 1e3  # cap inf for a finite feature
    return np.array(
        [desc.S2, desc.S3, np.tanh(xi / 50.0), desc.defect_density, desc.fa_mean],
        dtype=np.float64,
    )
