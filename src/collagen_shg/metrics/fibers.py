"""Family F — per-fibre descriptors and waviness.

When fibres are resolved and extracted, per-fibre and network descriptors are computed.
Straightness ``s = D/L`` (and tortuosity ``τ = L/D``) quantify global waviness; the persistence
length ``Lp`` follows from the tangent–tangent correlation (worm-like chain model). This family
is the richest but the most sensitive to extraction quality; it requires resolved fibres.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from collagen_shg.representations import conventions as cv

__all__ = [
    "FiberRecord",
    "FiberNetwork",
    "FiberMetricsResult",
    "fiber_metrics",
    "persistence_length",
]


@dataclass
class FiberRecord:
    id: int
    n_points: int
    length: float  # arc length L (µm)
    end_to_end: float  # straight-line distance D (µm)
    straightness: float  # s = D / L in [0, 1]
    tortuosity: float  # tau = L / D >= 1
    azimuth: float  # principal-axis azimuth phi in [0, pi)
    elevation: float  # principal-axis elevation theta in [-pi/2, pi/2]


@dataclass
class FiberNetwork:
    n_fibers: int
    total_length: float
    mean_straightness: float
    mean_tortuosity: float


@dataclass
class FiberMetricsResult:
    per_fiber: list[FiberRecord] = field(default_factory=list)
    network: FiberNetwork | None = None

    def to_table(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.per_fiber]


def _as_centerline(item: Any) -> tuple[int, np.ndarray]:
    """Accept a Fibril, an (N,3) array, or an (id, array) pair → (id, centerline)."""
    if hasattr(item, "centerline"):
        return int(getattr(item, "id", 0)), np.asarray(item.centerline, dtype=np.float64)
    if isinstance(item, tuple) and len(item) == 2:
        return int(item[0]), np.asarray(item[1], dtype=np.float64)
    return 0, np.asarray(item, dtype=np.float64)


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered
    return cv.major_eigenvector(cov)  # dominant direction of the point cloud


def fiber_metrics(centerlines: Any) -> FiberMetricsResult:
    """Per-fibre records + network statistics for a collection of centerlines."""
    records: list[FiberRecord] = []
    for i, item in enumerate(centerlines):
        fid, cl = _as_centerline(item)
        if cl.ndim != 2 or cl.shape[1] != 3 or cl.shape[0] < 2:
            raise ValueError(f"centerline {i} must be (N>=2, 3), got {cl.shape}")
        seg = np.diff(cl, axis=0)
        length = float(np.linalg.norm(seg, axis=1).sum())
        end_to_end = float(np.linalg.norm(cl[-1] - cl[0]))
        straightness = end_to_end / length if length > 0 else 0.0
        tortuosity = length / end_to_end if end_to_end > 0 else float("inf")
        axis = _principal_axis(cl)
        phi, theta = cv.angles_from_director(axis)
        records.append(
            FiberRecord(
                id=fid if fid else i,
                n_points=int(cl.shape[0]),
                length=length,
                end_to_end=end_to_end,
                straightness=straightness,
                tortuosity=tortuosity,
                azimuth=float(phi),
                elevation=float(theta),
            )
        )

    if records:
        total = float(sum(r.length for r in records))
        network = FiberNetwork(
            n_fibers=len(records),
            total_length=total,
            mean_straightness=float(np.mean([r.straightness for r in records])),
            mean_tortuosity=float(np.mean([r.tortuosity for r in records])),
        )
    else:
        network = FiberNetwork(0, 0.0, 0.0, 0.0)
    return FiberMetricsResult(per_fiber=records, network=network)


def persistence_length(centerline: np.ndarray, *, n_bins: int = 20) -> float:
    """Persistence length Lp from the tangent–tangent correlation (worm-like chain).

    ``<t(s)·t(s+Δ)> = exp(−Δ/Lp)``; fit over arc-length lag Δ. Returns ``inf`` for a straight
    fibre (no decay). Same length units as the centerline (µm).
    """
    cl = np.asarray(centerline, dtype=np.float64)
    if cl.ndim != 2 or cl.shape[1] != 3 or cl.shape[0] < 3:
        raise ValueError(f"centerline must be (N>=3, 3), got {cl.shape}")
    seg = np.diff(cl, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    keep = seglen > 0
    seg, seglen = seg[keep], seglen[keep]
    t = seg / seglen[:, None]
    s = np.cumsum(seglen) - seglen / 2.0  # arc length at tangent midpoints

    i, j = np.triu_indices(t.shape[0], k=0)
    lags = s[j] - s[i]
    dots = np.einsum("kd,kd->k", t[i], t[j])

    max_lag = float(lags.max())
    if max_lag <= 0:
        return float("inf")
    edges = np.linspace(0, max_lag, n_bins + 1)
    which = np.clip(np.digitize(lags, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(which, weights=dots, minlength=n_bins)
    counts = np.bincount(which, minlength=n_bins)
    valid = counts > 0
    centers = 0.5 * (edges[:-1] + edges[1:])[valid]
    corr = (sums[valid] / counts[valid])

    if corr.min() > 0.999:  # essentially straight
        return float("inf")

    def model(x, Lp):
        return np.exp(-x / Lp)

    try:
        popt, _ = curve_fit(
            model, centers, corr, p0=[max_lag / 2 or 1.0], bounds=(1e-6, 1e9), maxfev=10000
        )
        return float(popt[0])
    except (RuntimeError, ValueError):
        return float("inf")
