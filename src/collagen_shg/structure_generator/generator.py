"""Tier 0 — procedural 3D structure generator (Livrable 2).

Places fibrils as parametric curves (the source of ground truth; cheap and deterministic),
rasterizes them into a continuous density + director volume, and records the **known**
organization metrics (computed from the generated structure with the Livrable 1 metrics) so the
closed validation loop can check that analyzers recover them.

Scope of this first version: a single global mean orientation with von Mises (axial) dispersion
``kappa``, optional sinusoidal crimp, and a diameter distribution. Domain structure / explicit
correlation length ``xi`` is recorded from configuration and refined later. Deterministic for a
given ``{config, seed}``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from collagen_shg.config.models import StructureConfig
from collagen_shg.metrics.order import order_parameter_2d, order_tensor_3d
from collagen_shg.representations import conventions as cv
from collagen_shg.representations.phantom import (
    DirectorFields,
    Fibril,
    GlobalGT,
    OrganizationGT,
    Phantom,
)

__all__ = ["ProceduralStructureGenerator"]


class ProceduralStructureGenerator:
    """Generate a ground-truth :class:`Phantom` from a :class:`StructureConfig` + RNG.

    Parameters
    ----------
    shape_zyx, voxel_size_zyx
        Volume geometry (voxels) and voxel size (µm).
    n_fibrils
        Number of fibrils to place (default scales with the in-plane area).
    samples_per_um
        Centerline sampling density used for rasterization.
    """

    def __init__(
        self,
        shape_zyx: tuple[int, int, int],
        voxel_size_zyx: tuple[float, float, float],
        *,
        n_fibrils: int | None = None,
        samples_per_um: float = 2.0,
    ) -> None:
        self.shape_zyx = tuple(int(s) for s in shape_zyx)
        self.voxel_size_zyx = tuple(float(s) for s in voxel_size_zyx)
        self.n_fibrils = n_fibrils
        self.samples_per_um = float(samples_per_um)

    # ----------------------------------------------------------------- public API
    def generate(self, config: StructureConfig, rng: np.random.Generator) -> Phantom:
        z, y, x = self.shape_zyx
        dz, dy, dx = self.voxel_size_zyx
        extent = np.array([x * dx, y * dy, z * dz])  # physical (x, y, z) µm

        mean_phi = cv.deg2rad(_get(config.orientation, "mean_phi_deg", 0.0))
        kappa = float(_get(config.orientation, "kappa", 4.0) or 4.0)
        elev_sigma = float(_get(config.orientation, "elevation_sigma", 0.0) or 0.0)
        diam_mean = float(_get(config.fibril.diameter_um, "mean", 1.0) or 1.0)
        diam_disp = float(_get(config.fibril.diameter_um, "dispersion", 0.0) or 0.0)
        crimp_amp = float(_get(config.fibril.crimp, "amplitude_um", 0.0) or 0.0)
        crimp_period = float(_get(config.fibril.crimp, "period_um", 0.0) or 0.0)

        n = self.n_fibrils if self.n_fibrils is not None else max(8, (x * y) // 256)

        # 1. sample per-fibril axial orientations (von Mises on the doubled angle => axial).
        phi = cv.wrap_axial(rng.vonmises(2.0 * mean_phi, kappa, size=n) / 2.0)
        theta = rng.normal(0.0, elev_sigma, size=n) if elev_sigma > 0 else np.zeros(n)
        directions = cv.director_from_angles(phi, theta)  # (n, 3) in (x, y, z)
        diameters = np.clip(rng.normal(diam_mean, diam_mean * diam_disp, size=n), 0.05, None)
        polarity = rng.choice((-1, 1), size=n)

        length = float(np.linalg.norm(extent))  # span the volume diagonal
        fibrils: list[Fibril] = []
        density = np.zeros(self.shape_zyx, dtype=np.float64)
        dir_accum = np.zeros((3, *self.shape_zyx), dtype=np.float64)
        pol_accum = np.zeros(self.shape_zyx, dtype=np.float64)

        for i in range(n):
            centre = rng.uniform(0, 1, size=3) * extent
            centerline, tangents = _build_centerline(
                centre, directions[i], length, crimp_amp, crimp_period,
                self.samples_per_um, rng,
            )
            self._deposit(centerline, tangents, polarity[i], density, dir_accum, pol_accum)
            fibrils.append(
                Fibril(
                    id=i,
                    centerline=centerline[:: max(1, len(centerline) // 24)],
                    diameter=np.full(
                        len(centerline[:: max(1, len(centerline) // 24)]), diameters[i]
                    ),
                    polarity=int(polarity[i]),
                )
            )

        fields = self._finalize_fields(density, dir_accum, pol_accum, diam_mean)
        ground_truth = self._ground_truth(phi, directions, config)

        meta_seed = int(rng.integers(0, 2**31 - 1))
        phantom = Phantom.empty(
            self.shape_zyx, self.voxel_size_zyx, seed=meta_seed,
            tissue_preset=config.preset, with_fields=False,
        )
        return Phantom(
            meta=phantom.meta, geometry=fibrils, fields=fields, ground_truth=ground_truth
        )

    # ----------------------------------------------------------------- internals
    def _deposit(self, centerline, tangents, polarity, density, dir_accum, pol_accum) -> None:
        dz, dy, dx = self.voxel_size_zyx
        z, y, x = self.shape_zyx
        ix = np.round(centerline[:, 0] / dx).astype(int)
        iy = np.round(centerline[:, 1] / dy).astype(int)
        iz = np.round(centerline[:, 2] / dz).astype(int)
        ok = (iz >= 0) & (iz < z) & (iy >= 0) & (iy < y) & (ix >= 0) & (ix < x)
        iz, iy, ix, t = iz[ok], iy[ok], ix[ok], tangents[ok]
        idx = (iz, iy, ix)
        np.add.at(density, idx, 1.0)
        for c in range(3):
            np.add.at(dir_accum[c], idx, t[:, c])
        np.add.at(pol_accum, idx, float(polarity))

    def _finalize_fields(self, density, dir_accum, pol_accum, diam_mean) -> DirectorFields:
        dz, dy, dx = self.voxel_size_zyx
        radius_vox = max(0.5, 0.5 * diam_mean / ((dx + dy) / 2))
        dens = gaussian_filter(density, sigma=(radius_vox, radius_vox, radius_vox))
        if dens.max() > 0:
            dens = dens / dens.max()

        norm = np.linalg.norm(dir_accum, axis=0)
        director = np.divide(
            dir_accum, norm, out=np.zeros_like(dir_accum), where=norm > 0
        ).astype(np.float32)
        # local order proxy: resultant length of deposited tangents
        counts = np.maximum(density, 1.0)
        order_S = (norm / counts).astype(np.float32)
        polarity = np.divide(
            pol_accum, np.maximum(density, 1.0), out=np.zeros_like(pol_accum),
            where=density > 0,
        ).astype(np.float32)
        return DirectorFields(
            director=director, order_S=np.clip(order_S, 0, 1),
            density=dens.astype(np.float32), polarity=np.clip(polarity, -1, 1),
        )

    def _ground_truth(self, phi, directions, config: StructureConfig) -> OrganizationGT:
        op2 = order_parameter_2d(phi)
        op3 = order_tensor_3d(directions.T)  # expects [3, N]
        xi_um = _get(config.orientation, "xi_um", None)
        gt = GlobalGT(
            S2=op2.S2,
            S3=op3.S3,
            kappa=op2.kappa,
            xi_um=float(xi_um) if xi_um is not None else None,
        )
        # mean orientation stored as an extra field
        gt_dict = gt.model_dump()
        gt_dict["mean_phi"] = op2.theta_bar
        return OrganizationGT(global_=GlobalGT(**gt_dict))


def _build_centerline(centre, direction, length, crimp_amp, crimp_period, samples_per_um, rng):
    """Build a (possibly crimped) centerline through ``centre`` along ``direction`` (µm)."""
    n_pts = max(4, int(length * samples_per_um))
    s = np.linspace(-length / 2, length / 2, n_pts)
    base = centre[None, :] + s[:, None] * direction[None, :]

    if crimp_amp > 0 and crimp_period > 0:
        e1 = _perpendicular(direction, rng)
        lateral = crimp_amp * np.sin(2 * np.pi * s / crimp_period)
        pts = base + lateral[:, None] * e1[None, :]
    else:
        pts = base

    tangents = np.gradient(pts, axis=0)
    tn = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, tn, out=np.tile(direction, (n_pts, 1)), where=tn > 0)
    return pts, tangents


def _perpendicular(direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ref = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(direction, ref)
    norm = np.linalg.norm(e1)
    return e1 / norm if norm > 0 else np.array([0.0, 1.0, 0.0])


def _get(obj, name, default):
    """Read an attribute (pydantic model) or key (dict), falling back to ``default``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, default)
    return default if val is None else val
