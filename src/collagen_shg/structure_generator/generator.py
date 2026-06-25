"""Tier 0 — procedural 3D structure generator (Livrable 2, architecture-aware).

Grows fibrils by **following the architecture's mean-orientation field** (so arcades / tubular
sheaths produce curved fibrils), with per-fibril biaxial dispersion (``kappa_par``/``kappa_perp``),
worm-like waviness (persistence length ``Lp``) and optional periodic crimp. Fibrils are rasterized
as **solid binary tubes** (capsules of the fibril diameter) into an occupancy volume — real fibril
structures in a defined volume, not points. Records the known organization ground truth (order
tensor → S, biaxiality, mean direction; ξ; defect density; achieved volume fraction) for the
closed loop. Deterministic for a given ``{config, seed}``.
"""

from __future__ import annotations

import numpy as np

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
from collagen_shg.structure_generator.architecture import (
    build_architecture,
    sample_axial_directions,
)

__all__ = ["ProceduralStructureGenerator"]


class ProceduralStructureGenerator:
    """Generate a ground-truth :class:`Phantom` from a :class:`StructureConfig` + RNG.

    Parameters
    ----------
    shape_zyx, voxel_size_zyx
        Volume geometry (voxels) and voxel size (µm).
    n_fibrils
        Number of fibrils. If ``None``, derived from the target volume fraction ``density``
        (config ``volume_fraction``) and the mean fibril volume.
    step_um
        Centerline growth step (µm).
    """

    def __init__(
        self,
        shape_zyx: tuple[int, int, int],
        voxel_size_zyx: tuple[float, float, float],
        *,
        n_fibrils: int | None = None,
        step_um: float | None = None,
    ) -> None:
        self.shape_zyx = tuple(int(s) for s in shape_zyx)
        self.voxel_size_zyx = tuple(float(s) for s in voxel_size_zyx)
        self.n_fibrils = n_fibrils
        # Coarse enough for speed (capsules fill continuously between points), fine enough to
        # follow curved architectures (arcades / tubular).
        self.step_um = step_um or max(0.5, min(self.voxel_size_zyx))

    # ----------------------------------------------------------------- public API
    def generate(self, config: StructureConfig, rng: np.random.Generator) -> Phantom:
        z, y, x = self.shape_zyx
        dz, dy, dx = self.voxel_size_zyx
        extent = np.array([x * dx, y * dy, z * dz])  # physical (x, y, z) µm

        spec = _read_spec(config)
        populations = build_architecture(spec["architecture"], spec["params"],
                                         self.shape_zyx, self.voxel_size_zyx)
        weights = np.array([p.weight for p in populations], dtype=np.float64)
        weights = weights / weights.sum()

        diam_mean = spec["diameter_um"]
        length_mean = spec["length_um"] if spec["length_um"] > 0 else float(np.linalg.norm(extent))
        n = self.n_fibrils if self.n_fibrils is not None else _n_from_fraction(
            spec["volume_fraction"], extent, diam_mean, length_mean
        )

        occupancy = np.zeros(self.shape_zyx, dtype=np.float32)
        director_field = np.zeros((3, *self.shape_zyx), dtype=np.float32)
        polarity_field = np.zeros(self.shape_zyx, dtype=np.float32)
        fibrils: list[Fibril] = []
        sampled_dirs = np.zeros((n, 3))

        for i in range(n):
            pop = populations[rng.choice(len(populations), p=weights)]
            seed = rng.uniform(0, 1, size=3) * extent
            mean0 = pop.field.at(seed[None])[0]
            base = sample_axial_directions(
                mean0[None], spec["kappa_par"], spec["kappa_perp"], rng
            )[0]
            sampled_dirs[i] = base

            diameter = max(0.05, rng.normal(diam_mean, diam_mean * spec["diameter_cv"]))
            length = max(self.step_um * 3, _sample_length(length_mean, spec["length_cv"], rng))
            polarity = int(rng.choice((-1, 1)))

            centerline, tangents = _grow_fibril(
                pop.field, seed, base, mean0, length, self.step_um,
                spec["persistence_um"], spec["crimp_amplitude_um"], spec["crimp_period_um"], rng,
            )
            self._rasterize(centerline, tangents, diameter, polarity,
                            occupancy, director_field, polarity_field)
            keep = max(1, len(centerline) // 24)
            fibrils.append(
                Fibril(id=i, centerline=centerline[::keep],
                       diameter=np.full(len(centerline[::keep]), diameter), polarity=polarity)
            )

        fields = DirectorFields(
            director=director_field,
            order_S=occupancy.copy(),
            density=occupancy,
            polarity=polarity_field,
        )
        ground_truth = _ground_truth(sampled_dirs, spec, float(occupancy.mean()))

        meta_seed = int(rng.integers(0, 2**31 - 1))
        meta_phantom = Phantom.empty(self.shape_zyx, self.voxel_size_zyx, seed=meta_seed,
                                     tissue_preset=config.preset, with_fields=False)
        return Phantom(meta=meta_phantom.meta, geometry=fibrils, fields=fields,
                       ground_truth=ground_truth)

    # ----------------------------------------------------------------- rasterization
    def _rasterize(self, centerline, tangents, diameter, polarity,
                   occupancy, director_field, polarity_field) -> None:
        radius = max(0.5 * diameter, 0.3 * min(self.voxel_size_zyx))
        for k in range(len(centerline) - 1):
            _rasterize_capsule(occupancy, director_field, polarity_field,
                               centerline[k], centerline[k + 1], radius, tangents[k],
                               float(polarity), self.voxel_size_zyx, self.shape_zyx)


# --------------------------------------------------------------------------- fibril growth
def _grow_fibril(field, seed, base_dir, mean0, length, ds, persistence_um,
                 crimp_amp, crimp_period, rng):
    """Grow a centerline following ``field`` + a per-fibril offset, with Lp + crimp waviness."""
    offset = base_dir - mean0  # constant per-fibril tilt from the local mean
    n_half = max(2, int(0.5 * length / ds))

    def walk(sign: int):
        pts = []
        p = seed.copy()
        d = base_dir * sign
        for _ in range(n_half):
            local = field.at(p[None])[0]
            d = local * sign + offset
            if np.isfinite(persistence_um) and persistence_um > 0:
                wobble = rng.normal(0, np.sqrt(2 * ds / persistence_um), 3)
                d = d + wobble
            nd = np.linalg.norm(d)
            d = d / nd if nd > 0 else base_dir * sign
            p = p + d * ds
            pts.append(p.copy())
        return pts

    fwd = walk(+1)
    bwd = walk(-1)
    pts = np.array(bwd[::-1] + [seed] + fwd)

    if crimp_amp > 0 and crimp_period > 0 and len(pts) > 2:
        axis = pts[-1] - pts[0]
        an = np.linalg.norm(axis)
        if an > 0:
            axis = axis / an
            e1 = _perp(axis)
            s = np.arange(len(pts)) * ds
            pts = pts + (crimp_amp * np.sin(2 * np.pi * s / crimp_period))[:, None] * e1[None, :]

    tangents = np.gradient(pts, axis=0)
    tn = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, tn, out=np.tile(base_dir, (len(pts), 1)), where=tn > 0)
    return pts, tangents


def _rasterize_capsule(occupancy, director_field, polarity_field, a, b, radius, tangent,
                       polarity, voxel, shape):
    """Mark voxels within ``radius`` (µm) of segment a-b as occupied (a solid binary tube)."""
    dz, dy, dx = voxel
    z, y, x = shape
    lo = np.minimum(a, b) - radius
    hi = np.maximum(a, b) + radius
    ix0, ix1 = max(0, int(lo[0] / dx)), min(x - 1, int(np.ceil(hi[0] / dx)))
    iy0, iy1 = max(0, int(lo[1] / dy)), min(y - 1, int(np.ceil(hi[1] / dy)))
    iz0, iz1 = max(0, int(lo[2] / dz)), min(z - 1, int(np.ceil(hi[2] / dz)))
    if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return

    zc = np.arange(iz0, iz1 + 1) * dz
    yc = np.arange(iy0, iy1 + 1) * dy
    xc = np.arange(ix0, ix1 + 1) * dx
    zg, yg, xg = np.meshgrid(zc, yc, xc, indexing="ij")
    g = np.stack([xg, yg, zg], axis=-1)  # physical (x, y, z)

    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0:
        dist = np.linalg.norm(g - a, axis=-1)
    else:
        t = np.clip(((g - a) @ ab) / denom, 0.0, 1.0)
        proj = a + t[..., None] * ab
        dist = np.linalg.norm(g - proj, axis=-1)
    mask = dist <= radius
    if not mask.any():
        return

    occ = occupancy[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]
    occ[mask] = 1.0
    pol = polarity_field[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]
    pol[mask] = polarity
    for c in range(3):
        sub = director_field[c, iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]
        sub[mask] = tangent[c]


# --------------------------------------------------------------------------- helpers
def _perp(direction: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(direction, ref)
    n = np.linalg.norm(e1)
    return e1 / n if n > 0 else np.array([0.0, 1.0, 0.0])


def _sample_length(mean: float, cv: float, rng: np.random.Generator) -> float:
    if cv <= 0:
        return mean
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return float(rng.lognormal(mu, sigma))


def _n_from_fraction(volume_fraction, extent, diameter, length) -> int:
    vol = float(np.prod(extent))
    fiber_vol = np.pi * (0.5 * diameter) ** 2 * length
    if fiber_vol <= 0 or volume_fraction <= 0:
        return 64
    return max(1, int(volume_fraction * vol / fiber_vol))


def _ground_truth(sampled_dirs, spec, achieved_fraction) -> OrganizationGT:
    op2 = order_parameter_2d(cv.wrap_axial(np.arctan2(sampled_dirs[:, 1], sampled_dirs[:, 0])))
    ot = order_tensor_3d(sampled_dirs.T)
    eig = np.sort(np.linalg.eigvalsh(ot.Q))[::-1]  # descending
    S = float(eig[0])
    biaxiality = float(eig[1] - eig[2])
    phi0, theta0 = cv.angles_from_director(ot.director)

    gt = GlobalGT(S2=op2.S2, S3=ot.S3, kappa=op2.kappa, xi_um=spec["xi_um"])
    extra = gt.model_dump()
    extra.update(
        mean_phi=float(phi0), mean_theta=float(theta0),
        S=S, biaxiality=biaxiality, architecture=spec["architecture"],
        volume_fraction=float(achieved_fraction),
    )
    return OrganizationGT(global_=GlobalGT(**extra))


def _read_spec(config: StructureConfig) -> dict:
    """Extract architecture + dispersion + morphology params (backward compatible defaults)."""
    orient = config.orientation
    fibril = config.fibril
    arch = getattr(config, "architecture", None)
    if isinstance(arch, dict):
        arch_name = arch.get("type", "uniaxial")
        params = dict(arch)
        params.pop("type", None)
    else:
        arch_name = arch or "uniaxial"
        params = {}
    params.setdefault("mean_phi_deg", _get(orient, "mean_phi_deg", 0.0))

    kappa = float(_get(orient, "kappa", 4.0))
    if arch_name == "isotropic":
        kappa = 0.0
    return {
        "architecture": arch_name,
        "params": params,
        "kappa_par": float(_get(orient, "kappa_par", kappa)),
        "kappa_perp": float(_get(orient, "kappa_perp", kappa)),
        "xi_um": _get(orient, "xi_um", None),
        "diameter_um": float(_get(fibril.diameter_um, "mean", 1.0)),
        "diameter_cv": float(_get(fibril.diameter_um, "dispersion", 0.0)),
        "length_um": float(_get(fibril, "length_um", 0.0)),
        "length_cv": float(_get(fibril, "length_cv", 0.0)),
        "persistence_um": float(_get(fibril, "persistence_um", np.inf)),
        "crimp_amplitude_um": float(_get(fibril.crimp, "amplitude_um", 0.0)),
        "crimp_period_um": float(_get(fibril.crimp, "period_um", 0.0)),
        "volume_fraction": float(_get(config, "volume_fraction", 0.1)),
    }


def _get(obj, name, default):
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(name, default)
    else:
        v = getattr(obj, name, default)
    return default if v is None else v