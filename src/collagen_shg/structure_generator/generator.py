"""Tier 0 — procedural 3D structure generator (Livrable 2, architecture-aware).

Grows fibrils by **following the architecture's mean-orientation field** (so arcades / tubular
sheaths produce curved fibrils), with per-fibril biaxial dispersion (``kappa_par``/``kappa_perp``),
worm-like waviness (persistence length ``Lp``) and optional periodic crimp. Fibrils are rasterized
as **solid binary tubes** (capsules of the fibril diameter) into an occupancy volume.

Optional network features (all off by default) make the structures span more tissue archetypes:

- **volume exclusion** — fibrils do not interpenetrate (each voxel has a single owner) and growth
  stops at collisions; packing then caps the achievable volume fraction (cornea, tendon);
- **branching / crosslinking** — fibrils spawn child branches and short connectors bridge nearby
  fibrils (dermis basket-weave, gels), producing branch/crossing points and network connectivity;
- **hierarchy** — fibrils are grouped into fibers, fibers into fascicles (tendon-like bundling),
  recorded via ``fiber_id`` / ``fascicle_id``.

Records the known organization ground truth (order tensor → S, biaxiality, mean direction; ξ;
achieved volume fraction). Deterministic for a given ``{config, seed}``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


@dataclass
class _Seed:
    """A pending fibril to grow."""

    pos: np.ndarray
    base_dir: np.ndarray
    field: Any
    diameter: float
    length: float
    polarity: int
    fiber_id: int | None
    fascicle_id: int | None
    generation: int
    type: str


class ProceduralStructureGenerator:
    """Generate a ground-truth :class:`Phantom` from a :class:`StructureConfig` + RNG."""

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
        self.step_um = step_um or max(0.5, min(self.voxel_size_zyx))

    # ----------------------------------------------------------------- public API
    def generate(
        self,
        config: StructureConfig,
        rng: np.random.Generator,
        *,
        progress: Callable[[float], None] | None = None,
    ) -> Phantom:
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

        occupancy = np.zeros(self.shape_zyx, dtype=np.float32)
        director_field = np.zeros((3, *self.shape_zyx), dtype=np.float32)
        polarity_field = np.zeros(self.shape_zyx, dtype=np.float32)
        owner = np.full(self.shape_zyx, -1, dtype=np.int32) if spec["exclusion"] else None
        blocked = self._make_blocked(owner)

        # --- plan the initial fibrils (hierarchical bundles, or n flat fibrils) ---
        queue: deque[_Seed | None] = deque()
        if spec["hierarchy"]["enabled"]:
            for s in self._plan_hierarchy(spec, populations, weights, extent,
                                          diam_mean, length_mean, rng):
                queue.append(s)
        else:
            n = self.n_fibrils if self.n_fibrils is not None else _n_from_fraction(
                spec["volume_fraction"], extent, diam_mean, length_mean
            )
            queue.extend([None] * n)  # placeholders: flat seeds sampled at process time

        fibrils: list[Fibril] = []
        sampled: list[np.ndarray] = []
        next_id = 0
        processed = 0
        while queue:
            item = queue.popleft()
            seed = item if item is not None else self._flat_seed(
                spec, populations, weights, extent, diam_mean, length_mean, owner, rng
            )
            fib, base, children = self._grow_place(
                seed, spec, occupancy, director_field, polarity_field, owner, blocked, next_id, rng
            )
            if fib is not None:
                fibrils.append(fib)
                sampled.append(base)
                next_id += 1
                queue.extend(children)
            processed += 1
            if progress is not None and processed % 16 == 0:
                progress(min(0.98, processed / (processed + len(queue) + 1)))

        # --- crosslinks: short connectors between nearby fibrils ---
        if spec["crosslink_density"] > 0 and len(fibrils) >= 2:
            next_id = self._add_crosslinks(
                fibrils, spec, extent, occupancy, director_field, polarity_field,
                owner, next_id, rng
            )

        sampled_dirs = np.array(sampled) if sampled else np.array([[1.0, 0.0, 0.0]])
        fields = DirectorFields(director=director_field, order_S=occupancy.copy(),
                                density=occupancy, polarity=polarity_field)
        ground_truth = _ground_truth(sampled_dirs, spec, float(occupancy.mean()))
        if progress is not None:
            progress(1.0)

        meta_seed = int(rng.integers(0, 2**31 - 1))
        meta_phantom = Phantom.empty(self.shape_zyx, self.voxel_size_zyx, seed=meta_seed,
                                     tissue_preset=config.preset, with_fields=False)
        return Phantom(meta=meta_phantom.meta, geometry=fibrils, fields=fields,
                       ground_truth=ground_truth)

    # ----------------------------------------------------------------- seeding
    def _flat_seed(self, spec, populations, weights, extent, diam_mean, length_mean,
                   owner, rng) -> _Seed:
        pop = populations[rng.choice(len(populations), p=weights)]
        pos = self._free_seed(extent, owner, rng)
        base = sample_axial_directions(
            pop.field.at(pos[None])[0][None], spec["kappa_par"], spec["kappa_perp"], rng
        )[0]
        diameter = max(0.05, rng.normal(diam_mean, diam_mean * spec["diameter_cv"]))
        length = max(self.step_um * 3, _sample_length(length_mean, spec["length_cv"], rng))
        polarity = int(rng.choice((-1, 1)))
        return _Seed(pos, base, pop.field, diameter, length, polarity, None, None, 0, "fibril")

    def _free_seed(self, extent, owner, rng) -> np.ndarray:
        """Sample a seed point; with exclusion, reject points in already-occupied voxels."""
        if owner is None:
            return rng.uniform(0, 1, size=3) * extent
        dz, dy, dx = self.voxel_size_zyx
        z, y, x = self.shape_zyx
        p = rng.uniform(0, 1, size=3) * extent
        for _ in range(20):
            ix, iy, iz = int(p[0] / dx), int(p[1] / dy), int(p[2] / dz)
            if 0 <= iz < z and 0 <= iy < y and 0 <= ix < x and owner[iz, iy, ix] < 0:
                return p
            p = rng.uniform(0, 1, size=3) * extent
        return p

    def _plan_hierarchy(self, spec, populations, weights, extent, diam_mean, length_mean, rng):
        h = spec["hierarchy"]
        fasc_r = h["fascicle_radius_um"] or 0.2 * float(min(extent))
        fiber_r = h["fiber_radius_um"] or 0.35 * fasc_r
        seeds: list[_Seed] = []
        global_fiber = 0
        for fa in range(h["n_fascicles"]):
            fc = rng.uniform(0, 1, size=3) * extent
            pop = populations[rng.choice(len(populations), p=weights)]
            fasc_dir = sample_axial_directions(
                pop.field.at(fc[None])[0][None], spec["kappa_par"], spec["kappa_perp"], rng
            )[0]
            for _ in range(h["fibers_per_fascicle"]):
                fbc = np.clip(fc + _random_in_ball(fasc_r, rng), 0.0, extent)  # keep in-volume
                fiber_dir = sample_axial_directions(
                    fasc_dir[None], h["fiber_kappa"], h["fiber_kappa"], rng
                )[0]
                for _ in range(h["fibrils_per_fiber"]):
                    pos = np.clip(fbc + _random_in_ball(fiber_r, rng), 0.0, extent)
                    base = sample_axial_directions(
                        fiber_dir[None], 2 * h["fiber_kappa"], 2 * h["fiber_kappa"], rng
                    )[0]
                    diameter = max(0.05, rng.normal(diam_mean, diam_mean * spec["diameter_cv"]))
                    length = max(self.step_um * 3,
                                 _sample_length(length_mean, spec["length_cv"], rng))
                    polarity = int(rng.choice((-1, 1)))
                    seeds.append(_Seed(pos, base, pop.field, diameter, length, polarity,
                                       global_fiber, fa, 0, "fibril"))
                global_fiber += 1
        return seeds

    # ----------------------------------------------------------------- growth + placement
    def _grow_place(self, seed, spec, occupancy, director_field, polarity_field, owner,
                    blocked, next_id, rng):
        mean0 = seed.field.at(seed.pos[None])[0]
        centerline, tangents = _grow_fibril(
            seed.field, seed.pos, seed.base_dir, mean0, seed.length, self.step_um,
            spec["persistence_um"], spec["crimp_amplitude_um"], spec["crimp_period_um"], rng,
            blocked=blocked,
        )
        if len(centerline) < 2:
            return None, None, []
        claimed = self._rasterize(centerline, tangents, seed.diameter, seed.polarity,
                                  occupancy, director_field, polarity_field, owner, next_id)
        if claimed == 0:
            return None, None, []
        keep = max(1, len(centerline) // 24)
        fib = Fibril(id=next_id, fiber_id=seed.fiber_id, fascicle_id=seed.fascicle_id,
                     type=seed.type, centerline=centerline[::keep],
                     diameter=np.full(len(centerline[::keep]), seed.diameter),
                     polarity=seed.polarity)
        children = []
        if spec["branch_density_per_um"] > 0 and seed.generation < spec["max_generations"]:
            children = _spawn_branches(centerline, tangents, seed, spec, rng)
        return fib, seed.base_dir, children

    # ----------------------------------------------------------------- crosslinks
    def _add_crosslinks(self, fibrils, spec, extent, occupancy, director_field,
                        polarity_field, owner, next_id, rng) -> int:
        from scipy.spatial import cKDTree

        pts, labels = [], []
        for fib in fibrils:
            cl = np.asarray(fib.centerline)
            pts.append(cl)
            labels.append(np.full(len(cl), fib.id))
        pts = np.concatenate(pts)
        labels = np.concatenate(labels)
        tree = cKDTree(pts)

        n_links = int(spec["crosslink_density"] * float(np.prod(extent)))
        diameter = spec["crosslink_diameter_um"] or 0.5 * spec["diameter_um"]
        made = 0
        for _ in range(n_links):
            i = int(rng.integers(0, len(pts)))
            neigh = tree.query_ball_point(pts[i], spec["crosslink_max_um"])
            others = [j for j in neigh if labels[j] != labels[i]]
            if not others:
                continue
            j = int(others[int(rng.integers(0, len(others)))])
            a, b = pts[i], pts[j]
            d = b - a
            nd = np.linalg.norm(d)
            tangent = d / nd if nd > 0 else np.array([1.0, 0.0, 0.0])
            claimed = self._rasterize(np.stack([a, b]), np.stack([tangent, tangent]),
                                      diameter, 1, occupancy, director_field, polarity_field,
                                      owner, next_id)
            if claimed == 0:
                continue
            fibrils.append(Fibril(id=next_id, type="crosslink",
                                  centerline=np.stack([a, b]), diameter=np.full(2, diameter),
                                  polarity=1))
            next_id += 1
            made += 1
        return next_id

    # ----------------------------------------------------------------- rasterization
    def _make_blocked(self, owner):
        if owner is None:
            return None
        dz, dy, dx = self.voxel_size_zyx
        z, y, x = self.shape_zyx

        def blocked(pt: np.ndarray) -> bool:
            ix, iy, iz = int(round(pt[0] / dx)), int(round(pt[1] / dy)), int(round(pt[2] / dz))
            if 0 <= iz < z and 0 <= iy < y and 0 <= ix < x:
                return bool(owner[iz, iy, ix] >= 0)
            return False

        return blocked

    def _rasterize(self, centerline, tangents, diameter, polarity, occupancy, director_field,
                   polarity_field, owner, fibril_id) -> int:
        radius = max(0.5 * diameter, 0.3 * min(self.voxel_size_zyx))
        claimed = 0
        for k in range(len(centerline) - 1):
            claimed += _rasterize_capsule(
                occupancy, director_field, polarity_field, centerline[k], centerline[k + 1],
                radius, tangents[k], float(polarity), self.voxel_size_zyx, self.shape_zyx,
                owner, fibril_id,
            )
        return claimed


# --------------------------------------------------------------------------- fibril growth
def _grow_fibril(field, seed, base_dir, mean0, length, ds, persistence_um,
                 crimp_amp, crimp_period, rng, *, blocked=None):
    """Grow a centerline following ``field`` + a per-fibril offset, with Lp + crimp waviness."""
    offset = base_dir - mean0  # constant per-fibril tilt from the local mean
    n_half = max(2, int(0.5 * length / ds))

    def walk(sign: int):
        pts = []
        p = seed.copy()
        for _ in range(n_half):
            local = field.at(p[None])[0]
            # (local + offset) tilts the local field by the per-fibril offset; *sign grows the
            # two halves symmetrically (forward ~ +base_dir, backward ~ -base_dir).
            d = (local + offset) * sign
            if np.isfinite(persistence_um) and persistence_um > 0:
                d = d + rng.normal(0, np.sqrt(2 * ds / persistence_um), 3)
            nd = np.linalg.norm(d)
            d = d / nd if nd > 0 else base_dir * sign
            p_next = p + d * ds
            if blocked is not None and blocked(p_next):
                break
            p = p_next
            pts.append(p.copy())
        return pts

    fwd = walk(+1)
    bwd = walk(-1)
    pts = np.array(bwd[::-1] + [seed] + fwd)

    if len(pts) < 2:  # blocked immediately (exclusion) -> caller skips this fibril
        return pts, np.tile(base_dir, (len(pts), 1))

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


def _spawn_branches(centerline, tangents, parent: _Seed, spec, rng) -> list[_Seed]:
    """Spawn child fibrils branching off ``parent`` along its length."""
    seglen = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    total_len = float(seglen.sum())
    n_branch = int(rng.poisson(spec["branch_density_per_um"] * total_len))
    angle = np.deg2rad(spec["branch_angle_deg"])
    children: list[_Seed] = []
    for _ in range(n_branch):
        idx = int(rng.integers(0, len(centerline)))
        t = tangents[idx]
        perp = _perp(t)
        phi = rng.uniform(0, 2 * np.pi)
        e = perp * np.cos(phi) + np.cross(t, perp) * np.sin(phi)
        child_dir = _normalize(t * np.cos(angle) + e * np.sin(angle))
        children.append(_Seed(
            pos=centerline[idx].copy(), base_dir=child_dir, field=parent.field,
            diameter=parent.diameter * 0.8, length=parent.length * 0.6, polarity=parent.polarity,
            fiber_id=parent.fiber_id, fascicle_id=parent.fascicle_id,
            generation=parent.generation + 1, type="branch",
        ))
    return children


def _rasterize_capsule(occupancy, director_field, polarity_field, a, b, radius, tangent,
                       polarity, voxel, shape, owner, fibril_id) -> int:
    """Mark free voxels within ``radius`` (µm) of segment a-b; returns the # newly claimed."""
    dz, dy, dx = voxel
    z, y, x = shape
    lo = np.minimum(a, b) - radius
    hi = np.maximum(a, b) + radius
    ix0, ix1 = max(0, int(lo[0] / dx)), min(x - 1, int(np.ceil(hi[0] / dx)))
    iy0, iy1 = max(0, int(lo[1] / dy)), min(y - 1, int(np.ceil(hi[1] / dy)))
    iz0, iz1 = max(0, int(lo[2] / dz)), min(z - 1, int(np.ceil(hi[2] / dz)))
    if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return 0

    zc = np.arange(iz0, iz1 + 1) * dz
    yc = np.arange(iy0, iy1 + 1) * dy
    xc = np.arange(ix0, ix1 + 1) * dx
    zg, yg, xg = np.meshgrid(zc, yc, xc, indexing="ij")
    g = np.stack([xg, yg, zg], axis=-1)

    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0:
        dist = np.linalg.norm(g - a, axis=-1)
    else:
        t = np.clip(((g - a) @ ab) / denom, 0.0, 1.0)
        dist = np.linalg.norm(g - (a + t[..., None] * ab), axis=-1)
    mask = dist <= radius
    if not mask.any():
        return 0

    sl = (slice(iz0, iz1 + 1), slice(iy0, iy1 + 1), slice(ix0, ix1 + 1))
    if owner is not None:  # volume exclusion: claim only free voxels
        mask = mask & (owner[sl] < 0)
        if not mask.any():
            return 0
        owner[sl][mask] = fibril_id
    occupancy[sl][mask] = 1.0
    polarity_field[sl][mask] = polarity
    for c in range(3):
        director_field[c][sl][mask] = tangent[c]
    return int(mask.sum())


# --------------------------------------------------------------------------- helpers
def _perp(direction: np.ndarray) -> np.ndarray:
    ref = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(direction, ref)
    n = np.linalg.norm(e1)
    return e1 / n if n > 0 else np.array([0.0, 1.0, 0.0])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _random_in_ball(radius: float, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v) or 1.0
    return v * (rng.uniform() ** (1.0 / 3.0)) * radius


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
    """Extract architecture + dispersion + morphology + network params (backward compatible)."""
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
    kappa_par = float(_get(orient, "kappa_par", kappa))
    kappa_perp = float(_get(orient, "kappa_perp", kappa))
    if arch_name == "isotropic":
        kappa_par = kappa_perp = 0.0

    branching = getattr(config, "branching", None) or {}
    crosslinks = getattr(config, "crosslinks", None) or {}
    hierarchy = getattr(config, "hierarchy", None) or {}

    return {
        "architecture": arch_name,
        "params": params,
        "kappa_par": kappa_par,
        "kappa_perp": kappa_perp,
        "xi_um": _get(orient, "xi_um", None),
        "diameter_um": float(_get(fibril.diameter_um, "mean", 1.0)),
        "diameter_cv": float(_get(fibril.diameter_um, "dispersion", 0.0)),
        "length_um": float(_get(fibril, "length_um", 0.0)),
        "length_cv": float(_get(fibril, "length_cv", 0.0)),
        "persistence_um": float(_get(fibril, "persistence_um", np.inf)),
        "crimp_amplitude_um": float(_get(fibril.crimp, "amplitude_um", 0.0)),
        "crimp_period_um": float(_get(fibril.crimp, "period_um", 0.0)),
        "volume_fraction": float(_get(config, "volume_fraction", 0.1)),
        # network features (off by default)
        "exclusion": bool(_get(config, "exclusion", False)),
        "branch_density_per_um": float(_get(branching, "density_per_um", 0.0)),
        "branch_angle_deg": float(_get(branching, "angle_deg", 30.0)),
        "max_generations": int(_get(branching, "max_generations", 2)),
        "crosslink_density": float(_get(crosslinks, "density_per_um3", 0.0)),
        "crosslink_max_um": float(_get(crosslinks, "max_um", 2.0)),
        "crosslink_diameter_um": float(_get(crosslinks, "diameter_um", 0.0)),
        "hierarchy": {
            "enabled": bool(_get(hierarchy, "enabled", False)),
            "n_fascicles": int(_get(hierarchy, "n_fascicles", 3)),
            "fibers_per_fascicle": int(_get(hierarchy, "fibers_per_fascicle", 4)),
            "fibrils_per_fiber": int(_get(hierarchy, "fibrils_per_fiber", 8)),
            "fascicle_radius_um": float(_get(hierarchy, "fascicle_radius_um", 0.0)),
            "fiber_radius_um": float(_get(hierarchy, "fiber_radius_um", 0.0)),
            "fiber_kappa": float(_get(hierarchy, "fiber_kappa", 50.0)),
        },
    }


def _get(obj, name, default):
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(name, default)
    else:
        v = getattr(obj, name, default)
    return default if v is None else v