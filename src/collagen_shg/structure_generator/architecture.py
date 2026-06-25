"""Architecture templates — the spatially-varying mean-orientation field n₀(r).

The macro-organization of a fibre network is encoded by **how the local mean direction varies in
space**. Each architecture is a director field ``n₀(r)`` (a function from physical points to unit
axial directions); fibrils are grown by following it, so curved architectures (arcades, tubular)
produce naturally curved fibrils. Multi-axis architectures (basket-weave, crossed helix) are
expressed as several **populations**, each its own field, split by weight.

Local disorder around ``n₀`` is added separately (:func:`sample_axial_directions`, a small-angle
biaxial / Bingham model with in-plane ``kappa_par`` and out-of-plane ``kappa_perp``), and spatial
coherence by a correlation length ξ (in the generator). Together these span tendon (uniaxial +
crimp), skin (biaxial / Langer, high disorder), cornea (lamellae), cartilage (arcades) and
arterial sheaths (tubular ± helix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from collagen_shg.representations import conventions as cv

__all__ = [
    "DirectorField",
    "Uniaxial",
    "Lamellar",
    "Arcade",
    "Tubular",
    "Population",
    "build_architecture",
    "sample_axial_directions",
    "ARCHITECTURES",
]

ARCHITECTURES = ("uniaxial", "biaxial", "lamellar", "arcade", "tubular", "isotropic")


class DirectorField:
    """Base class: ``at(points_xyz)`` returns unit axial directions at physical points (µm)."""

    def at(self, points_xyz: NDArray[np.float64]) -> NDArray[np.float64]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Uniaxial(DirectorField):
    """Constant direction (tendon)."""

    phi0: float = 0.0
    theta0: float = 0.0

    def at(self, points_xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        n = cv.director_from_angles(self.phi0, self.theta0)
        return np.broadcast_to(n, points_xyz.shape).copy()


@dataclass
class Lamellar(DirectorField):
    """In-plane azimuth steps with depth z — stacked lamellae (cornea)."""

    thickness_um: float = 2.0
    dphi: float = np.pi / 2  # azimuth increment per lamella (orthogonal by default)
    phi_start: float = 0.0
    theta0: float = 0.0

    def at(self, points_xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        z = points_xyz[..., 2]
        layer = np.floor(z / max(self.thickness_um, 1e-9))
        phi = self.phi_start + self.dphi * layer
        theta = np.full_like(phi, self.theta0)
        return cv.director_from_angles(phi, theta)


@dataclass
class Arcade(DirectorField):
    """Elevation gradient with depth — Benninghoff arcades (cartilage).

    Vertical (θ≈90°) in the deep zone (large z), curving to in-plane (θ≈0°) near the surface
    (z≈0); ``phi0`` is the in-plane projection direction.
    """

    theta_deep: float = np.pi / 2
    theta_surface: float = 0.0
    z_max_um: float = 10.0
    phi0: float = 0.0

    def at(self, points_xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        z = points_xyz[..., 2]
        frac = np.clip(z / max(self.z_max_um, 1e-9), 0.0, 1.0)
        theta = self.theta_surface + (self.theta_deep - self.theta_surface) * frac
        phi = np.full_like(theta, self.phi0)
        return cv.director_from_angles(phi, theta)


@dataclass
class Tubular(DirectorField):
    """Circumferential / helical field around a central axis (arterial sheath, IVD annulus).

    Default axis is ``z``; at each point the director is the circumferential direction (⊥ to the
    radial vector, in the x-y plane) tilted toward the axis by the helix angle ``beta`` (β=0 →
    purely circumferential rings).
    """

    center_xy: tuple[float, float] = (0.0, 0.0)
    beta: float = 0.0  # helix angle (rad)

    def at(self, points_xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        x = points_xyz[..., 0] - self.center_xy[0]
        y = points_xyz[..., 1] - self.center_xy[1]
        r = np.hypot(x, y)
        with np.errstate(invalid="ignore", divide="ignore"):
            rx = np.where(r > 0, x / r, 1.0)
            ry = np.where(r > 0, y / r, 0.0)
        circ = np.stack([-ry, rx, np.zeros_like(rx)], axis=-1)  # circumferential, in-plane
        axis = np.zeros_like(circ)
        axis[..., 2] = 1.0
        director = np.cos(self.beta) * circ + np.sin(self.beta) * axis
        norm = np.linalg.norm(director, axis=-1, keepdims=True)
        return director / np.where(norm > 0, norm, 1.0)


@dataclass
class Population:
    """A weighted fibril population following one director field."""

    weight: float
    field: DirectorField


def build_architecture(
    name: str,
    params: dict,
    shape_zyx: tuple[int, int, int],
    voxel_size_zyx: tuple[float, float, float],
) -> list[Population]:
    """Build the population list for an architecture (uses volume geometry for defaults)."""
    name = name.lower()
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; choose from {ARCHITECTURES}")

    z, y, x = shape_zyx
    dz, dy, dx = voxel_size_zyx
    z_max, y_max, x_max = z * dz, y * dy, x * dx
    center = (x_max / 2.0, y_max / 2.0)

    def deg(key: str, default: float) -> float:
        return float(np.deg2rad(params.get(key, default)))

    if name in ("uniaxial", "isotropic"):
        return [Population(1.0, Uniaxial(deg("mean_phi_deg", 0.0), deg("mean_theta_deg", 0.0)))]

    if name == "biaxial":
        phi_a = deg("phi_a_deg", 0.0)
        phi_b = deg("phi_b_deg", 90.0)
        mix = float(params.get("mix", 0.5))
        return [Population(mix, Uniaxial(phi_a)), Population(1.0 - mix, Uniaxial(phi_b))]

    if name == "lamellar":
        return [
            Population(
                1.0,
                Lamellar(
                    thickness_um=float(params.get("lamella_thickness_um", 2.0)),
                    dphi=deg("lamella_dphi_deg", 90.0),
                    phi_start=deg("mean_phi_deg", 0.0),
                ),
            )
        ]

    if name == "arcade":
        return [
            Population(
                1.0,
                Arcade(
                    theta_deep=deg("theta_deep_deg", 90.0),
                    theta_surface=deg("theta_surface_deg", 0.0),
                    z_max_um=z_max,
                    phi0=deg("mean_phi_deg", 0.0),
                ),
            )
        ]

    # tubular
    beta = deg("helix_beta_deg", 0.0)
    if bool(params.get("crossed", False)):
        return [
            Population(0.5, Tubular(center, beta)),
            Population(0.5, Tubular(center, -beta)),
        ]
    return [Population(1.0, Tubular(center, beta))]


def sample_axial_directions(
    mean_dirs: NDArray[np.float64],
    kappa_par: float,
    kappa_perp: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample axial directions around ``mean_dirs`` with biaxial small-angle spread.

    ``kappa_par`` controls the in-plane spread (about the horizontal perpendicular), ``kappa_perp``
    the out-of-plane spread. ``kappa→0`` gives an isotropic (uniform-sphere) sample; large κ gives
    tight alignment. ``mean_dirs`` is ``(..., 3)``; returns unit directions of the same shape.
    """
    mean = np.asarray(mean_dirs, dtype=np.float64)
    flat = mean.reshape(-1, 3)
    n = flat.shape[0]

    kpar = max(float(kappa_par), 0.0)
    kperp = max(float(kappa_perp), 0.0)

    # Tangent frame at each mean direction: e1 in-plane, e2 out-of-plane.
    z_hat = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(flat, z_hat)
    bad = np.linalg.norm(e1, axis=1) < 1e-8
    e1[bad] = np.cross(flat[bad], np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(flat, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)

    # Polar deviation gamma from the axis: axial Watson, governed by the LOWER concentration
    # (the broadest direction sets how far it spreads). u = cos(gamma) ~ exp(kpolar u^2) on [0,1].
    kpolar = min(kpar, kperp)
    u = _sample_watson_u(kpolar, n, rng)
    gamma = np.arccos(np.clip(u, 0.0, 1.0))

    # Azimuth of the deviation in the tangent plane: concentrate toward the lower-kappa axis so
    # the spread is anisotropic (biaxial). Equal kappa -> uniform azimuth (axisymmetric Watson).
    if abs(kpar - kperp) < 1e-9:
        psi = rng.uniform(0.0, 2.0 * np.pi, n)
    else:
        conc = min(abs(kperp - kpar), 200.0)
        mu2 = 0.0 if kpar <= kperp else np.pi  # favor e1 (in-plane) or e2 (out-of-plane)
        psi = rng.vonmises(mu2, conc, n) / 2.0
    uhat = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2

    sign = rng.choice(np.array([-1.0, 1.0]), n)  # fibres are axial (n == -n)
    out = sign[:, None] * (flat * np.cos(gamma)[:, None] + uhat * np.sin(gamma)[:, None])
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out.reshape(mean.shape)


def _sample_watson_u(kappa: float, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Sample ``u = cos(gamma) in [0, 1]`` with density ``∝ exp(kappa·u²)`` (vectorized rejection).

    ``kappa = 0`` → uniform u → ``gamma`` distributed as ``sin gamma`` (isotropic hemisphere).
    """
    if kappa < 1e-6:
        return rng.uniform(0.0, 1.0, n)
    res = np.empty(n)
    need = np.ones(n, dtype=bool)
    for _ in range(200):
        m = int(need.sum())
        if m == 0:
            break
        prop = rng.uniform(0.0, 1.0, m)
        accept = rng.uniform(0.0, 1.0, m) < np.exp(kappa * (prop**2 - 1.0))
        idx = np.flatnonzero(need)[accept]
        res[idx] = prop[accept]
        need[idx] = False
    res[need] = 1.0  # fallback for very high kappa: fully concentrated
    return res
