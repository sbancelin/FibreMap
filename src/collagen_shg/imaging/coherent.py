"""Tier 3 — coherent SHG imaging (first-order model, Livrable 2).

SHG is coherent: the second-harmonic field sums **with phase** along the optical (z) axis, so the
forward/backward ratio reports on the sub-wavelength axial arrangement of emitters (polarity),
not just their number. This is a tractable scalar first-order model (no vectorial Richards–Wolf,
no GPU — those are later): for each lateral column the forward field is (near) phase-matched
(``Δk_f ≈ 0``) while the backward field carries a large momentum mismatch ``Δk_b = 8π n / λ``, so
a bulk of same-polarity emitters radiates forward and backward signal appears only where the
emitter polarity has axial structure near the QPM period (cornea/cartilage-type order).

The coherent detector integrates along z to a 2D image, so the rendered bundle is a single plane
``[1, Y, X]`` and does not carry the (volumetric) phantom as ground truth; use :meth:`fields` for
the forward/backward intensities and ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collagen_shg.config.models import DegradationConfig, MicroscopeConfig
from collagen_shg.representations.image_bundle import (
    BundleMetadata,
    ImageBundle,
    MicroscopeMeta,
)
from collagen_shg.representations.phantom import Phantom

__all__ = ["CoherentFields", "CoherentImager"]


@dataclass
class CoherentFields:
    forward: np.ndarray  # [Y, X] forward SHG intensity
    backward: np.ndarray  # [Y, X] backward SHG intensity
    fb_ratio: float  # total forward / total backward energy


class CoherentImager:
    """First-order coherent SHG imager (forward/backward via axial phase summation)."""

    def __init__(self, *, n: float = 1.33, dispersion_dk: float = 0.0) -> None:
        self.n = float(n)
        self.dispersion_dk = float(dispersion_dk)  # forward mismatch (≈0, ignoring dispersion)

    def fields(self, phantom: Phantom, microscope: MicroscopeConfig) -> CoherentFields:
        if phantom.fields is None:
            raise ValueError("CoherentImager requires a phantom with voxelized fields")
        density = np.asarray(phantom.fields.density, dtype=np.float64)
        polarity = np.asarray(phantom.fields.polarity, dtype=np.float64)
        amplitude = density * np.where(polarity != 0, np.sign(polarity), 1.0)  # signed χ(2)

        dz = phantom.meta.voxel_size_zyx[0]
        lam_exc = float(microscope.wavelength_nm or 900.0) / 1000.0
        dk_b = 8.0 * np.pi * self.n / lam_exc  # backward momentum mismatch (µm^-1)
        z = np.arange(amplitude.shape[0]) * dz

        phase_f = np.exp(1j * self.dispersion_dk * z)[:, None, None]
        phase_b = np.exp(1j * dk_b * z)[:, None, None]
        forward = np.abs((amplitude * phase_f).sum(axis=0)) ** 2
        backward = np.abs((amplitude * phase_b).sum(axis=0)) ** 2

        total_b = float(backward.sum())
        fb_ratio = float(forward.sum() / total_b) if total_b > 0 else float("inf")
        return CoherentFields(forward=forward, backward=backward, fb_ratio=fb_ratio)

    def render(
        self,
        phantom: Phantom,
        microscope: MicroscopeConfig,
        degradation: DegradationConfig,
        rng: np.random.Generator,
        *,
        add_noise: bool = True,
    ) -> ImageBundle:
        """Render the detected-direction 2D image as a single-plane ``[1, Y, X]`` bundle."""
        cf = self.fields(phantom, microscope)
        detected = cf.backward if microscope.detection == "backward" else cf.forward
        peak_val = detected.max()
        norm = detected / peak_val if peak_val > 0 else detected

        photons = float(_noise(degradation, "photons_peak") or 1000.0)
        lam = norm * photons
        counts = rng.poisson(lam).astype(np.float64) if add_noise else lam
        image = np.clip(counts, 0.0, None).astype(np.float32)[None, :, :]  # [1, Y, X]

        ny, nx = detected.shape
        metadata = BundleMetadata(
            kind="synthetic",
            shape_zyx=(1, ny, nx),
            voxel_size_zyx=phantom.meta.voxel_size_zyx,
            microscope=MicroscopeMeta(
                mode="coherent",
                NA=microscope.NA,
                wavelength_nm=microscope.wavelength_nm,
                detection=microscope.detection,
                pixel_size_um=microscope.pixel_size_um,
                psf_model=microscope.psf_model,
            ),
        )
        # 2D projection: shape differs from the volume, so no volumetric GT passthrough.
        return ImageBundle(image=image, metadata=metadata, phantom=None)


def _noise(degradation: DegradationConfig, name: str) -> float | None:
    noise = getattr(degradation, "noise", None)
    if noise is None:
        return None
    return getattr(noise, name, None)
