"""Tier 1 — fast incoherent imaging (Livrable 2).

The high-throughput generator: the emitter intensity field (the phantom density) is convolved
with a Gaussian PSF (from NA / λ), attenuated with depth (Beer–Lambert, round-trip in epi), then
corrupted by shot (Poisson) and detector (Gaussian read) noise. Fast, sufficient for analyzing
resolved tissues, and the ground truth is preserved (carried through on the bundle).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from collagen_shg.config.models import DegradationConfig, MicroscopeConfig
from collagen_shg.representations.image_bundle import (
    AcquisitionMeta,
    BundleMetadata,
    ImageBundle,
    MicroscopeMeta,
)
from collagen_shg.representations.phantom import Phantom

__all__ = ["IncoherentImager", "psf_sigma_um"]

_FWHM_TO_SIGMA = 1.0 / 2.354820045


def psf_sigma_um(
    NA: float, wavelength_um: float, *, n: float = 1.33, detection: str = "backward"
) -> tuple[float, float]:
    """Gaussian PSF standard deviations ``(σ_axial, σ_lateral)`` in µm from NA and λ.

    Lateral FWHM ≈ ``0.51 λ / NA``; axial FWHM ≈ ``0.88 λ / (n − √(n² − NA²))``.
    """
    NA = max(float(NA), 1e-3)
    lateral_fwhm = 0.51 * wavelength_um / NA
    denom = n - np.sqrt(max(n**2 - NA**2, 1e-6))
    axial_fwhm = 0.88 * wavelength_um / max(denom, 1e-3)
    return axial_fwhm * _FWHM_TO_SIGMA, lateral_fwhm * _FWHM_TO_SIGMA


class IncoherentImager:
    """Tier 1 incoherent imager: ``Phantom + Microscope + Degradation + RNG → ImageBundle``."""

    def signal(
        self, phantom: Phantom, microscope: MicroscopeConfig, degradation: DegradationConfig
    ) -> np.ndarray:
        """Noise-free signal: PSF-blurred, depth-attenuated intensity in roughly [0, 1]."""
        if phantom.fields is None:
            raise ValueError("IncoherentImager requires a phantom with voxelized fields")
        intensity = np.asarray(phantom.fields.density, dtype=np.float64)

        dz, dy, dx = phantom.meta.voxel_size_zyx
        na = float(microscope.NA or 0.8)
        wavelength_um = float(microscope.wavelength_nm or 900.0) / 1000.0
        s_ax, s_lat = psf_sigma_um(na, wavelength_um, detection=microscope.detection)
        sigma = (s_ax / dz, s_lat / dy, s_lat / dx)
        blurred = gaussian_filter(intensity, sigma=sigma)

        ell = _attenuation_length(degradation)
        if ell is not None and ell > 0:
            nz = intensity.shape[0]
            z_um = np.arange(nz) * dz
            k = 2.0 if microscope.detection == "backward" else 1.0  # epi round-trip
            blurred = blurred * np.exp(-k * z_um / ell)[:, None, None]
        return np.clip(blurred, 0.0, None)

    def render(
        self,
        phantom: Phantom,
        microscope: MicroscopeConfig,
        degradation: DegradationConfig,
        rng: np.random.Generator,
        *,
        add_noise: bool = True,
    ) -> ImageBundle:
        """Render the full noisy image bundle (carrying the phantom as ground truth)."""
        signal = self.signal(phantom, microscope, degradation)

        peak = float(_noise(degradation, "photons_peak") or 1000.0)
        read = float(_noise(degradation, "read_noise_e") or 0.0)
        lam = signal * peak
        if add_noise:
            counts = rng.poisson(lam).astype(np.float64)
            if read > 0:
                counts = counts + rng.normal(0.0, read, size=counts.shape)
        else:
            counts = lam
        image = np.clip(counts, 0.0, None).astype(np.float32)

        metadata = self._metadata(phantom, microscope)
        bundle = ImageBundle(image=image, metadata=metadata, phantom=phantom)
        return bundle

    def _metadata(self, phantom: Phantom, microscope: MicroscopeConfig) -> BundleMetadata:
        return BundleMetadata(
            kind="synthetic",
            shape_zyx=phantom.meta.shape_zyx,
            voxel_size_zyx=phantom.meta.voxel_size_zyx,
            microscope=MicroscopeMeta(
                mode="incoherent",
                NA=microscope.NA,
                wavelength_nm=microscope.wavelength_nm,
                detection=microscope.detection,
                pixel_size_um=microscope.pixel_size_um,
                psf_model=microscope.psf_model or "gaussian",
            ),
            acquisition=AcquisitionMeta(bit_depth=16),
        )


def _attenuation_length(degradation: DegradationConfig) -> float | None:
    depth = getattr(degradation, "depth", None)
    if depth is None:
        return None
    val = getattr(depth, "attenuation_length_um", None)
    return float(val) if val else None


def _noise(degradation: DegradationConfig, name: str) -> float | None:
    noise = getattr(degradation, "noise", None)
    if noise is None:
        return None
    return getattr(noise, name, None)
