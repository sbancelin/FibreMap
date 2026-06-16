"""Tier 1 incoherent imaging: PSF blur, depth attenuation, noise statistics."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.config.models import DegradationConfig, MicroscopeConfig
from collagen_shg.imaging import IncoherentImager, psf_sigma_um
from collagen_shg.representations.phantom import DirectorFields, Phantom, PhantomMeta

SHAPE = (16, 48, 48)
VOXEL = (0.5, 0.2, 0.2)


def _phantom_with_density(density):
    meta = PhantomMeta(shape_zyx=SHAPE, voxel_size_zyx=VOXEL)
    fields = DirectorFields(
        director=np.zeros((3, *SHAPE), dtype=np.float32),
        order_S=np.zeros(SHAPE, dtype=np.float32),
        density=density.astype(np.float32),
        polarity=np.zeros(SHAPE, dtype=np.float32),
    )
    return Phantom(meta=meta, fields=fields)


def test_psf_sigma_scales_with_NA():
    s_ax_lo, s_lat_lo = psf_sigma_um(0.5, 0.9)
    s_ax_hi, s_lat_hi = psf_sigma_um(1.2, 0.9)
    assert s_lat_hi < s_lat_lo  # higher NA -> tighter PSF
    assert s_ax_hi < s_ax_lo
    assert s_ax_lo > s_lat_lo  # axial PSF broader than lateral


def test_point_source_is_blurred():
    density = np.zeros(SHAPE)
    density[8, 24, 24] = 1.0
    phantom = _phantom_with_density(density)
    sig = IncoherentImager().signal(
        phantom, MicroscopeConfig(NA=0.9, wavelength_nm=900), DegradationConfig()
    )
    # energy spreads to neighbours; peak no longer the only nonzero voxel
    assert np.count_nonzero(sig > sig.max() * 0.01) > 5
    assert sig[8, 24, 24] == sig.max()


def test_depth_attenuation_decreases_with_z():
    density = np.ones(SHAPE)
    phantom = _phantom_with_density(density)
    deg = DegradationConfig.model_validate({"depth": {"attenuation_length_um": 5.0}})
    sig = IncoherentImager().signal(
        phantom, MicroscopeConfig(NA=0.9, wavelength_nm=900, detection="backward"), deg
    )
    profile = sig.mean(axis=(1, 2))
    assert np.all(np.diff(profile) < 1e-9)  # monotonically non-increasing with depth
    assert profile[-1] < profile[0]


def test_render_is_deterministic_and_poisson_like():
    density = np.full(SHAPE, 0.5)
    phantom = _phantom_with_density(density)
    mic = MicroscopeConfig(NA=0.9, wavelength_nm=900)
    deg = DegradationConfig.model_validate({"noise": {"photons_peak": 200.0, "read_noise_e": 0.0}})
    imager = IncoherentImager()
    a = imager.render(phantom, mic, deg, np.random.default_rng(0))
    b = imager.render(phantom, mic, deg, np.random.default_rng(0))
    assert np.array_equal(a.image, b.image)
    assert a.metadata.kind == "synthetic"
    assert a.phantom is phantom
    # Poisson: in a uniform interior, variance ~ mean (no read noise)
    interior = a.image[4:-4, 8:-8, 8:-8]
    assert interior.var() == pytest.approx(interior.mean(), rel=0.3)


def test_render_no_noise_matches_signal():
    density = np.full(SHAPE, 0.5)
    phantom = _phantom_with_density(density)
    mic = MicroscopeConfig(NA=0.9, wavelength_nm=900)
    deg = DegradationConfig.model_validate({"noise": {"photons_peak": 100.0}})
    imager = IncoherentImager()
    sig = imager.signal(phantom, mic, deg)
    img = imager.render(phantom, mic, deg, np.random.default_rng(0), add_noise=False)
    assert np.allclose(img.image, sig * 100.0, atol=1e-3)
