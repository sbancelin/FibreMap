"""Tier 3 coherent imaging: forward/backward ratio physics (qualitative)."""

from __future__ import annotations

import numpy as np

from collagen_shg.config.models import DegradationConfig, MicroscopeConfig
from collagen_shg.imaging import CoherentImager
from collagen_shg.representations.phantom import DirectorFields, Phantom, PhantomMeta


def _phantom(density, polarity, dz):
    shape = density.shape
    meta = PhantomMeta(shape_zyx=shape, voxel_size_zyx=(dz, 0.2, 0.2))
    fields = DirectorFields(
        director=np.zeros((3, *shape), dtype=np.float32),
        order_S=np.zeros(shape, dtype=np.float32),
        density=density.astype(np.float32),
        polarity=polarity.astype(np.float32),
    )
    return Phantom(meta=meta, fields=fields)


def test_single_plane_has_unit_fb_ratio():
    density = np.ones((1, 4, 4))
    polarity = np.ones((1, 4, 4))
    cf = CoherentImager().fields(_phantom(density, polarity, dz=0.2), MicroscopeConfig())
    assert cf.fb_ratio == 1.0


def test_thick_uniform_slab_is_forward_dominant():
    # fine axial sampling so the phase-matched forward field adds coherently
    density = np.ones((200, 2, 2))
    polarity = np.ones((200, 2, 2))
    cf = CoherentImager().fields(
        _phantom(density, polarity, dz=0.05), MicroscopeConfig(wavelength_nm=900)
    )
    assert cf.fb_ratio > 10.0  # forward >> backward for a uniform bulk


def test_quasi_phase_matched_polarity_enhances_backward():
    n_med = 1.33
    lam = 0.9
    dk_b = 8 * np.pi * n_med / lam  # backward mismatch (µm^-1)
    period_um = 2 * np.pi / dk_b
    dz = period_um / 4  # resolve the QPM period
    nz = 240
    z = np.arange(nz) * dz
    density = np.ones((nz, 2, 2))

    uniform_pol = np.ones((nz, 2, 2))
    qpm_pol = np.sign(np.sin(dk_b * z))[:, None, None] * np.ones((nz, 2, 2))

    imager = CoherentImager()
    mic = MicroscopeConfig(wavelength_nm=900)
    r_uniform = imager.fields(_phantom(density, uniform_pol, dz), mic).fb_ratio
    r_qpm = imager.fields(_phantom(density, qpm_pol, dz), mic).fb_ratio
    assert r_qpm < r_uniform  # QPM polarity boosts the backward signal


def test_render_produces_2d_bundle_without_volumetric_gt():
    density = np.ones((10, 6, 6))
    polarity = np.ones((10, 6, 6))
    bundle = CoherentImager().render(
        _phantom(density, polarity, dz=0.1),
        MicroscopeConfig(detection="backward", wavelength_nm=900),
        DegradationConfig.model_validate({"noise": {"photons_peak": 100.0}}),
        np.random.default_rng(0),
    )
    assert bundle.image.shape == (1, 6, 6)
    assert bundle.metadata.microscope.mode == "coherent"
    assert bundle.phantom is None
