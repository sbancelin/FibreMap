"""Interactive GUI logic (pure helpers; no napari/Qt required)."""

from __future__ import annotations

import numpy as np

from collagen_shg.gui.interactive import (
    analyze_auto,
    degradation_config_from_params,
    descriptors_summary,
    generate_structure,
    image_phantom,
    microscope_config_from_params,
    refine_placeholder,
    skeleton_paths,
    structure_config_from_params,
)

SHAPE = (8, 32, 32)
VOXEL = (0.5, 0.2, 0.2)

_STRUCT = dict(
    mean_phi_deg=90.0, kappa=20.0, diameter_um=1.0, dispersion=0.3,
    crimp_amplitude_um=1.0, crimp_period_um=20.0, xi_um=40.0,
)


def test_config_builders():
    sc = structure_config_from_params(**_STRUCT)
    assert sc.orientation.kappa == 20.0
    assert sc.fibril.diameter_um.mean == 1.0
    mc = microscope_config_from_params(NA=0.9, wavelength_nm=900, detection="backward")
    assert mc.NA == 0.9 and mc.detection == "backward"
    dc = degradation_config_from_params(attenuation_length_um=80, photons_peak=2000, read_noise_e=2)
    assert dc.depth.attenuation_length_um == 80
    assert dc.noise.photons_peak == 2000


def test_generate_structure_is_deterministic():
    a = generate_structure(SHAPE, VOXEL, n_fibrils=40, seed=1, **_STRUCT)
    b = generate_structure(SHAPE, VOXEL, n_fibrils=40, seed=1, **_STRUCT)
    assert np.array_equal(a.fields.director, b.fields.director)
    assert a.ground_truth.global_.S2 > 0.7  # high kappa -> aligned
    # most fibrils placed (a few may fall entirely outside a small volume and are dropped)
    assert 36 <= len(a.geometry) <= 40
    assert len(a.geometry) == len(b.geometry)  # deterministic count


def test_image_phantom_and_skeleton():
    phantom = generate_structure(SHAPE, VOXEL, n_fibrils=40, seed=2, **_STRUCT)
    mic = microscope_config_from_params(NA=0.9, wavelength_nm=900, detection="backward")
    deg = degradation_config_from_params(
        attenuation_length_um=80, photons_peak=2000, read_noise_e=2
    )
    bundle = image_phantom(phantom, mic, deg, seed=3)
    assert bundle.image.shape == SHAPE
    paths = skeleton_paths(phantom, max_fibrils=10)
    assert 0 < len(paths) <= 10
    assert paths[0].shape[1] == 3  # [z, y, x] points


def test_refine_placeholder_preserves_shape():
    img = np.abs(np.random.default_rng(0).standard_normal(SHAPE)).astype(np.float32)
    out = refine_placeholder(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0


def test_analyze_auto_3d_volume():
    phantom = generate_structure((16, 48, 48), VOXEL, n_fibrils=200, seed=4,
                                 **{**_STRUCT, "mean_phi_deg": 90.0, "kappa": 30.0})
    mic = microscope_config_from_params(NA=0.9, wavelength_nm=900, detection="backward")
    deg = degradation_config_from_params(
        attenuation_length_um=80, photons_peak=4000, read_noise_e=0
    )
    bundle = image_phantom(phantom, mic, deg, seed=5)
    res = analyze_auto(np.asarray(bundle.image), VOXEL)
    assert res["ndim"] == 3
    assert "S3" in res["descriptors"]
    assert res["orientation"].shape == (16, 48, 48)


def test_analyze_auto_2d_image():
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.sin(2 * np.pi * xx / 8.0)  # stripes -> fibre along y (phi=pi/2)
    res = analyze_auto(img, VOXEL)
    assert res["ndim"] == 2
    assert res["orientation"].shape == (n, n)
    d = abs(np.angle(np.exp(1j * 2 * (res["descriptors"]["mean_phi"] - np.pi / 2))) / 2)
    assert d < np.deg2rad(4)


def test_descriptors_summary():
    s = descriptors_summary({"S2": 0.8, "S3": 0.7, "mean_phi": np.pi / 2, "xi_um": 30.0})
    assert "S2=0.800" in s and "phi=90.0deg" in s
