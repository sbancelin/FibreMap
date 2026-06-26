"""Tabbed GUI pure logic (no Qt): volume table, architecture params, generation."""

from __future__ import annotations

import numpy as np

from collagen_shg.gui.tabs import (
    arch_params_for,
    build_structure_config,
    build_volume,
    generate_structure_phantom,
    skeleton_volume,
    voxel_counts,
)


def test_voxel_counts_derived():
    assert voxel_counts((20.0, 20.0, 10.0), (0.2, 0.2, 0.5)) == (100, 100, 20)
    assert voxel_counts((10.0, 5.0, 3.0), (0.5, 0.5, 0.5)) == (20, 10, 6)


def test_build_volume_axis_order():
    shape_zyx, voxel_zyx = build_volume((20.0, 16.0, 8.0), (0.2, 0.2, 0.5))
    assert shape_zyx == (16, 80, 100)  # (Z, Y, X)
    assert voxel_zyx == (0.5, 0.2, 0.2)  # (dz, dy, dx)


def test_arch_params_selection():
    flat = {"mean_phi_deg": 30, "phi_a_deg": 0, "phi_b_deg": 90, "helix_beta_deg": 20,
            "crossed": True, "lamella_thickness_um": 3, "theta_deep_deg": 80}
    assert arch_params_for("uniaxial", flat)["mean_phi_deg"] == 30
    assert set(arch_params_for("biaxial", flat)) == {"phi_a_deg", "phi_b_deg", "mix"}
    assert arch_params_for("tubular", flat)["crossed"] is True
    assert "lamella_thickness_um" in arch_params_for("lamellar", flat)


def test_build_structure_config_roundtrips_into_generator():
    cfg = build_structure_config(
        "tubular", {"helix_beta_deg": 20.0, "crossed": True},
        kappa_par=15, kappa_perp=15, xi_um=40, diameter_um=1.0, diameter_cv=0.2,
        length_um=0.0, length_cv=0.0, persistence_um=1e6,
        crimp_amplitude_um=0.0, crimp_period_um=0.0,
    )
    assert cfg.orientation.kappa_par == 15
    assert cfg.architecture["type"] == "tubular"


def test_generate_structure_phantom_binary():
    phantom = generate_structure_phantom(
        (12.0, 12.0, 4.0), (0.25, 0.25, 0.5), "uniaxial",
        {"mean_phi_deg": 90.0, "mean_theta_deg": 0.0}, seed=0, n_fibrils=40,
        kappa_par=30, kappa_perp=30, xi_um=40, diameter_um=1.0, diameter_cv=0.2,
        length_um=0.0, length_cv=0.0, persistence_um=1e6,
        crimp_amplitude_um=0.0, crimp_period_um=0.0,
    )
    dens = np.asarray(phantom.fields.density)
    assert set(np.round(np.unique(dens), 6)).issubset({0.0, 1.0})  # binary
    assert dens.max() == 1.0
    assert phantom.ground_truth.global_.S2 > 0.7  # aligned


def test_generate_with_volume_fraction():
    # volume-fraction path produces a valid binary structure (count derived from phi_v)
    phantom = generate_structure_phantom(
        (16.0, 16.0, 6.0), (0.2, 0.2, 0.5), "uniaxial", {"mean_phi_deg": 0.0},
        seed=1, n_fibrils=None,
        kappa_par=20, kappa_perp=20, xi_um=40, diameter_um=1.0, diameter_cv=0.2,
        length_um=0.0, length_cv=0.0, persistence_um=1e6,
        crimp_amplitude_um=0.0, crimp_period_um=0.0, volume_fraction=0.15,
    )
    assert len(phantom.geometry) > 0
    dens = np.asarray(phantom.fields.density)
    assert set(np.round(np.unique(dens), 6)).issubset({0.0, 1.0})


def test_network_features_through_gui_path():
    # exclusion + branching + hierarchy all wired from the GUI helper into the generator
    phantom = generate_structure_phantom(
        (12.0, 12.0, 4.0), (0.25, 0.25, 0.5), "uniaxial",
        {"mean_phi_deg": 0.0, "mean_theta_deg": 0.0}, seed=0, n_fibrils=30,
        kappa_par=20, kappa_perp=20, xi_um=40, diameter_um=0.4, diameter_cv=0.2,
        length_um=8.0, length_cv=0.3, persistence_um=1e6,
        crimp_amplitude_um=0.0, crimp_period_um=0.0,
        exclusion=True,
        branching={"density_per_um": 0.2, "angle_deg": 40},
    )
    assert np.asarray(phantom.fields.density).max() == 1.0
    assert any(f.type == "branch" for f in phantom.geometry)


def test_hierarchy_through_gui_path():
    phantom = generate_structure_phantom(
        (12.0, 12.0, 4.0), (0.25, 0.25, 0.5), "uniaxial", {"mean_phi_deg": 0.0},
        seed=1, n_fibrils=None,
        kappa_par=20, kappa_perp=20, xi_um=40, diameter_um=0.4, diameter_cv=0.2,
        length_um=8.0, length_cv=0.3, persistence_um=1e6,
        crimp_amplitude_um=0.0, crimp_period_um=0.0,
        hierarchy={"enabled": True, "n_fascicles": 2, "fibers_per_fascicle": 2,
                   "fibrils_per_fiber": 4},
    )
    assert {f.fascicle_id for f in phantom.geometry} == {0, 1}


def test_skeleton_is_binary_and_inside_the_tubes():
    phantom = generate_structure_phantom(
        (12.0, 12.0, 4.0), (0.25, 0.25, 0.5), "uniaxial",
        {"mean_phi_deg": 90.0, "mean_theta_deg": 0.0}, seed=0, n_fibrils=30,
        kappa_par=30, kappa_perp=30, xi_um=40, diameter_um=1.5, diameter_cv=0.2,
        length_um=0.0, length_cv=0.0, persistence_um=1e6,
        crimp_amplitude_um=0.0, crimp_period_um=0.0,
    )
    skel = skeleton_volume(phantom)
    dens = np.asarray(phantom.fields.density)
    assert skel.shape == dens.shape
    assert set(np.round(np.unique(skel), 6)).issubset({0.0, 1.0})  # binary
    assert skel.sum() > 0
    # the centerline (skeleton) lies inside the fibril tubes: most skeleton voxels are occupied
    inside = float((dens[skel > 0] > 0).mean())
    assert inside > 0.9


def test_isotropic_many_fibrils_low_order():
    phantom = generate_structure_phantom(
        (12.0, 12.0, 4.0), (0.25, 0.25, 0.5), "isotropic", {"mean_phi_deg": 0.0},
        seed=2, n_fibrils=400,
        kappa_par=0, kappa_perp=0, xi_um=40, diameter_um=1.0, diameter_cv=0.2,
        length_um=0.0, length_cv=0.0, persistence_um=1e6,
        crimp_amplitude_um=0.0, crimp_period_um=0.0,
    )
    assert phantom.ground_truth.global_.S2 < 0.35  # isotropic with good statistics