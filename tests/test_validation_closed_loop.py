"""Closed-loop validation: analyzers recover the generator's known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.config import load_config_dict
from collagen_shg.representations.phantom import GlobalGT, OrganizationGT, Phantom
from collagen_shg.validation import compare, run_closed_loop


def _config(mean_phi_deg, kappa, seed=20260616):
    return load_config_dict(
        {
            "run": {"name": "loop", "seed": seed},
            "volume": {"shape_zyx": [16, 64, 64], "voxel_size_zyx_um": [0.5, 0.2, 0.2]},
            "structure": {
                "preset": "tendon",
                "overrides": {"orientation": {"mean_phi_deg": mean_phi_deg, "kappa": kappa}},
            },
            "microscope": {
                "preset": "default",
                "overrides": {"NA": 0.9, "wavelength_nm": 900, "detection": "backward"},
            },
            "degradation": {"noise": {"photons_peak": 4000.0, "read_noise_e": 0.0}},
        }
    )


def test_compare_computes_bias():
    p = Phantom.empty((4, 8, 8), (0.5, 0.2, 0.2))
    p.ground_truth = OrganizationGT(global_=GlobalGT(S2=0.8, S3=0.7))
    p.ground_truth.global_.mean_phi = 1.0  # extra field (extra="allow")
    rep = compare(p, {"S2": 0.7, "S3": 0.65, "mean_phi": 1.05})
    assert rep.bias["S2"] == pytest.approx(-0.1)
    assert rep.bias["S3"] == pytest.approx(-0.05)
    assert rep.bias["mean_phi"] == pytest.approx(0.05)


def test_closed_loop_recovers_orientation_when_aligned():
    rep = run_closed_loop(_config(mean_phi_deg=90.0, kappa=30.0), n_fibrils=300)
    # measured mean orientation matches the generated mean within tolerance
    assert abs(rep.bias["mean_phi"]) < np.deg2rad(15)
    # measured order is appreciable (aligned sample)
    assert rep.measured["S2"] > 0.4
    assert "S3" in rep.ground_truth


def test_closed_loop_isotropic_has_low_measured_order():
    rep = run_closed_loop(_config(mean_phi_deg=0.0, kappa=0.05), n_fibrils=500)
    assert rep.measured["S2"] < 0.4
    assert rep.ground_truth["S2"] < 0.3


def test_aligned_measures_higher_order_than_isotropic():
    aligned = run_closed_loop(_config(mean_phi_deg=45.0, kappa=30.0), n_fibrils=300)
    iso = run_closed_loop(_config(mean_phi_deg=45.0, kappa=0.05), n_fibrils=500)
    assert aligned.measured["S2"] > iso.measured["S2"]
