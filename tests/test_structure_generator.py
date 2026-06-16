"""Tier 0 structure generator: determinism, valid fields, known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.config.models import StructureConfig
from collagen_shg.representations import conventions as cv
from collagen_shg.structure_generator import ProceduralStructureGenerator

SHAPE = (12, 64, 64)
VOXEL = (0.5, 0.2, 0.2)


def _config(mean_phi_deg=90.0, kappa=20.0):
    return StructureConfig.model_validate(
        {
            "preset": "tendon",
            "orientation": {"mean_phi_deg": mean_phi_deg, "kappa": kappa, "xi_um": 40.0},
            "fibril": {
                "diameter_um": {"mean": 1.0, "dispersion": 0.2},
                "crimp": {"amplitude_um": 1.0, "period_um": 20.0},
            },
        }
    )


def _generate(cfg, seed=0, n_fibrils=150):
    gen = ProceduralStructureGenerator(SHAPE, VOXEL, n_fibrils=n_fibrils)
    return gen.generate(cfg, np.random.default_rng(seed))


def test_is_deterministic():
    cfg = _config()
    a = _generate(cfg, seed=42)
    b = _generate(cfg, seed=42)
    assert np.array_equal(a.fields.director, b.fields.director)
    assert np.array_equal(a.fields.density, b.fields.density)
    assert a.ground_truth.global_.S2 == b.ground_truth.global_.S2
    assert a.ground_truth.global_.S3 == b.ground_truth.global_.S3


def test_fields_valid_and_density_normalized():
    p = _generate(_config())
    assert p.fields.shape_zyx == SHAPE
    assert p.fields.director.shape == (3, *SHAPE)
    assert 0.0 <= p.fields.density.min() and p.fields.density.max() == pytest.approx(1.0)
    assert len(p.geometry) == 150
    # director is unit (or zero) everywhere
    norm = np.linalg.norm(p.fields.director, axis=0)
    assert np.all((np.abs(norm - 1.0) < 1e-4) | (norm < 1e-6))


def test_ground_truth_aligned_high_kappa():
    p = _generate(_config(mean_phi_deg=90.0, kappa=30.0))
    gt = p.ground_truth.global_
    assert gt.S2 > 0.8
    assert gt.S3 > 0.7
    mean_phi = gt.mean_phi
    d = abs(np.angle(np.exp(1j * 2 * (mean_phi - np.pi / 2))) / 2)
    assert d < np.deg2rad(6)


def test_ground_truth_isotropic_low_kappa():
    p = _generate(_config(mean_phi_deg=0.0, kappa=0.02), n_fibrils=400)
    assert p.ground_truth.global_.S2 < 0.25


def test_director_field_azimuth_follows_mean_for_high_kappa():
    p = _generate(_config(mean_phi_deg=0.0, kappa=40.0), n_fibrils=300)
    dirf = p.fields.director
    mask = p.fields.density > 0.3
    phi = cv.wrap_axial(np.arctan2(dirf[1][mask], dirf[0][mask]))
    # circular mean of the doubled angle should be near 0 (mean_phi = 0)
    mean_phi = cv.angle_from_doubled(np.mean(np.cos(2 * phi)), np.mean(np.sin(2 * phi)))
    d = abs(np.angle(np.exp(1j * 2 * (float(mean_phi) - 0.0))) / 2)
    assert d < np.deg2rad(12)
