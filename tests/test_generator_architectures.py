"""Architecture-aware generator: binary tubes, biaxiality, density, all templates run."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.config.models import StructureConfig
from collagen_shg.structure_generator.architecture import ARCHITECTURES
from collagen_shg.structure_generator.generator import ProceduralStructureGenerator

VOXEL = (0.5, 0.2, 0.2)


def _config(architecture, params=None, kappa=20.0, diameter=1.0):
    return StructureConfig.model_validate(
        {
            "architecture": {"type": architecture, **(params or {})},
            "orientation": {"mean_phi_deg": 0.0, "kappa": kappa, "xi_um": 40.0},
            "fibril": {"diameter_um": {"mean": diameter, "dispersion": 0.2}},
        }
    )


def _gen(config, shape=(8, 48, 48), n=60, seed=0):
    gen = ProceduralStructureGenerator(shape, VOXEL, n_fibrils=n)
    return gen.generate(config, np.random.default_rng(seed))


def test_volume_is_binary():
    p = _gen(_config("uniaxial"))
    vals = np.unique(p.fields.density)
    assert set(np.round(vals, 6)).issubset({0.0, 1.0})
    assert p.fields.density.max() == 1.0
    assert (p.fields.density == 0).any()


def test_fibrils_are_tubes_not_points():
    p = _gen(_config("uniaxial", diameter=1.0))
    occupied = int((p.fields.density > 0).sum())
    # each fibril is a multi-voxel tube: far more occupied voxels than fibrils
    assert occupied > 20 * len(p.geometry)


def test_director_defined_only_where_occupied():
    p = _gen(_config("uniaxial"))
    occ = p.fields.density > 0
    norm = np.linalg.norm(p.fields.director, axis=0)
    assert np.all(norm[~occ] < 1e-6)  # empty voxels have no director
    assert np.all(np.abs(norm[occ] - 1.0) < 1e-4)  # occupied -> unit director


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_all_architectures_run(architecture):
    params = {
        "biaxial": {"phi_a_deg": 0, "phi_b_deg": 90},
        "lamellar": {"lamella_thickness_um": 1.5, "lamella_dphi_deg": 90},
        "arcade": {"theta_deep_deg": 90, "theta_surface_deg": 0},
        "tubular": {"helix_beta_deg": 20, "crossed": True},
    }.get(architecture, {})
    p = _gen(_config(architecture, params), n=40)
    assert p.fields.density.max() == 1.0
    assert p.ground_truth.global_.S is not None


def test_biaxiality_uniaxial_below_biaxial():
    uni = _gen(_config("uniaxial", kappa=40)).ground_truth.global_.biaxiality
    bi = _gen(
        _config("biaxial", {"phi_a_deg": 0, "phi_b_deg": 90}, kappa=40)
    ).ground_truth.global_.biaxiality
    assert bi > uni + 0.2


def test_volume_fraction_controls_count():
    def count(frac):
        cfg = StructureConfig.model_validate(
            {
                "architecture": {"type": "uniaxial"},
                "orientation": {"mean_phi_deg": 0.0, "kappa": 20.0},
                "fibril": {"diameter_um": {"mean": 1.0}},
                "volume_fraction": frac,
            }
        )
        gen = ProceduralStructureGenerator((8, 64, 64), VOXEL, n_fibrils=None)
        return len(gen.generate(cfg, np.random.default_rng(0)).geometry)

    assert count(0.3) > count(0.05)


def test_ground_truth_has_full_descriptor_set():
    gt = _gen(_config("uniaxial", kappa=30)).ground_truth.global_
    for key in ("S2", "S3", "S", "biaxiality", "mean_phi", "mean_theta", "architecture",
                "volume_fraction"):
        assert getattr(gt, key) is not None