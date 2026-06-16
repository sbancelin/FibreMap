"""Config loading, preset resolution and validation (phase0 acceptance: config drives a run)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from collagen_shg.config import Config, default_configs_root, load_config, load_config_dict


def test_load_demo_run():
    path = default_configs_root() / "runs" / "demo_tendon.yaml"
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert cfg.run.name == "demo_tendon"
    assert cfg.run.seed == 20260612
    assert cfg.volume.shape_zyx == (64, 512, 512)
    assert cfg.volume.voxel_size_zyx_um == (0.5, 0.2, 0.2)
    # resolved structure: preset name kept, overrides applied
    assert cfg.structure.preset == "tendon"
    assert cfg.structure.orientation.kappa == 12
    assert cfg.structure.fibril.diameter_um.mean == 1.5
    # microscope preset + overrides
    assert cfg.microscope.preset == "default"
    assert cfg.microscope.NA == 0.95
    assert cfg.microscope.mode == "incoherent"
    assert cfg.microscope.psf_model == "gaussian"  # inherited from preset (not overridden)
    # degradation
    assert cfg.degradation.depth.attenuation_length_um == 80
    assert cfg.degradation.noise.photons_peak == 500


def test_preset_inheritance_and_override():
    raw = {
        "run": {"name": "t", "seed": 1},
        "volume": {"shape_zyx": [4, 8, 8], "voxel_size_zyx_um": [1.0, 1.0, 1.0]},
        "structure": {"preset": "tendon", "overrides": {"orientation": {"kappa": 99}}},
    }
    cfg = load_config_dict(raw)
    assert cfg.structure.orientation.kappa == 99  # overridden
    assert cfg.structure.orientation.xi_um == 40.0  # inherited from tendon preset
    assert cfg.structure.orientation.mean_phi_deg == 90.0  # inherited, not wiped by sibling override
    assert cfg.structure.fibril.diameter_um.mean == 1.5  # inherited deep value


def test_missing_preset_raises():
    raw = {
        "run": {"name": "t", "seed": 1},
        "volume": {"shape_zyx": [4, 8, 8], "voxel_size_zyx_um": [1.0, 1.0, 1.0]},
        "structure": {"preset": "does_not_exist"},
    }
    with pytest.raises(FileNotFoundError):
        load_config_dict(raw)


def test_invalid_volume_rejected():
    raw = {
        "run": {"name": "t", "seed": 1},
        "volume": {"shape_zyx": [0, 8, 8], "voxel_size_zyx_um": [1.0, 1.0, 1.0]},
    }
    with pytest.raises(ValidationError):
        load_config_dict(raw)


def test_config_without_presets():
    raw = {
        "run": {"name": "bare", "seed": 5},
        "volume": {"shape_zyx": [2, 4, 4], "voxel_size_zyx_um": [0.5, 0.2, 0.2]},
        "microscope": {"overrides": {"NA": 1.2}},
    }
    cfg = load_config_dict(raw)
    assert cfg.microscope.NA == 1.2
    assert cfg.microscope.preset is None