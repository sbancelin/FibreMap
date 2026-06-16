"""End-to-end smoke test — the Phase 0 "null run" (phase0 acceptance criterion).

Empty phantom → white image → trivial analysis → bundle written and read back. Proves the
config + seeds + representations + I/O interfaces wire together, and that a given
{config + seed} reproduces identical output (Tier 0-1 determinism).
"""

from __future__ import annotations

from collagen_shg.config import load_config_dict
from collagen_shg.validation import run_null_pipeline

_RAW = {
    "run": {"name": "nullrun", "seed": 20260616},
    "volume": {"shape_zyx": [2, 16, 16], "voxel_size_zyx_um": [0.5, 0.2, 0.2]},
    "structure": {"preset": "tendon"},
    "microscope": {"preset": "default"},
}


def test_null_run_wires_interfaces(tmp_path):
    cfg = load_config_dict(_RAW)
    report = run_null_pipeline(cfg, output_path=tmp_path / "nullrun.bundle")

    # empty phantom
    assert report["phantom_shape"] == (2, 16, 16)
    assert report["n_fibrils"] == 0
    # white image -> trivial analysis
    assert report["image_mean"] == 1.0
    assert report["descriptors"]["min"] == 1.0
    assert report["descriptors"]["max"] == 1.0
    # bundle round-trip held
    assert report["roundtrip_ok"] is True
    # seeds logged for provenance
    assert report["seed_provenance"]["master_seed"] == 20260616
    assert set(report["seed_provenance"]["children"]) == {"structure", "noise"}


def test_null_run_is_reproducible(tmp_path):
    cfg = load_config_dict(_RAW)
    r1 = run_null_pipeline(cfg, output_path=tmp_path / "a.bundle")
    r2 = run_null_pipeline(cfg, output_path=tmp_path / "b.bundle")
    assert r1["descriptors"] == r2["descriptors"]
    assert r1["seed_provenance"] == r2["seed_provenance"]