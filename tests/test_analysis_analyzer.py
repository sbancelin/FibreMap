"""ResolvedAnalyzer on synthetic bundles: closed-loop recovery of known ground truth."""

from __future__ import annotations

import numpy as np

from collagen_shg.analysis_resolved.analyzer import ResolvedAnalyzer
from collagen_shg.config.models import DegradationConfig, MicroscopeConfig, StructureConfig
from collagen_shg.config.seeds import SeedManager
from collagen_shg.imaging.incoherent import IncoherentImager
from collagen_shg.structure_generator.generator import ProceduralStructureGenerator
from collagen_shg.validation import compare

SHAPE = (16, 64, 64)
VOXEL = (0.5, 0.2, 0.2)


def _bundle(mean_phi_deg, kappa, seed=20260616, n_fibrils=300):
    cfg = StructureConfig.model_validate(
        {"preset": "tendon", "orientation": {"mean_phi_deg": mean_phi_deg, "kappa": kappa}}
    )
    seeds = SeedManager(seed)
    gen = ProceduralStructureGenerator(SHAPE, VOXEL, n_fibrils=n_fibrils)
    phantom = gen.generate(cfg, seeds.generator("structure"))
    mic = MicroscopeConfig(NA=0.9, wavelength_nm=900, detection="backward")
    deg = DegradationConfig.model_validate({"noise": {"photons_peak": 4000.0}})
    bundle = IncoherentImager().render(phantom, mic, deg, seeds.generator("noise"))
    return bundle


def test_analyzer_recovers_orientation_aligned():
    bundle = _bundle(mean_phi_deg=90.0, kappa=30.0)
    result = ResolvedAnalyzer().analyze_bundle(bundle)
    report = compare(bundle.phantom, result.measured())
    assert abs(report.bias["mean_phi"]) < np.deg2rad(15)
    assert result.descriptors.S2 > 0.4
    assert result.director.shape == (3, *SHAPE)
    assert result.orientation.shape == SHAPE


def test_analyzer_isotropic_low_order():
    bundle = _bundle(mean_phi_deg=0.0, kappa=0.05, n_fibrils=500)
    result = ResolvedAnalyzer().analyze_bundle(bundle)
    assert result.descriptors.S2 < 0.4


def test_analyzer_aligned_above_isotropic():
    aligned = ResolvedAnalyzer().analyze_bundle(_bundle(45.0, 30.0)).descriptors.S2
    iso = ResolvedAnalyzer().analyze_bundle(_bundle(45.0, 0.05, n_fibrils=500)).descriptors.S2
    assert aligned > iso


def test_analyzer_bootstrap_ci():
    bundle = _bundle(mean_phi_deg=90.0, kappa=30.0)
    result = ResolvedAnalyzer(bootstrap=True, n_boot=80).analyze_bundle(bundle)
    assert result.ci is not None
    lo, hi = result.ci["S2"]
    assert 0.0 <= lo <= hi <= 1.0


def test_analyze_rejects_2d():
    import pytest

    with pytest.raises(ValueError, match="3D volume"):
        ResolvedAnalyzer().analyze(np.zeros((8, 8)))
