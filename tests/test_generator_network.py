"""Network features: volume exclusion, branching, crosslinking, hierarchy."""

from __future__ import annotations

import numpy as np

from collagen_shg.config.models import StructureConfig
from collagen_shg.structure_generator.generator import ProceduralStructureGenerator

VOXEL = (0.4, 0.2, 0.2)
SHAPE = (8, 48, 48)


def _cfg(**extra):
    base = {
        "architecture": {"type": "uniaxial"},
        "orientation": {"mean_phi_deg": 0.0, "kappa": 20.0},
        "fibril": {"diameter_um": {"mean": 0.6, "dispersion": 0.2}},
    }
    base.update(extra)
    return StructureConfig.model_validate(base)


def _gen(n=None):
    return ProceduralStructureGenerator(SHAPE, VOXEL, n_fibrils=n)


def test_volume_exclusion_caps_packing():
    gen = _gen(n=300)
    vf_overlap = gen.generate(_cfg(), np.random.default_rng(0)).fields.density.mean()
    vf_excl = gen.generate(_cfg(exclusion=True), np.random.default_rng(0)).fields.density.mean()
    # exclusion stops growth at collisions and leaves gaps -> less filled than free overlap
    assert vf_excl < vf_overlap
    assert vf_excl < 0.95


def test_volume_exclusion_still_binary():
    p = _gen(n=100).generate(_cfg(exclusion=True), np.random.default_rng(1))
    vals = np.unique(p.fields.density)
    assert set(np.round(vals, 6)).issubset({0.0, 1.0})


def test_branching_adds_child_fibrils():
    plain = _gen(n=20).generate(_cfg(), np.random.default_rng(0))
    branched = _gen(n=20).generate(
        _cfg(branching={"density_per_um": 0.25, "angle_deg": 40, "max_generations": 2}),
        np.random.default_rng(0),
    )
    assert len(branched.geometry) > len(plain.geometry)
    assert any(f.type == "branch" for f in branched.geometry)


def test_crosslinks_connect_fibrils():
    p = _gen(n=40).generate(
        _cfg(crosslinks={"density_per_um3": 0.1, "max_um": 5.0, "diameter_um": 0.1}),
        np.random.default_rng(2),
    )
    crosslinks = [f for f in p.geometry if f.type == "crosslink"]
    assert len(crosslinks) > 0
    # a crosslink is a short 2-point connector
    assert all(len(f.centerline) == 2 for f in crosslinks)


def test_hierarchy_assigns_fiber_and_fascicle_ids():
    p = _gen().generate(
        _cfg(hierarchy={"enabled": True, "n_fascicles": 2, "fibers_per_fascicle": 3,
                        "fibrils_per_fiber": 5}),
        np.random.default_rng(3),
    )
    assert len(p.geometry) == 2 * 3 * 5
    assert {f.fascicle_id for f in p.geometry} == {0, 1}
    assert len({f.fiber_id for f in p.geometry}) == 2 * 3  # 6 distinct fibers


def test_hierarchy_fibrils_of_a_fiber_are_clustered():
    p = _gen().generate(
        _cfg(hierarchy={"enabled": True, "n_fascicles": 1, "fibers_per_fascicle": 2,
                        "fibrils_per_fiber": 8, "fascicle_radius_um": 3.0, "fiber_radius_um": 0.6}),
        np.random.default_rng(4),
    )
    centroids = {}
    for f in p.geometry:
        centroids.setdefault(f.fiber_id, []).append(np.asarray(f.centerline).mean(axis=0))
    fibers = {k: np.array(v) for k, v in centroids.items()}
    within = np.mean([np.linalg.norm(v - v.mean(0), axis=1).mean() for v in fibers.values()])
    across = np.linalg.norm(
        np.array([v.mean(0) for v in fibers.values()])[0]
        - np.array([v.mean(0) for v in fibers.values()])[1]
    )
    assert within < across  # tighter within a fiber than between fibers