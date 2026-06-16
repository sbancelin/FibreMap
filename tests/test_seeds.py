"""Seed reproducibility and independence (phase0 acceptance: same {config+seed} => identical)."""

from __future__ import annotations

import numpy as np

from collagen_shg.config.seeds import SeedManager, derive_generator, name_key


def test_derive_generator_is_deterministic():
    a = derive_generator(123, "noise").random(8)
    b = derive_generator(123, "noise").random(8)
    assert np.array_equal(a, b)


def test_named_children_are_independent():
    sm = SeedManager(2026)
    a = sm.generator("structure").random(1000)
    b = sm.generator("noise").random(1000)
    assert not np.array_equal(a, b)
    # independent streams should be ~uncorrelated
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.1


def test_master_seed_changes_streams():
    a = SeedManager(1).generator("noise").random(8)
    b = SeedManager(2).generator("noise").random(8)
    assert not np.array_equal(a, b)


def test_name_key_stable_and_distinct():
    assert name_key("structure") == name_key("structure")
    assert name_key("structure") != name_key("noise")
    assert isinstance(name_key("x"), int)


def test_adding_a_child_does_not_perturb_others():
    # Keying by name (not draw order) => a new component never shifts existing streams.
    sm1 = SeedManager(7)
    s1 = sm1.generator("structure").random(8)
    sm2 = SeedManager(7)
    _ = sm2.generator("noise").random(8)  # request a different child first
    s2 = sm2.generator("structure").random(8)
    assert np.array_equal(s1, s2)


def test_provenance_logs_children():
    sm = SeedManager(42)
    sm.generator("structure")
    sm.generator("noise")
    prov = sm.provenance()
    assert prov["rng"] == "PCG64"
    assert prov["master_seed"] == 42
    assert set(prov["children"]) == {"structure", "noise"}