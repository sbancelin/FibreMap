"""Family F — per-fibre metrics and persistence length on known curves."""

from __future__ import annotations

import numpy as np

from collagen_shg.metrics.fibers import fiber_metrics, persistence_length


def _straight_along(axis, n=100, length=50.0):
    t = np.linspace(0, length, n)
    pts = np.zeros((n, 3))
    pts[:, axis] = t
    return pts


def _crimped(amplitude, n=400, length=100.0, period=20.0):
    t = np.linspace(0, length, n)
    x = t
    y = amplitude * np.sin(2 * np.pi * t / period)
    z = np.zeros_like(t)
    return np.stack([x, y, z], axis=-1)


def test_straight_fiber_metrics():
    res = fiber_metrics([_straight_along(0)])
    rec = res.per_fiber[0]
    assert rec.straightness == 1.0
    assert rec.tortuosity == 1.0
    # principal axis along x -> azimuth 0, elevation 0
    d = abs(np.angle(np.exp(1j * 2 * (rec.azimuth - 0.0))) / 2)
    assert d < 1e-6
    assert abs(rec.elevation) < 1e-6


def test_crimped_fiber_is_wavy():
    res = fiber_metrics([_crimped(amplitude=3.0)])
    rec = res.per_fiber[0]
    assert rec.straightness < 0.98
    assert rec.tortuosity > 1.0
    # mean orientation still along x (the crimp is symmetric about the x axis)
    assert abs(np.angle(np.exp(1j * 2 * rec.azimuth)) / 2) < np.deg2rad(5)


def test_network_statistics():
    res = fiber_metrics([_straight_along(0), _straight_along(1), _crimped(2.0)])
    assert res.network.n_fibers == 3
    assert res.network.total_length > 0
    assert 0 < res.network.mean_straightness <= 1.0


def test_persistence_length_straight_is_infinite():
    assert np.isinf(persistence_length(_straight_along(0)))


def test_persistence_length_decreases_with_waviness():
    lp_low = persistence_length(_crimped(amplitude=0.5))
    lp_high = persistence_length(_crimped(amplitude=3.0))
    assert np.isfinite(lp_high)
    assert lp_high < lp_low  # more amplitude -> shorter persistence length


def test_fiber_metrics_accepts_fibril_objects():
    from collagen_shg.representations import Fibril

    f = Fibril(id=7, centerline=_straight_along(2), diameter=np.ones(100))
    res = fiber_metrics([f])
    assert res.per_fiber[0].id == 7
    # along z -> elevation pi/2
    assert abs(abs(res.per_fiber[0].elevation) - np.pi / 2) < 1e-6
