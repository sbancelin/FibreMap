"""Family G — topological defects: ±1/2 disclinations detected by winding number."""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.metrics.defects import defect_density
from collagen_shg.representations import conventions as cv


def _disclination(n, charge):
    """Axial field θ = charge · atan2(y−cy, x−cx) with the core inside one plaquette."""
    c = n / 2 - 0.5
    yy, xx = np.mgrid[0:n, 0:n]
    return cv.wrap_axial(charge * np.arctan2(yy - c, xx - c))


def test_uniform_field_has_no_defects():
    res = defect_density(np.full((64, 64), 0.4))
    assert res.n_defects == 0
    assert res.density == 0.0


def test_plus_half_disclination():
    res = defect_density(_disclination(64, 0.5))
    assert res.n_defects == 1
    peak = res.defect_map.flat[np.argmax(np.abs(res.defect_map))]
    assert peak == pytest.approx(0.5, abs=0.05)


def test_minus_half_disclination():
    res = defect_density(_disclination(64, -0.5))
    assert res.n_defects == 1
    peak = res.defect_map.flat[np.argmax(np.abs(res.defect_map))]
    assert peak == pytest.approx(-0.5, abs=0.05)


def test_density_scales_with_count():
    one = defect_density(_disclination(64, 0.5))
    assert one.density == pytest.approx(1.0 / one.defect_map.size, abs=1e-12)
