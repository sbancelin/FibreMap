"""GUI generation path and orientation RGB (pure; no napari)."""

from __future__ import annotations

import numpy as np

from collagen_shg.config import load_config_dict
from collagen_shg.gui.app import generate_bundle
from collagen_shg.gui.orientation import director_to_rgb, hsv_to_rgb


def test_hsv_to_rgb_primaries():
    # hue 0 -> red, 1/3 -> green, 2/3 -> blue (full saturation/value)
    one = np.array(1.0)
    zero = np.array(0.0)
    assert np.allclose(hsv_to_rgb(np.array(0.0), one, one), [1, 0, 0])
    assert np.allclose(hsv_to_rgb(np.array(1 / 3), one, one), [0, 1, 0], atol=1e-6)
    assert np.allclose(hsv_to_rgb(np.array(2 / 3), one, one), [0, 0, 1], atol=1e-6)
    assert np.allclose(hsv_to_rgb(np.array(0.0), zero, one), [1, 1, 1])  # desaturated -> white


def test_director_to_rgb_shape_and_range():
    director = np.zeros((3, 2, 4, 4))
    director[0] = 1.0  # all along +x
    rgb = director_to_rgb(director, weight=np.ones((2, 4, 4)))
    assert rgb.shape == (2, 4, 4, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    # constant azimuth -> constant hue
    assert np.allclose(rgb, rgb[0, 0, 0])


def test_director_to_rgb_hue_varies_with_azimuth():
    dx = np.zeros((3, 1, 1, 1))
    dx[0] = 1.0  # azimuth 0
    dy = np.zeros((3, 1, 1, 1))
    dy[1] = 1.0  # azimuth pi/2
    assert not np.allclose(director_to_rgb(dx), director_to_rgb(dy))


def test_generate_bundle_from_config():
    cfg = load_config_dict(
        {
            "run": {"name": "gen", "seed": 7},
            "volume": {"shape_zyx": [8, 32, 32], "voxel_size_zyx_um": [0.5, 0.2, 0.2]},
            "structure": {"preset": "tendon", "overrides": {"orientation": {"kappa": 20}}},
            "microscope": {"preset": "default"},
        }
    )
    bundle = generate_bundle(cfg, n_fibrils=60)
    assert bundle.shape_zyx == (8, 32, 32)
    assert bundle.phantom is not None
    assert bundle.phantom.ground_truth.global_.S2 is not None
    assert bundle.image.dtype == np.float32
