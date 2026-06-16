"""GUI layer-building tests — pure, no napari dependency required."""

from __future__ import annotations

import numpy as np

from collagen_shg.gui.app import build_layers, load_any
from collagen_shg.representations import ImageBundle, Phantom, write_bundle


def test_build_layers_image_only():
    b = ImageBundle.white((2, 4, 4), (0.5, 0.2, 0.2))
    layers = build_layers(b)
    assert len(layers) == 1
    data, kwargs, ltype = layers[0]
    assert ltype == "image"
    assert kwargs["name"] == "image"
    assert kwargs["scale"] == (0.5, 0.2, 0.2)
    assert np.array_equal(data, b.image)


def test_build_layers_with_ground_truth_fields():
    p = Phantom.empty((2, 4, 4), (0.5, 0.2, 0.2))
    p.fields.order_S[...] = 0.5
    b = ImageBundle.white((2, 4, 4), (0.5, 0.2, 0.2), phantom=p)
    layers = build_layers(b)
    names = [kw["name"] for _, kw, _ in layers]
    assert names == ["image", "order_S (GT)", "density (GT)", "|director| (GT)"]
    # director magnitude is computed over the channel axis
    mag = layers[-1][0]
    assert mag.shape == (2, 4, 4)


def test_load_any_roundtrips_bundle(tmp_path):
    p = Phantom.empty((2, 4, 4), (0.5, 0.2, 0.2))
    b = ImageBundle.white((2, 4, 4), (0.5, 0.2, 0.2), phantom=p)
    path = tmp_path / "x.bundle"
    write_bundle(b, path)
    loaded = load_any(path)
    assert loaded.shape_zyx == (2, 4, 4)
    assert np.array_equal(loaded.image, b.image)