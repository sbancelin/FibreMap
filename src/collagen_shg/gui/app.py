"""Minimal launchable napari shell for inspecting a bundle (Phase 0).

This is the seed of the GUI that will grow with each Livrable (Génération / Analyse tabs). For
now it opens a ``dataset.bundle`` (or an OME-TIFF) and shows the image plus any ground-truth
fields as napari layers. The napari import is deferred so the package imports without the
optional ``gui`` dependency installed; :func:`build_layers` is pure and unit-tested without
napari.

Usage::

    collagen-shg-gui path/to/dataset.bundle
    collagen-shg-gui path/to/image.ome.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from collagen_shg.representations.image_bundle import ImageBundle
from collagen_shg.representations.io import read_bundle, read_ome_tiff

__all__ = ["build_layers", "load_any", "add_bundle_to_viewer", "launch", "main"]

# A napari "LayerData"-style tuple: (data, kwargs, layer_type).
LayerData = tuple[np.ndarray, dict[str, Any], str]


def load_any(path: str | Path) -> ImageBundle:
    """Load a bundle directory or an OME-TIFF file into an ``ImageBundle``."""
    path = Path(path)
    if path.is_dir():
        return read_bundle(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        return read_ome_tiff(path)
    raise ValueError(f"unsupported path (expected a *.bundle dir or OME-TIFF): {path}")


def build_layers(bundle: ImageBundle) -> list[LayerData]:
    """Build napari layer specs for a bundle (pure; no napari import).

    Always yields the image; adds ground-truth ``order_S`` and ``density`` scalar fields, and a
    director-magnitude field, when a phantom with voxelized fields is present.
    """
    dz, dy, dx = bundle.metadata.voxel_size_zyx
    scale = (dz, dy, dx)
    layers: list[LayerData] = [
        (np.asarray(bundle.image), {"name": "image", "scale": scale, "colormap": "gray"},
         "image"),
    ]

    fields = bundle.phantom.fields if bundle.phantom is not None else None
    if fields is not None:
        layers.append(
            (np.asarray(fields.order_S),
             {"name": "order_S (GT)", "scale": scale, "colormap": "viridis", "visible": False},
             "image")
        )
        layers.append(
            (np.asarray(fields.density),
             {"name": "density (GT)", "scale": scale, "colormap": "magma", "visible": False},
             "image")
        )
        director_mag = np.linalg.norm(np.asarray(fields.director), axis=0)
        layers.append(
            (director_mag,
             {"name": "|director| (GT)", "scale": scale, "colormap": "inferno",
              "visible": False},
             "image")
        )
    return layers


def add_bundle_to_viewer(viewer: Any, bundle: ImageBundle) -> Any:
    """Add a bundle's layers to a napari viewer; returns the viewer."""
    for data, kwargs, layer_type in build_layers(bundle):
        adder = getattr(viewer, f"add_{layer_type}")
        adder(data, **kwargs)
    return viewer


def launch(path: str | Path, *, show: bool = True, block: bool = True) -> Any:
    """Open a bundle/OME-TIFF in a napari viewer. Returns the viewer.

    ``show=False`` builds the viewer without displaying it (useful for tests). ``block=True``
    runs the napari event loop.
    """
    try:
        import napari
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "napari is required for the GUI shell. Install it with:\n"
            '    pip install -e ".[gui]"'
        ) from exc

    bundle = load_any(path)
    viewer = napari.Viewer(show=show, title=f"collagen-shg — {Path(path).name}")
    add_bundle_to_viewer(viewer, bundle)
    if block and show:
        napari.run()
    return viewer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collagen-shg-gui",
        description="Open a collagen-shg bundle (or OME-TIFF) in a napari viewer.",
    )
    parser.add_argument("path", help="path to a *.bundle directory or an OME-TIFF file")
    args = parser.parse_args(argv)
    try:
        launch(args.path)
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())