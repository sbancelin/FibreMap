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

from collagen_shg.gui.orientation import director_to_rgb
from collagen_shg.representations.image_bundle import ImageBundle
from collagen_shg.representations.io import read_bundle, read_ome_tiff

__all__ = [
    "build_layers",
    "load_any",
    "generate_bundle",
    "add_bundle_to_viewer",
    "launch",
    "main",
]

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
        rgb = director_to_rgb(np.asarray(fields.director), np.asarray(fields.order_S))
        layers.append(
            (rgb, {"name": "orientation (GT)", "scale": scale, "rgb": True, "visible": False},
             "image")
        )
    return layers


def generate_bundle(config: Any, *, n_fibrils: int | None = None) -> ImageBundle:
    """Generate a synthetic bundle from a validated ``Config`` (structure → incoherent image).

    Convenience for the GUI's generation path; the heavy modules are imported lazily.
    """
    from collagen_shg.config.seeds import SeedManager
    from collagen_shg.imaging.incoherent import IncoherentImager
    from collagen_shg.structure_generator.generator import ProceduralStructureGenerator

    seeds = SeedManager(config.run.seed)
    generator = ProceduralStructureGenerator(
        config.volume.shape_zyx, config.volume.voxel_size_zyx_um, n_fibrils=n_fibrils
    )
    phantom = generator.generate(config.structure, seeds.generator("structure"))
    return IncoherentImager().render(
        phantom, config.microscope, config.degradation, seeds.generator("noise")
    )


def add_bundle_to_viewer(viewer: Any, bundle: ImageBundle) -> Any:
    """Add a bundle's layers to a napari viewer; returns the viewer."""
    for data, kwargs, layer_type in build_layers(bundle):
        adder = getattr(viewer, f"add_{layer_type}")
        adder(data, **kwargs)
    return viewer


def view_bundle(bundle: ImageBundle, *, title: str = "collagen-shg", show: bool = True,
                block: bool = True) -> Any:
    """Open an in-memory bundle in a napari viewer (lazy napari import)."""
    try:
        import napari
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "napari is required for the GUI shell. Install it with:\n"
            '    pip install -e ".[gui]"'
        ) from exc
    viewer = napari.Viewer(show=show, title=title)
    add_bundle_to_viewer(viewer, bundle)
    if block and show:
        napari.run()
    return viewer


def launch(path: str | Path, *, show: bool = True, block: bool = True) -> Any:
    """Open a bundle/OME-TIFF in a napari viewer. Returns the viewer."""
    bundle = load_any(path)
    return view_bundle(bundle, title=f"collagen-shg — {Path(path).name}", show=show, block=block)


def launch_generated(config_path: str | Path, *, show: bool = True, block: bool = True) -> Any:
    """Generate a bundle from a config YAML and open it in napari (the generation path)."""
    from collagen_shg.config.loader import load_config

    config = load_config(config_path)
    bundle = generate_bundle(config)
    return view_bundle(bundle, title=f"collagen-shg — generated {config.run.name}",
                       show=show, block=block)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collagen-shg-gui",
        description="Interactive collagen-shg GUI (generation / imaging / analysis). "
        "With no argument, launches the full interactive app; or quickly view a bundle/"
        "OME-TIFF, or generate one from a config.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("path", nargs="?", help="path to a *.bundle directory or an OME-TIFF file")
    group.add_argument("--generate", metavar="CONFIG.yaml",
                       help="generate a synthetic bundle from a run config and view it")
    args = parser.parse_args(argv)
    try:
        if args.generate:
            launch_generated(args.generate)
        elif args.path:
            launch(args.path)
        else:
            from collagen_shg.gui.interactive import run_app

            run_app()
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())