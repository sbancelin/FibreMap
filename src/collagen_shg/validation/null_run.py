"""The Phase 0 end-to-end *null run* — proof that the interfaces wire together.

Pipeline (phase0 §10, acceptance criterion): **empty phantom → white image → trivial
analysis**, optionally written to a bundle and read back. It composes the trivial/null
implementations of the real module interfaces (``NullStructureGenerator``, ``NullImager``,
``TrivialAnalyzer``) and the seed manager, so a single deterministic call demonstrates that
config + seeds + representations + I/O are correctly connected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from collagen_shg.analysis_resolved import TrivialAnalyzer
from collagen_shg.config.models import Config
from collagen_shg.config.seeds import SeedManager
from collagen_shg.imaging import NullImager
from collagen_shg.representations.io import read_bundle, write_bundle
from collagen_shg.structure_generator import NullStructureGenerator

__all__ = ["run_null_pipeline"]


def run_null_pipeline(
    config: Config,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the null pipeline for a validated ``Config``; return a small report dict.

    If ``output_path`` is given, the resulting bundle is written and read back, and the report
    includes ``roundtrip_ok`` (bit-exact image identity + metadata equality).
    """
    seeds = SeedManager(config.run.seed)
    shape = config.volume.shape_zyx
    voxel = config.volume.voxel_size_zyx_um

    # 1. empty phantom (Tier 0 null generator)
    generator = NullStructureGenerator(shape, voxel)
    phantom = generator.generate(config.structure, seeds.generator("structure"))

    # 2. white image (Tier 1 null imager)
    imager = NullImager(fill=1.0)
    bundle = imager.render(
        phantom, config.microscope, config.degradation, seeds.generator("noise")
    )
    bundle.config = config.model_dump(mode="json")
    bundle.metadata.provenance.seed = config.run.seed
    bundle.metadata.provenance.library_versions = _library_versions()

    # 3. trivial analysis
    analyzer = TrivialAnalyzer()
    descriptors = analyzer.analyze(bundle.image)

    report: dict[str, Any] = {
        "phantom_shape": phantom.meta.shape_zyx,
        "n_fibrils": len(phantom.geometry),
        "image_mean": descriptors["mean"],
        "descriptors": descriptors,
        "seed_provenance": seeds.provenance(),
        "roundtrip_ok": None,
    }

    # 4. optional bundle round-trip
    if output_path is not None:
        path = write_bundle(bundle, output_path, overwrite=overwrite)
        reread = read_bundle(path)
        import numpy as np

        report["roundtrip_ok"] = bool(
            np.array_equal(reread.image, bundle.image)
            and reread.metadata == bundle.metadata
        )
        report["bundle_path"] = str(path)

    return report


def _library_versions() -> dict[str, str]:
    import numpy
    import pydantic
    import zarr

    return {
        "numpy": numpy.__version__,
        "pydantic": pydantic.VERSION,
        "zarr": zarr.__version__,
    }