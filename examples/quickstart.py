"""End-to-end quickstart: generate -> image -> save bundle -> reload -> analyze -> compare.

Run it (from the repo root, with the project venv active or via its python)::

    python examples/quickstart.py                          # uses configs/runs/demo_small.yaml
    python examples/quickstart.py configs/runs/demo_tendon.yaml   # full-size (slow/large)

It writes a reproducible *.bundle, re-reads it, runs the resolved-tissue analyzer, and prints
the measured organization next to the known ground truth (the closed validation loop).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from collagen_shg.analysis_resolved import ResolvedAnalyzer
from collagen_shg.config import load_config
from collagen_shg.config.seeds import SeedManager
from collagen_shg.imaging import IncoherentImager
from collagen_shg.representations import read_bundle, write_bundle
from collagen_shg.structure_generator import ProceduralStructureGenerator
from collagen_shg.validation import compare


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    print(f"# config: {config_path}  (run={cfg.run.name}, seed={cfg.run.seed})")
    print(f"# volume: shape_zyx={cfg.volume.shape_zyx} voxel_um={cfg.volume.voxel_size_zyx_um}")

    seeds = SeedManager(cfg.run.seed)

    # 1. Tier 0 — generate a ground-truth phantom
    generator = ProceduralStructureGenerator(
        cfg.volume.shape_zyx, cfg.volume.voxel_size_zyx_um
    )
    phantom = generator.generate(cfg.structure, seeds.generator("structure"))
    gt = phantom.ground_truth.global_
    print(f"\n[1] generated phantom: {len(phantom.geometry)} fibrils")
    print(f"    ground truth: S2={gt.S2:.3f} S3={gt.S3:.3f} "
          f"mean_phi={np.rad2deg(gt.mean_phi):.1f} deg")

    # 2. Tier 1 — incoherent image
    bundle = IncoherentImager().render(
        phantom, cfg.microscope, cfg.degradation, seeds.generator("noise")
    )
    img = bundle.image
    print(f"\n[2] image: shape={img.shape} dtype={img.dtype} "
          f"mean={img.mean():.1f} max={img.max():.1f}")

    # 3. Save + reload the bundle (lossless round-trip)
    out = Path(cfg.run.output or "datasets/demo.bundle")
    write_bundle(bundle, out, overwrite=True)
    reread = read_bundle(out)
    print(f"\n[3] bundle written to: {out.resolve()}")
    print(f"    reload image identical: {np.array_equal(reread.image, bundle.image)}")

    # 4. Livrable 3 — analyze and compare to the ground truth (closed loop)
    result = ResolvedAnalyzer().analyze_bundle(reread)
    report = compare(reread.phantom, result.measured())
    print("\n[4] measured vs ground truth:")
    measured = result.measured()
    for key in ("S2", "S3", "mean_phi"):
        meas, truth, bias = measured[key], report.ground_truth[key], report.bias[key]
        if key == "mean_phi":
            meas, truth, bias = np.rad2deg(meas), np.rad2deg(truth), np.rad2deg(bias)
            unit = " deg"
        else:
            unit = ""
        print(f"    {key:9s} measured={meas:7.3f}{unit}  truth={truth:7.3f}{unit}  "
              f"bias={bias:+.3f}{unit}")
    return 0


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/runs/demo_small.yaml"
    raise SystemExit(main(config))
