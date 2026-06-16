"""Closed validation loop — the project's differentiator.

Structure (known organization) → image → analysis → measured organization → comparison to the
ground truth (bias). This is what makes metric validation quantitative: the generator stores the
true descriptors, the analyzer measures them on the rendered image, and ``compare`` reports the
bias. The full SNR/depth/dispersion sweeps build on :func:`run_closed_loop`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from collagen_shg.config.models import Config
from collagen_shg.config.seeds import SeedManager
from collagen_shg.imaging.incoherent import IncoherentImager
from collagen_shg.metrics.order import order_parameter_2d, order_tensor_3d
from collagen_shg.metrics.structure_tensor import structure_tensor_3d
from collagen_shg.representations import conventions as cv
from collagen_shg.representations.phantom import Phantom
from collagen_shg.structure_generator.generator import ProceduralStructureGenerator

__all__ = ["ComparisonReport", "ClosedLoopReport", "compare", "analyze_image_3d", "run_closed_loop"]


@dataclass
class ComparisonReport:
    """Measured vs ground-truth descriptors and their bias (measured − truth)."""

    measured: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    bias: dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ClosedLoopReport:
    measured: dict[str, Any]
    ground_truth: dict[str, Any]
    bias: dict[str, float]
    phantom: Phantom | None = None
    bundle: Any = None


def compare(phantom: Phantom, analysis_output: dict[str, Any]) -> ComparisonReport:
    """Compare measured descriptors against a phantom's organization ground truth."""
    gt = phantom.ground_truth.global_
    truth: dict[str, Any] = {}
    for key in ("S2", "S3", "kappa", "xi_um"):
        val = getattr(gt, key, None)
        if val is not None:
            truth[key] = float(val)
    mean_phi = getattr(gt, "mean_phi", None)
    if mean_phi is not None:
        truth["mean_phi"] = float(mean_phi)

    measured = dict(analysis_output)
    bias: dict[str, float] = {}
    for key, true_val in truth.items():
        meas = measured.get(key)
        if meas is None:
            continue
        if key == "mean_phi":
            bias[key] = float(np.angle(np.exp(1j * 2 * (meas - true_val))) / 2)  # axial circular
        else:
            bias[key] = float(meas - true_val)
    return ComparisonReport(measured=measured, ground_truth=truth, bias=bias)


def analyze_image_3d(image: np.ndarray, *, sigma: float = 1.0, rho: float = 2.0) -> dict[str, Any]:
    """Measure orientation/order from a 3D image via the structure tensor + order parameters."""
    img = np.asarray(image, dtype=np.float64)
    st = structure_tensor_3d(img, sigma, rho)
    imgn = img / img.max() if img.max() > 0 else img
    weights = st.fa * imgn  # emphasise anisotropic, bright (fibre) regions
    if weights.sum() <= 0:
        weights = st.fa + 1e-9

    ot = order_tensor_3d(st.director, weights=weights)
    azimuth = cv.wrap_axial(np.arctan2(st.director[1], st.director[0]))
    op = order_parameter_2d(azimuth, weights=weights)
    return {
        "S2": op.S2,
        "S3": ot.S3,
        "mean_phi": op.theta_bar,
        "kappa": op.kappa,
        "mean_director": ot.director,
    }


def run_closed_loop(
    config: Config,
    *,
    sigma: float = 1.0,
    rho: float = 2.0,
    n_fibrils: int | None = None,
    imager: IncoherentImager | None = None,
) -> ClosedLoopReport:
    """Run generate → image → analyze → compare for a validated ``Config``."""
    seeds = SeedManager(config.run.seed)
    generator = ProceduralStructureGenerator(
        config.volume.shape_zyx, config.volume.voxel_size_zyx_um, n_fibrils=n_fibrils
    )
    phantom = generator.generate(config.structure, seeds.generator("structure"))

    imager = imager or IncoherentImager()
    bundle = imager.render(
        phantom, config.microscope, config.degradation, seeds.generator("noise")
    )

    measured = analyze_image_3d(np.asarray(bundle.image), sigma=sigma, rho=rho)
    report = compare(phantom, measured)
    return ClosedLoopReport(
        measured=report.measured,
        ground_truth=report.ground_truth,
        bias=report.bias,
        phantom=phantom,
        bundle=bundle,
    )
