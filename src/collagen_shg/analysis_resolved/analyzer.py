"""Resolved-tissue analyzer (Livrable 3).

The field-based analysis chain for resolved tissues (skin/dermis, tendon, vocal fold, bone):
preprocess → multi-scale orientation field → organization descriptors (+ optional bootstrap
CIs). Validated on synthetic data with known ground truth (the closed loop). The learned
extractor (trained on synthetic centerlines) is a separate, later component.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from collagen_shg.analysis_resolved.descriptors import (
    OrganizationDescriptors,
    bootstrap_order_ci,
    descriptor_vector,
    organization_descriptors_3d,
)
from collagen_shg.analysis_resolved.orientation_field import multiscale_orientation_3d
from collagen_shg.analysis_resolved.preprocess import preprocess
from collagen_shg.representations import conventions as cv

__all__ = ["AnalysisResult", "ResolvedAnalyzer"]


@dataclass
class AnalysisResult:
    orientation: np.ndarray  # azimuth map [Z, Y, X] in [0, pi)
    coherence: np.ndarray  # fractional anisotropy [Z, Y, X]
    director: np.ndarray  # [3, Z, Y, X]
    descriptors: OrganizationDescriptors
    descriptor_vector: np.ndarray
    ci: dict[str, tuple[float, float]] | None = None

    def measured(self) -> dict[str, Any]:
        """Descriptors as a dict aligned with the phantom ground-truth keys (for ``compare``)."""
        d = self.descriptors
        return {"S2": d.S2, "S3": d.S3, "mean_phi": d.mean_phi, "xi_um": d.xi_um}


class ResolvedAnalyzer:
    """Field-based analyzer: ``Image -> orientation maps + OrganizationDescriptors``."""

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        rhos: Sequence[float] = (1.0, 2.0, 4.0),
        flat_field: bool = True,
        denoise_sigma: float = 0.0,
        subtract_bg: bool = False,
        max_r: int = 16,
        bootstrap: bool = False,
        n_boot: int = 200,
    ) -> None:
        self.sigma = sigma
        self.rhos = tuple(rhos)
        self.flat_field = flat_field
        self.denoise_sigma = denoise_sigma
        self.subtract_bg = subtract_bg
        self.max_r = max_r
        self.bootstrap = bootstrap
        self.n_boot = n_boot

    def analyze(
        self, image: np.ndarray, voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> AnalysisResult:
        vol = np.asarray(image, dtype=np.float64)
        if vol.ndim != 3:
            raise ValueError(f"ResolvedAnalyzer expects a 3D volume [Z, Y, X], got {vol.shape}")
        vol = preprocess(
            vol,
            flat_field=self.flat_field,
            subtract_bg=self.subtract_bg,
            denoise_sigma=self.denoise_sigma,
        )
        field = multiscale_orientation_3d(vol, sigma=self.sigma, rhos=self.rhos)
        desc = organization_descriptors_3d(
            field.director, field.fa, voxel_size_zyx, max_r=self.max_r
        )
        azimuth = cv.wrap_axial(np.arctan2(field.director[1], field.director[0]))
        ci = (
            bootstrap_order_ci(field.director, field.fa, n_boot=self.n_boot)
            if self.bootstrap
            else None
        )
        return AnalysisResult(
            orientation=azimuth,
            coherence=field.fa,
            director=field.director,
            descriptors=desc,
            descriptor_vector=descriptor_vector(desc),
            ci=ci,
        )

    def analyze_bundle(self, bundle: Any) -> AnalysisResult:
        """Analyze an ``ImageBundle`` (uses its voxel size from metadata)."""
        return self.analyze(np.asarray(bundle.image), bundle.metadata.voxel_size_zyx)
