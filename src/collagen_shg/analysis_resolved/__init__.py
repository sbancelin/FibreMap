"""Resolved-tissue image analysis (Livrable 3).

Contract (phase0 Tableau 4): ``Image -> maps + OrganizationDescriptors``. Covers skin/dermis,
tendon, vocal fold, bone (resolved or partially resolved). The field-based chain (preprocess →
multi-scale orientation field → order parameters / ξ / defects → inter-tissue comparison) is
implemented in :class:`ResolvedAnalyzer`. The learned extractor (trained on synthetic
centerlines, surpassing CT-FIRE in dense regions) is a separate, training-heavy component
(:class:`LearnedExtractor`, stub for now).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .analyzer import AnalysisResult, ResolvedAnalyzer
from .compare_tissues import PCA, NearestCentroidClassifier, feature_matrix, standardize
from .descriptors import (
    OrganizationDescriptors,
    bootstrap_order_ci,
    descriptor_vector,
    organization_descriptors_3d,
)
from .orientation_field import multiscale_orientation_3d, orientation_field_2d
from .preprocess import preprocess

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "Analyzer",
    "ResolvedAnalyzer",
    "AnalysisResult",
    "LearnedExtractor",
    "TrivialAnalyzer",
    # pipeline pieces
    "preprocess",
    "multiscale_orientation_3d",
    "orientation_field_2d",
    "organization_descriptors_3d",
    "descriptor_vector",
    "bootstrap_order_ci",
    "OrganizationDescriptors",
    # inter-tissue comparison
    "PCA",
    "NearestCentroidClassifier",
    "feature_matrix",
    "standardize",
]


@runtime_checkable
class Analyzer(Protocol):
    def analyze(self, image: np.ndarray) -> Any: ...


class LearnedExtractor:
    """Deep fibre extractor trained on synthetic ground truth (centerlines). Later component.

    Surpasses CT-FIRE in dense/crossing regions; requires a training pipeline + GPU, so it is a
    documented stub here. Field-based organization analysis (:class:`ResolvedAnalyzer`) does not
    depend on it.
    """

    def analyze(self, image):  # noqa: ANN001
        raise NotImplementedError(
            "The learned fibre extractor (trained on synthetic centerlines) is a later component."
        )


class TrivialAnalyzer:
    """Trivial analyzer returning basic intensity statistics — used by the Phase 0 null run."""

    def analyze(self, image):  # noqa: ANN001
        import numpy as np

        a = np.asarray(image)
        return {
            "shape": tuple(int(s) for s in a.shape),
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max()),
        }
