"""Resolved-tissue image analysis (Livrable 3) — interface only.

Contract (phase0 Tableau 4): ``Image -> maps + OrganizationDescriptors``. Covers skin/dermis,
tendon, vocal fold, bone (resolved or partially resolved). Pipeline (orientation field →
order parameters / ξ / defects → inter-tissue comparison) lands in Livrable 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Analyzer", "ResolvedAnalyzer"]


@runtime_checkable
class Analyzer(Protocol):
    def analyze(self, image: "np.ndarray") -> Any: ...


class ResolvedAnalyzer:
    """Field + learned-extractor analysis for resolved tissues. Implemented in Livrable 3."""

    def analyze(self, image):  # noqa: ANN001
        raise NotImplementedError("Resolved-tissue analysis lands in Livrable 3.")