"""Resolved-tissue image analysis (Livrable 3) — interface only.

Contract (phase0 Tableau 4): ``Image -> maps + OrganizationDescriptors``. Covers skin/dermis,
tendon, vocal fold, bone (resolved or partially resolved). Pipeline (orientation field →
order parameters / ξ / defects → inter-tissue comparison) lands in Livrable 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Analyzer", "ResolvedAnalyzer", "TrivialAnalyzer"]


@runtime_checkable
class Analyzer(Protocol):
    def analyze(self, image: np.ndarray) -> Any: ...


class ResolvedAnalyzer:
    """Field + learned-extractor analysis for resolved tissues. Implemented in Livrable 3."""

    def analyze(self, image):  # noqa: ANN001
        raise NotImplementedError("Resolved-tissue analysis lands in Livrable 3.")


class TrivialAnalyzer:
    """Trivial analyzer returning basic intensity statistics — used by the Phase 0 null run.

    Conforms to :class:`Analyzer`; it does no organization analysis (that lands in Livrable 3),
    only enough to prove the generator → imaging → analysis interfaces wire together.
    """

    def analyze(self, image):  # noqa: ANN001
        import numpy as np

        a = np.asarray(image)
        return {
            "shape": tuple(int(s) for s in a.shape),
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max()),
        }