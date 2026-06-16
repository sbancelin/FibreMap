"""P-SHG analysis for unresolved tissues — cornea, cartilage (Livrable 4). Interface only.

Contract (phase0 Tableau 4): ``PolarizationStack + PSHGConfig -> maps (phi, rho); sub-resolution
order``. Separable module: it shares only the sub-voxel emitter representation (orientation +
polarity) with the rest. Implemented in Livrable 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = ["PSHGAnalyzer", "LinearPSHGAnalyzer"]


@runtime_checkable
class PSHGAnalyzer(Protocol):
    def analyze(self, stack: "np.ndarray", angles_rad: "np.ndarray") -> Any: ...


class LinearPSHGAnalyzer:
    """Pixelwise least-squares baseline (phi, rho). Implemented in Livrable 4."""

    def analyze(self, stack, angles_rad):  # noqa: ANN001
        raise NotImplementedError("P-SHG analysis lands in Livrable 4.")