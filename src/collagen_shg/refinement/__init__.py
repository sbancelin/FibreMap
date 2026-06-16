"""Tier 2 — learned realistic refinement (Phase 3). Interface only.

Contract (phase0 Tableau 4): ``ImageBundle (Tier 1) -> ImageBundle (realistic)``. An
image-to-image translation (conditional diffusion preferred) that adds last-mile realism
while preserving the ground truth carried by the input. Not part of Phase 0; GPU deps
(PyTorch/CuPy) are intentionally excluded here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collagen_shg.representations.image_bundle import ImageBundle

__all__ = ["Refiner", "LearnedRefiner"]


@runtime_checkable
class Refiner(Protocol):
    def refine(self, bundle: ImageBundle) -> ImageBundle: ...


class LearnedRefiner:
    """Conditional image-to-image refinement. Implemented in Phase 3."""

    def refine(self, bundle):  # noqa: ANN001
        raise NotImplementedError("Tier 2 learned refinement lands in Phase 3.")