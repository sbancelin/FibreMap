"""Image formation — microscope model (Livrable 2). Interface only.

Contract (phase0 Tableau 4): ``Phantom + MicroscopeConfig + DegradationConfig -> ImageBundle``
(image plus ground-truth passthrough). Two fidelities: ``incoherent`` (Tier 1, projected
intensity ⊛ PSF + depth degradation + Poisson noise) and ``coherent`` (Tier 3, complex-field
summation for SHG / P-SHG). Implemented in Livrable 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from collagen_shg.config.models import DegradationConfig, MicroscopeConfig
    from collagen_shg.representations.image_bundle import ImageBundle
    from collagen_shg.representations.phantom import Phantom

__all__ = ["Imager", "IncoherentImager", "CoherentImager"]


@runtime_checkable
class Imager(Protocol):
    """Renders a ``Phantom`` into an ``ImageBundle`` under a microscope/degradation model."""

    def render(
        self,
        phantom: "Phantom",
        microscope: "MicroscopeConfig",
        degradation: "DegradationConfig",
        rng: "np.random.Generator",
    ) -> "ImageBundle": ...


class IncoherentImager:
    """Tier 1 fast incoherent imaging. Implemented in Livrable 2."""

    def render(self, phantom, microscope, degradation, rng):  # noqa: ANN001
        raise NotImplementedError("Tier 1 incoherent imaging lands in Livrable 2.")


class CoherentImager:
    """Tier 3 coherent SHG imaging. Implemented in Livrable 2/4."""

    def render(self, phantom, microscope, degradation, rng):  # noqa: ANN001
        raise NotImplementedError("Tier 3 coherent SHG imaging lands in Livrable 2/4.")