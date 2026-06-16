"""Image formation — microscope model (Livrable 2). Interface only.

Contract (phase0 Tableau 4): ``Phantom + MicroscopeConfig + DegradationConfig -> ImageBundle``
(image plus ground-truth passthrough). Two fidelities: ``incoherent`` (Tier 1, projected
intensity ⊛ PSF + depth degradation + Poisson noise) and ``coherent`` (Tier 3, complex-field
summation for SHG / P-SHG). Implemented in Livrable 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .incoherent import IncoherentImager, psf_sigma_um

if TYPE_CHECKING:
    import numpy as np

    from collagen_shg.config.models import DegradationConfig, MicroscopeConfig
    from collagen_shg.representations.image_bundle import ImageBundle
    from collagen_shg.representations.phantom import Phantom

__all__ = ["Imager", "IncoherentImager", "CoherentImager", "NullImager", "psf_sigma_um"]


@runtime_checkable
class Imager(Protocol):
    """Renders a ``Phantom`` into an ``ImageBundle`` under a microscope/degradation model."""

    def render(
        self,
        phantom: Phantom,
        microscope: MicroscopeConfig,
        degradation: DegradationConfig,
        rng: np.random.Generator,
    ) -> ImageBundle: ...


class CoherentImager:
    """Tier 3 coherent SHG imaging. Implemented in the coherent-imaging commit."""

    def render(self, phantom, microscope, degradation, rng):  # noqa: ANN001
        raise NotImplementedError("Tier 3 coherent SHG imaging lands in the next commit.")


class NullImager:
    """Trivial imager producing a uniform ("white") image — used by the Phase 0 null run.

    Conforms to :class:`Imager`, carries the microscope parameters into the bundle metadata,
    and passes the source phantom through as ground truth.
    """

    def __init__(self, fill: float = 1.0) -> None:
        self.fill = float(fill)

    def render(self, phantom, microscope, degradation, rng):  # noqa: ANN001
        import numpy as np

        from collagen_shg.representations.image_bundle import (
            BundleMetadata,
            ImageBundle,
            MicroscopeMeta,
        )

        shape = phantom.meta.shape_zyx
        voxel = phantom.meta.voxel_size_zyx
        metadata = BundleMetadata(
            kind="synthetic",
            shape_zyx=shape,
            voxel_size_zyx=voxel,
            microscope=MicroscopeMeta(
                mode=getattr(microscope, "mode", "incoherent"),
                NA=getattr(microscope, "NA", None),
                wavelength_nm=getattr(microscope, "wavelength_nm", None),
                detection=getattr(microscope, "detection", "backward"),
                pixel_size_um=getattr(microscope, "pixel_size_um", None),
                psf_model=getattr(microscope, "psf_model", None),
            ),
        )
        image = np.full(shape, self.fill, dtype=np.float32)
        return ImageBundle(image=image, metadata=metadata, phantom=phantom)