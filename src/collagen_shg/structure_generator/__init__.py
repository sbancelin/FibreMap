"""Tier 0 — procedural 3D structure generator (Livrable 2). Interface only.

Contract (phase0 Tableau 4): ``StructureConfig -> Phantom``. Deterministic for a given
``{config, seed}``. The concrete procedural placement of fibrils (centerlines, crimp,
packing, director field) lands in Livrable 2; only the stable interface is fixed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid runtime coupling to other modules
    import numpy as np

    from collagen_shg.config.models import StructureConfig
    from collagen_shg.representations.phantom import Phantom

__all__ = ["StructureGenerator", "ProceduralStructureGenerator", "NullStructureGenerator"]


@runtime_checkable
class StructureGenerator(Protocol):
    """A structure generator turns a typed config + RNG into a ground-truth ``Phantom``."""

    def generate(self, config: StructureConfig, rng: np.random.Generator) -> Phantom: ...


class ProceduralStructureGenerator:
    """Tier 0 deterministic generator. Implemented in Livrable 2."""

    def generate(self, config: StructureConfig, rng: np.random.Generator) -> Phantom:
        raise NotImplementedError("Tier 0 procedural structure generation lands in Livrable 2.")


class NullStructureGenerator:
    """Trivial generator producing an **empty** phantom — used by the Phase 0 null run.

    It conforms to the :class:`StructureGenerator` interface so the null run exercises the
    real contract; it holds the volume geometry (shape + voxel size) the empty phantom needs.
    """

    def __init__(
        self,
        shape_zyx: tuple[int, int, int],
        voxel_size_zyx: tuple[float, float, float],
    ) -> None:
        self.shape_zyx = tuple(int(s) for s in shape_zyx)
        self.voxel_size_zyx = tuple(float(s) for s in voxel_size_zyx)

    def generate(self, config: StructureConfig, rng: np.random.Generator) -> Phantom:
        from collagen_shg.representations.phantom import Phantom

        tissue = getattr(config, "preset", None)
        return Phantom.empty(self.shape_zyx, self.voxel_size_zyx, tissue_preset=tissue)