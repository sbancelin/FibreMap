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

__all__ = ["StructureGenerator", "ProceduralStructureGenerator"]


@runtime_checkable
class StructureGenerator(Protocol):
    """A structure generator turns a typed config + RNG into a ground-truth ``Phantom``."""

    def generate(self, config: "StructureConfig", rng: "np.random.Generator") -> "Phantom": ...


class ProceduralStructureGenerator:
    """Tier 0 deterministic generator. Implemented in Livrable 2."""

    def generate(self, config: "StructureConfig", rng: "np.random.Generator") -> "Phantom":
        raise NotImplementedError("Tier 0 procedural structure generation lands in Livrable 2.")