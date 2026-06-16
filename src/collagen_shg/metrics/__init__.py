"""Organization metrics (Livrable 1) — families A–G.

Implements the Livrable 1 contract (Tableau 3) on top of the Phase 0 conventions and types.
The functions below are the stable public surface; richer structured results live on the
returned dataclasses. The comparison/scoring harness (coupling the generator to these metrics,
the "metrics of metrics") arrives with Livrable 2 once the generator exists.

Families
--------
- A — structure tensor (orientation + coherence/FA): :mod:`.structure_tensor`
- B — order parameters & circular statistics (S2/S3, κ, Saupe Q): :mod:`.order`
- C — orientation correlation & correlation length ξ: :mod:`.correlation`
- D — Fourier / power spectrum (orientation + spacing): :mod:`.fourier`
- E — texture (GLCM / LBP / Gabor): :mod:`.texture`
- F — per-fibre descriptors & persistence length: :mod:`.fibers`
- G — topological defect density: :mod:`.defects`
"""

from __future__ import annotations

from .correlation import OrientationCorrelation, orientation_correlation
from .defects import DefectResult, defect_density
from .fibers import (
    FiberMetricsResult,
    FiberNetwork,
    FiberRecord,
    fiber_metrics,
    persistence_length,
)
from .fourier import PowerSpectrumResult, power_spectrum_orientation
from .order import (
    OrderParameter2D,
    OrderTensor3D,
    order_parameter_2d,
    order_tensor_3d,
    vonmises_kappa_from_R,
)
from .structure_tensor import (
    StructureTensor2DResult,
    StructureTensor3DResult,
    structure_tensor_2d,
    structure_tensor_3d,
)
from .texture import GaborEnergy, gabor_energy, glcm_features, lbp_histogram

__all__ = [
    # Family A
    "structure_tensor_2d",
    "structure_tensor_3d",
    "StructureTensor2DResult",
    "StructureTensor3DResult",
    # Family B
    "order_parameter_2d",
    "order_tensor_3d",
    "vonmises_kappa_from_R",
    "OrderParameter2D",
    "OrderTensor3D",
    # Family C
    "orientation_correlation",
    "OrientationCorrelation",
    # Family D
    "power_spectrum_orientation",
    "PowerSpectrumResult",
    # Family E
    "glcm_features",
    "lbp_histogram",
    "gabor_energy",
    "GaborEnergy",
    # Family F
    "fiber_metrics",
    "persistence_length",
    "FiberRecord",
    "FiberNetwork",
    "FiberMetricsResult",
    # Family G
    "defect_density",
    "DefectResult",
]
