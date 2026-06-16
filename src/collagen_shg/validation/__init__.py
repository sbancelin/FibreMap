"""Closed-loop validation harness (skeleton).

Contract (phase0 Tableau 4): ``(Phantom, analysis output) -> comparison report
(bias/variance)``. In Phase 0 only the skeleton and the end-to-end *null run* are wired (see
``null_run``); the parameter sweeps (SNR / depth / dispersion) arrive with the relevant
Livrables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collagen_shg.representations.phantom import Phantom

__all__ = ["ComparisonReport", "compare"]


@dataclass
class ComparisonReport:
    """Result of comparing measured descriptors against a phantom's ground truth."""

    measured: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    bias: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def compare(phantom: "Phantom", analysis_output: Any) -> ComparisonReport:
    """Compare an analyzer output against the phantom ground truth. Implemented per-Livrable."""
    raise NotImplementedError("Quantitative bias/variance comparison lands with the metrics.")