"""Closed-loop validation harness.

Contract (phase0 Tableau 4): ``(Phantom, analysis output) -> comparison report
(bias/variance)``. The closed loop (generate → image → analyze → compare) is in
:mod:`.closed_loop`; the Phase 0 end-to-end *null run* (interface wiring) is in :mod:`.null_run`.
The parameter sweeps (SNR / depth / dispersion) build on ``run_closed_loop``.
"""

from __future__ import annotations

from .closed_loop import (
    ClosedLoopReport,
    ComparisonReport,
    analyze_image_3d,
    compare,
    run_closed_loop,
)
from .null_run import run_null_pipeline

__all__ = [
    "ComparisonReport",
    "ClosedLoopReport",
    "compare",
    "analyze_image_3d",
    "run_closed_loop",
    "run_null_pipeline",
]
