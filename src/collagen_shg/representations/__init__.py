"""Shared representations: ground-truth ``Phantom``, ``ImageBundle``, conventions, and I/O.

This is the project's foundation (phase0 Tableau 4): it depends on no generation or analysis
algorithm. Submodules are wired up incrementally across Phase 0 commits.
"""

from __future__ import annotations

from . import conventions

__all__ = ["conventions"]