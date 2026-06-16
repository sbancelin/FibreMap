"""Typed, YAML-driven configuration and seed management.

Pydantic models fully describe a run; presets (tissues/microscopes) are named, overridable
fragments. Seeds are derived master → named children via ``SeedSequence`` / ``PCG64`` for
reproducibility. Submodules are wired up in the Phase 0 config commit.
"""

from __future__ import annotations

__all__: list[str] = []