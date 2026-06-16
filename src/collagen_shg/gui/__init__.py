"""Napari-based GUI.

Built incrementally alongside the project (Génération / Analyse tabs grow with each Livrable).
Phase 0 ships a minimal launchable shell that opens and displays a bundle (see ``app``).
The napari dependency is optional (``pip install -e ".[gui]"``); importing this package does
not require napari until the shell is actually launched.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``collagen-shg-gui`` console script (lazy import of napari)."""
    from .app import main as _main

    return _main(argv)