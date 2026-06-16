"""collagen-shg — collagen organization quantification for SHG microscopy.

Phase 0 exposes the shared infrastructure: representations (``Phantom``,
``ImageBundle``, conventions, I/O), typed configuration, and seed management.
Computational modules are present as stubs with stable interfaces.
"""

from __future__ import annotations

from .version import SCHEMA_VERSION, __version__

__all__ = ["__version__", "SCHEMA_VERSION"]