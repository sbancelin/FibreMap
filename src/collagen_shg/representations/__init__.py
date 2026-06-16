"""Shared representations: ground-truth ``Phantom``, ``ImageBundle``, conventions, and I/O.

This is the project's foundation (phase0 Tableau 4): it depends on no generation or analysis
algorithm.
"""

from __future__ import annotations

from . import conventions
from .image_bundle import (
    AcquisitionMeta,
    BundleMetadata,
    ImageBundle,
    MicroscopeMeta,
    Provenance,
)
from .phantom import (
    DirectorFields,
    Fibril,
    GlobalGT,
    OrganizationGT,
    Phantom,
    PhantomMeta,
    Units,
)

__all__ = [
    "conventions",
    # phantom
    "Phantom",
    "PhantomMeta",
    "Fibril",
    "DirectorFields",
    "GlobalGT",
    "OrganizationGT",
    "Units",
    # image bundle
    "ImageBundle",
    "BundleMetadata",
    "MicroscopeMeta",
    "AcquisitionMeta",
    "Provenance",
]