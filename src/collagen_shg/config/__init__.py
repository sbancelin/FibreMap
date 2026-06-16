"""Typed, YAML-driven configuration and seed management (phase0 §7-8).

Pydantic models fully describe a run; presets (tissues/microscopes) are named, overridable
fragments resolved by :mod:`~collagen_shg.config.loader`. Seeds are derived master → named
children via ``SeedSequence`` / ``PCG64`` for reproducibility.
"""

from __future__ import annotations

from .loader import default_configs_root, load_config, load_config_dict, resolve_presets
from .models import (
    Config,
    DegradationConfig,
    MicroscopeConfig,
    RunMeta,
    StructureConfig,
    VolumeConfig,
)
from .seeds import SeedManager, derive_generator

__all__ = [
    "Config",
    "RunMeta",
    "VolumeConfig",
    "StructureConfig",
    "MicroscopeConfig",
    "DegradationConfig",
    "load_config",
    "load_config_dict",
    "resolve_presets",
    "default_configs_root",
    "SeedManager",
    "derive_generator",
]