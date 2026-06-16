"""YAML loading and preset resolution for run configurations (phase0 §7).

The on-disk run YAML references named presets and overrides them::

    structure:  { preset: tendon,  overrides: { ... } }
    microscope: { preset: default, overrides: { ... } }

:func:`load_config` reads the YAML, resolves each preset (tissue / microscope) by deep-merging
its ``overrides`` on top of the preset fragment, and validates the result into a typed
:class:`~collagen_shg.config.models.Config`. The original presets live under ``configs/``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from collagen_shg.config.models import Config

__all__ = ["load_config", "load_config_dict", "resolve_presets", "default_configs_root"]


def default_configs_root() -> Path:
    """Repository ``configs/`` directory (resolved relative to the installed package)."""
    # src/collagen_shg/config/loader.py -> repo root is three parents up from the package.
    return Path(__file__).resolve().parents[3] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base`` (dicts merged, others replaced)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping, got {type(data).__name__}")
    return data


def _resolve_section(
    section: dict[str, Any] | None, preset_dir: Path
) -> dict[str, Any] | None:
    """Resolve one ``{preset, overrides}`` section against ``preset_dir``."""
    if section is None:
        return None
    section = copy.deepcopy(section)
    preset = section.pop("preset", None)
    overrides = section.pop("overrides", {}) or {}
    if preset is not None:
        preset_path = preset_dir / f"{preset}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"preset '{preset}' not found at {preset_path}")
        base = _load_yaml(preset_path)
        base.pop("name", None)  # preset name field is informational only
        merged = _deep_merge(base, overrides)
        merged["preset"] = preset
    else:
        merged = _deep_merge(section, overrides)
    return merged


def resolve_presets(
    raw: dict[str, Any], configs_root: Path | None = None
) -> dict[str, Any]:
    """Resolve tissue/microscope presets in a raw run-config dict, returning a flat dict."""
    configs_root = configs_root or default_configs_root()
    out = copy.deepcopy(raw)
    if "structure" in out:
        out["structure"] = _resolve_section(out["structure"], configs_root / "tissues")
    if "microscope" in out:
        out["microscope"] = _resolve_section(out["microscope"], configs_root / "microscopes")
    return out


def load_config_dict(raw: dict[str, Any], configs_root: Path | None = None) -> Config:
    """Validate an already-parsed run-config dict (resolving presets) into a ``Config``."""
    return Config.model_validate(resolve_presets(raw, configs_root))


def load_config(path: str | Path, configs_root: Path | None = None) -> Config:
    """Load + resolve + validate a run-config YAML file into a typed ``Config``."""
    path = Path(path)
    raw = _load_yaml(path)
    return load_config_dict(raw, configs_root)