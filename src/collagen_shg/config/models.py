"""Typed configuration models (phase0 §7).

A run is fully described by typed, validated Pydantic models. On disk it is human-readable
YAML; in memory it is these objects. The configuration is hierarchical — ``run``, ``volume``,
``structure``, ``microscope``, ``degradation`` — and tissue/microscope archetypes are named,
overridable fragments (*presets*) resolved by :mod:`collagen_shg.config.loader`.

Degrees may appear at this YAML/GUI boundary (e.g. ``mean_phi_deg``); everything internal is
radians/µm per the conventions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "RunMeta",
    "VolumeConfig",
    "OrientationParams",
    "DiameterParams",
    "CrimpParams",
    "FibrilParams",
    "StructureConfig",
    "MicroscopeConfig",
    "DepthParams",
    "NoiseParams",
    "DegradationConfig",
    "Config",
]


class _Base(BaseModel):
    # Extra keys are allowed so presets can carry forward-compatible fields without breaking.
    model_config = ConfigDict(extra="allow")


class RunMeta(_Base):
    name: str
    seed: int
    output: str | None = None


class VolumeConfig(_Base):
    shape_zyx: tuple[int, int, int]
    voxel_size_zyx_um: tuple[float, float, float]

    @field_validator("shape_zyx")
    @classmethod
    def _shape_positive(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(int(s) <= 0 for s in v):
            raise ValueError(f"shape_zyx must be strictly positive, got {v}")
        return v

    @field_validator("voxel_size_zyx_um")
    @classmethod
    def _voxel_positive(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(float(s) <= 0 for s in v):
            raise ValueError(f"voxel_size_zyx_um must be strictly positive, got {v}")
        return v


class OrientationParams(_Base):
    mean_phi_deg: float | None = None  # GUI/YAML boundary: degrees
    kappa: float | None = None
    xi_um: float | None = None


class DiameterParams(_Base):
    mean: float | None = None
    dispersion: float | None = None


class CrimpParams(_Base):
    amplitude_um: float | None = None
    period_um: float | None = None


class FibrilParams(_Base):
    diameter_um: DiameterParams = Field(default_factory=DiameterParams)
    crimp: CrimpParams = Field(default_factory=CrimpParams)


class StructureConfig(_Base):
    """Tier 0 structure parameters (resolved from a tissue preset + overrides)."""

    preset: str | None = None
    orientation: OrientationParams = Field(default_factory=OrientationParams)
    fibril: FibrilParams = Field(default_factory=FibrilParams)


class MicroscopeConfig(_Base):
    """Microscope model parameters (resolved from a microscope preset + overrides)."""

    preset: str | None = None
    mode: str = "incoherent"  # incoherent | coherent
    NA: float | None = None
    wavelength_nm: float | None = None
    detection: str = "backward"  # backward | forward
    pixel_size_um: float | None = None
    psf_model: str | None = None


class DepthParams(_Base):
    attenuation_length_um: float | None = None


class NoiseParams(_Base):
    photons_peak: float | None = None
    read_noise_e: float | None = None


class DegradationConfig(_Base):
    depth: DepthParams = Field(default_factory=DepthParams)
    noise: NoiseParams = Field(default_factory=NoiseParams)


class Config(_Base):
    """The full, validated run configuration. Snapshotted into every bundle."""

    run: RunMeta
    volume: VolumeConfig
    structure: StructureConfig = Field(default_factory=StructureConfig)
    microscope: MicroscopeConfig = Field(default_factory=MicroscopeConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)