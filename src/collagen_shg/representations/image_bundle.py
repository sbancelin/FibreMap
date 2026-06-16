"""The ``ImageBundle`` — an imaged volume plus its metadata and (optional) ground truth.

Real microscope images and synthetic volumes flow through the **same** structure (phase0 §5):
a real OME-TIFF becomes an ``ImageBundle`` with ``kind="real"`` and empty/partial ground
truth, while a synthetic volume carries its source ``Phantom`` as ground truth. The metadata
schema mirrors the phase0 ``metadata.json`` table (microscope / acquisition / provenance).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from collagen_shg.representations.phantom import Phantom
from collagen_shg.version import SCHEMA_VERSION, __version__

__all__ = [
    "MicroscopeMeta",
    "AcquisitionMeta",
    "Provenance",
    "BundleMetadata",
    "ImageBundle",
]


class MicroscopeMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["incoherent", "coherent"] = "incoherent"
    NA: float | None = None
    wavelength_nm: float | None = None
    psf_model: str | None = None  # e.g. "gaussian" / "richards-wolf"
    detection: Literal["backward", "forward"] = "backward"
    pixel_size_um: float | None = None


class AcquisitionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    polarization_angles_rad: list[float] = Field(default_factory=list)
    bit_depth: int | None = None


class Provenance(BaseModel):
    """Everything needed to regenerate the artifact: {config + seed + code version}."""

    model_config = ConfigDict(extra="allow")
    seed: int | None = None
    code_version: str = __version__
    library_versions: dict[str, str] = Field(default_factory=dict)
    rng: str = "PCG64"


class BundleMetadata(BaseModel):
    """Structured sidecar metadata (the phase0 ``metadata.json`` schema)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    kind: Literal["synthetic", "real"] = "synthetic"
    shape_zyx: tuple[int, int, int]
    voxel_size_zyx: tuple[float, float, float]
    microscope: MicroscopeMeta = Field(default_factory=MicroscopeMeta)
    acquisition: AcquisitionMeta = Field(default_factory=AcquisitionMeta)
    tissue: dict[str, Any] | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class ImageBundle(BaseModel):
    """An imaged volume (``[Z,Y,X]`` or ``[C,Z,Y,X]``) with metadata and optional ground truth."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    image: np.ndarray
    metadata: BundleMetadata
    phantom: Phantom | None = None  # ground-truth source for synthetic bundles
    config: dict[str, Any] | None = None  # config.yaml snapshot

    @field_validator("image", mode="before")
    @classmethod
    def _coerce_image(cls, v: Any) -> np.ndarray:
        a = np.asarray(v)
        if a.ndim not in (3, 4):
            raise ValueError(f"image must be [Z,Y,X] or [C,Z,Y,X], got ndim={a.ndim}")
        return np.ascontiguousarray(a)

    @model_validator(mode="after")
    def _shape_consistency(self) -> "ImageBundle":
        spatial = tuple(int(s) for s in self.image.shape[-3:])
        if spatial != self.metadata.shape_zyx:
            raise ValueError(
                f"image spatial shape {spatial} != metadata.shape_zyx {self.metadata.shape_zyx}"
            )
        if self.phantom is not None and self.phantom.meta.shape_zyx != self.metadata.shape_zyx:
            raise ValueError(
                f"phantom shape {self.phantom.meta.shape_zyx} != "
                f"metadata.shape_zyx {self.metadata.shape_zyx}"
            )
        return self

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.image.shape[-3:])  # type: ignore[return-value]

    @classmethod
    def white(
        cls,
        shape_zyx: tuple[int, int, int],
        voxel_size_zyx: tuple[float, float, float],
        *,
        fill: float = 1.0,
        dtype: Any = np.float32,
        phantom: Phantom | None = None,
        metadata: BundleMetadata | None = None,
    ) -> "ImageBundle":
        """A uniform ("white") image bundle — the Phase 0 null-run image.

        ``fill`` is the constant intensity (default ``1.0`` for float). Pass a ``phantom`` to
        carry ground truth through, and/or a ready ``metadata``; otherwise minimal synthetic
        metadata is generated.
        """
        z, y, x = (int(s) for s in shape_zyx)
        image = np.full((z, y, x), fill, dtype=dtype)
        if metadata is None:
            metadata = BundleMetadata(
                kind="synthetic", shape_zyx=shape_zyx, voxel_size_zyx=voxel_size_zyx
            )
        return cls(image=image, metadata=metadata, phantom=phantom)