"""The ``Phantom`` — the project's single source of ground truth (Pydantic v2 schema).

A phantom carries up to three complementary things (phase0 §4):

1. **geometry** — an explicit list of parametric ``Fibril`` (centerlines in µm, diameter,
   polarity ±1) for resolved tissues and exact training centerlines;
2. **fields** — voxelized continuous fields (``director [3,Z,Y,X]``, ``order_S``, ``density``,
   ``polarity``) for the sub-resolution regime and fast incoherent imaging;
3. **ground_truth** — the known organization metrics (``S2, S3, kappa, xi_um,
   defect_density``, domains, per-region) that make the closed validation loop possible.

Scalars/metadata are validated and (de)serialized by Pydantic; large arrays are attached as
NumPy (validated for shape/dtype) and persisted by :mod:`collagen_shg.representations.io`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from collagen_shg.representations import conventions as cv
from collagen_shg.version import SCHEMA_VERSION, __version__

__all__ = [
    "Units",
    "PhantomMeta",
    "Fibril",
    "DirectorFields",
    "GlobalGT",
    "OrganizationGT",
    "Phantom",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Units(BaseModel):
    """Internal units are fixed: lengths in µm, angles in radians."""

    model_config = ConfigDict(extra="forbid")
    length: Literal["um"] = "um"
    angle: Literal["rad"] = "rad"


class PhantomMeta(BaseModel):
    """Root metadata for a phantom (phase0 PhantomMeta table)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    created: datetime = Field(default_factory=_utcnow)
    code_version: str = __version__
    seed: int | None = None
    tissue_preset: str | None = None
    units: Units = Field(default_factory=Units)
    shape_zyx: tuple[int, int, int]
    voxel_size_zyx: tuple[float, float, float]
    bounds_um: tuple[float, float, float, float, float, float] | None = None

    @field_validator("shape_zyx")
    @classmethod
    def _shape_positive(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(int(s) <= 0 for s in v):
            raise ValueError(f"shape_zyx must be strictly positive, got {v}")
        return v

    @field_validator("voxel_size_zyx")
    @classmethod
    def _voxel_positive(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(float(s) <= 0 for s in v):
            raise ValueError(f"voxel_size_zyx must be strictly positive, got {v}")
        return v

    @model_validator(mode="after")
    def _fill_bounds(self) -> "PhantomMeta":
        if self.bounds_um is None:
            object.__setattr__(
                self, "bounds_um", cv.bounds_um(self.shape_zyx, self.voxel_size_zyx)
            )
        return self


class Fibril(BaseModel):
    """One parametric fibril: a polyline centerline in µm with a diameter profile."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: int
    centerline: np.ndarray  # (N, 3) float64, physical (x, y, z) in µm
    diameter: np.ndarray  # (N,) float64, µm
    polarity: Literal[-1, 1] = 1
    fiber_id: int | None = None
    fascicle_id: int | None = None
    type: str | None = None

    @field_validator("centerline", mode="before")
    @classmethod
    def _coerce_centerline(cls, v: Any) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"centerline must have shape (N, 3), got {a.shape}")
        return np.ascontiguousarray(a)

    @field_validator("diameter", mode="before")
    @classmethod
    def _coerce_diameter(cls, v: Any) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError(f"diameter must have shape (N,), got {a.shape}")
        if np.any(a < 0):
            raise ValueError("diameter values must be non-negative")
        return np.ascontiguousarray(a)

    @model_validator(mode="after")
    def _consistent_length(self) -> "Fibril":
        if self.centerline.shape[0] != self.diameter.shape[0]:
            raise ValueError(
                f"centerline ({self.centerline.shape[0]}) and diameter "
                f"({self.diameter.shape[0]}) lengths differ"
            )
        return self


class DirectorFields(BaseModel):
    """Voxelized continuous fields. ``director`` is ``[3, Z, Y, X]``; the rest ``[Z, Y, X]``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    director: np.ndarray  # [3, Z, Y, X] float32
    order_S: np.ndarray  # [Z, Y, X] float32
    density: np.ndarray  # [Z, Y, X] float32
    polarity: np.ndarray  # [Z, Y, X] float32, net polarity in [-1, 1]

    @field_validator("director", mode="before")
    @classmethod
    def _coerce_director(cls, v: Any) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 4 or a.shape[0] != 3:
            raise ValueError(f"director must have shape (3, Z, Y, X), got {a.shape}")
        return np.ascontiguousarray(a)

    @field_validator("order_S", "density", "polarity", mode="before")
    @classmethod
    def _coerce_scalar_field(cls, v: Any) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 3:
            raise ValueError(f"scalar field must have shape (Z, Y, X), got {a.shape}")
        return np.ascontiguousarray(a)

    @model_validator(mode="after")
    def _consistent_shapes(self) -> "DirectorFields":
        zyx = self.director.shape[1:]
        for name in ("order_S", "density", "polarity"):
            if getattr(self, name).shape != zyx:
                raise ValueError(
                    f"{name} shape {getattr(self, name).shape} != director ZYX {zyx}"
                )
        return self

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.director.shape[1:])  # type: ignore[return-value]

    @classmethod
    def zeros(cls, shape_zyx: tuple[int, int, int]) -> "DirectorFields":
        """Empty fields: zero director, zero order/density/polarity."""
        z, y, x = (int(s) for s in shape_zyx)
        return cls(
            director=np.zeros((3, z, y, x), dtype=np.float32),
            order_S=np.zeros((z, y, x), dtype=np.float32),
            density=np.zeros((z, y, x), dtype=np.float32),
            polarity=np.zeros((z, y, x), dtype=np.float32),
        )


class GlobalGT(BaseModel):
    """Global organization ground truth. Extra (future) descriptors are allowed."""

    model_config = ConfigDict(extra="allow")
    S2: float | None = None
    S3: float | None = None
    kappa: float | None = None
    xi_um: float | None = None
    defect_density: float | None = None


class OrganizationGT(BaseModel):
    """Known organization metrics — the same quantities the metrics module will measure."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    global_: GlobalGT = Field(default_factory=GlobalGT, alias="global")
    domains: list[dict[str, Any]] = Field(default_factory=list)
    per_region: dict[str, Any] = Field(default_factory=dict)


class Phantom(BaseModel):
    """The central ground-truth structure: metadata + geometry + fields + organization GT."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    meta: PhantomMeta
    geometry: list[Fibril] = Field(default_factory=list)
    fields: DirectorFields | None = None
    ground_truth: OrganizationGT = Field(default_factory=OrganizationGT)

    @model_validator(mode="after")
    def _fields_match_meta(self) -> "Phantom":
        if self.fields is not None and self.fields.shape_zyx != self.meta.shape_zyx:
            raise ValueError(
                f"fields shape {self.fields.shape_zyx} != meta.shape_zyx {self.meta.shape_zyx}"
            )
        return self

    @classmethod
    def empty(
        cls,
        shape_zyx: tuple[int, int, int],
        voxel_size_zyx: tuple[float, float, float],
        *,
        seed: int | None = None,
        tissue_preset: str | None = None,
        with_fields: bool = True,
    ) -> "Phantom":
        """An empty phantom: no fibrils, zeroed fields, default (empty) ground truth.

        Used by the Phase 0 *null run* and as a minimal valid construction.
        """
        meta = PhantomMeta(
            shape_zyx=shape_zyx,
            voxel_size_zyx=voxel_size_zyx,
            seed=seed,
            tissue_preset=tissue_preset,
        )
        fields = DirectorFields.zeros(shape_zyx) if with_fields else None
        return cls(meta=meta, geometry=[], fields=fields, ground_truth=OrganizationGT())