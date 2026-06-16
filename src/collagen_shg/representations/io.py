"""Reproducible *bundle* I/O — the canonical on-disk dataset format (phase0 §5).

A bundle is one reproducible unit (synthetic or real)::

    dataset.bundle/
    ├─ image.zarr/              # imaged volume (Zarr; axis metadata in attrs)
    ├─ ground_truth/
    │    ├─ fields.zarr/        # director, order_S, density, polarity
    │    ├─ geometry.parquet    # fibril list (centerlines, diameters, ...)
    │    ├─ phantom_meta.json   # PhantomMeta (so the phantom round-trips)
    │    └─ organization.json   # OrganizationGT (known organization metrics)
    ├─ metadata.json            # BundleMetadata (microscope/acquisition/provenance)
    ├─ config.yaml              # run config snapshot (optional)
    └─ provenance.json          # seed, versions, rng (mirror of metadata.provenance)

Round-trip is **lossless**: arrays come back bit-for-bit identical and metadata compares
equal. Real OME-TIFF images enter through :func:`read_ome_tiff` into the *same* structure
(``kind="real"``, empty/partial ground truth).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
import yaml
import zarr

from collagen_shg.representations.image_bundle import BundleMetadata, ImageBundle
from collagen_shg.representations.phantom import (
    DirectorFields,
    Fibril,
    OrganizationGT,
    Phantom,
    PhantomMeta,
)

__all__ = ["write_bundle", "read_bundle", "read_ome_tiff", "write_ome_tiff"]

# Canonical filenames within a bundle.
_IMAGE = "image.zarr"
_GT_DIR = "ground_truth"
_FIELDS = "fields.zarr"
_GEOMETRY = "geometry.parquet"
_PHANTOM_META = "phantom_meta.json"
_ORGANIZATION = "organization.json"
_METADATA = "metadata.json"
_CONFIG = "config.yaml"
_PROVENANCE = "provenance.json"

_AXES = ["z", "y", "x"]

_GEOMETRY_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("cx", pa.list_(pa.float64())),
        ("cy", pa.list_(pa.float64())),
        ("cz", pa.list_(pa.float64())),
        ("diameter", pa.list_(pa.float64())),
        ("polarity", pa.int64()),
        ("fiber_id", pa.int64()),
        ("fascicle_id", pa.int64()),
        ("type", pa.string()),
    ]
)


# --------------------------------------------------------------------------------- zarr helpers
def _write_zarr_array(path: Path, array: np.ndarray, attrs: dict[str, Any] | None = None) -> None:
    array = np.ascontiguousarray(array)
    store = zarr.storage.LocalStore(str(path))
    z = zarr.create_array(store=store, shape=array.shape, dtype=array.dtype)
    z[...] = array
    if attrs:
        for k, v in attrs.items():
            z.attrs[k] = v


def _read_zarr_array(path: Path) -> np.ndarray:
    store = zarr.storage.LocalStore(str(path))
    z = zarr.open_array(store=store, mode="r")
    return z[...]


# ---------------------------------------------------------------------------------- fields (zarr)
def _write_fields(path: Path, fields: DirectorFields) -> None:
    store = zarr.storage.LocalStore(str(path))
    g = zarr.open_group(store=store, mode="w")
    g.attrs["axes"] = ["c", *_AXES]
    for name, arr in (
        ("director", fields.director),
        ("order_S", fields.order_S),
        ("density", fields.density),
        ("polarity", fields.polarity),
    ):
        arr = np.ascontiguousarray(arr)
        a = g.create_array(name, shape=arr.shape, dtype=arr.dtype)
        a[...] = arr


def _read_fields(path: Path) -> DirectorFields:
    store = zarr.storage.LocalStore(str(path))
    g = zarr.open_group(store=store, mode="r")
    return DirectorFields(
        director=g["director"][...],
        order_S=g["order_S"][...],
        density=g["density"][...],
        polarity=g["polarity"][...],
    )


# ------------------------------------------------------------------------------ geometry (parquet)
def _write_geometry(path: Path, fibrils: list[Fibril]) -> None:
    rows: dict[str, list[Any]] = {k: [] for k in _GEOMETRY_SCHEMA.names}
    for f in fibrils:
        cl = np.asarray(f.centerline, dtype=np.float64)
        rows["id"].append(int(f.id))
        rows["cx"].append(cl[:, 0].tolist())
        rows["cy"].append(cl[:, 1].tolist())
        rows["cz"].append(cl[:, 2].tolist())
        rows["diameter"].append(np.asarray(f.diameter, dtype=np.float64).tolist())
        rows["polarity"].append(int(f.polarity))
        rows["fiber_id"].append(f.fiber_id)
        rows["fascicle_id"].append(f.fascicle_id)
        rows["type"].append(f.type)
    table = pa.table(rows, schema=_GEOMETRY_SCHEMA)
    pq.write_table(table, str(path))


def _read_geometry(path: Path) -> list[Fibril]:
    table = pq.read_table(str(path))
    d = table.to_pydict()
    fibrils: list[Fibril] = []
    for i in range(table.num_rows):
        centerline = np.stack([d["cx"][i], d["cy"][i], d["cz"][i]], axis=-1)
        fibrils.append(
            Fibril(
                id=d["id"][i],
                centerline=centerline,
                diameter=np.asarray(d["diameter"][i], dtype=np.float64),
                polarity=d["polarity"][i],
                fiber_id=d["fiber_id"][i],
                fascicle_id=d["fascicle_id"][i],
                type=d["type"][i],
            )
        )
    return fibrils


# ------------------------------------------------------------------------------------ json helpers
def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------- bundle IO
def write_bundle(bundle: ImageBundle, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write an ``ImageBundle`` to a bundle directory at ``path``. Returns the bundle path."""
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)

    _write_zarr_array(path / _IMAGE, bundle.image, attrs={"axes": _image_axes(bundle.image.ndim)})

    _write_json(path / _METADATA, bundle.metadata.model_dump(mode="json"))
    _write_json(path / _PROVENANCE, bundle.metadata.provenance.model_dump(mode="json"))

    if bundle.config is not None:
        (path / _CONFIG).write_text(
            yaml.safe_dump(bundle.config, sort_keys=False), encoding="utf-8"
        )

    if bundle.phantom is not None:
        gt_dir = path / _GT_DIR
        gt_dir.mkdir()
        p = bundle.phantom
        _write_json(gt_dir / _PHANTOM_META, p.meta.model_dump(mode="json"))
        _write_json(gt_dir / _ORGANIZATION, p.ground_truth.model_dump(mode="json", by_alias=True))
        _write_geometry(gt_dir / _GEOMETRY, p.geometry)
        if p.fields is not None:
            _write_fields(gt_dir / _FIELDS, p.fields)

    return path


def read_bundle(path: str | Path) -> ImageBundle:
    """Read a bundle directory back into an ``ImageBundle`` (lossless inverse of write)."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {path}")

    image = _read_zarr_array(path / _IMAGE)
    metadata = BundleMetadata.model_validate(_read_json(path / _METADATA))

    config = None
    if (path / _CONFIG).exists():
        config = yaml.safe_load((path / _CONFIG).read_text(encoding="utf-8"))

    phantom = None
    gt_dir = path / _GT_DIR
    if gt_dir.is_dir():
        meta = PhantomMeta.model_validate(_read_json(gt_dir / _PHANTOM_META))
        organization = OrganizationGT.model_validate(_read_json(gt_dir / _ORGANIZATION))
        geometry = _read_geometry(gt_dir / _GEOMETRY) if (gt_dir / _GEOMETRY).exists() else []
        fields = _read_fields(gt_dir / _FIELDS) if (gt_dir / _FIELDS).exists() else None
        phantom = Phantom(
            meta=meta, geometry=geometry, fields=fields, ground_truth=organization
        )

    return ImageBundle(image=image, metadata=metadata, phantom=phantom, config=config)


def _image_axes(ndim: int) -> list[str]:
    return _AXES if ndim == 3 else ["c", *_AXES]


# ----------------------------------------------------------------------------- OME-TIFF interchange
def write_ome_tiff(bundle: ImageBundle, path: str | Path) -> Path:
    """Export a bundle's image as an OME-TIFF (interchange / Fiji), writing physical sizes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dz, dy, dx = bundle.metadata.voxel_size_zyx
    tifffile.imwrite(
        str(path),
        bundle.image,
        photometric="minisblack",
        metadata={"axes": "ZYX" if bundle.image.ndim == 3 else "CZYX",
                  "PhysicalSizeZ": dz, "PhysicalSizeY": dy, "PhysicalSizeX": dx,
                  "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
                  "PhysicalSizeZUnit": "µm"},
    )
    return path


def read_ome_tiff(
    path: str | Path,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    *,
    tissue: dict[str, Any] | None = None,
) -> ImageBundle:
    """Ingest a real OME-TIFF into an ``ImageBundle`` (``kind="real"``, no ground truth).

    Voxel size is taken from the argument when given, else parsed from the OME ``PhysicalSize``
    metadata, else defaults to ``(1, 1, 1)`` µm. This is the shared entry path for real images.
    """
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tif:
        image = tif.asarray()
        ome_xml = tif.ome_metadata

    if image.ndim == 2:
        image = image[np.newaxis, ...]  # promote single plane to [Z=1, Y, X]

    if voxel_size_zyx is None:
        voxel_size_zyx = _parse_physical_sizes(ome_xml)

    metadata = BundleMetadata(
        kind="real",
        shape_zyx=tuple(int(s) for s in image.shape[-3:]),
        voxel_size_zyx=voxel_size_zyx,
        tissue=tissue,
    )
    return ImageBundle(image=np.ascontiguousarray(image), metadata=metadata, phantom=None)


def _parse_physical_sizes(ome_xml: str | None) -> tuple[float, float, float]:
    """Best-effort extraction of (dz, dy, dx) µm from an OME-XML string; default (1, 1, 1)."""
    if not ome_xml:
        return (1.0, 1.0, 1.0)

    def _get(axis: str) -> float:
        m = re.search(rf'PhysicalSize{axis}="([0-9eE+\-.]+)"', ome_xml)
        return float(m.group(1)) if m else 1.0

    return (_get("Z"), _get("Y"), _get("X"))