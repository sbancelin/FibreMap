"""Bundle round-trip and OME-TIFF ingestion (phase0 acceptance: lossless I/O; real ingestion).

The acceptance criterion is bit-for-bit identity of arrays and equality of metadata after a
write -> read cycle.
"""

from __future__ import annotations

import numpy as np
import pytest

from collagen_shg.representations import (
    BundleMetadata,
    Fibril,
    ImageBundle,
    MicroscopeMeta,
    Phantom,
    read_bundle,
    read_ome_tiff,
    write_bundle,
    write_ome_tiff,
)


def _rich_bundle(rng: np.random.Generator) -> ImageBundle:
    shape = (3, 8, 8)
    voxel = (0.5, 0.2, 0.2)
    p = Phantom.empty(shape, voxel, seed=7, tissue_preset="tendon")
    # populate non-trivial fields so the array round-trip is meaningful
    p.fields.director[...] = rng.standard_normal((3, *shape)).astype(np.float32)
    p.fields.order_S[...] = rng.random(shape, dtype=np.float32)
    p.fields.density[...] = rng.random(shape, dtype=np.float32)
    # two fibrils with different-length centerlines
    p.geometry.append(
        Fibril(id=0, centerline=[[0, 0, 0], [1, 0.5, 0.2], [2, 1, 0.4]],
               diameter=[1.5, 1.4, 1.3], polarity=1, fiber_id=10, type="I")
    )
    p.geometry.append(
        Fibril(id=1, centerline=[[0, 1, 0], [0.5, 1.5, 0.1]],
               diameter=[1.0, 1.1], polarity=-1, fascicle_id=3)
    )
    p.ground_truth.global_.S2 = 0.42
    p.ground_truth.global_.xi_um = 35.0

    image = (rng.random(shape) * 1000).astype(np.uint16)
    metadata = BundleMetadata(
        kind="synthetic",
        shape_zyx=shape,
        voxel_size_zyx=voxel,
        microscope=MicroscopeMeta(
            mode="incoherent", NA=0.95, wavelength_nm=900, detection="backward"
        ),
    )
    metadata.provenance.seed = 7
    return ImageBundle(image=image, metadata=metadata, phantom=p,
                       config={"run": {"name": "rt", "seed": 7}})


def test_bundle_roundtrip_bit_exact(tmp_path):
    rng = np.random.default_rng(0)
    b = _rich_bundle(rng)
    path = tmp_path / "ds.bundle"
    write_bundle(b, path)
    r = read_bundle(path)

    # arrays: bit-for-bit identity
    assert r.image.dtype == b.image.dtype
    assert np.array_equal(r.image, b.image)
    assert np.array_equal(r.phantom.fields.director, b.phantom.fields.director)
    assert np.array_equal(r.phantom.fields.order_S, b.phantom.fields.order_S)
    assert np.array_equal(r.phantom.fields.density, b.phantom.fields.density)
    assert np.array_equal(r.phantom.fields.polarity, b.phantom.fields.polarity)

    # metadata: equality
    assert r.metadata == b.metadata
    assert r.config == b.config

    # geometry: structural identity
    assert len(r.phantom.geometry) == 2
    for fr, fb in zip(r.phantom.geometry, b.phantom.geometry, strict=True):
        assert fr.id == fb.id
        assert fr.polarity == fb.polarity
        assert fr.fiber_id == fb.fiber_id
        assert fr.fascicle_id == fb.fascicle_id
        assert fr.type == fb.type
        assert np.array_equal(fr.centerline, fb.centerline)
        assert np.array_equal(fr.diameter, fb.diameter)

    # organization ground truth
    assert r.phantom.ground_truth.global_.S2 == 0.42
    assert r.phantom.ground_truth.global_.xi_um == 35.0
    # phantom meta (incl. the UUID and tissue preset)
    assert r.phantom.meta.id == b.phantom.meta.id
    assert r.phantom.meta.tissue_preset == "tendon"


def test_bundle_roundtrip_empty_phantom(tmp_path):
    p = Phantom.empty((2, 4, 4), (0.5, 0.2, 0.2))
    b = ImageBundle.white((2, 4, 4), (0.5, 0.2, 0.2), phantom=p)
    path = tmp_path / "empty.bundle"
    write_bundle(b, path)
    r = read_bundle(path)
    assert np.array_equal(r.image, b.image)
    assert r.metadata == b.metadata
    assert r.phantom.geometry == []


def test_write_refuses_existing_without_overwrite(tmp_path):
    b = ImageBundle.white((2, 2, 2), (1.0, 1.0, 1.0))
    path = tmp_path / "x.bundle"
    write_bundle(b, path)
    with pytest.raises(FileExistsError):
        write_bundle(b, path)
    write_bundle(b, path, overwrite=True)  # should not raise


def test_ome_tiff_ingestion(tmp_path):
    # Write a synthetic OME-TIFF then ingest it through the shared real-image path.
    rng = np.random.default_rng(1)
    image = (rng.random((4, 16, 16)) * 255).astype(np.uint8)
    src = ImageBundle(
        image=image,
        metadata=BundleMetadata(kind="synthetic", shape_zyx=(4, 16, 16),
                                voxel_size_zyx=(0.7, 0.3, 0.3)),
    )
    tif_path = tmp_path / "real.ome.tif"
    write_ome_tiff(src, tif_path)

    ingested = read_ome_tiff(tif_path)
    assert ingested.metadata.kind == "real"
    assert ingested.shape_zyx == (4, 16, 16)
    assert ingested.phantom is None
    assert np.array_equal(ingested.image, image)
    # physical sizes recovered from OME metadata
    assert ingested.metadata.voxel_size_zyx == pytest.approx((0.7, 0.3, 0.3))

    # and a real bundle writes/reads with empty ground truth
    out = tmp_path / "real.bundle"
    write_bundle(ingested, out)
    back = read_bundle(out)
    assert back.metadata.kind == "real"
    assert np.array_equal(back.image, image)
    assert back.phantom is None