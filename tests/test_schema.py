"""Schema validation tests for Phantom and ImageBundle (phase0 acceptance: schema versioned)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from collagen_shg.representations import (
    BundleMetadata,
    DirectorFields,
    Fibril,
    ImageBundle,
    OrganizationGT,
    Phantom,
    PhantomMeta,
)
from collagen_shg.version import SCHEMA_VERSION


# ----------------------------------------------------------------------------------- PhantomMeta
def test_phantom_meta_defaults_and_bounds():
    m = PhantomMeta(shape_zyx=(4, 5, 6), voxel_size_zyx=(0.5, 0.2, 0.2))
    assert m.schema_version == SCHEMA_VERSION
    assert m.units.length == "um" and m.units.angle == "rad"
    assert m.bounds_um == (0.0, 3 * 0.5, 0.0, 4 * 0.2, 0.0, 5 * 0.2)
    assert isinstance(m.id, str)


@pytest.mark.parametrize("shape", [(0, 4, 4), (4, -1, 4)])
def test_phantom_meta_rejects_bad_shape(shape):
    with pytest.raises(ValidationError):
        PhantomMeta(shape_zyx=shape, voxel_size_zyx=(1.0, 1.0, 1.0))


def test_phantom_meta_rejects_bad_voxel():
    with pytest.raises(ValidationError):
        PhantomMeta(shape_zyx=(4, 4, 4), voxel_size_zyx=(0.0, 1.0, 1.0))


# ---------------------------------------------------------------------------------------- Fibril
def test_fibril_valid_and_coercion():
    f = Fibril(id=1, centerline=[[0, 0, 0], [1, 1, 1]], diameter=[1.0, 1.2], polarity=-1)
    assert f.centerline.shape == (2, 3)
    assert f.centerline.dtype == np.float64
    assert f.polarity == -1


def test_fibril_rejects_shape_mismatch():
    with pytest.raises(ValidationError):
        Fibril(id=1, centerline=[[0, 0, 0], [1, 1, 1]], diameter=[1.0])


def test_fibril_rejects_bad_centerline():
    with pytest.raises(ValidationError):
        Fibril(id=1, centerline=[[0, 0], [1, 1]], diameter=[1.0, 1.0])


def test_fibril_rejects_bad_polarity():
    with pytest.raises(ValidationError):
        Fibril(id=1, centerline=[[0, 0, 0]], diameter=[1.0], polarity=2)


# ---------------------------------------------------------------------------------- DirectorFields
def test_director_fields_zeros():
    df = DirectorFields.zeros((2, 3, 4))
    assert df.director.shape == (3, 2, 3, 4)
    assert df.director.dtype == np.float32
    assert df.shape_zyx == (2, 3, 4)
    assert np.all(df.order_S == 0)


def test_director_fields_rejects_inconsistent_shapes():
    with pytest.raises(ValidationError):
        DirectorFields(
            director=np.zeros((3, 2, 3, 4), dtype=np.float32),
            order_S=np.zeros((2, 3, 5), dtype=np.float32),  # wrong X
            density=np.zeros((2, 3, 4), dtype=np.float32),
            polarity=np.zeros((2, 3, 4), dtype=np.float32),
        )


def test_director_fields_rejects_bad_director_shape():
    with pytest.raises(ValidationError):
        DirectorFields(
            director=np.zeros((2, 2, 3, 4), dtype=np.float32),  # first dim must be 3
            order_S=np.zeros((2, 3, 4), dtype=np.float32),
            density=np.zeros((2, 3, 4), dtype=np.float32),
            polarity=np.zeros((2, 3, 4), dtype=np.float32),
        )


# ---------------------------------------------------------------------------------------- Phantom
def test_phantom_empty():
    p = Phantom.empty((2, 4, 4), (0.5, 0.2, 0.2), seed=123, tissue_preset="tendon")
    assert p.meta.shape_zyx == (2, 4, 4)
    assert p.meta.seed == 123
    assert p.geometry == []
    assert p.fields is not None and p.fields.shape_zyx == (2, 4, 4)
    assert isinstance(p.ground_truth, OrganizationGT)


def test_phantom_rejects_field_shape_mismatch():
    meta = PhantomMeta(shape_zyx=(2, 4, 4), voxel_size_zyx=(1.0, 1.0, 1.0))
    bad_fields = DirectorFields.zeros((2, 4, 8))
    with pytest.raises(ValidationError):
        Phantom(meta=meta, fields=bad_fields)


def test_organization_gt_global_alias():
    gt = OrganizationGT.model_validate({"global": {"S2": 0.9, "xi_um": 40.0}})
    assert gt.global_.S2 == 0.9
    assert gt.global_.xi_um == 40.0
    # round-trip through alias
    dumped = gt.model_dump(by_alias=True)
    assert "global" in dumped


# ------------------------------------------------------------------------------------ ImageBundle
def test_image_bundle_white():
    b = ImageBundle.white((2, 4, 4), (0.5, 0.2, 0.2), fill=1.0)
    assert b.shape_zyx == (2, 4, 4)
    assert np.all(b.image == 1.0)
    assert b.metadata.kind == "synthetic"
    assert b.metadata.schema_version == SCHEMA_VERSION


def test_image_bundle_rejects_shape_mismatch():
    meta = BundleMetadata(kind="synthetic", shape_zyx=(2, 4, 4), voxel_size_zyx=(1.0, 1.0, 1.0))
    with pytest.raises(ValidationError):
        ImageBundle(image=np.zeros((2, 4, 8), dtype=np.float32), metadata=meta)


def test_image_bundle_carries_phantom_gt():
    p = Phantom.empty((2, 4, 4), (0.5, 0.2, 0.2))
    b = ImageBundle.white((2, 4, 4), (0.5, 0.2, 0.2), phantom=p)
    assert b.phantom is not None
    assert b.phantom.meta.shape_zyx == b.metadata.shape_zyx