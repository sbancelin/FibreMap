"""Inter-tissue comparison: PCA + nearest-centroid separate aligned vs isotropic descriptors."""

from __future__ import annotations

import numpy as np

from collagen_shg.analysis_resolved.compare_tissues import (
    PCA,
    NearestCentroidClassifier,
    feature_matrix,
    standardize,
)
from collagen_shg.analysis_resolved.descriptors import (
    descriptor_vector,
    organization_descriptors_3d,
)
from collagen_shg.representations import conventions as cv

VOXEL = (0.5, 0.2, 0.2)


def _descriptor_vec(mean_phi, sigma_phi, rng):
    shape = (6, 24, 24)
    phi = cv.wrap_axial(rng.normal(mean_phi, sigma_phi, size=shape))
    d = np.moveaxis(cv.director_from_angles(phi, np.zeros(shape)), -1, 0)
    desc = organization_descriptors_3d(d, np.ones(shape), VOXEL)
    return descriptor_vector(desc)


def _dataset(rng):
    aligned = [_descriptor_vec(0.5, 0.15, rng) for _ in range(5)]  # high order
    isotropic = [_descriptor_vec(0.0, 5.0, rng) for _ in range(5)]  # near-isotropic
    X = feature_matrix(aligned + isotropic)
    y = np.array(["aligned"] * 5 + ["isotropic"] * 5)
    return X, y


def test_pca_separates_classes_on_first_component():
    rng = np.random.default_rng(0)
    X, y = _dataset(rng)
    Xs, _, _ = standardize(X)
    pcs = PCA(n_components=2).fit_transform(Xs)
    pc1_aligned = pcs[y == "aligned", 0].mean()
    pc1_iso = pcs[y == "isotropic", 0].mean()
    assert abs(pc1_aligned - pc1_iso) > 1.0  # classes separated along PC1


def test_pca_explained_variance_is_ordered():
    rng = np.random.default_rng(1)
    X, _ = _dataset(rng)
    Xs, _, _ = standardize(X)
    pca = PCA(n_components=3).fit(Xs)
    ev = pca.explained_variance_ratio_
    assert np.all(np.diff(ev) <= 1e-9)  # non-increasing
    assert ev[0] > 0.4


def test_nearest_centroid_classifies():
    rng = np.random.default_rng(2)
    X, y = _dataset(rng)
    Xs, mean, std = standardize(X)
    clf = NearestCentroidClassifier().fit(Xs, y)
    assert (clf.predict(Xs) == y).mean() == 1.0
    # a fresh aligned sample classifies as aligned
    new = _descriptor_vec(0.5, 0.15, rng)[None, :]
    new_s, _, _ = standardize(new, mean, std)
    assert clf.predict(new_s)[0] == "aligned"
