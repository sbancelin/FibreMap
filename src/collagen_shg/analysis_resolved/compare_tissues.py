"""Inter-tissue comparison (Livrable 3).

Assemble per-image organization descriptor vectors, standardize them, reduce with PCA, and
classify (nearest centroid) — the "same language of organization" used to compare and separate
tissues. PCA is a plain NumPy SVD (no scikit-learn dependency); UMAP is an optional later add.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["standardize", "PCA", "NearestCentroidClassifier", "feature_matrix"]


def feature_matrix(vectors) -> np.ndarray:
    """Stack descriptor vectors (1D arrays) into a 2D ``(n_samples, n_features)`` matrix."""
    return np.vstack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])


def standardize(
    X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features. Returns ``(standardized, mean, std)``; reuse mean/std to transform."""
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    safe = np.where(std > 0, std, 1.0)
    return (X - mean) / safe, mean, safe


@dataclass
class PCA:
    """Minimal PCA via SVD on centered data."""

    n_components: int = 2

    def fit(self, X: np.ndarray) -> PCA:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self.components_ = vt[:k]
        var = s**2 / max(1, X.shape[0] - 1)
        total = var.sum()
        self.explained_variance_ratio_ = (var[:k] / total) if total > 0 else var[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NearestCentroidClassifier:
    """Classify by nearest class centroid in feature space."""

    def fit(self, X: np.ndarray, labels) -> NearestCentroidClassifier:
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self.centroids_ = np.vstack([X[labels == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(dists, axis=1)]
