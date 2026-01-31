"""A small ordinary least squares and ridge regression estimator."""

from __future__ import annotations

import numpy as np


class LinearRegression:
    """Fit a linear model with a closed-form least squares solution."""

    def __init__(self, *, fit_intercept: bool = True, l2: float = 0.0) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.fit_intercept = fit_intercept
        self.l2 = float(l2)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if y.ndim not in (1, 2):
            raise ValueError("y must be a one-dimensional or two-dimensional array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("X and y cannot be empty")

        design = np.column_stack((np.ones(X.shape[0]), X)) if self.fit_intercept else X
        penalty = np.eye(design.shape[1]) * self.l2
        if self.fit_intercept:
            penalty[0, 0] = 0.0

        weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = np.asarray(weights[1:]).T
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(weights).T
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        return X @ np.asarray(self.coef_).T + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        residual = y - self.predict(X)
        total = y - y.mean(axis=0)
        denominator = float(np.sum(total**2))
        return 1.0 - float(np.sum(residual**2)) / denominator if denominator else 1.0

