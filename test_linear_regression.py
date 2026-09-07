import unittest

import numpy as np

from linear_regression import LinearRegression


class LinearRegressionTests(unittest.TestCase):
    def test_recovers_known_line(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = 3.0 * X[:, 0] + 2.0
        model = LinearRegression().fit(X, y)
        self.assertAlmostEqual(float(model.coef_[0]), 3.0)
        self.assertAlmostEqual(float(model.intercept_), 2.0)
        self.assertAlmostEqual(model.score(X, y), 1.0)

    def test_ridge_shrinks_coefficients(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = 4.0 * X[:, 0]
        ordinary = LinearRegression(fit_intercept=False).fit(X, y)
        ridge = LinearRegression(fit_intercept=False, l2=100.0).fit(X, y)
        self.assertLess(abs(float(ridge.coef_[0])), abs(float(ordinary.coef_[0])))

    def test_rejects_mismatched_rows(self):
        with self.assertRaises(ValueError):
            LinearRegression().fit(np.ones((3, 1)), np.ones(2))


if __name__ == "__main__":
    unittest.main()

