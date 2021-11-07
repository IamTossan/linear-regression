import numpy as np

from utils import cost_function, coef_determination


class TestCostFunction:
    def test_duplicate(self):
        xs = np.array([[1], [2], [3]])

        assert cost_function(xs, xs) == 0

    def test_simple(self):
        pred = np.array([[1], [2], [3], [4]])
        ys = np.array([[1], [2], [3], [5]])

        assert cost_function(pred, ys) == 1 / 8


class TestCoefDetermination:
    def test_duplicate(self):
        ys = np.array([[1], [2], [3]])

        assert coef_determination(ys, ys) == 1

    def test_simple(self):
        ys = np.array([[1], [2], [3]])
        pred = np.array([[1], [2], [4]])

        assert coef_determination(ys, pred) == 0.5
