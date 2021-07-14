import numpy as np

from . import model, cost_function, grad, gradient_descent, coef_determination


class TestModel:
    def test_id(self):
        """f(x) = x"""
        xs = np.array([[1], [2], [3]])

        theta = np.ones((1, 3))

        expected_result = np.array([[1], [2], [3]])

        assert (model(xs, theta) == expected_result).all()

    def test_slope(self):
        """f(x) = 2x"""
        xs = np.array([[1], [2], [3]])

        theta = np.array([[2]])

        expected_result = np.array([[2], [4], [6]])

        assert (model(xs, theta) == expected_result).all()

    def test_len_two_theta(self):
        """f(x) = 2x + 1"""
        xs = np.array([[1, 1], [2, 1], [3, 1]])

        theta = np.array([[2], [1]])

        expected_result = np.array([[3], [5], [7]])

        assert (model(xs, theta) == expected_result).all()


class TestCostFunction:
    def test_duplicate(self):
        xs = np.array([[1], [2], [3]])

        assert cost_function(xs, xs) == 0

    def test_simple(self):
        pred = np.array([[1], [2], [3], [4]])
        ys = np.array([[1], [2], [3], [5]])

        assert cost_function(pred, ys) == 1 / 8


class TestGradient:
    def test_duplicate(self):
        xs = np.array([[1, 1], [2, 1], [3, 1]])
        ys = np.array([[1], [2], [3]])

        assert (grad(xs, ys, ys) == np.array([[0], [0]])).all()

    def test_simple(self):
        """
        grad[0] = 1/3 * 1 * x
        where x = 3
        grad[1] = 1/3 * 1
        """
        xs = np.array([[1, 1], [2, 1], [3, 1]])
        pred = np.array([[1], [2], [4]])
        ys = np.array([[1], [2], [3]])

        assert (grad(xs, pred, ys) == np.array([[1], [1 / 3]])).all()


class TestGradientDescent:
    def test_no_descent(self):
        xs = np.array([[1, 1], [2, 1], [3, 1]])
        ys = np.array([[1], [2], [3]])
        theta = np.array([[1], [0]])

        final_theta, history = gradient_descent(xs, ys, theta, 1, 1)

        assert (final_theta == theta).all()

    def test_simple(self):
        """
        da = 1/3 * (1(2-1) + 2(4-2) + 3(6-3))
        db = 1/3 * ((2-1) + (4-2) + (6-3))
        """
        xs = np.array([[1, 1], [2, 1], [3, 1]])
        ys = np.array([[1], [2], [3]])
        theta = np.array([[2], [0]])

        final_theta, history = gradient_descent(xs, ys, theta, 1, 1)

        assert format(final_theta[0][0], ".3f") == format(2 - (14 / 3), ".3f")
        assert final_theta[1][0] == -6 / 3


class TestCoefDetermination:
    def test_duplicate(self):
        ys = np.array([[1], [2], [3]])

        assert coef_determination(ys, ys) == 1

    def test_simple(self):
        ys = np.array([[1], [2], [3]])
        pred = np.array([[1], [2], [4]])

        assert coef_determination(ys, pred) == 0.5
