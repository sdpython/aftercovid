"""
Unit tests for ``optim``.
"""
import io
from contextlib import redirect_stdout
import unittest
import numpy
from aftercovid.optim import SGDOptimizer


def fct_loss(c, X, y):
    return numpy.linalg.norm(X @ c - y) ** 2


def fct_grad(c, x, y, i):
    return x * (x @ c - y) * 0.1


class TestOptim(unittest.TestCase):

    def test_sgd_optimizer(self):
        coef = numpy.array([0.5, 0.6, 0.7])

        X = numpy.random.randn(10, 3)
        y = X @ coef

        ls = fct_loss(coef, X, y)
        self.assertLess(ls, 1e-10)

        gr = fct_grad(coef, X[0, :], y[0], 0)
        no = numpy.linalg.norm(gr)
        self.assertLess(no, 1e-10)

        gr = fct_grad(numpy.array([0., 0., 0.]), X[0, :], y[0], 0)
        no = numpy.linalg.norm(gr)
        self.assertGreater(no, 0.001)

        sgd = SGDOptimizer(numpy.array([0., 0., 0.]))

        buf = io.StringIO()
        with redirect_stdout(buf):
            ls = sgd.train(X, y, fct_loss, fct_grad, max_iter=15, verbose=True)
        out = buf.getvalue()
        self.assertIn("15/15: loss", out)
        self.assertLess(ls, 0.1)
        self.assertEqual(sgd.learning_rate, 0.1)

        sgd = SGDOptimizer(numpy.array([0., 0., 0.]), lr_schedule='invscaling')
        buf = io.StringIO()
        with redirect_stdout(buf):
            ls = sgd.train(X, y, fct_loss, fct_grad, max_iter=15, verbose=True)
        out = buf.getvalue()
        self.assertIn("15/15: loss", out)
        self.assertLess(ls, 1)
        self.assertLess(sgd.learning_rate, 0.1)

    def test_sgd_optimizer_raise(self):
        coef = numpy.array([0.5, 0.6, 0.7])

        rs = numpy.random.RandomState(seed=0)
        X = rs.randn(10, 3)
        y = X @ coef

        ls = fct_loss(coef, X, y)
        self.assertLess(ls, 1e-10)

        gr = fct_grad(coef, X[0, :], y[0], 0)
        no = numpy.linalg.norm(gr)
        self.assertLess(no, 1e-10)

        gr = fct_grad(numpy.array([0., 0., 0.]), X[0, :], y[0], 0)
        no = numpy.linalg.norm(gr)
        self.assertGreater(no, 0.0007)

        with self.assertRaises(TypeError):
            SGDOptimizer({})
        sgd = SGDOptimizer(numpy.array([0., 0., 0.]))
        with self.assertRaises(ValueError):
            sgd.update_coef(numpy.array([0., 0., 0., 0.]))
        with self.assertRaises(TypeError):
            sgd.train(X, {}, fct_loss, fct_grad)
        with self.assertRaises(TypeError):
            sgd.train({}, y, fct_loss, fct_grad)
        with self.assertRaises(ValueError):
            sgd.train(X[:4], y, fct_loss, fct_grad)
        with self.assertRaises(TypeError):
            sgd.train(X, y, fct_loss, fct_grad, max_iter=15, min_threshold="e")
        with self.assertRaises(TypeError):
            sgd.train(X, y, fct_loss, fct_grad, max_iter=15, max_threshold="e")
        buf = io.StringIO()
        with redirect_stdout(buf):
            X[0, 0] = numpy.nan
            with self.assertRaises(ValueError):
                sgd.train(X, y, fct_loss, fct_grad, max_iter=15, verbose=True)
            X[0, 0] = 1.
            y[0] = numpy.nan
            with self.assertRaises(ValueError):
                sgd.train(X, y, fct_loss, fct_grad, max_iter=15, verbose=True)


if __name__ == '__main__':
    unittest.main()
