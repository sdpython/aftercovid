"""
Unit tests for ``CovidSIRD``.
"""
import io
from contextlib import redirect_stdout
import unittest
import numpy
from numpy.testing import assert_almost_equal
from aftercovid.models import CovidSIRDMixture


class TestModelsCovidSirMixture(unittest.TestCase):

    def test_covid_sir(self):
        model = CovidSIRDMixture()
        rst = model.to_rst()
        self.assertIn('\\frac', rst)
        self.assertIn('I', rst)
        par = model.get()
        self.assertIn('I1', par)
        self.assertIn('I2', par)
        p = {'I1': 5.}
        model.update(**p)
        par = model.get()
        self.assertEqual(par['I1'], 5.)
        dot = model.to_dot(verbose=True)
        self.assertIn("I1 -> R", dot)
        self.assertNotIn("I1 -> I1", dot)
        self.assertIn('beta', dot)
        self.assertNotIn('-beta1', dot)
        dot = model.to_dot(verbose=True, full=True)
        self.assertIn("I1 -> I1", dot)
        self.assertIn('-beta1', dot)
        model['beta1'] = 0.5
        model['beta2'] = 0.5
        self.assertEqual(model['beta1'], 0.5)
        self.assertEqual(model['beta2'], 0.5)
        model['N'] = 100000
        self.assertEqual(model['N'], 100000)
        ht = model._repr_html_()
        self.assertIn("{equation}", ht)
        res = model.correctness()
        self.assertEqual(res.min(), res.max())
        self.assertEqual(res.min(), 0.)
        model.rnd()

    def test_covid_sir_eval(self):
        model = CovidSIRDMixture()
        cst = model.cst_param
        self.assertEqual(cst, {'N': 10000.0,
                               'beta1': 0.5, 'beta2': 0.7,
                               'mu': 0.07142857142857142,
                               'nu': 0.047619047619047616})
        ev = model.eval_diff()
        self.assertLess(abs(ev['S'] - (-5.3946)), 1e-5)
        self.assertEqual(len(ev), 5)

    def test_covid_sir_loop(self):
        model = CovidSIRDMixture()
        sim = list(model.iterate())
        self.assertEqual(len(sim), 10)
        self.assertGreater(sim[-1]['S'], 9500)
        self.assertLess(sim[-1]['S'], 10000)
        r0 = model.R0()
        self.assertEqual(r0, 10.08)

    def test_predict(self):
        model = CovidSIRDMixture()
        sim = list(model.iterate(derivatives=True))
        self.assertIsInstance(sim, list)
        X = model.iterate2array(derivatives=False)
        self.assertIsInstance(X, numpy.ndarray)
        X, y = model.iterate2array(derivatives=True)
        self.assertEqual(X.shape, y.shape)
        y2 = model.predict(X)
        assert_almost_equal(y / 100, y2 / 100, decimal=6)
        with self.assertRaises(TypeError):
            model.predict({})
        with self.assertRaises(ValueError):
            model.predict(numpy.array([4]))
        with self.assertRaises(ValueError):
            model.predict(numpy.array([[4, 5, 6, 7, 8]]))
        X2 = X.copy()
        X2[0, 0] -= 50
        with self.assertRaises(ValueError):
            model.predict(X2)
        X2 = X.copy()
        X2[:, 0] -= 50
        with self.assertRaises(ValueError):
            model.predict(X2)

    def test_prefit(self):
        model = CovidSIRDMixture()
        losses = model._losses_sympy()
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), 5)
        grads = model._grads_sympy()
        self.assertIsInstance(grads, list)
        for row in grads:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 4)

    def test_fit(self):
        model = CovidSIRDMixture()
        X, y = model.iterate2array(derivatives=True)
        with self.assertRaises(TypeError):
            model.fit(X, {})
        with self.assertRaises(ValueError):
            model.fit(X, numpy.array([4]))
        with self.assertRaises(ValueError):
            model.fit(X, numpy.array([[4, 5, 6, 7, 8]]))
        X2 = X.copy()
        X2[0, 0] = -5
        with self.assertRaises(ValueError):
            model.fit(X2, y)
        exp = numpy.array(
            [model['beta1'], model['beta2'], model['nu'], model['mu']])
        model.fit(X, y, verbose=False, max_iter=10)
        coef = numpy.array(
            [model['beta1'], model['beta2'], model['nu'], model['mu']])
        err = numpy.linalg.norm(exp - coef)
        self.assertLess(err, 1e-5)
        model['nu'] = model['mu'] = model['beta1'] = model['beta2'] = 0.1
        buf = io.StringIO()
        with redirect_stdout(buf):
            model.fit(X, y, verbose=True, max_iter=20)
        out = buf.getvalue()
        self.assertIn('20/20', out)
        coef = numpy.array(
            [model['beta1'], model['beta2'], model['nu'], model['mu']])
        err = numpy.linalg.norm(exp - coef)
        self.assertLess(err, 3e-1)
        loss = model.score(X, y)
        self.assertGreater(loss, 0)

    def test_noise(self):
        model = CovidSIRDMixture()
        X, y = model.iterate2array(derivatives=True)
        X2 = model.add_noise(X)
        diff = numpy.abs(X - X2).max()
        self.assertEqual(X.shape, X.shape)
        s1 = numpy.sum(X, axis=1)
        s2 = numpy.sum(X2, axis=1)
        assert_almost_equal(s1, s2)
        self.assertGreater(diff, 1)

    def test_eval_diff(self):
        model = CovidSIRDMixture()
        df1 = model.eval_diff()
        df2 = model._eval_diff_sympy()
        self.assertEqual(df1, df2)


if __name__ == '__main__':
    unittest.main()
