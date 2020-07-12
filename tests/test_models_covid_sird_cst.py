"""
Unit tests for ``CovidSIRDc``.
"""
import io
from contextlib import redirect_stdout
import unittest
import numpy
from numpy.testing import assert_almost_equal
from aftercovid.models import CovidSIRDc


class TestModelsCovidSIRDcst(unittest.TestCase):

    def test_covid_sir(self):
        model = CovidSIRDc()
        rst = model.to_rst()
        self.assertIn('\\frac', rst)
        self.assertIn('I', rst)
        par = model.get()
        self.assertIn('I', par)
        p = {'I': 5.}
        model.update(**p)
        par = model.get()
        self.assertEqual(par['I'], 5.)
        dot = model.to_dot(verbose=True)
        self.assertIn("I -> R", dot)
        self.assertNotIn("I -> I", dot)
        self.assertIn('beta', dot)
        self.assertNotIn('-beta', dot)
        dot = model.to_dot(verbose=True, full=True)
        self.assertIn("I -> I", dot)
        self.assertIn('-beta', dot)
        model['beta'] = 0.5
        self.assertEqual(model['beta'], 0.5)
        model['N'] = 100000
        self.assertEqual(model['N'], 100000)
        ht = model._repr_html_()
        self.assertIn("{equation}", ht)

    def test_covid_sir_eval(self):
        model = CovidSIRDc()
        cst = model.cst_param
        self.assertEqual(cst, {'N': 10000.0, 'beta': 0.5,
                               'mu': 0.07142857142857142,
                               'nu': 0.047619047619047616,
                               'cR': 0.01, 'cS': 0.01})
        ev = model.eval_diff()
        self.assertEqual(-4.9949995, ev['S'])
        self.assertEqual(len(ev), 4)

    def test_covid_sir_loop(self):
        model = CovidSIRDc()
        sim = list(model.iterate())
        self.assertEqual(len(sim), 10)
        self.assertGreater(sim[-1]['S'], 9500)
        self.assertLess(sim[-1]['S'], 10000)
        r0 = model.R0()
        self.assertEqual(r0, 4.2)

    def test_predict(self):
        model = CovidSIRDc()
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
        model = CovidSIRDc()
        losses = model._losses_sympy()
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), 4)
        grads = model._grads_sympy()
        self.assertIsInstance(grads, list)
        for row in grads:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 5)

    def test_fit(self):
        model = CovidSIRDc()
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
            model.fit(X2, y, learning_rate_init=0.01)
        exp = numpy.array([model['beta'], model['nu'], model['mu'],
                           model['cS'], model['cR']])
        model.fit(X, y, verbose=False, max_iter=10,
                  learning_rate_init=0.01)
        coef = numpy.array([model['beta'], model['nu'], model['mu'],
                            model['cS'], model['cR']])
        err = numpy.linalg.norm(exp - coef)
        self.assertLess(err, 1e-5)
        model['nu'] = model['mu'] = model['beta'] = 0.1
        buf = io.StringIO()
        with redirect_stdout(buf):
            model.fit(X, y, verbose=True, max_iter=20)
        out = buf.getvalue()
        self.assertIn('20/20', out)
        coef = numpy.array([model['beta'], model['nu'], model['mu'],
                            model['cS'], model['cR']])
        err = numpy.linalg.norm(exp - coef)
        self.assertLess(err, 1)
        loss = model.score(X, y)
        self.assertGreater(loss, 0)

    def test_noise(self):
        model = CovidSIRDc()
        X, y = model.iterate2array(derivatives=True)
        X2 = model.add_noise(X)
        diff = numpy.abs(X - X2).max()
        self.assertEqual(X.shape, X.shape)
        s1 = numpy.sum(X, axis=1)
        s2 = numpy.sum(X2, axis=1)
        assert_almost_equal(s1, s2)
        self.assertGreater(diff, 1)

    def test_eval_diff(self):
        model = CovidSIRDc()
        df1 = model.eval_diff()
        df2 = model._eval_diff_sympy()
        self.assertEqual(df1, df2)


if __name__ == '__main__':
    unittest.main()
