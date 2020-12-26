"""
Unit tests for ``CovidSIRD``.
"""
import io
from contextlib import redirect_stdout
import unittest
import numpy
from numpy.testing import assert_almost_equal
from aftercovid.models._base_sir import BaseSIR
from aftercovid.models import CovidSIRD


class TestModelsCovidSir(unittest.TestCase):

    def test_base_sir(self):
        with self.assertRaises(TypeError):
            BaseSIR(('p', 0.5, 'PP'), [('q', 0.6, 'QQ')])
        with self.assertRaises(TypeError):
            BaseSIR([('p', 0.5, 'PP')], ('q', 0.6, 'QQ'))
        with self.assertRaises(TypeError):
            BaseSIR([('p', 0.5, 'PP')], [('q', 0.6, 'QQ')],
                    ('N', 0.6, 'NN'))
        with self.assertRaises(TypeError):
            BaseSIR([('p', 0.5, 'PP')], [('q', 0.6, 'QQ')],
                    [('N', 0.6, 'NN')], "r")
        models = BaseSIR([('p', 0.5, 'PP')], [('q', 0.6, 'QQ')],
                         [('N', 0.6, 'NN')])
        with self.assertRaises(NotImplementedError):
            models.R0()
        names = models.names
        self.assertEqual(names, ['N', 'p', 'q'])
        self.assertEqual(models['p'], 0.5)
        self.assertEqual(models['q'], 0.6)
        self.assertEqual(models['N'], 0.6)
        self.assertEqual(models.P, [('p', 0.5, 'PP')])
        self.assertEqual(models.Q, [('q', 0.6, 'QQ')])
        self.assertEqual(models.C, [('N', 0.6, 'NN')])
        with self.assertRaises(ValueError):
            models['qq']
        models['q'] = 6.1
        self.assertEqual(models['q'], 6.1)
        rst = models.to_rst()
        self.assertIn('*q*: QQ', rst)
        self.assertEqual(models.vect_names, ['q', 'p', 'N', 't'])

    def test_covid_sir(self):
        model = CovidSIRD()
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
        res = model.correctness()
        self.assertEqual(res.min(), res.max())
        self.assertEqual(res.min(), 0.)

    def test_covid_sir_eval(self):
        model = CovidSIRD()
        cst = model.cst_param
        self.assertEqual(cst, {'N': 10000.0, 'beta': 0.5,
                               'mu': 0.07142857142857142,
                               'nu': 0.047619047619047616})
        ev = model.eval_diff()
        self.assertEqual(ev['S'], -4.995)
        self.assertEqual(len(ev), 4)

    def test_covid_sir_loop(self):
        model = CovidSIRD()
        sim = list(model.iterate())
        self.assertEqual(len(sim), 10)
        self.assertGreater(sim[-1]['S'], 9500)
        self.assertLess(sim[-1]['S'], 10000)
        r0 = model.R0()
        self.assertEqual(r0, 4.2)

    def test_predict(self):
        model = CovidSIRD()
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
        model = CovidSIRD()
        losses = model._losses_sympy()
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), 4)
        grads = model._grads_sympy()
        self.assertIsInstance(grads, list)
        for row in grads:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 3)

    def test_fit(self):
        model = CovidSIRD()
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
        exp = numpy.array([model['beta'], model['nu'], model['mu']])
        model.fit(X, y, verbose=False, max_iter=10)
        coef = numpy.array([model['beta'], model['nu'], model['mu']])
        err = numpy.linalg.norm(exp - coef)
        self.assertLess(err, 1e-5)
        model['nu'] = model['mu'] = model['beta'] = 0.1
        buf = io.StringIO()
        with redirect_stdout(buf):
            model.fit(X, y, verbose=True, max_iter=20)
        out = buf.getvalue()
        self.assertIn('20/20', out)
        coef = numpy.array([model['beta'], model['nu'], model['mu']])
        err = numpy.linalg.norm(exp - coef)
        self.assertLess(err, 1e-1)
        loss = model.score(X, y)
        self.assertGreater(loss, 0)

    def test_noise(self):
        model = CovidSIRD()
        X, y = model.iterate2array(derivatives=True)
        X2 = model.add_noise(X)
        diff = numpy.abs(X - X2).max()
        self.assertEqual(X.shape, X.shape)
        s1 = numpy.sum(X, axis=1)
        s2 = numpy.sum(X2, axis=1)
        assert_almost_equal(s1, s2)
        self.assertGreater(diff, 1)

    def test_eval_diff(self):
        model = CovidSIRD()
        df1 = model.eval_diff()
        df2 = model._eval_diff_sympy()
        self.assertEqual(df1, df2)


if __name__ == '__main__':
    unittest.main()
