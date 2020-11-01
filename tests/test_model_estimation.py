"""
Unit tests for ``model_estimation``.
"""
import os
import io
from contextlib import redirect_stdout
import unittest
import numpy
import pandas
from aftercovid.models import rolling_estimation


class TestModelEstimation(unittest.TestCase):

    def test_model_estimation_sird(self):
        this = os.path.dirname(__file__)
        data = os.path.join(this, 'data_france_preprocessed.csv')
        df = pandas.read_csv(data)
        df = df.tail(14)

        cols = ['safe', 'infected', 'recovered', 'deaths']
        data = df[cols].values.astype(numpy.float32)
        X = data[:-1]
        y = data[1:] - data[:-1]
        buf = io.StringIO()
        with redirect_stdout(buf):
            roll, mo = rolling_estimation(X, y, lrs=(1, 1e-2), delay=7,
                                          verbose=1, max_iter=100)
        pr = buf.getvalue()
        self.assertIn("k=6 iter=100", pr)
        self.assertIn('EpidemicRegressor', str(mo))
        self.assertIn("model='SIRD'", str(mo))
        self.assertEqual(['k', 'loss', 'it', 'R0', 'lr', 'beta', 'mu', 'nu'],
                         list(roll.columns))
        self.assertEqual(roll.shape, (6, 8))

    def test_model_estimation_sirdc(self):
        this = os.path.dirname(__file__)
        data = os.path.join(this, 'data_france_preprocessed.csv')
        df = pandas.read_csv(data)
        df = df.tail(14)

        cols = ['safe', 'infected', 'recovered', 'deaths']
        data = df[cols].values.astype(numpy.float32)
        X = data[:-1]
        y = data[1:] - data[:-1]
        buf = io.StringIO()
        with redirect_stdout(buf):
            roll, mo = rolling_estimation(X, y, lrs=(1, 1e-2), delay=7,
                                          verbose=1, max_iter=100,
                                          model_name="SIRDc")
        pr = buf.getvalue()
        self.assertIn("k=6 iter=100", pr)
        self.assertIn("model='SIRDc'", str(mo))
        self.assertIn('EpidemicRegressor', str(mo))
        self.assertEqual(['k', 'loss', 'it', 'R0', 'lr', 'beta', 'mu', 'nu', 'b'],
                         list(roll.columns))
        self.assertEqual(roll.shape, (6, 9))


if __name__ == '__main__':
    unittest.main()
