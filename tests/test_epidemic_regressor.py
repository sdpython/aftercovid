"""
Unit tests for ``CovidSir``.
"""
import io
import pickle
import unittest
from numpy.testing import assert_almost_equal
from aftercovid.models import EpidemicRegressor, CovidSIR


class TestEpidemicRegressor(unittest.TestCase):

    def test_fit(self):
        model = CovidSIR()
        X, y = model.iterate2array(derivatives=True)
        epi = EpidemicRegressor(max_iter=10)
        with self.assertRaises(RuntimeError):
            epi.score(X, y)
        with self.assertRaises(RuntimeError):
            epi.predict(X)
        epi.fit(X, y)
        loss = epi.score(X, y)
        self.assertGreater(loss, 0)

    def test_clone(self):
        model = CovidSIR()
        model['beta'] = 0.4
        X, y = model.iterate2array(derivatives=True)
        epi = EpidemicRegressor(max_iter=10)
        epi.fit(X, y)
        self.assertGreater(epi.iter_, 0)
        pred1 = epi.predict(X)

        f = io.BytesIO()
        pickle.dump(epi, f)
        epi2 = pickle.load(io.BytesIO(f.getvalue()))
        pred2 = epi2.predict(X)
        assert_almost_equal(pred1, pred2)


if __name__ == '__main__':
    unittest.main()
