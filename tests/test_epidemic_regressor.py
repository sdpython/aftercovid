"""
Unit tests for ``CovidSir``.
"""
import io
import os
import pickle
import unittest
import warnings
import numpy
from numpy.testing import assert_almost_equal
from pandas import read_csv
from aftercovid.models import EpidemicRegressor, CovidSIR


def find_best_model(Xt, yt, lrs, th):
    best_est, best_loss = None, None
    for lr in lrs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EpidemicRegressor(
                'SIR',
                learning_rate_init=lr,
                max_iter=500,
                early_th=1)
            m.fit(Xt, yt)
            loss = m.score(Xt, yt)
            if numpy.isnan(loss):
                continue
        if best_est is None or best_loss > loss:
            best_est = m
            best_loss = loss
        if best_loss < th:
            return best_est, best_loss
    return best_est, best_loss


class TestEpidemicRegressor(unittest.TestCase):

    def test_fit(self):
        model = CovidSIR()
        X, y = model.iterate2array(derivatives=True)
        with self.assertRaises(ValueError):
            EpidemicRegressor('sir2')
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

    def test_real_data(self):
        this = os.path.join(os.path.dirname(__file__), "data_france.csv")
        df = read_csv(this)
        data = df[['total', 'confirmed', 'recovered', 'deaths']].values
        X = data[:-1]
        y = data[1:] - data[:-1]
        self.assertEqual(X.shape, y.shape)

        coefs = []
        for k in range(0, X.shape[0] - 9, 20):
            end = min(k + 10, X.shape[0])
            Xt, yt = X[k:end], y[k:end]
            m, loss = find_best_model(Xt, yt, [1e-2, 1e-3], 10)
            loss = m.score(Xt, yt)
            obs = dict(k=k, loss=loss, it=m.iter_, R0=m.model_.R0())
            obs.update({k: v for k, v in zip(
                m.model_.param_names, m.model_._val_p)})
            coefs.append(obs)

        self.assertGreater(len(coefs), 1)


if __name__ == '__main__':
    unittest.main()
