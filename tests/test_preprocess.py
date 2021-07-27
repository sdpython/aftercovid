"""
Unit tests for ``preprocess``.
"""
import unittest
import numpy
from numpy import nan as nnan
from numpy.testing import assert_almost_equal
import pandas
from aftercovid.preprocess import (
    ts_moving_average, ts_normalise_negative_values,
    ts_remove_decreasing_values)


class TestPreprocess(unittest.TestCase):

    def test_ts_moving_average(self):
        ts = numpy.array([1] * 10)
        with self.assertRaises(ValueError):
            ts_moving_average(ts, center=True, n=6)
        res = ts_moving_average(ts, n=7, center=False)
        assert_almost_equal(ts, res)
        res = ts_moving_average(ts, n=7, center=True)
        assert_almost_equal(ts, res)
        data = pandas.DataFrame(ts)
        res = ts_moving_average(data, n=7, center=True)
        assert_almost_equal(ts, res.values.ravel())

    def test_ts_moving_average_0(self):
        ts_ = numpy.array([1] * 10, dtype=numpy.float64)
        for i in range(0, len(ts_)):
            with self.subTest(i=i):
                ts = ts_.copy()
                ts[i] = 0
                res = ts_moving_average(ts, n=7, center=False)
                assert_almost_equal(ts[:i], res[:i])
                assert_almost_equal(ts[i + 7:], res[i + 7:])
                self.assertTrue(res[i] < 1)
                res = ts_moving_average(ts, n=7, center=True)
                self.assertTrue(res[i] < 1)
                if i >= 3:
                    assert_almost_equal(ts[:i - 3], res[:i - 3])
                    assert_almost_equal(ts[i + 3:], res[i + 3:])

    def test_ts_moving_average_nan(self):
        ts_ = numpy.array([1] * 10, dtype=numpy.float64)
        for i in range(0, len(ts_)):
            with self.subTest(i=i):
                ts = ts_.copy()
                ts[i] = numpy.nan
                res = ts_moving_average(ts, n=7, center=False)
                assert_almost_equal(ts[:i], res[:i])
                assert_almost_equal(ts[i + 7:], res[i + 7:])
                res = ts_moving_average(ts, n=7, center=True)
                self.assertEqual(res[i], 1)
                if i >= 3:
                    assert_almost_equal(ts[:i - 3], res[:i - 3])
                    assert_almost_equal(ts[i + 3:], res[i + 3:])

    def test_ts_moving_average_neg(self):
        ts = numpy.array([1, -1] * 6)
        res = ts_moving_average(ts, n=5, center=False)
        exp = numpy.array([1., 0., 0.3333333, 0., 0.2, -
                           0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
        assert_almost_equal(exp, res)
        res = ts_moving_average(ts, n=5, center=True)
        exp = numpy.array([0, 0.3333333, 0., -0.2, 0.2, -
                           0.2, 0.2, -0.2, 0.2, 0, -0.3333333, 0.])
        assert_almost_equal(exp, res)

    def test_ts_moving_average_dim2(self):
        ts = numpy.ones((10, 2))
        res = ts_moving_average(ts, n=7, center=False)
        assert_almost_equal(ts, res)
        res = ts_moving_average(ts, n=7, center=True)
        assert_almost_equal(ts, res)

    def test_ts_normalise_negative_values(self):
        ts = numpy.array([1] * 10)
        ts[5] = -1
        ts[-1] = 2
        with self.assertRaises(ValueError):
            ts_normalise_negative_values(ts, n=6)
        res = ts_normalise_negative_values(ts, n=7)
        self.assertEqual(ts.sum(), res.sum())
        self.assertEqual(res[0], 0.84)
        self.assertEqual(res.min(), 0.6)
        self.assertEqual(res[-1], 1.68)

    def test_normalise_nan_values(self):
        ts = numpy.array([1] * 10, dtype=numpy.float64)
        ts[5] = numpy.nan
        ts[-1] = 2
        with self.assertRaises(ValueError):
            ts_normalise_negative_values(ts, n=6)
        res = ts_normalise_negative_values(ts, n=7)
        self.assertEqual(10, res.sum())
        self.assertEqual(res[0], 0.9090909090909091)
        self.assertEqual(res.min(), 0.9090909090909091)
        self.assertEqual(res[-1], 1.8181818181818181)

    def test_normalise_negative_pandas_values(self):
        df = pandas.DataFrame(
            numpy.ones((10, 2), dtype=numpy.float64), columns=["c1", "c2"])
        df['c2'] *= 2
        res = ts_normalise_negative_values(df, n=7)
        assert_almost_equal(df.values, res.values)
        self.assertEqual(list(df.columns), list(res.columns))

    def test_normalise_negative_pandas_values_series(self):
        df = pandas.DataFrame(
            numpy.ones((10, 2), dtype=numpy.float64), columns=["c1", "c2"])
        df['c2'] *= 2
        for c in df.columns:
            res = ts_normalise_negative_values(df[c])
            self.assertIsInstance(res, df[c].__class__)

    def test_moving_nan_final(self):
        ts = numpy.array(
            [nnan, 5.5625, 6.4375, 9.375, 11.5, 11.875, 16.3125, 16.875,
             16.8125, 5.5625, 6.4375, 9.375, 11.5, 11.875, 16.3125,
             16.875, 16.8125, 11.625, 5.375, 5.25, 3.875, 5.0, 5.0625,
             nnan, nnan, nnan, nnan, nnan, nnan, nnan, nnan, nnan, nnan,
             nnan, nnan, nnan])
        mov = ts_moving_average(ts)
        self.assertFalse(numpy.isnan(mov[0]))
        self.assertTrue(numpy.isnan(mov[-1]))

    def test_normalize_spain(self):
        data = numpy.array([
            nnan, 7.125, 8.5625, 13.5625, 21.6875, 33.625, 49.1875,
            67.625, 84.375, 103.5, 129.375, 157.0, 202.0625, 260.8125,
            327.3125, 416.6875, 506.9375, 591.0625, 679.125,
            734.375, 781.75, 812.9375, 827.375, 851.6875, 859.875,
            869.6875, 860.25, 825.75, 786.0625, 746.1875,
            718.1875, 702.375, 684.5, 657.75, 626.6875, 578.875,
            550.1875, 530.75, 526.8125, 527.5625, 505.25, 469.6875,
            409.3125, 389.75, 387.1875, 395.625, 416.875, 404.3125,
            383.0625, 359.6875, 344.25, 336.75, 315.375, 314.125,
            290.125, 260.875, 263.875, 231.3125, 219.875, 219.0,
            206.1875, 206.625, 198.375, 182.75, 167.8125, 163.8125,
            166.375, 167.75, 157.3125, 137.8125, 113.0625, 93.1875,
            88.0625, 121.25, 157.5625, 191.25, 98.5625, -53.3125,
            -207.125, -363.1875, -276.125, -181.1875, -83.0, 19.5625,
            2.0, 1.6875, 1.4375, 1.375, 1.5, 1.8125, 1.625, 1.3125,
            0.875, 0.375, 0.1875, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0,
            73.6875, 147.8125, 222.0, 296.25, 223.1875, 149.375,
            75.625, 2.25, 2.625, 3.3125, 3.875, 4.0625, 4.5, 5.0625,
            6.5, 7.3125, 7.1875, 6.875, 4.875, 3.6875, 3.375, 3.0625,
            3.3125, 3.0625, 2.375, 1.8125, 1.75, 2.125, 2.6875, 3.0,
            2.8125, 2.375, 1.75, 1.4375, 1.5625, 1.8125, 2.0625, 2.0625,
            nnan, nnan, nnan, nnan, nnan, nnan, nnan, nnan,
            nnan, nnan, nnan, nnan, nnan])
        norm = ts_normalise_negative_values(data, extreme=2)
        self.assertFalse(any(norm < 0))

    def test_normalise_defect(self):
        data = numpy.array([
            nnan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 3, 2, 8, 0, 14, 15,
            0, 31, 12, 0, 57, 0, 0, 95, 207, 112, 112, 186, 240,
            231, 365, 299, 319, 292, 418, 499, 880, 984, 1120,
            1053, 518, 833, 1417, 541, 1341, 987, 635, 561, 574,
            745, 1436, 753, 760, 642, 391, 546, 525, 544, 516, 389,
            369, 242, 437, 367, 427, 289, 218, 166, 135, 304, 330,
            274, 177, 243, 79, 70, 263, 347, 81, 349, 104, -2, 579,
            131, -217, 108, 83, 74, 43, 33, 90, 73, 66, 65, 52, 57,
            31, 28, 107, 81, 43, 46, 31, 13, 53, 84, 23, 27, 28, 24,
            7, 29, 109, 28, 28, 14, 14, 6, 21, 57, 9, 19, 25, -1, 0,
            32, 27, 17, 14, 18, 0, 1, 23, 11, -1, 43, 23, -3, 0, 22,
            -2, 89, 17, 14, -3, 0, 23, -13, 7, 9, 9, 0, 0])
        norm = ts_normalise_negative_values(data, extreme=2)
        self.assertFalse(any(norm < 0))
        self.assertFalse(any(numpy.isnan(norm)))
        data = pandas.DataFrame(data.reshape((-1, 1)), columns=['r'])
        norm = ts_normalise_negative_values(data['r'], extreme=2)
        self.assertFalse(any(norm < 0))
        self.assertFalse(any(numpy.isnan(norm)))

    def test_ts_remove_decreasing_values(self):
        values = numpy.array([0, 1, 10, 100, 1000, 901, 2000])
        with self.assertRaises(NotImplementedError):
            ts_remove_decreasing_values(values.astype(float))
        new_values = ts_remove_decreasing_values(values)
        assert_almost_equal(
            values, numpy.array([0, 1, 10, 100, 1000, 901, 2000]))
        assert_almost_equal(
            new_values, numpy.array([0, 1, 8, 41, 881, 901, 2000]))

        values = numpy.array([0, 1, 10, 950, 1000, 901, 2000])
        new_values = ts_remove_decreasing_values(values)
        assert_almost_equal(
            new_values, numpy.array([0, 1, 7, 873, 881, 901, 2000]))

        values = numpy.array([0, 1, 10, 950, 1000, 901, 890, 2000])
        new_values = ts_remove_decreasing_values(values)
        assert_almost_equal(
            new_values, numpy.array([0, 1, 3, 865, 868, 888, 890, 2000]))


if __name__ == '__main__':
    unittest.main()
