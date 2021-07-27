"""
Unit tests for ``data``.
"""
import unittest
import numpy
from aftercovid.data import (
    download_hopkins_data, extract_hopkins_data, preprocess_hopkins_data)


class TestDataHopkins(unittest.TestCase):

    def test_download_data(self):
        df = download_hopkins_data()
        self.assertEqual(df.shape[1], 1)
        self.assertEqual(df.columns, ['deaths'])

    def test_extract_data(self):
        df, df0 = extract_hopkins_data(raw=True)
        for c in df0.columns:
            conf = df0[c].diff().dropna().values
            if min(conf) < 0:
                raise AssertionError(
                    "Columns %r has decreasing values\n%s"
                    "" % (c, "\n".join(map(str, df0[c]))))

        logy = numpy.log(df[['infected']].astype(numpy.float64))
        diff = list(logy['infected'].diff().dropna().values[1:])
        if min(diff) < -5 or max(diff) > 5:
            raise AssertionError(
                "Wrong smoothing\n%s" % "\n".join(map(str, diff)))

        self.assertEqual(df.shape[1], 5)
        self.assertEqual(list(df.columns),
                         'deaths confirmed recovered infected safe'.split())
        diff, df = preprocess_hopkins_data(df)
        self.assertEqual(df.shape[1], 5)
        self.assertEqual(list(df.columns),
                         'deaths confirmed recovered infected safe'.split())
        self.assertEqual(diff.shape[1], 5)
        self.assertEqual(list(diff.columns),
                         'deaths confirmed recovered infected safe'.split())
        # df.to_csv('france.csv', index=True)
        logy = numpy.log(df[['infected']])
        diff = list(logy['infected'].diff().dropna().values[1:])
        if min(diff) < -5 or max(diff) > 5:
            raise AssertionError(
                "Wrong smoothing\n%s" % "\n".join(map(str, diff)))


if __name__ == '__main__':
    unittest.main()
