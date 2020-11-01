"""
Unit tests for ``data``.
"""
import unittest
from aftercovid.data import (
    download_hopkins_data, extract_hopkins_data, preprocess_hopkins_data)


class TestDataHopkins(unittest.TestCase):

    def test_download_data(self):
        df = download_hopkins_data()
        self.assertEqual(df.shape[1], 1)
        self.assertEqual(df.columns, ['deaths'])

    def test_extract_data(self):
        df = extract_hopkins_data()
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


if __name__ == '__main__':
    unittest.main()
