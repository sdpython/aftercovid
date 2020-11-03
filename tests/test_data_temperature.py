"""
Unit tests for ``data``.
"""
import unittest
from aftercovid.data import load_temperatures


class TestDataHopkins(unittest.TestCase):

    def test_data_temperature(self):
        with self.assertRaises(ValueError):
            load_temperatures('Belgium')
        df = load_temperatures()
        self.assertEqual(df.shape[1], 7)
        df.to_excel('t.xlsx', index=False)
        maxmo = df[['year', 'month', 'day']].groupby(
            ['year', 'month'], as_index=False).max()
        by = {}
        for row in maxmo.to_dict(orient='records'):
            by[row['year'], row['month']] = row['day']
        self.assertEqual(by[2020, 1], 31)
        self.assertEqual(by[2020, 2], 29)
        self.assertEqual(by[2020, 3], 31)
        self.assertEqual(by[2020, 4], 30)
        self.assertEqual(by[2020, 5], 31)
        self.assertEqual(by[2020, 6], 30)
        self.assertEqual(by[2020, 7], 31)
        self.assertEqual(by[2020, 8], 31)


if __name__ == '__main__':
    unittest.main()
