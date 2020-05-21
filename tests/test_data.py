"""
Unit tests for ``data``.
"""
import unittest
from aftercovid.data import load_grid_image


class TestData(unittest.TestCase):

    def test_france(self):
        img = load_grid_image()
        self.assertEqual(img.shape, (356, 386))
        self.assertEqual(img.min(), 0)
        self.assertEqual(img.max(), 1)
        su = img.sum()
        self.assertEqual(su, 72987)


if __name__ == '__main__':
    unittest.main()
