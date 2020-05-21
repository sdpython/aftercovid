"""
Unit tests for ``data``.
"""
import unittest
from aftercovid.data import load_grid_image
from aftercovid.data.image_helper import reduce_image


class TestData(unittest.TestCase):

    def test_france(self):
        img = load_grid_image()
        self.assertEqual(img.shape, (356, 386))
        self.assertEqual(img.min(), 0)
        self.assertEqual(img.max(), 1)
        su = img.sum()
        self.assertEqual(su, 72987)

    def test_reduce_img(self):
        img = load_grid_image()
        small = reduce_image(img, (8, 10))
        self.assertEqual(small.shape, (8, 10))
        self.assertEqual(small.min(), 0)
        self.assertEqual(small.max(), 1)


if __name__ == '__main__':
    unittest.main()
