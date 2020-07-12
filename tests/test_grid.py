"""
Unit tests for ``data``.
"""
import unittest
from numpy.testing import assert_almost_equal
from aftercovid.models import CovidSIRD
from aftercovid.grid import GridMapSIR


class TestGrid(unittest.TestCase):

    def test_grid_sir_france(self):
        grid = GridMapSIR(CovidSIRD())
        met = grid.metric()
        self.assertEqual(met.shape, (10, 12))
        assert_almost_equal(grid.grid * 9990., met)
        su = grid['S']
        self.assertEqual(su, met.sum())
        self.assertEqual(su, 649350.0)


if __name__ == '__main__':
    unittest.main()
