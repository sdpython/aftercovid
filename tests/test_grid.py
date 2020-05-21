"""
Unit tests for ``data``.
"""
import unittest
from numpy.testing import assert_almost_equal
from aftercovid.models import CovidSIR
from aftercovid.grid import GridMap


class TestGrid(unittest.TestCase):

    def test_grid_sir_france(self):
        grid = GridMap(CovidSIR())
        met = grid.metric()
        self.assertEqual(met.shape, (10, 12))
        assert_almost_equal(grid.grid * 9990., met)


if __name__ == '__main__':
    unittest.main()
