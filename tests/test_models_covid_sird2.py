"""
Unit tests for ``CovidSIRD2``.
"""
import unittest
from aftercovid.models import CovidSIRD2


class TestModelsCovidSir2(unittest.TestCase):

    def test_covid_sir(self):
        with self.assertRaises(NotImplementedError):
            CovidSIRD2()


if __name__ == '__main__':
    unittest.main()
