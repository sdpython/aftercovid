"""
Unit tests for ``data``.
"""
import unittest
from aftercovid.data import (
    data_covid_france_departments_hospitals,
    data_covid_france_departments_tests,
    data_france_departments)
from pyquickhelper.pycode import skipif_appveyor, ignore_warnings
from pandas.errors import DtypeWarning


class TestDataInsee(unittest.TestCase):

    @skipif_appveyor("connectivity issue")
    @ignore_warnings((DtypeWarning, DeprecationWarning))
    def test_data_covid_france_departments_hospitals(self):
        cache = "temp_hosp"
        df = data_covid_france_departments_hospitals(cache, metropole=True)
        exp_cols = ['dep', 'sexe', 'jour', 'hosp', 'rea', 'HospConv',
                    'SSR_USLD', 'autres', 'rad', 'dc']
        self.assertEqual(list(df.columns), exp_cols)
        df = data_covid_france_departments_hospitals(cache)
        self.assertEqual(list(df.columns), exp_cols)

    @skipif_appveyor("connectivity issue")
    @ignore_warnings((DtypeWarning, DeprecationWarning))
    def test_data_covid_france_departments_tests(self):
        cache = "temp_tests"
        df = data_covid_france_departments_tests(cache, metropole=True)
        exp_cols = ['dep', 'jour', 'P', 'T', 'cl_age90', 'pop']
        self.assertEqual(list(df.columns), exp_cols)
        df = data_covid_france_departments_tests(cache)
        self.assertEqual(list(df.columns), exp_cols)

    @skipif_appveyor("connectivity issue")
    @ignore_warnings((DtypeWarning, DeprecationWarning))
    def test_data_france_departments(self):
        cache = "temp_dep"
        df = data_france_departments(cache, metropole=True)
        exp_cols = ['code_depart', 'departement', 'code_region',
                    'region', 'code_ancien', 'ancienne_re', 'geometry']
        self.assertEqual(list(sorted(df.columns)), list(sorted(exp_cols)))
        df = data_france_departments(cache)
        self.assertEqual(list(sorted(df.columns)), list(sorted(exp_cols)))


if __name__ == '__main__':
    unittest.main()
