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
    def test_data_covid_france_departments_hospitals(self):
        cache = "temp_hosp"
        df = data_covid_france_departments_hospitals(cache, metropole=True)
        self.assertEqual(
            list(
                df.columns), [
                'dep', 'sexe', 'jour', 'hosp', 'rea', 'rad', 'dc'])
        df = data_covid_france_departments_hospitals(cache)
        self.assertEqual(
            list(
                df.columns), [
                'dep', 'sexe', 'jour', 'hosp', 'rea', 'rad', 'dc'])

    @skipif_appveyor("connectivity issue")
    @ignore_warnings(DtypeWarning)
    def test_data_covid_france_departments_tests(self):
        cache = "temp_tests"
        df = data_covid_france_departments_tests(cache, metropole=True)
        self.assertEqual(
            list(
                df.columns), [
                'dep', 'jour', 'P', 'T', 'cl_age90', 'pop'])
        df = data_covid_france_departments_tests(cache)
        self.assertEqual(
            list(
                df.columns), [
                'dep', 'jour', 'P', 'T', 'cl_age90', 'pop'])

    @skipif_appveyor("connectivity issue")
    def test_data_france_departments(self):
        cache = "temp_dep"
        df = data_france_departments(cache, metropole=True)
        self.assertEqual(
            list(sorted(df.columns)),
            list(sorted(['code_depart', 'departement', 'code_region',
                         'region', 'code_ancien', 'ancienne_re',
                         'geometry'])))
        df = data_france_departments(cache)
        self.assertEqual(
            list(sorted(df.columns)),
            list(sorted(['code_depart', 'departement', 'code_region',
                         'region', 'code_ancien', 'ancienne_re',
                         'geometry'])))


if __name__ == '__main__':
    unittest.main()
