"""
Unit tests for ``data``.
"""
import unittest
from aftercovid.data import (
    data_covid_france_departments_hospitals,
    data_covid_france_departments_tests,
    data_france_departments)


class TestDataInsee(unittest.TestCase):

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

    def test_data_covid_france_departments_tests(self):
        cache = "temp_tests"
        df = data_covid_france_departments_tests(cache, metropole=True)
        self.assertEqual(
            list(
                df.columns), [
                'dep', 'jour', 'P', 'T', 'cl_age90', 'pop.x', 'pop.y'])
        df = data_covid_france_departments_tests(cache)
        self.assertEqual(
            list(
                df.columns), [
                'dep', 'jour', 'P', 'T', 'cl_age90', 'pop.x', 'pop.y'])

    def test_data_france_departments(self):
        cache = "temp_dep"
        df = data_france_departments(cache, metropole=True)
        self.assertEqual(list(sorted(df.columns)),
                         list(sorted(['code_depart',
                                      'departement',
                                      'code_region',
                                      'region',
                                      'code_ancien',
                                      'ancienne_re',
                                      'geometry'])))
        df = data_france_departments(cache)
        self.assertEqual(list(sorted(df.columns)),
                         list(sorted(['code_depart',
                                      'departement',
                                      'code_region',
                                      'region',
                                      'code_ancien',
                                      'ancienne_re',
                                      'geometry'])))


if __name__ == '__main__':
    unittest.main()
