# coding: utf-8
"""
Loads data from :epkg:`INSEE`.
"""
from pandas import to_datetime
from .pandas_cache import read_csv_cache, geo_read_csv_cache


def data_france_departments(cache='dep_france', metropole=False):
    """
    Retrieves data from
    `Contours géographiques des départements
    <https://www.data.gouv.fr/en/datasets/
    contours-geographiques-des-departements/>`_.

    :param metropole: only for the metropole
    :param cache: cache name
    :return: geodataframe
    """
    url = ("https://www.data.gouv.fr/en/datasets/r/"
           "ed02b655-4307-4db4-b1ca-7939145dc20f")
    df = geo_read_csv_cache(cache, url)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    if metropole:
        codes = [_ for _ in set(df.code_depart) if len(_) < 3]
        return df[df.code_depart.isin(codes)]
    return df


def data_covid_france_departments_hospitals(
        cache='covid_france_hosp', metropole=False):
    """
    Retrieves data from
    `Données hospitalières relatives à l'épidémie de COVID-19
    <https://www.data.gouv.fr/fr/datasets/
    donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/>`_.

    :param cache: cache name
    :param metropole: only for the metropole
    :return: dataframe
    """
    url = ("https://www.data.gouv.fr/fr/datasets/r/"
           "63352e38-d353-4b54-bfd1-f1b3ee1cabd7")
    df = read_csv_cache(cache, url, sep=';')
    df['jour'] = to_datetime(df['jour'])
    if metropole:
        codes = [_ for _ in set(df.dep) if len(_) < 3]
        return df[df.dep.isin(codes)]
    return df


def data_covid_france_departments_tests(
        cache='covid_france_test',
        metropole=False):
    """
    Retrieves data from
    `Données relatives aux résultats des tests virologiques COVID-19 SI-DEP
    <https://www.data.gouv.fr/fr/datasets/
    donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/>`_.

    :param cache: cache name
    :param metropole: only for the metropole
    :return: geodatafrale
    """
    def trylen(v):
        try:
            return len(v)
        except TypeError as e:
            raise TypeError("Issue with '{}'".format(v)) from e
    url = ("https://www.data.gouv.fr/fr/datasets/r/"
           "406c6a23-e283-4300-9484-54e78c8ae675")
    df = read_csv_cache(cache, url, sep=';')
    df['jour'] = to_datetime(df['jour'])
    df['dep'] = df.dep.astype(str)
    if metropole:
        codes = [_ for _ in set(df.dep) if trylen(_) < 3]
        return df[df.dep.isin(codes)]
    return df
