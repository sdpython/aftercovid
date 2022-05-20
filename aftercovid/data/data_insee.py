# coding: utf-8
"""
Loads data from :epkg:`INSEE`.
"""
import os
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
    backup = os.path.join(os.path.dirname(__file__),
                          "data_france_dep.geojson")
    df = geo_read_csv_cache(cache, url, backup=backup)
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
    `Données de laboratoires pour le dépistage
    (A COMPTER DU 18/05/2022) - SI-DEP
    <https://www.data.gouv.fr/fr/datasets/
    donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/>`_.

    :param cache: cache name
    :param metropole: only for the metropole
    :return: geodatafrale
    """
    def trylen(v):
        try:
            return len(v)
        except TypeError as e:  # pragma: no cover
            raise TypeError("Issue with '{}'".format(v)) from e
    url = ("https://www.data.gouv.fr/fr/datasets/r/"
           "674bddab-6d61-4e59-b0bd-0be535490db0")
    df = read_csv_cache(cache, url, sep=';')
    df['jour'] = to_datetime(df['jour'])
    df['dep'] = df.dep.astype(str)

    def plus0(s):
        if len(s) < 2:
            return "0" * (2 - len(s)) + s
        return s

    df['dep'] = df['dep'].apply(plus0)

    if metropole:
        codes = [_ for _ in set(df.dep) if trylen(_) < 3]
        return df[df.dep.isin(codes)]
    return df
