"""
Loads data from :epkg:`CSSE Johns Hopkins`.
"""
import numpy
import pandas
from ..preprocess import ts_normalise_negative_values, ts_moving_average


population = {
    'Belgium': 11.5e6,
    'France': 67e6,
    'Germany': 83e6,
    'Spain': 47e6,
    'Italy': 60e6,
    'UK': 67e6,
}


def download_hopkins_data(kind='deaths', country='France'):
    """
    Downloads data from :epkg:`CSSE Johns Hopkins`
    for a particular country.

    :param kind: `'deaths'`, `'confirmed'` or `'recovered'`
    :param country: `'France'`, `'UK'`, ...
    :return: dataframe

    .. runpython::
        :showcode:

        from aftercovid.data import download_hopkins_data
        df = download_hopkins_data()
        print(df.tail())
    """
    url = (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
        "master/csse_covid_19_data/"
        "csse_covid_19_time_series/time_series_covid19_%s_global.csv" %
        kind)
    df = pandas.read_csv(url)
    eur = df[df['Country/Region'].isin([country])
             & df['Province/State'].isna()]
    tf = eur.T.iloc[4:]
    tf.columns = [kind]
    return tf


def extract_hopkins_data(kinds=('deaths', 'confirmed', 'recovered'),
                         country='France', delay=21):
    """
    Downloads data from :epkg:`CSSE Johns Hopkins` and infers
    the number of current positive cases in a very simple way.

    :param kinds: series to extracts, by default
        `('deaths', 'confirmed', 'recovered')`
    :param country: `'France'`, `'UK'`, ...
    :param delay: the function assumes after 21 days, a confirmed
        case moves is not positive anymore
    :return: dataframe

    .. runpython::
        :showcode:

        from aftercovid.data import extract_hopkins_data
        df = extract_hopkins_data()
        print(df.tail())
    """
    total = population[country]
    dfs = []
    for k in kinds:
        df = download_hopkins_data(k, country)
        dfs.append(df)
    conc = pandas.concat(dfs, axis=1)
    infected = conc['confirmed'] - (conc['deaths'] + conc['recovered'])
    conf30 = infected[:-delay]
    recovered = conc['recovered'].values.copy()
    recovered[delay:] += conf30
    delta_conf = conc['confirmed'].values[1:] - conc['confirmed'].values[:-1]
    infected = conc['confirmed'].values * 0
    infected[:] = conc['confirmed'] - (conc['deaths'] + recovered)
    infected[1:] = numpy.maximum(1, numpy.maximum(infected[1:], delta_conf))
    infected[20:] = numpy.maximum(10, infected[20:])
    infected[60:] = numpy.maximum(100, infected[60:])
    conc['recovered'] = recovered
    conc['infected'] = infected
    conc['safe'] = total - conc.drop('confirmed', axis=1).sum(axis=1)
    return conc


def preprocess_hopkins_data(df):
    """
    Improves the differentiated series by removing negative values.

    :param df: dataframe returned by :func:`extract_hopkins_data
        <aftercovid.data.extract_hopkins_data>`
    :return: (smoothed differentiated series,
        preprocessed dataframe)
    """
    total = df.drop('confirmed', axis=1).sum(axis=1)
    total = list(total)[0]
    diff = df.diff()
    diff['deaths'] = ts_normalise_negative_values(diff['deaths'], extreme=2)
    diff['recovered'] = ts_normalise_negative_values(
        diff['recovered'], extreme=2)
    diff['confirmed'] = ts_normalise_negative_values(
        diff['confirmed'], extreme=2)
    mov = ts_moving_average(diff, n=7, center=True)
    df2 = mov.cumsum()
    df2['safe'] = total - df2.drop(['confirmed', 'safe'], axis=1).sum(axis=1)
    return mov, df2
