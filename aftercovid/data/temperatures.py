"""
Loads data about temperatures.
"""
import os
import numpy
import pandas


def load_temperatures(country='France'):
    """
    Loads a dataframe containing temperatures.
    :param name: picture name

    Source:

    * `temperature_france.xlsx`:
      `meteociel <https://www.meteociel.fr/climatologie/obs_villes.php?
      code2=75107005&mois=11&annee=2020>`_
    """
    this = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(this, "temperature_2020_%s.xlsx" % country.lower())
    if not os.path.exists(filename):
        raise ValueError(
            "Unable to load data for country %r." % country)

    def to_float(val, c, cls=float):
        if val == '---':
            return numpy.nan
        if isinstance(val, (str, numpy.str)):
            return cls(val.split()[c])
        return val

    def _process(df, month):
        if len(df.columns) != 5:
            raise ValueError(  # pragma: no cover
                "Unexpected number of columns %r for month %r." % (
                    df.columns, month))

        df.columns = ["day", "tmax", "tmin", "rain", "sun"]
        df['day'] = df['day'].apply(lambda c: to_float(c, -1, int))
        df['tmax'] = df['tmax'].apply(lambda c: to_float(c, 0))
        df['tmin'] = df['tmin'].apply(lambda c: to_float(c, 0))
        return df

    dfs = []
    for month in range(1, 13):
        sheet = "%02d" % month
        df = pandas.read_excel(filename, sheet_name=sheet, header=1)
        if df.shape[0] == 0:
            continue
        df = _process(df, month)
        df['month'] = month
        df['year'] = 2020
        dfs.append(df)
    res = pandas.concat(dfs)
    res = res[(~res['tmin'].isna()) & (~res['day'].isna())].copy()
    return res
