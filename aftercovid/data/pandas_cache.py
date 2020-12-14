"""
Caches a file updated every day.
"""
import os
from datetime import datetime
import pandas


def read_csv_cache(cache, url, **kwargs):
    """
    Checks that the data is not cached before loading it
    again.

    :param cache: filename
    :param url: data url
    :param kwargs: see :epkg:`pandas:read_csv`
    :return:  see :epkg:`pandas:read_csv`
    """
    now = datetime.now()
    ext = "%s-%04d-%02d-%02d.csv" % (cache, now.year, now.month, now.day)
    if os.path.exists(ext):
        return pandas.read_csv(ext, **kwargs)
    df = pandas.read_csv(url, **kwargs)
    df.to_csv(ext, sep=kwargs.get('sep', ','), index=False)
    return df


def geo_read_csv_cache(cache, url, **kwargs):
    """
    Checks that the data is not cached before loading it
    again.

    :param cache: filename
    :param url: data url
    :param kwargs: see :epkg:`pandas:read_csv`
    :return:  see :epkg:`pandas:read_csv`
    """
    import geopandas
    now = datetime.now()
    ext = "%s-%04d-%02d-%02d.geojson" % (cache, now.year, now.month, now.day)
    if os.path.exists(ext):
        with open(ext, 'r', encoding='utf-8'):
            return geopandas.read_file(ext, **kwargs)
    df = geopandas.read_file(url, **kwargs)
    with open(ext, 'w', encoding='utf-8') as f:
        f.write(df.to_json(), **kwargs)
    return df
