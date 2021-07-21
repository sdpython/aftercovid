"""
Caches a file updated every day.
"""
import os
from datetime import datetime
from urllib.error import HTTPError
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
    df = pandas.read_csv(url, **kwargs)  # pragma: no cover
    df.to_csv(ext, sep=kwargs.get('sep', ','), index=False)  # pragma: no cover
    return df  # pragma: no cover


def geo_read_csv_cache(cache, url, backup=None, **kwargs):
    """
    Checks that the data is not cached before loading it
    again.

    :param cache: filename
    :param url: data url
    :param backup: backup file (geojson),
        used when the connection has failed
    :param kwargs: see :epkg:`pandas:read_csv`
    :return:  see :epkg:`pandas:read_csv`
    """
    import geopandas
    now = datetime.now()
    ext = "%s-%04d-%02d-%02d.geojson" % (cache, now.year, now.month, now.day)
    if os.path.exists(ext):
        with open(ext, 'r', encoding='utf-8'):
            return geopandas.read_file(ext, **kwargs)
    try:  # pragma: no cover
        df = geopandas.read_file(url, **kwargs)  # pragma: no cover
    except HTTPError as e:  # pragma: no cover
        if backup is None:
            raise e
        # use a backup in case the connection failed.
        df = geopandas.read_file(backup, **kwargs)
    with open(ext, 'w', encoding='utf-8') as f:  # pragma: no cover
        f.write(df.to_json(), **kwargs)
    return df  # pragma: no cover
