"""
Preprocesses timeseries about COVID.
"""
import numpy


def ts_moving_average(series, n=7, center=True):
    """
    Computes the moving average of a differential series.
    The function handles nan as well. The outputs
    does not contain any nan unless there are too many
    consecutive nans.

    :param series: timeseries
    :param n: window
    :param center: centered average
    :return: moving average (of same size)
    """
    if hasattr(series, 'values'):
        cls = series.__class__
        columns = getattr(series, 'columns', None)
        name = getattr(series, 'name', None)
        index = series.index
        series = series.values
        as_df = True
    else:
        as_df = False
        cls = None

    if center and n % 2 != 1:
        raise ValueError("If center is True, n should be odd.")

    dtype = numpy.float64 if series.dtype != numpy.float32 else numpy.float32

    series = series.astype(dtype)
    weights = numpy.ones(series.shape, dtype=dtype)
    isna = numpy.isnan(series)
    weights[isna] = 0
    series[isna] = 0

    ret = numpy.cumsum(series.astype(dtype), axis=0)
    wet = numpy.cumsum(weights.astype(dtype), axis=0)
    res = numpy.zeros(ret.shape, dtype)
    if center:
        d = n // 2
        res[d + 1:-d] = (ret[n:] - ret[:-n]) / (wet[n:] - wet[:-n])
        for i in range(0, d + 1):
            res[i] = numpy.divide(ret[i + d - 1], wet[i + d - 1])
            res[-i - 1] = numpy.divide(ret[-1] - ret[-(i + d) - 1],
                                       wet[-1] - wet[-(i + d) - 1])
    else:
        res[n:] = (ret[n:] - ret[:-n]) / (wet[n:] - wet[:-n])
        for i in range(0, n):
            res[i] = numpy.divide(ret[i], wet[i])

    if as_df:
        if columns is not None:
            return cls(series, columns=columns, index=index)
        if name is not None:
            return cls(series, name=name, index=index)
    return res


def ts_normalise_negative_values(series, n=7, extreme=4):
    """
    *series* is a differential series which should not
    have any negative values. The function removes
    unexpected high value and negative value. These extremes
    are replaced by a local average.
    The function handles nan as well. The outputs
    does not contain any nan unless there are too many
    consecutive nans.

    :param series: differential values
    :param n: moving average
    :param extreme: removes extreme values,
        if the series is higher or lower than its moverage * th or / th
    :return: corrected series
    """
    if hasattr(series, 'values'):
        cls = series.__class__
        columns = getattr(series, 'columns', None)
        name = getattr(series, 'name', None)
        index = series.index
        series = series.values
        as_df = True
    else:
        as_df = False

    mov = ts_moving_average(series, n=n, center=True)
    series = series.astype(mov.dtype)
    isna = numpy.isnan(series)
    series_raw = series
    series = series.copy().astype(mov.dtype)
    series[isna] = 0
    total = numpy.sum(series, axis=0)
    rep = (numpy.isnan(series_raw) | (series < 0) |
           (mov / extreme > series) | (mov * extreme < series))
    series[rep] = mov[rep]
    nonan = series.copy()
    isna = numpy.isnan(nonan)
    nonan[isna] = 0
    series = numpy.maximum(series, 0)
    new_total = numpy.sum(nonan, axis=0)
    series *= total / new_total
    if as_df:
        if columns is not None:
            return cls(series, columns=columns, index=index)
        if name is not None:
            return cls(series, name=name, index=index)
    return series
