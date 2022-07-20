# coding: utf-8
"""
Optimizes a model over true data.
"""
import warnings
import numpy
import pandas
from .epidemic_regressor import EpidemicRegressor


def find_best_model(Xt, yt, lrs, stop_loss, verbose=0,
                    init=None, model_name='SIRD',
                    max_iter=500):
    """
    Finds the best model over a short period of time.

    :param Xt: matrix Nx4 with times series
    :param Yt: matrix Nx4 with expected differentiated time series
    :param lrs: learning rates to try
    :param stop_loss: stops trying other learning rate if the loss is below
        this threshold
    :param verbose: display progress information (uses `print`)
    :param init: initialized model to start the optimisation from
        this parameters
    :param model_name: name of the model to optimize, `'SIRD'`, `'SIRDc'`,
        see :class:`EpidemicRegressor <aftercovid.models.EpidemicRegressor>`
    :param max_iter: maximum number of iterator to train every model
    :return: (best model, best loss, best learning rate)
    """
    best_est, best_loss, best_lr = None, None, None
    m = None
    for ilr, lr in enumerate(lrs):
        if verbose:
            print(  # pragma: no cover
                f"--- TRY {ilr + 1}/{len(lrs)}: {lr}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EpidemicRegressor(
                model_name, learning_rate_init=lr, max_iter=max_iter,
                early_th=stop_loss, verbose=verbose, init=m)
            try:
                m.fit(Xt, yt)
            except RuntimeError as e:  # pragma: no cover
                if verbose:
                    print(f'ERROR: {e}')
                continue
            loss = m.score(Xt, yt)
            if numpy.isnan(loss):
                continue  # pragma: no cover
        if best_est is None or best_loss > loss:
            best_est = m
            best_loss = loss
            best_lr = lr
        if best_loss < stop_loss:
            return best_est, best_loss, best_lr  # pragma: no cover
    return best_est, best_loss, best_lr


def rolling_estimation(X, y,
                       lrs=(1e8, 1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6),
                       delay=21, stop_loss=1, init=None,
                       model_name='SIRD', max_iter=500, verbose=0,
                       dates=None):
    """
    Estimates a model over a rolling windows whose size is
    *delay* days. See :ref:`l-example-rolling-estimation` to
    see how to use that function.

    :param Xt: matrix Nx4 with times series
    :param Yt: matrix Nx4 with expected differentiated time series
    :param lrs: learning rates to try
    :param delay: size of the rolling window
    :param stop_loss: stops trying other learning rate if the loss is below
        this threshold
    :param verbose: display progress information (uses `print`)
    :param init: initialized model to start the optimisation from
        this parameters
    :param model_name: name of the model to optimize, `'SIRD'`, `'SIRDc'`,
        see :class:`EpidemicRegressor <aftercovid.models.EpidemicRegressor>`
    :param max_iter: maximum number of iterator to train every model
    :param dates: dates for every row in X, can be None
    :return: (results, last best model)
    """
    coefs = []
    m = None
    kdates = (
        list(range(0, X.shape[0] - delay + 1 - 28, 7)) +
        list(range(X.shape[0] - delay + 1 - 28,
                   X.shape[0] - delay + 1 - 7, 2)) +
        list(range(X.shape[0] - delay + 1 - 7,
                   X.shape[0] - delay + 1, 1)))
    kdates = [d for d in kdates if d > 0]
    for k in kdates:
        end = min(k + delay, X.shape[0])
        Xt, yt = X[k:end], y[k:end]
        if any(numpy.isnan(Xt.ravel())) or any(numpy.isnan(yt.ravel())):
            continue  # pragma: no cover
        m, loss, lr = find_best_model(
            Xt, yt, lrs, stop_loss, init=m, model_name=model_name,
            max_iter=max_iter)
        if m is None:
            if verbose:  # pragma: no cover
                print(f"k={k} loss=nan")
            find_best_model(
                Xt, yt, [1e8, 1e6, 1e4, 1e2, 1,
                         1e-2, 1e-4, 1e-6], 10,
                init=m, verbose=True)
            continue
        loss = m.score(Xt, yt)
        loss_l1 = m.score(Xt, yt, 'l1')
        if verbose:
            print("k={} iter={} loss={:1.3f} l1={:1.3g} coef={} R0={} "
                  "lr={} cn={}".format(
                      k, m.iter_, loss, loss_l1,
                      m.model_._val_p, m.model_.R0(), lr,
                      m.model_.correctness().sum()))
        obs = dict(k=k, loss=loss, loss_l1=loss_l1, it=m.iter_,
                   R0=m.model_.R0(), lr=lr,
                   correctness=m.model_.correctness().sum(),
                   date=k if dates is None else dates[end - 1])
        obs.update({k: v for k, v in zip(
            m.model_.param_names, m.model_._val_p)})
        coefs.append(obs)

    dfcoef = pandas.DataFrame(coefs)
    dfcoef = dfcoef.set_index("date")
    return dfcoef, m
