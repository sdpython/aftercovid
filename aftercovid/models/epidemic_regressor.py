# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
import numpy
from sklearn.base import BaseEstimator, RegressorMixin
from .covid_sird import CovidSIRD
from .covid_sird_cst import CovidSIRDc


class EpidemicRegressor(BaseEstimator, RegressorMixin):
    """
    Follows :epkg:`scikit-learn` API.
    Trains a model on observed data from an epidemic.

    :param model: model to train, `'SIR'` refers to
        `CovidSIRD <aftercovid.models.CovidSIRD>`,
        `SIRDc` refers to `CovidSIRDc
        <aftercovid.models.CovidSIRDc>`
    :param t: implicit feature
    :param max_iter: number of iteration
    :param learning_rate_init: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`
    :param lr_schedule: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`
    :param momentum: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`
    :param power_t: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`
    :param early_th: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`
    :param verbose: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`
    :param min_threshold: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`, if `'auto'`,
        the value depends on the models, if is `0.01`
        for model `SIR`, it means every coefficient must
        be greater than 0.01.
    :param max_threshold: see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`, upper bound
    :param init: dictionary, initializes the model
        with this parameters

    Once trained the model holds a member `model_`
    which contains the trained model and `iter_`
    which holds the number of training iteration.
    It also keep track of the coefficients in a dictionary
    in attribute `coef_`.
    """

    def __init__(self, model='SIR', t=0, max_iter=100,
                 learning_rate_init=0.1, lr_schedule='constant',
                 momentum=0.9, power_t=0.5, early_th=None,
                 min_threshold='auto', max_threshold='auto',
                 verbose=False, init=None):
        if init is not None:
            if isinstance(init, EpidemicRegressor):
                if hasattr(init, 'coef_'):
                    init = init.coef_.copy()
                else:
                    init = None
            elif not isinstance(init, dict):
                raise TypeError(
                    "init must be a dictionary not {}.".format(type(init)))
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)
        self.t = t
        self.model = model
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.power_t = power_t
        self.early_th = early_th
        self.verbose = verbose
        if min_threshold == 'auto':
            if model.upper() in ('SIR', 'SIRD'):
                min_threshold = 0.0001
            elif model.upper() in ('SIRC', 'SIRDC'):
                pmin = dict(beta=0.001, nu=0.0001, mu=0.0001,
                            a=-1., b=0., c=0.)
                min_threshold = numpy.array(
                    [pmin[k[0]] for k in CovidSIRDc.P0])
        if max_threshold == 'auto':
            if model.upper() in ('SIR', 'SIRD'):
                max_threshold = 1.
            elif model.upper() in ('SIRC', 'SIRDC'):
                pmax = dict(beta=1., nu=0.5, mu=0.5,
                            a=0., b=4., c=2.)
                max_threshold = numpy.array(
                    [pmax[k[0]] for k in CovidSIRDc.P0])
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._get_model()
        self.init = init
        if init is not None:
            self.coef_ = init

    def _get_model(self):
        if self.model.lower() in ('sir', 'sird'):
            return CovidSIRD()
        if self.model.lower() in ('sirc', 'sirdc'):
            return CovidSIRDc()
        raise ValueError(
            "Unknown model name '{}'.".format(self.model))

    def fit(self, X, y):
        """
        Trains a model to approximate its derivative as much as
        possible.
        """
        if not hasattr(self, 'model_'):
            self.model_ = self._get_model()
            self.model_.rnd()
        total = numpy.sum(X, axis=1)
        mi, ma = total.min(), total.max()
        err = (ma - mi) / mi
        if err > 1e-5:
            raise RuntimeError(  # pragma: no cover
                "Population is not constant, in [{}, {}].".format(
                    mi, ma))
        if self.init is not None:
            for k, v in self.init.items():
                self.model_[k] = v
        self.model_['N'] = (ma + mi) / 2
        self.model_.fit(
            X, y, learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter, early_th=self.early_th,
            verbose=self.verbose, lr_schedule=self.lr_schedule,
            power_t=self.power_t, momentum=self.momentum,
            min_threshold=self.min_threshold,
            max_threshold=self.max_threshold)
        self.iter_ = self.model_.iter_
        self.coef_ = self.model_.params_dict
        return self

    def predict(self, X):
        """
        Predicts the derivatives.
        """
        if not hasattr(self, 'model_'):
            raise RuntimeError("Model was not trained.")
        return self.model_.predict(X)

    def simulate(self, X, n=7):
        """
        Predicts and simulates the epidemics.
        Every row of *X* is a starting point,
        the function then simulates the epidemics for the next
        *n* days for every starting point.

        :param X: data
        :param n: number of days
        :return: quantities, matrix of shape
            *(X.shape[0], n, number of parameters)*
        """
        if not hasattr(self, "model_"):
            raise RuntimeError(  # pragma: no cover
                "Model was not trained.")
        clq = self.model_.quantity_names
        if len(clq) != X.shape[1]:
            raise RuntimeError(  # pragma: no cover
                "Unapexected shape for X ({}), expecting {} columns."
                "".format(X.shape, len(clq)))
        res = None
        for i in range(X.shape[0]):
            for k, v in zip(clq, X[i]):
                self.model_[k] = v
            pred = self.model_.iterate2array(n=n, derivatives=False)
            if res is None:
                res = numpy.zeros((X.shape[0], ) + pred.shape)
            res[i, :, :] = pred
        return res

    def predict_many(self, X, n=7):
        """
        Predicts the derivatives and the series
        for many days.

        :param X: series
        :param n: number of days
        :return: derivates and series, return shape is
            *(X.shape[0], number of parameters, n)*
        """
        if not hasattr(self, 'model_'):
            raise RuntimeError("Model was not trained.")  # pragma: no cover
        deri = numpy.empty(X.shape + (n, ))
        curv = numpy.empty(X.shape + (n, ))
        for i in range(0, n):
            d = self.predict(X)
            deri[:, :, i] = d
            X += d
            curv[:, :, i] = X
        return deri, curv

    def score(self, X, y, norm=None):
        """
        Scores the prediction of the derivatives.

        :param X: data
        :param y: expected derivatives
        :param norm: norm to return the norm used to optimize (L2)
            or 'L1' to return the L1 norm
        :return: score
        """
        if not hasattr(self, 'model_'):
            raise RuntimeError(  # pragma: no cover
                "Model was not trained.")
        if norm is None:
            return self.model_.score(X, y)
        if norm.lower() == 'l1':
            return self.model_.score_l1(X, y)
        raise ValueError("Unexpected norm %r." % norm)
