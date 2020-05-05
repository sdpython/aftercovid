# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
import numpy
from sklearn.base import BaseEstimator, RegressorMixin
from .covid_sir import CovidSIR


class EpidemicRegressor(BaseEstimator, RegressorMixin):
    """
    Follows :epkg:`scikit-learn` API.
    Trains a model on observed data from an epidemic.

    :param model: model to train, `'SIR'` refers to
        `CovidSIR <aftercovid.models.CovidSIR>`
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
        <aftercovid.optim.SGDOptimizer>`, the value depends on the
        models, if is `0.01` for model `SIR`, it means
        every coefficient must be greater than 0.01.

    Once trained the model holds a member`model_`
    which contains the trained model and `iter_`
    which holds the number of training iteration.
    """

    def __init__(self, model='SIR', t=0, max_iter=100,
                 learning_rate_init=0.1, lr_schedule='constant',
                 momentum=0.9, power_t=0.5, early_th=None,
                 min_threshold='auto', verbose=False):
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
            if model == 'SIR':
                min_threshold = 0.01
        self.min_threshold = min_threshold
        self._get_model()

    def _get_model(self):
        if self.model.lower() == 'sir':
            return CovidSIR()
        raise ValueError("Unknown model name '{}'.".format(self.model))

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
            raise RuntimeError(
                "Population is not constant, in [{}, {}].".format(
                    mi, ma))
        self.model_['N'] = (ma + mi) / 2
        self.model_.fit(
            X, y, learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter, early_th=self.early_th,
            verbose=self.verbose, lr_schedule=self.lr_schedule,
            power_t=self.power_t, momentum=self.momentum,
            min_threshold=self.min_threshold)
        self.iter_ = self.model_.iter_
        return self

    def predict(self, X):
        """
        Predicts the derivatives.
        """
        if not hasattr(self, 'model_'):
            raise RuntimeError("Model was not trained.")
        return self.model_.predict(X)

    def score(self, X, y):
        """
        Predicts the derivatives.
        """
        if not hasattr(self, 'model_'):
            raise RuntimeError("Model was not trained.")
        return self.model_.score(X, y)
