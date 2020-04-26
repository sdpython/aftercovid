# coding: utf-8
"""
Common methods about training, predicting for :epkg:`SIR` models.
"""
import pprint
import numpy
from sympy import Symbol, diff as sympy_diff
from sympy.core.numbers import Zero
from ..optim import SGDOptimizer


class BaseSIRSklAPI:
    """
    Common methods about training, predicting for :epkg:`SIR` models.
    """

    def _check_fit_predict(self, X, y=None):
        if not isinstance(X, numpy.ndarray):
            raise TypeError("X must be a numpy array.")
        if len(X.shape) != 2:
            raise ValueError("X must be a matrix.")
        clq = self.quantity_names
        if len(clq) != X.shape[1]:
            raise ValueError(
                "Unexpected number of columns, got {}, expected {}.".format(
                    X.shape[1], len(clq)))
        sums = numpy.sum(X, axis=1)
        mi, ma = sums.min(), sums.max()
        df = abs(ma - mi)
        if df > abs(ma) * 1e-3:
            raise ValueError(
                "All rows must sum up to the same amount. Current "
                "range: [{}, {}].".format(mi, max))
        if y is not None:
            if not isinstance(y, numpy.ndarray):
                raise TypeError("y must be a numpy array.")
            if y.shape != X.shape:
                raise ValueError(
                    "Unexpected shape of y, got {}, expected {}.".format(
                        y.shape, X.shape))
        return clq, ma

    def fit(self, X, y, t=0, max_iter=100,
            learning_rate_init=0.1, lr_schedule='constant',
            momentum=0.9, power_t=0.5, verbose=False):
        """
        Fits a model :class:`BaseSIR <aftercovid.models._base_sir.BaseSIR>`.

        :param X: known values for every quantity at time *t*,
            every column is mapped to the list returned by
            :meth:`quantity_names <aftercovid.models._base_sir.quantity_names>`
        :param y: known derivative for every quantity at time *t*,
            comes in the same order as *X*
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
        :param verbose: see :class:`SGDOptimizer
            <aftercovid.optim.SGDOptimizer>`
        Both *X* and *y* have the same shape.

        The training needs two steps. The first one creates a training
        datasets. The second one estimates the coefficients by using
        a stochastic gradient descent (see :class:`SGDOptimizer
        <aftercovid.optim.SGDOptimizer>`).
        Let's use a SIDR model (see :class:`CovidSIR
        <aftercovid.models.CovidSIR>`).as an example.
        Let's denote the parameters as :math:`\\Omega`
        and :math:`Z_1=S`, ...,  :math:`Z_4=R`.
        The model is defined by
        :math:`\\frac{dZ_i}{dt} = f_i(\\Omega, Z)`
        where :math:`Z=(Z_1, ..., Z_4)`.
        *y* is used to compute the expected derivates
        :math:`\\frac{dZ_i}{dt}`. The loss function is defined as

        .. math::

            L(\\Omega,Z) = \\sum_{i=1}^4 \\left( f_i(\\Omega,Z) -
            \\frac{dZ_i}{dt}\\right)^2

        Then the gradient is:

        .. math::

            \\frac{\\partial L(\\Omega,Z)}{\\partial\\Omega} =
            2 \\sum_{i=1}^4 \\frac{\\partial f_i(\\Omega,Z)}{\\partial\\Omega}
            \\left( f_i(\\Omega,Z) - \\frac{dZ_i}{dt} \\right)

        A stochastic gradient descent takes care of the rest.
        """
        self._fit(
            X,
            y,
            t=0,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            lr_schedule=lr_schedule,
            momentum=momentum,
            power_t=power_t,
            verbose=verbose)
        return self

    def _losses_sympy(self):
        """
        Builds the loss functions using :epkg:`sympy`,
        one for each quantity.
        """
        clq = self.quantity_names
        res = []
        for name in clq:
            sym = Symbol('d' + name)
            eq = self._eq[name]
            res.append((eq - sym) ** 2)
        return res

    def _grads_sympy(self):
        """
        Builds the gradient for every parameter,
        exactly number of quantities multiplied by
        the number of parameters.
        """
        losses = self._losses_sympy()
        clq = self.quantity_names
        prn = self.param_names
        res = []
        for name, eq in zip(clq, losses):
            row = []
            for pn in prn:
                pns = Symbol(pn)
                df = sympy_diff(eq, pns)
                row.append(df)
            res.append(row)
        return res

    def predict(self, X, t=0):
        """
        Predicts the derivative at time *t*.

        :param X: known values for every quantity at time *t*,
            every column is mapped to the list returned by
            :meth:`quantity_names <aftercovid.models._base_sir.quantity_names>`
        :param t: implicit feature
        :return: predictive derivative
        """
        cache = self._eval_cache()
        clq, N = self._check_fit_predict(X)
        if N != self['N']:
            raise ValueError(
                "All rows must sum up to {} not {}.".format(self['N'], N))
        pos = {n: i for i, n in enumerate(clq)}
        pred = numpy.empty(X.shape, dtype=X.dtype)

        for i in range(0, X.shape[0]):
            svalues = {self._syms['t']: t + i}
            svalues.update(cache)
            for n, v in zip(clq, X[i, :]):
                svalues[self._syms[n]] = v

            for k, v in self._eq.items():
                ev = v.evalf(subs=svalues)
                pred[i, pos[k]] = ev

        return pred

    def _fit(self, X, y, t, max_iter,
             learning_rate_init, lr_schedule,
             momentum, power_t, verbose):
        '''
        See method :meth:`fit
        <aftercovid.models._base_sir_estimation.BaseSIRSklAPI.fit>`
        and :class:`SGDOptimizer <aftercovid.optim.SGDOptimizer>`.
        '''
        clq, N = self._check_fit_predict(X, y)
        pnames = self.param_names
        self['N'] = N

        symbol_clq = [Symbol(n) for n in clq]
        symbol_d_clq = [Symbol('d' + n) for n in clq]
        symbol_N = Symbol('N')
        symbol_params = [Symbol(n) for n in pnames]

        # loss and gradients functions
        losses = self._losses_sympy()
        grads = self._grads_sympy()

        def pformat(d):
            nd = {str(k): (str(v), type(v), type(k)) for k, v in d.items()}
            return pprint.pformat(nd)

        def fct_loss(coef, X, y):
            'Computes the loss function for every X and y.'
            svalues = {symbol_N: N}
            for p, c in zip(symbol_params, coef):
                svalues[p] = c

            res = 0.
            for i in range(X.shape[0]):
                for n, v in zip(symbol_clq, X[i, :]):
                    svalues[n] = v
                for n, v in zip(symbol_d_clq, y[i, :]):
                    svalues[n] = v
                for loss in losses:
                    try:
                        res += loss.evalf(subs=svalues)
                    except (AttributeError, TypeError, IndexError) as e:
                        raise RuntimeError(
                            'Unable to calculate loss for [{}] with '
                            'values={}.'.format(
                                loss, pformat(svalues))) from e
            return res

        def fct_grad(coef, x, y):
            'Computes the gradient function for every X and y.'
            svalues = {symbol_N: N}
            for p, c in zip(symbol_params, coef):
                svalues[p] = c

            res = numpy.zeros((len(pnames), ), dtype=x.dtype)
            for n, v in zip(symbol_clq, x):
                svalues[n] = v
            for n, v in zip(symbol_d_clq, y):
                svalues[n] = v
            for row in grads:
                for i, g in enumerate(row):
                    if isinstance(g, Zero):
                        continue
                    try:
                        res[i] = g.evalf(subs=svalues)
                    except (AttributeError, TypeError, IndexError) as e:
                        raise RuntimeError(
                            'Unable to calculate gradient for [{}] with '
                            'values={}.'.format(
                                g, pformat(svalues))) from e
            return res

        coef = numpy.array([self[p] for p in pnames])
        lrn = 1. / (N ** 1.5)
        sgd = SGDOptimizer(
            coef, learning_rate_init=learning_rate_init * lrn,
            lr_schedule=lr_schedule, momentum=momentum,
            power_t=power_t)

        sgd.train(X, y, fct_loss, fct_grad, max_iter=max_iter,
                  verbose=verbose)

        # uses trained coefficients
        coef = sgd.coef
        for n, c in zip(pnames, coef):
            self[n] = c
