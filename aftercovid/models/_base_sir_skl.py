# coding: utf-8
"""
Common methods about training, predicting for :epkg:`SIR` models.
"""
import numpy
# from sympy import Symbol, diff as sympy_diff


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

    def fit(self, X, y, t=0):
        """
        Fits a model :class:`BaseSIR <aftercovid.models._base_sir.BaseSIR>`.

        :param X: known values for every quantity at time *t*,
            every column is mapped to the list returned by
            :meth:`quantity_names <aftercovid.models._base_sir.quantity_names>`
        :param y: known derivative for every quantity at time *t*,
            comes in the same order as *X*
        :param t: implicit feature
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
        clq, N = self._check_fit_predict(X, y)
        self['N'] = N

        # compute derivatives
        pos = {n: i for i, n in enumerate(clq)}
        train = numpy.empty((X.shape[0], ), dtype=X.dtype)
        assert train is not None
        assert pos is not None
        raise NotImplementedError("Fit method must be rewritten.")

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
