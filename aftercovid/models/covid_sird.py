# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
import numpy.random
from ._base_sir import BaseSIR


class CovidSIRD(BaseSIR):
    """
    Inspiration `Modelling some COVID-19 data
    <http://webpopix.org/covidix19.html>`_.

    .. runpython::
        :showcode:
        :rst:

        from aftercovid.models import CovidSIRD

        model = CovidSIRD()
        print(model.to_rst())

    .. exref::
        :title: SIRD simulation and plotting

        .. plot::

            from pandas import DataFrame
            import matplotlib.pyplot as plt
            from aftercovid.models import CovidSIRD

            model = CovidSIRD()
            sims = list(model.iterate(60))
            df = DataFrame(sims)
            print(df.head())
            ax = df.plot(y=['S', 'I', 'R', 'D'], kind='line')
            ax.set_xlabel("jours")
            ax.set_ylabel("population")
            r0 = model.R0()
            ax.set_title("Simulation SIRD\\nR0=%f" % r0)

            plt.show()

    Visual representation:

    .. gdot::
        :script:

        from aftercovid.models import CovidSIRD
        model = CovidSIRD()
        print(model.to_dot())

    See :ref:`l-base-model-sir` to get the methods
    common to SIRx models.
    """

    P0 = [
        ('beta', 0.5, 'taux de transmission dans la population'),
        ('mu', 1 / 14., '1/. : durée moyenne jusque la guérison'),
        ('nu', 1 / 21., '1/. : durée moyenne jusqu\'au décès'),
    ]

    Q0 = [
        ('S', 9990., 'personnes non contaminées'),
        ('I', 10., 'nombre de personnes malades ou contaminantes'),
        ('R', 0., 'personnes guéries (recovered)'),
        ('D', 0., 'personnes décédées'),
    ]

    C0 = [
        ('N', 10000., 'population'),
    ]

    eq = {
        'S': '- beta * S / N * I',
        'I': 'beta * S / N * I - mu * I - nu * I',
        'R': 'mu * I',
        'D': 'nu * I'
    }

    def __init__(self):
        BaseSIR.__init__(
            self,
            p=CovidSIRD.P0.copy(),
            q=CovidSIRD.Q0.copy(),
            c=CovidSIRD.C0.copy(),
            eq=CovidSIRD.eq.copy())

    def R0(self, t=0):
        '''
        Returns R0 coefficient.

        :param t: unused

        This coefficients tells how many people one
        contagious infects in average.
        Let's consider the contagious population :math:`I_t`.
        A time *t+1*, there are :math:`\\beta I_t` new cases
        and :math:`(\\nu + \\mu)I_t` disappear. So at time
        *t+2*, there :math:`\\beta I_t(1 - \\nu - \\mu)`
        new cases, at time *t+d+1*, it is
        :math:`\\beta I_t(1 - \\nu - \\mu)^d`.
        We finally find that:
        :math:`R_0=\\beta \\sum_d (1 - \\nu - \\mu)^d` and
        :math:`R_0=\\frac{\\beta}{\\nu + \\mu}`.
        '''
        return self['beta'] / (self['nu'] + self['mu'])

    def correctness(self, X=None):
        """
        Unused.
        """
        if X is None:
            X = self.vect().reshape((1, -1))
        return numpy.zeros(X.shape)

    def rnd(self):
        '''
        Draws random parameters.
        Not perfect.
        '''
        self['beta'] = numpy.random.randn(1) * 0.1 + 0.5
        self['mu'] = numpy.random.randn(1) * 0.1 + 1. / 14
        self['nu'] = numpy.random.randn(1) * 0.1 + 1. / 21

    @staticmethod
    def add_noise(X, epsilon=1.):
        """
        Tries to add reasonable noise to the quantities stored in *X*.

        :param epsilon: amplitude
        :return: new X
        """
        rnd = numpy.random.randn(*X.shape) * epsilon + 1.
        rnd = numpy.maximum(rnd, 0)

        X2 = X.copy().astype(numpy.float64)
        grad = (X2[1:] - X2[:-1])
        grad[:, 3] *= rnd[1:, 3]
        grad[:, 2] *= (rnd[1:, 2] - 1) / 5 + 1.

        X2[1:] = X2[0, :] + numpy.cumsum(grad, axis=0)

        fact = numpy.multiply(numpy.sum(X2, axis=1), 1. / numpy.sum(X, axis=1))
        X2 = numpy.multiply(X2, fact.reshape(X.shape[0], 1))
        delta = numpy.sum(X, axis=1) - numpy.sum(X2, axis=1)
        delta /= X.shape[1]
        X2 = numpy.add(X2, delta.reshape(-1, 1))
        return X2
