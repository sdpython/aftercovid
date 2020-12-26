# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
import numpy.random
from sklearn.linear_model import ElasticNet
from ._base_sir import BaseSIR
from .covid_sird import CovidSIRD


class CovidSIRDc(BaseSIR):
    """
    Inspiration `Modelling some COVID-19 data
    <http://webpopix.org/covidix19.html>`_.
    This model considers that observed data are not the
    true ones.

    .. math::

        \\begin{array}{rcl}
        S &=& S_{obs} (1 + a)\\\\
        I &=& I_{obs} (1 + b)\\\\
        R &=& R_{obs} (1 + c)\\\\
        D &=& D_{obs}
        \\end{array}

    Where :math:`S`, :math:`I`, :math:`R` are
    :math:`R_{obs}` are observed.
    hidden, only :math:`S_{obs}`, :math:`I_{obs}`,
    The following equality should be verified:
    :math:`S + I + R + D = N = S_{obs} + I_{obs} + R_{obs} + D_{obs}`.
    We also get from the previous equations:

    .. math::

        \\begin{array}{rcl}
        dS &=& dS_{obs} (1 + a) = - \\beta \\frac{IS}{N} =
        - \\beta \\frac{I_{obs}S_{obs}}{N}(1+a)(1+b) \\\\
        \\Longrightarrow dS_{obs} &=& - \\beta \\frac{I_{obs}S_{obs}}{N}(1+b)
        \\end{array}

    And also:

    .. math::

        \\begin{array}{rcl}
        dD &=& dD_{obs} = \\nu I = \\nu I_{obs} (1+b)
        \\end{array}

    And as well:

    .. math::

        \\begin{array}{rcl}
        dR &=& dR_{obs} (1 + c) = \\mu I = \\mu (1 + b) I_{obs}  \\\\
        \\Longrightarrow dR_{obs} &=& - \\nu I_{obs} \\frac{1+b}{1+c}
        \\end{array}

    And finally:

    .. math::

        \\begin{array}{rcl}
        dI &=& dI_{obs} (1 + b) = -dR - dS - dD =
        - \\mu \\frac{1 + b}{1+ c} I_{obs} - \\nu (1+b) I_{obs} -
        - \\beta I_{obs}\\frac{S_{obs}}{N} (1 + a)(1 + b)
        \\\\
        \\Longrightarrow dI_{obs} &=& - \\nu I_{obs} - \\mu I_{obs}
        - \\beta I_{obs}\\frac{S_{obs}}{N} (1 + a)
        \\end{array}

    This model should still verify:

    .. math::

        \\begin{array}{rcl}
        S_{obs} + I_{obs} + R_{obs} + D_{obs} &=& N = S + I + R + D \\\\
        &=& S_{obs}(1+a) + I_{obs}(1+b) + R_{obs}(1+c) + D_{obs}
        \\end{array}

    That gives :math:`a S_{obs} + b I_{obs} + c R_{obs} = 0`.

    .. runpython::
        :showcode:
        :rst:

        from aftercovid.models import CovidSIRDc

        model = CovidSIRDc()
        print(model.to_rst())

    .. exref::
        :title: SIRDc simulation and plotting

        .. plot::

            from pandas import DataFrame
            import matplotlib.pyplot as plt
            from aftercovid.models import CovidSIRDc

            model = CovidSIRDc()
            sims = list(model.iterate(60))
            df = DataFrame(sims)
            print(df.head())
            ax = df.plot(y=['S', 'I', 'R', 'D'], kind='line')
            ax.set_xlabel("jours")
            ax.set_ylabel("population")
            r0 = model.R0()
            ax.set_title("Simulation SIRDc\\nR0=%f" % r0)

            plt.show()

    Visual representation:

    .. gdot::
        :script:

        from aftercovid.models import CovidSIRDc
        model = CovidSIRDc()
        print(model.to_dot())

    See :ref:`l-base-model-sir` to get the methods
    common to SIRx models. This model is not really working
    better than :class:`CovidSIRD <aftercovid.covid_sird.CovidSIRD>`.
    """

    P0 = [
        ('beta', 0.5, 'taux de transmission dans la population'),
        ('mu', 1 / 14., '1/. : durée moyenne jusque la guérison'),
        ('nu', 1 / 21., '1/. : durée moyenne jusqu\'au décès'),
        ('a', -1.5025135094805093e-08,
         'paramètre gérant les informations cachées (S)'),
        ('b', 1e-5, 'paramètre gérant les informations cachées (I)'),
        ('c', 1e-5, 'paramètre gérant les informations cachées (R)'),
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

    # c = b (nu + mu) / (mu - b * nu)
    # (1 + b) / (1 + c) = 1 - nu * b / mu
    eq = {
        'S': '-beta * (1 + b) * I * S / N',
        'I': ('beta * (1 + a) * I * S / N'
              '- nu * I - mu * I'),
        'R': 'mu * (1 + b) * I / (1 + c)',
        'D': 'nu * (1 + b) * I'}

    def __init__(self):
        BaseSIR.__init__(
            self,
            p=CovidSIRDc.P0.copy(),
            q=CovidSIRDc.Q0.copy(),
            c=CovidSIRDc.C0.copy(),
            eq=CovidSIRDc.eq.copy())

    def R0(self, t=0):
        '''Returns R0 coefficient.
        See :meth:`CovidSIRD.R0 <aftercovid.models.CovidSIRD.R0>`'''
        return self['beta'] / (self['nu'] + self['mu'])

    def correctness(self, X=None):
        '''
        Returns :math:`a S_{obs} + b I_{obs} + c R_{obs} = 0`.

        :param X: None to use inner quantities
        :return: a number
        '''
        if X is None:
            X = self.vect().reshape((1, -1))
        return (X[:, 0] * self['a'] + X[:, 1] * self['b'] +
                X[:, 2] * self['c']) / self['N']

    def update_abc(self, X=None, update=True, alpha=1.0, l1_ratio=0.5):
        '''
        Updates coefficients *a*, *b*, *c* so that method
        :meth:`correctness <aftercovid.models.CovidSIRDc.correctness>`
        returns 0. It uses `ElasticNet
        <https://scikit-learn.org/stable/modules/generated/
        sklearn.linear_model.ElasticNet.html>`_.

        :param X: None to use inner quantities
        :param update: True to update to the coefficients
            or False to just return the results
        :param alpha: see ElasticNet
        :param l1_ratio: see ElasticNet
        :return: dictionary
        '''
        if X is None:
            X = self.vect().reshape((1, -1))
        X = X / self['N']
        cst = - self.correctness(X)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
        model.fit(X, cst)
        coef = model.coef_
        res = dict(a=self['a'] + coef[0], b=self['b'] + coef[1],
                   c=self['c'] + coef[2])
        if update:
            self.update(**res)
        return res

    def rnd(self):
        '''
        Draws random parameters.
        Not perfect.
        '''
        self['beta'] = numpy.random.randn(1) * 0.1 + 0.5
        self['mu'] = numpy.random.randn(1) * 0.1 + 1. / 14
        self['nu'] = numpy.random.randn(1) * 0.1 + 1. / 21
        self['a'] = numpy.random.rand() * 1e-5
        self['b'] = numpy.random.rand() * 1e-5
        self['c'] = numpy.random.rand() * 1e-5

    @staticmethod
    def add_noise(X, epsilon=1.):
        """
        Tries to add reasonable noise to the quantities stored in *X*.

        :param epsilon: amplitude
        :return: new X
        """
        return CovidSIRD.add_noise(X, epsilon=epsilon)
