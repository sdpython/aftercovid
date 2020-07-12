# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
import numpy.random
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
        S_{obs} &=& S + S_c \\\\
        I_{obs} &=& I - I_c \\\\
        R_{obs} &=& R - R_c \\\\
        D_{obs} &=& D
        \\end{array}

    Where :math:`S_c`, :math:`I_c`, :math:`R_c` are
    hidden, only :math:`S_{obs}`, :math:`I_{obs}`,
    :math:`R_{obs}` are observed.
    As :math:`S + I + R + D = N = S_{obs} + I_{obs} + R_{obs} + D_{obs}`,
    we get :math:`S_c = I_c + R_c`.

    .. runpython::
        :showcode:
        :rst:

        from aftercovid.models import CovidSIRDc

        model = CovidSIRDc()
        print(model.to_rst())

    .. exref::
        :title: SIRDC simulation and plotting

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
            ax.set_title("Simulation SIR\\nR0=%f" % r0)

            plt.show()

    Visual representation:

    .. gdot::
        :script:

        from aftercovid.models import CovidSIRDc
        model = CovidSIRDc()
        print(model.to_dot())

    See :ref:`l-base-model-sir` to get the methods
    common to SIRx models.
    """

    P0 = [
        ('beta', 0.5, 'taux de transmission dans la population'),
        ('mu', 1 / 14., '1/. : durée moyenne jusque la guérison'),
        ('nu', 1 / 21., '1/. : durée moyenne jusqu\'au décès'),
        ('cS', 1e-2, 'personnes non infectées et cachées'),
        ('cR', 1e-2, 'personnes guéries et cachées'),
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
        'S': '- beta / N * (S - cS * N * 1e-5) * (I + (cS - cR) * N * 1e-5)',
        'I': ('beta / N * (S - cS * N * 1e-5) * (I + (cS - cR) * N * 1e-5)'
              '- mu * I '
              '- nu * (I + (cS - cR) * N * 1e-5)'),
        'R': 'mu * I',
        'D': 'nu * (I + (cS - cR) * N * 1e-5)'}

    def __init__(self):
        BaseSIR.__init__(
            self,
            p=CovidSIRDc.P0.copy(),
            q=CovidSIRDc.Q0.copy(),
            c=CovidSIRDc.C0.copy(),
            eq=CovidSIRDc.eq.copy())

    def R0(self, t=0):
        '''Returns R0 coefficient.'''
        return self['beta'] / (self['nu'] + self['mu'])

    def rnd(self):
        '''
        Draws random parameters.
        Not perfect.
        '''
        self['beta'] = numpy.random.randn(1) * 0.1 + 0.5
        self['mu'] = numpy.random.randn(1) * 0.1 + 1. / 14
        self['nu'] = numpy.random.randn(1) * 0.1 + 1. / 21
        self['cR'] = numpy.random.rand() * 1e-4
        self['cS'] = numpy.random.rand() * 1e-4

    @staticmethod
    def add_noise(X, epsilon=1.):
        """
        Tries to add reasonable noise to the quantities stored in *X*.

        :param epsilon: amplitude
        :return: new X
        """
        return CovidSIRD.add_noise(X, epsilon=epsilon)
