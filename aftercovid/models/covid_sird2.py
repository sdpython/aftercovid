# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
import numpy.random
from ._base_sir import BaseSIR


class CovidSIRD2(BaseSIR):
    """
    Inspiration `Modelling some COVID-19 data
    <http://webpopix.org/covidix19.html>`_.
    This model considers that observed data are not the
    true ones. First, we assume the dynamic follows
    a model SIRD:

    .. math::

        \\begin{array}{rcl}
        \\frac{dS}{dt} &=& \\frac{-IS}{N}\\beta \\\\
        \\frac{dI}{dt} &=& \\frac{IS}{N}\\beta  - I\\nu - I\\mu\\\\
        \\frac{dR}{dt} &=& I \\mu \\\\
        \\frac{dD}{dt} &=& I \\nu
        \\end{array}

    We split the population into observed
    (:math:`S_1, I_1, R_1, D`) and not observed
    (:math:`S_2, I_2, R_2, D_2`).

    .. math::

        \\begin{array}{rcl}
        S &=& S_1 + S_2 \\\\
        I &=& I_1 + I_2 \\\\
        R &=& R_1 + R_2 \\\\
        D &=& D_1 + D_2
        \\end{array}


    .. math::

        \\begin{array}{rcl}
        \\frac{dS}{dt} &=& \\frac{-(I_1 + I_2)(S_1 + S_2)}{N}\\beta \\\\
        \\frac{dI}{dt} &=& \\frac{(I_1 + I_2)(S_1 + S_2)}{N}\\beta -
        (I_1 + I_2)(\\nu + \\mu)\\\\
        \\frac{dR}{dt} &=& (I_1 + I_2) \\mu\\\\
        \\frac{dD}{dt} &=& (I_1 + I_2) \\nu\\\\
        \\frac{dR_1}{dt} &=& I_1 \\mu \\\\
        \\frac{dR_2}{dt} &=& I_2 \\mu \\\\
        \\frac{dD_1}{dt} &=& I_1 \\nu \\\\
        \\frac{dD_2}{dt} &=& I_2 \\nu
        \\end{array}



    .. runpython::
        :showcode:
        :rst:

        from aftercovid.models import CovidSIRDc

        model = CovidSIRD2()
        print(model.to_rst())

    .. exref::
        :title: SIRDC simulation and plotting

        .. plot::

            from pandas import DataFrame
            import matplotlib.pyplot as plt
            from aftercovid.models import CovidSIRD2

            model = CovidSIRD2()
            sims = list(model.iterate(60))
            df = DataFrame(sims)
            print(df.head())
            ax = df.plot(y=['S1', 'I1', 'R1', 'D1',
                            'S2', 'I2', 'R2', 'D2'], kind='line')
            ax.set_xlabel("jours")
            ax.set_ylabel("population")
            r0 = model.R0()
            ax.set_title("Simulation SIRD2\\nR0=%f" % r0)

            plt.show()

    Visual representation:

    .. gdot::
        :script:

        from aftercovid.models import CovidSIRD2
        model = CovidSIRD2()
        print(model.to_dot())

    This is unfinished work.
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
        ('S1', 9990., 'personnes non contaminées, population 1'),
        ('I1', 10., 'nombre de personnes malades ou contaminantes, '
                    'population 1'),
        ('R1', 0., 'personnes guéries (recovered), population 1'),
        ('D1', 0., 'personnes décédées, population 1'),
        ('S2', 0., 'personnes non contaminées, population 2'),
        ('I2', 0., 'nombre de personnes malades ou contaminantes, '
                   'population 2'),
        ('R2', 0., 'personnes guéries (recovered), population 2'),
        ('D2', 0., 'personnes décédées, population 2'),
    ]

    Cst0 = [
        ('S', 'S1 + S2'),
        ('I', 'I1 + I2'),
        ('R', 'R1 + R2'),
        ('D', 'D1 + D2'),
        ('N', 'S + I + R + D'),
    ]

    Obs0 = ['S1', 'I1', 'R1', 'D']

    C0 = [
        ('N', 10000., 'population'),
    ]

    eq = {
        'S': '- beta * S / N * I',
        'I': 'beta * S / N * I - mu * I - nu * I',
        'R': 'mu * I',
        'R1': 'mu * I1',
        'R2': 'mu * I2',
        'D': 'nu * I',
        'D1': 'nu * I1',
        'D2': 'nu * I2',
    }

    def __init__(self):
        BaseSIR.__init__(
            self,
            p=CovidSIRD2.P0.copy(),
            q=CovidSIRD2.Q0.copy(),
            c=CovidSIRD2.C0.copy(),
            cst=CovidSIRD2.Cst0.copy(),
            obs=CovidSIRD2.Obs0.copy()
        )

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

    @staticmethod
    def add_noise(X, epsilon=1.):
        """
        Tries to add reasonable noise to the quantities stored in *X*.

        :param epsilon: amplitude
        :return: new X
        """
        raise NotImplementedError()
