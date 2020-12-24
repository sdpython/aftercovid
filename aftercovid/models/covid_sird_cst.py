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
        S &=& S_{obs} (1 + a)\\\\
        I &=& I_{obs} (1 + b)\\\\
        R &=& R_{obs} (1 + c)\\\\
        D &=& D_{obs}
        \\end{array}

    Where :math:`S`, :math:`I`, :math:`R` are
    hidden, only :math:`S_{obs}`, :math:`I_{obs}`,
    :math:`R_{obs}` are observed.
    As :math:`S + I + R + D = N = S_{obs} + I_{obs} + R_{obs} + D_{obs}`.
    Let's see where it goes. First gradient:

    .. math::

        \\begin{array}{ll}
        &\\frac{dR}{dt} = \\frac{dR_{obs}}{dt} (1 + c) =
        I \\mu = \\mu(1+b) I_{obs} \\\\
        \\Longrightarrow &\\frac{dR_{obs}}{dt} =
        \\frac{\\mu(1+b)}{1+c} I_{obs}
        \\end{array}

    Second gradient:

    .. math::

        \\begin{array}{ll}
        &\\frac{dD_{obs}}{dt} = \\frac{dD}{dt} =
        I \\nu = \\nu(1+b) I_{obs} \\\\
        \\Longrightarrow & \\frac{dD_{obs}}{dt} = \\nu (1+b) I_{obs}
        \\end{array}

    Third gradient:

    .. math::

        \\begin{array}{ll}
        &\\frac{dS}{dt} = \\frac{dS_{obs}}{dt} (1 + a) =
        -\\beta\\frac{IS}{N} =
        -\\beta\\frac{I_{obs}(1+b)S_{obs}(1+a)}{N} \\\\
        \\Longrightarrow &\\frac{dS_{obs}}{dt} =
        -\\beta\\frac{I_{obs}(1+b)S_{obs}}{N}
        \\end{array}

    :math:`S + I + R + D = N = S_{obs} + I_{obs} + R_{obs} + D_{obs}`
    implies that the derivatives verify the following equality:

    .. math::

        \\begin{array}{ll}
        & a \\frac{dS_{obs}}{dt} + b \\frac{dI_{obs}}{dt} +
        c \\frac{dR_{obs}}{dt}  + \\frac{dD_{obs}}{dt} =
        \\frac{dS_{obs}}{dt} + \\frac{dI_{obs}}{dt} +
        \\frac{dR_{obs}}{dt} + \\frac{dD_{obs}}{dt} = 0 \\\\
        \\Longrightarrow & (1 - a) \\frac{dS_{obs}}{dt} + (1 - b) \\frac{dI_{obs}}{dt} +
        (1 - c) \\frac{dR_{obs}}{dt} = 0
        \\end{array}

    Then:

    .. math::

        \\Longrightarrow \\frac{dI_{obs}}{dt} =
        \\beta\\frac{I_{obs}(1+b)S_{obs}}{N} - \\nu (1+b) I_{obs} -
        \\frac{\\mu(1+b)}{1+c} I_{obs} =
        \\frac{\\mu(1+b)c}{b(1+c)} I_{obs} +
        \\beta\\frac{I_{obs}a(1+b)S_{obs}}{Nb}

    And:

    .. math::

        \\begin{array}{lll}
        V_1(a,b,c)  &= (1+b)\\left(\\begin{array}{cc}
        \\frac{\\beta}{N} &,& -\\nu - \\frac{\\mu}{1+c}
        \\end{array}\\right) \\\\
        V_2(a,b,c) &=  (1+b)\\left(\\begin{array}{cc}
        \\frac{\\beta a}{Nb} &,& \\frac{\\mu c}{b(1+c)}
        \\end{array}\\right) \\\\
        X &= (I_{obs} S_{obs}, I_{obs}) \\\\
        \\Longrightarrow  &\\forall X, \\; V_1(a,b,c) X = V_1(a,b,c) X
        \\end{array}

    And we get :math:`a=b` and
    :math:`c = \\frac{b(\\nu + \\mu)}{\\mu - b\\nu}`.

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
        ('b', 1e-5, 'paramètres gérant les informations cachées'),
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
        'I': ('beta * (1 + b) * I * S / N'
              '- nu * (1 + b) * I - (mu - nu * b) * I'),
        'R': '(mu - nu * b) * I',
        'D': 'nu * (1 + b) * I'}

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
        self['b'] = numpy.random.rand() * 1e-5

    @staticmethod
    def add_noise(X, epsilon=1.):
        """
        Tries to add reasonable noise to the quantities stored in *X*.

        :param epsilon: amplitude
        :return: new X
        """
        return CovidSIRD.add_noise(X, epsilon=epsilon)
