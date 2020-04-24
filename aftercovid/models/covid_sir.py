# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
from ._base_sir import BaseSIR


class CovidSIR(BaseSIR):
    """
    Inspiration `Modelling some COVID-19 data
    <http://webpopix.org/covidix19.html>`_.

    .. runpython::
        :showcode:
        :rst:

        from aftercovid.models import CovidSIR

        model = CovidSIR()
        print(model.to_rst())

    .. plot::

        from pandas import DataFrame
        import matplotlib.pyplot as plt
        from aftercovid.models import CovidSIR

        model = CovidSIR()
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

        from aftercovid.models import CovidSIR
        model = CovidSIR()
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
        ('S', 9990., 'personnes non contaminés'),
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
            p=CovidSIR.P0.copy(),
            q=CovidSIR.Q0.copy(),
            c=CovidSIR.C0.copy(),
            eq=CovidSIR.eq.copy())

    def R0(self, t=0):
        '''Returns R0 coefficient.'''
        return self['beta'] / (self['nu'] + self['mu'])
