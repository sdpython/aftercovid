# coding: utf-8
"""
Implémentation d'un modèle pour la propagation de l'épidémie.
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
        from aftercovid.models import CovidSIR

        model = CovidSIR()
        sims = list(model.iterate(60))
        df = DataFrame(sims)
        print(df.head())
        ax = df.plot(y=['St', 'It', 'Rt', 'Dt'], kind='line')
        ax.set_xlabel("jours")
        ax.set_ylabel("population")
        ax.set_title("Simulation SIR")

        import matplotlib.pyplot as plt
        plt.show()
    """

    P0 = [
        ('beta', 0.5, 'taux de transmission dans la population'),
        ('mu', 1/14., '1/. : durée moyenne jusque la guérison'),
        ('nu', 1/21., '1/. : durée moyenne jusqu\'au décès'),
    ]

    Q0 = [
        ('St', 9990., 'personnes non contaminés'),
        ('It', 10., 'nombre de personnes malades ou contaminantes'),
        ('Rt', 0., 'personnes guéries (recovered)'),
        ('Dt', 0., 'personnes décédées'),
    ]

    C0 = [
        ('N', 10000., 'population'),
    ]

    eq = {
        'St': '- beta * St / N * It',
        'It': 'beta * St / N * It - mu * It - nu * It',
        'Rt': 'mu * It',
        'Dt': 'nu * It'
    }

    def __init__(self):
        BaseSIR.__init__(
            self,
            p=CovidSIR.P0.copy(),
            q=CovidSIR.Q0.copy(),
            c=CovidSIR.C0.copy(),
            eq=CovidSIR.eq.copy())
