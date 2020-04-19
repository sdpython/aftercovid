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
    """

    P0 = [
        ('beta', 2.5, 'taux de transmission dans la population'),
        ('mu', 1/14., '1/. : durée moyenne jusque la guérison'),
        ('nu', 1/10., '1/. : durée moyenne jusqu\'au décès'),
    ]

    Q0 = [
        ('S', None, 'personnes non contaminés'),
        ('I', None, 'nombre de personnes malades ou contaminantes'),
        ('R', None, 'personnes guéries (recovered)'),
        ('D', None, 'personnes décédées'),
    ]

    C0 = [
        ('N', None, 'population'),
    ]

    eq = {
        'S': '- beta * S(t) / N * I(t)',
        'I': 'beta * S(t) / N * I(t) - mu * I(t) - nu * I(t)',
        'R': 'mu * I(t)',
        'D': 'nu * I(t)'
    }

    def __init__(self):
        BaseSIR.__init__(
            self,
            p=CovidSIR.P0.copy(),
            q=CovidSIR.Q0.copy(),
            c=CovidSIR.C0.copy(),
            eq=CovidSIR.eq.copy())
