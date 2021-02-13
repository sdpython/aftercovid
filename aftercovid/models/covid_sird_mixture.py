# coding: utf-8
"""
Implementation of a model for epidemics propagation.
"""
import numpy.random
from ._base_sir import BaseSIR
from .covid_sird import CovidSIRD


class CovidSIRDMixture(BaseSIR):
    """
    The model extends model @see cl CovidSIRD and assumes
    there are two variants of the same virus.
    The term `beta1 * beta2 * S * I1 / N * I2 / N` means
    that for all people being in contact with both
    virus, the second one wins as it is more contagious.

    .. runpython::
        :showcode:
        :rst:

        from aftercovid.models import CovidSIRDMixture

        model = CovidSIRDMixture()
        print(model.to_rst())

    .. exref::
        :title: Mixture of SIRD simulation and plotting

        .. plot::

            from pandas import DataFrame
            import matplotlib.pyplot as plt
            from aftercovid.models import CovidSIRDMixture

            model = CovidSIRDMixture()
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

        from aftercovid.models import CovidSIRDMixture
        model = CovidSIRDMixture()
        print(model.to_dot())

    See :ref:`l-base-model-sir` to get the methods
    common to SIRx models.
    """

    P0 = [
        ('beta1', 0.5, 'taux de transmission dans la population'),
        ('beta2', 0.7, 'second taux de transmission dans la population'),
        ('mu', 1 / 14., '1/. : durée moyenne jusque la guérison'),
        ('nu', 1 / 21., '1/. : durée moyenne jusqu\'au décès'),
    ]

    Q0 = [
        ('S', 9990., 'personnes non contaminées'),
        ('I1', 8., 'nombre de personnes malades ou contaminantes '
         'pour le premier variant'),
        ('I2', 2., 'nombre de personnes malades ou contaminantes '
                   'pour le second variant'),
        ('R', 0., 'personnes guéries (recovered)'),
        ('D', 0., 'personnes décédées'),
    ]

    C0 = [
        ('N', 10000., 'population'),
    ]

    eq = {
        'S': ('- beta1 * S / N * I1 - beta2 * S / N * I2 '
              '+ beta1 * beta2 * S * I1 / N * I2 / N'),
        'I1': ('beta1 * S / N * I1 - mu * I1 - nu * I1 '
               '- beta1 * beta2 * S * I1 / N * I2 / N'),
        'I2': 'beta2 * S / N * I2 - mu * I2 - nu * I2',
        'R': 'mu * (I1 + I2)',
        'D': 'nu * (I1 + I2)'
    }

    def __init__(self):
        BaseSIR.__init__(
            self,
            p=CovidSIRDMixture.P0.copy(),
            q=CovidSIRDMixture.Q0.copy(),
            c=CovidSIRDMixture.C0.copy(),
            eq=CovidSIRDMixture.eq.copy())

    def R0(self, t=0):
        '''
        Returns R0 coefficient.

        :param t: unused
        '''
        return (self['beta1'] + self['beta2']) / (self['nu'] + self['mu'])

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
        self['beta1'] = numpy.random.randn(1) * 0.1 + 0.5
        self['beta2'] = numpy.random.randn(1) * 0.1 + 0.7
        self['mu'] = numpy.random.randn(1) * 0.1 + 1. / 14
        self['nu'] = numpy.random.randn(1) * 0.1 + 1. / 21

    @staticmethod
    def add_noise(X, epsilon=1.):
        """
        Tries to add reasonable noise to the quantities stored in *X*.

        :param epsilon: amplitude
        :return: new X
        """
        return CovidSIRD.add_noise(X, epsilon=epsilon)
