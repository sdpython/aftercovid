# coding: utf-8
"""
Estimation des paramètres d'un mélange de modèles SIRD
======================================================

Le virus mute et certains variantes sont plus contagieuses.
Dans ce modèle, on simule une population au sein de laquelle
circulent deux virus avec le modèle suivant :
:class:`CovidSIRDMixture <aftercovid.models.CovidSIRDMixture>`.
L'idée est ensuite d'estimer un module plus simple
sur cette population pour comprendre comment le paramètre
:math:`\\beta` de ce modèle simple évoluerait.

.. contents::
    :local:

Simulation des données
++++++++++++++++++++++
"""

import warnings
from pprint import pprint
import numpy
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
import matplotlib.pyplot as plt
import pandas
from aftercovid.models import EpidemicRegressor, CovidSIRDMixture

model = CovidSIRDMixture()
model


###########################################
# Mise à jour des coefficients.

model['beta1'] = 0.15
model['beta2'] = 0.25
model["mu"] = 0.06
model["nu"] = 0.04
pprint(model.P)

###################################
# Point de départ
pprint(model.Q)

###################################
# On part d'un point de départ un peu plus conséquent
# car l'estimation n'est pas très fiable au départ de l'épidémie
# comme le montre l'exemple :ref:`l-estim-sird-theory`.
model.update(S=9100, I1=80, I2=20)
pprint(model.Q)


###################################
# Simulation

X, y = model.iterate2array(90, derivatives=True)
data = {_[0]: x for _, x in zip(model.Q, X.T)}
data.update({('d' + _[0]): c for _, c in zip(model.Q, y.T)})
df = pandas.DataFrame(data)
df.tail()

######################################
# Visualisation

df.drop('S', axis=1).plot(title="Simulation SIRDMixture")


###########################################
# Estimation
# ++++++++++
#
# Le module implémente la class :class:`EpidemicRegressor
# <aftercovid.models.EpidemicRegressor>` qui réplique
# l'API de :epkg:`scikit-learn`. Il faut additionner
# I1 et I2.

X2 = numpy.empty((X.shape[0], 4), dtype=X.dtype)
X2[:, 0] = X[:, 0]
X2[:, 1] = X[:, 1] + X[:, 2]
X2[:, 2] = X[:, 3]
X2[:, 3] = X[:, 4]

y2 = numpy.empty((y.shape[0], 4), dtype=X.dtype)
y2[:, 0] = y[:, 0]
y2[:, 1] = y[:, 1] + y[:, 2]
y2[:, 2] = y[:, 3]
y2[:, 3] = y[:, 4]

X, y = X2, y2

m = EpidemicRegressor('SIRD', verbose=True, learning_rate_init=1e-3,
                      max_iter=15, early_th=1)
m.fit(X, y)
pprint(m.model_.P)


###############################################
# La réaction de la population n'est pas constante
# tout au long de l'épidémie. Il est possible qu'elle
# change de comportement tout au long de la propagation.
# On estime alors les coefficients du modèle sur une
# fenêtre glissante.


def find_best_model(Xt, yt, lrs, th):
    best_est, best_loss = None, None
    for lr in lrs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EpidemicRegressor(
                'SIRDc',
                learning_rate_init=lr,
                max_iter=500,
                early_th=1)
            m.fit(Xt, yt)
            loss = m.score(Xt, yt)
            if numpy.isnan(loss):
                continue
        if best_est is None or best_loss > loss:
            best_est = m
            best_loss = loss
        if best_loss < th:
            return best_est, best_loss
    return best_est, best_loss

###############################################
# On estime les coefficients du modèle tous les 5 jours
# sur les 10 derniers jours.


coefs = []
for k in range(0, X.shape[0] - 9):
    end = min(k + 10, X.shape[0])
    Xt, yt = X[k:end], y[k:end]
    m, loss = find_best_model(Xt, yt, [1e-2, 1e-3], 10)
    loss = m.score(Xt, yt)
    print(f"k={k} iter={m.iter_} loss={loss:1.3f} coef={m.model_._val_p}")
    obs = dict(k=k, loss=loss, it=m.iter_, R0=m.model_.R0())
    obs.update({k: v for k, v in zip(m.model_.param_names, m.model_._val_p)})
    coefs.append(obs)

#######################################
# Résumé

dfcoef = pandas.DataFrame(coefs).set_index('k')
dfcoef

#############################################################
# On visualise.


with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    fig, ax = plt.subplots(2, 3, figsize=(14, 6))
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0"]].plot(ax=ax[0, 2])
    df.plot(ax=ax[1, 1], logy=True)
    fig.suptitle('Estimation de R0 tout au long de la simulation', fontsize=12)
