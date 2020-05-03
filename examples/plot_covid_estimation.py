# coding: utf-8
"""
Estimation des paramètres
=========================

On part d'un modèle :class:`CovidSIR <aftercovid.models.CovidSIR>`
qu'on utilise pour simuler des données. On regarde s'il est possible
de réestimer les paramètres du modèle à partir des observations.

.. contents::
    :local:

Simulation des données
++++++++++++++++++++++
"""


import numpy
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
import matplotlib.pyplot as plt
import warnings
from aftercovid.models import EpidemicRegressor
import pandas
from aftercovid.models import CovidSIR

model = CovidSIR()
model


###########################################
# Mise à jour des coefficients.

model['beta'] = 0.4
model["mu"] = 0.06
model["nu"] = 0.04
print(model.P)

###################################
# Point de départ
print(model.Q)


###################################
# Simulation

X, y = model.iterate2array(50, derivatives=True)
df = pandas.DataFrame({_[0]: x for _, x in zip(model.Q, X.T)})
df.tail()

######################################
# Visualisation

df.plot(title="Simulation SIR")


###########################################
# Estimation
# ++++++++++
#
# Le module implémente la class :class:`EpidemicRegressor
# <aftercovid.models.EpidemicRegressor>` qui réplique
# l'API de :epkg:`scikit-learn`.


m = EpidemicRegressor('SIR', verbose=True, learning_rate_init=1e-2,
                      max_iter=10, early_th=1)
m.fit(X, y)
print(m.model_.P)


###############################################
# La réaction de la population n'est pas constante
# tout au long de l'épidémie. Il est possible qu'elle
# change de comportement tout au long de la propagation.
# On estime alors les coefficients du modèle sur une
# fenêtre glissante.


def find_best_model(X, y, lrs, th):
    "Détermine le meilleur modèle pour différentes valeurs de paramètres."
    best_est, best_loss = None, None
    for lr in lrs:
        m = EpidemicRegressor('SIR', learning_rate_init=lr,
                              max_iter=100, early_th=1)
        m.fit(Xt, yt)
        loss = m.score(Xt, yt)
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
for k in range(0, X.shape[0], 5):
    end = min(k + 10, X.shape[0])
    Xt, yt = X[k:end], y[k:end]
    m, loss = find_best_model(Xt, yt, [5e-2, 5e-3], 10)
    loss = m.score(Xt, yt)
    print("k={} iter={} loss={:1.3f} coef={}".format(
        k, m.iter_, loss, m.model_._val_p))
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
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0])
    dfcoef[["beta"]].plot(ax=ax[0, 1])
    dfcoef[["loss"]].plot(ax=ax[1, 0])
    dfcoef[["R0"]].plot(ax=ax[0, 2])
    df.plot(ax=ax[1, 1])
    fig.suptitle('Estimation de R0 tout au long de la simulation', fontsize=12)


######################################################
# Données bruitées
# ++++++++++++++++
#
# L'idée est de voir si l'estimation se comporte
# aussi bien sur des données bruitées.

Xeps = CovidSIR.add_noise(X, epsilon=2.)
yeps = numpy.vstack([Xeps[1:] - Xeps[:-1], y[-1:]])


if False:
    # trop long
    coefs = []
    for k in range(0, X.shape[0], 5):
        end = min(k + 10, X.shape[0])
        Xt, yt = Xeps[k:end], yeps[k:end]
        m, loss = find_best_model(Xt, yt, [5e-2, 5e-3, 5e-4], 10)
        loss = m.score(Xt, yt)
        print(
            "k={} iter={} loss={:1.3f} coef={}".format(
                k,
                m.iter_,
                loss,
                m.model_._val_p))
        obs = dict(k=k, loss=loss, it=m.iter_, R0=m.model_.R0())
        obs.update({k: v for k, v in zip(
            m.model_.param_names, m.model_._val_p)})
        coefs.append(obs)

    dfcoef = pandas.DataFrame(coefs).set_index('k')
    print(dfcoef)

    dfeps = pandas.DataFrame({_[0]: x for _, x in zip(model.Q, Xeps.T)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
        fig, ax = plt.subplots(2, 3, figsize=(14, 6))
        dfcoef[["mu", "nu"]].plot(ax=ax[0, 0])
        dfcoef[["beta"]].plot(ax=ax[0, 1])
        dfcoef[["loss"]].plot(ax=ax[1, 0])
        dfcoef[["R0"]].plot(ax=ax[0, 2])
        dfeps.plot(ax=ax[1, 1])
        fig.suptitle(
            'Estimation de R0 tout au long de la simulation',
            fontsize=12)
