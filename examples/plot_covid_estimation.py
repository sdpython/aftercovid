# coding: utf-8
"""
Estimation des paramètres d'un modèle SIR
=========================================

On part d'un modèle :class:`CovidSIR <aftercovid.models.CovidSIR>`
qu'on utilise pour simuler des données. On regarde s'il est possible
de réestimer les paramètres du modèle à partir des observations.

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
from aftercovid.models import EpidemicRegressor, CovidSIR

model = CovidSIR()
model


###########################################
# Mise à jour des coefficients.

model['beta'] = 0.4
model["mu"] = 0.06
model["nu"] = 0.04
pprint(model.P)

###################################
# Point de départ
pprint(model.Q)


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


m = EpidemicRegressor('SIR', verbose=True, learning_rate_init=1e-3,
                      max_iter=10, early_th=1)
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
                'SIR',
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
for k in range(0, X.shape[0] - 9, 2):
    end = min(k + 10, X.shape[0])
    Xt, yt = X[k:end], y[k:end]
    m, loss = find_best_model(Xt, yt, [1e-2, 1e-3], 10)
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
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0"]].plot(ax=ax[0, 2])
    df.plot(ax=ax[1, 1])
    fig.suptitle('Estimation de R0 tout au long de la simulation', fontsize=12)

#################################
# L'estimation des coefficients est plus compliquée
# au début et à la fin de l'expérience. Il faudrait sans
# doute changer de stratégie.

####################################################
# Différentes tailles d'estimation
# ++++++++++++++++++++++++++++++++
#
# Le paramètre `beta` a été estimé sur une période de 10 jours.
# Est-ce que cela change sur une période plus courte ou plus longue ?
# Sur des données parfaites (sans bruit), cela ne devrait pas changer
# grand chose.

coefs = []
for delay in [4, 5, 6, 7, 8, 9, 10]:
    print('delay', delay)
    for k in range(0, X.shape[0] - delay, 4):
        end = min(k + delay, X.shape[0])
        Xt, yt = X[k:end], y[k:end]
        m, loss = find_best_model(Xt, yt, [1e-3, 1e-4], 10)
        loss = m.score(Xt, yt)
        if k == 0:
            print(
                "k={} iter={} loss={:1.3f} coef={}".format(
                    k, m.iter_, loss, m.model_._val_p))
        obs = dict(k=k, loss=loss, it=m.iter_, R0=m.model_.R0(), delay=delay)
        obs.update({k: v for k, v in zip(
            m.model_.param_names, m.model_._val_p)})
        coefs.append(obs)

#############################################
# Résumé

dfcoef = pandas.DataFrame(coefs)
dfcoef

################################################
# Graphes

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    fig, ax = plt.subplots(2, 3, figsize=(14, 6))
    for delay in sorted(set(dfcoef['delay'])):
        dfcoef.pivot('k', 'delay', 'mu').plot(
            ax=ax[0, 0], logy=True, legend=False).set_title('mu')
        dfcoef.pivot('k', 'delay', 'nu').plot(
            ax=ax[0, 1], logy=True, legend=False).set_title('nu')
        dfcoef.pivot('k', 'delay', 'beta').plot(
            ax=ax[0, 2], logy=True, legend=False).set_title('beta')
        dfcoef.pivot('k', 'delay', 'R0').plot(
            ax=ax[1, 2], logy=True, legend=False).set_title('R0')
        dfcoef.pivot('k', 'delay', 'loss').plot(
            ax=ax[1, 0], logy=True, legend=False).set_title('loss')
    df.plot(ax=ax[1, 1])
    fig.suptitle('Estimation de R0 tout au long de la simulation '
                 'avec différentes tailles de fenêtre', fontsize=12)

###################################
# Le graphique manque de légende.
# Ce sera pour plus tard.

######################################################
# Données bruitées
# ++++++++++++++++
#
# L'idée est de voir si l'estimation se comporte
# aussi bien sur des données bruitées.

Xeps = CovidSIR.add_noise(X, epsilon=1.)
yeps = numpy.vstack([Xeps[1:] - Xeps[:-1], y[-1:]])

###########################################
# On recommence.

coefs = []
for k in range(0, X.shape[0] - 9, 2):
    end = min(k + 10, X.shape[0])
    Xt, yt = Xeps[k:end], yeps[k:end]
    m, loss = find_best_model(Xt, yt, [1e-2, 1e-3, 1e-4], 10)
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
dfcoef

##########################
# Graphes.

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
        'Estimation de R0 tout au long de la simulation sur '
        'des données bruitées', fontsize=12)
