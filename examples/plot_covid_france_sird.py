# coding: utf-8
"""
.. _l-sir-france-example:

Estimation des paramètres d'un modèle SIRD pour la France
=========================================================

On récupère les données réelles pour un pays
et on cherche à estimer un modèle
:class:`CovidSIRD <aftercovid.models.CovidSIRD>`.
Il y a deux problèmes avec les données. Le premier est
que le nombre de cas positifs est largement sous-estimé.
La principale raison durant le premier confinement fut le
manque de tests, limité à 5000 pendant cette période.
La seconde raison est que les personnes asymptomatiques
ne le savent pas et ne se font pas toujours tester.
Le délai d'attente des résultats des tests,
les longues queues devant les laboratoires ont peut-être
aussi joué puisqu'aux mois d'août, septembre.

.. contents::
    :local:

Récupération des données
++++++++++++++++++++++++
"""
import matplotlib.gridspec as gridspec
from aftercovid.data import extract_hopkins_data, preprocess_hopkins_data
from aftercovid.models import CovidSIRD, rolling_estimation
import numpy
import warnings
import matplotlib.pyplot as plt
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning


data = extract_hopkins_data()
diff, df = preprocess_hopkins_data(data)
df.tail()

#####################################
# Séries différenciées.

diff.tail()


########################################
# Graphes.
fig = plt.figure(tight_layout=True, figsize=(12, 10))
gs = gridspec.GridSpec(2, 2)
axs = []
ax = fig.add_subplot(gs[0, :])
df.plot(logy=True, title="Données COVID", ax=ax)
axs.append(ax)
ax = fig.add_subplot(gs[1, 0])
df[['recovered', 'confirmed', 'infected']].diff().plot(
    title="Différences", ax=ax)
axs.append(ax)
ax = fig.add_subplot(gs[1, 1])
df[['deaths']].diff().plot(title="Différences", ax=ax)
axs.append(ax)
for a in axs:
    for tick in a.get_xticklabels():
        tick.set_rotation(30)


################################################
# On voit qu'en France, les données sont difficilement
# exploitables en l'état. Et on sait qu'en France
# la pénurie de tests implique une sous-estimation
# du nombre de cas positifs. L'estimation du modèle
# est très compromise.

############################################
# _l-example-rolling-estimation:
#
# Estimation d'un modèle
# ++++++++++++++++++++++
#
# Sur 21 jours.
# ^^^^^^^^^^^^^

model = CovidSIRD()
print(model.quantity_names)

cols = ['safe', 'infected', 'recovered', 'deaths']
data = df[cols].values.astype(numpy.float32)
print(data[:5])

X = data[:-1]
y = data[1:] - data[:-1]
dates = df.index[:-1]

#########################################
# Estimation.


# 3 semaines car les séries sont cycliques
dfcoef, model = rolling_estimation(
    X, y, delay=21, dates=dates, verbose=1)
dfcoef.head(n=10)

#############################################
# Fin de la période.

dfcoef.tail(n=10)

#############################################
# Statistiques.

dfcoef.describe()

#############################################
# Fin de la période.

df.tail(n=10)

#############################################
# Graphe.

dfcoef['R0=1'] = 1

fig, ax = plt.subplots(2, 3, figsize=(14, 6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0", "R0=1"]].plot(ax=ax[0, 2])
    df.drop('safe', axis=1).plot(ax=ax[1, 1], logy=True)
ax[0, 2].set_ylim(0, 5)
fig.suptitle('Estimation de R0 tout au long de la période\n'
             'Estimation sur 3 semaines', fontsize=12)
plt.show()

#############################################
# Graphe sur les dernières valeurs.

dfcoeflast = dfcoef.iloc[-30:, :]
dflast = df.iloc[-30:, :]

fig, ax = plt.subplots(2, 3, figsize=(14, 6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    dfcoeflast[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoeflast[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoeflast[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoeflast[["R0", "R0=1"]].plot(ax=ax[0, 2])
    dflast.drop('safe', axis=1).plot(ax=ax[1, 1], logy=True)
ax[0, 2].set_ylim(0, 5)
fig.suptitle('Estimation de R0 sur la fin de la période', fontsize=12)
plt.show()

#################################################
# Taille fenêtre glissante
# ++++++++++++++++++++++++
#
# On fait varier le paramètre *delay* pour voir comment
# le modèle réagit. Sur 7 jours d'abord.
#
# Sur 7 jours.
# ^^^^^^^^^^^^^

dfcoef, model = rolling_estimation(X, y, delay=7, verbose=1)
dfcoef.tail()

#######################################
# Graphe.

dfcoef['R0=1'] = 1


fig, ax = plt.subplots(2, 3, figsize=(14, 6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0", "R0=1"]].plot(ax=ax[0, 2])
    df.drop('safe', axis=1).plot(ax=ax[1, 1], logy=True)
ax[0, 2].set_ylim(0, 5)
fig.suptitle('Estimation de R0 tout au long de la période\n'
             'Estimation sur 1 semaine', fontsize=12)
plt.show()

#######################################
# Sur 14 jours.
# ^^^^^^^^^^^^^

dfcoef, model = rolling_estimation(X, y, delay=14, verbose=1)
dfcoef.tail()

#######################################
# Graphe.

dfcoef['R0=1'] = 1

fig, ax = plt.subplots(2, 3, figsize=(14, 6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0", "R0=1"]].plot(ax=ax[0, 2])
    df.drop('safe', axis=1).plot(ax=ax[1, 1], logy=True)
ax[0, 2].set_ylim(0, 5)
fig.suptitle('Estimation de R0 tout au long de la période\n'
             'Estimation sur 2 semaines', fontsize=12)
plt.show()

##############################################
# Sur 4 semaines.
# ^^^^^^^^^^^^^^^

dfcoef, model = rolling_estimation(X, y, delay=28, verbose=1)
dfcoef.tail()

#########################################
# Graphe.

dfcoef['R0=1'] = 1

fig, ax = plt.subplots(2, 3, figsize=(14, 6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0", "R0=1"]].plot(ax=ax[0, 2])
    df.drop('safe', axis=1).plot(ax=ax[1, 1], logy=True)
ax[0, 2].set_ylim(0, 5)
fig.suptitle('Estimation de R0 tout au long de la période\n'
             'Estimation sur 4 semaines', fontsize=12)
plt.show()
