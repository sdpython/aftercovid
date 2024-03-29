# coding: utf-8
"""
Estimation des paramètres d'un modèle SIRD étendu pour la France
================================================================

Le modèle proposé dans l'exemple :ref:`l-sir-france-example`
ne fonctionne pas très bien. Les données collectées sont erronées
pour le recensement des personnes infectées. Comme les malades n'étaient
pas testées au début de l'épidémie, le nombre officiel de personnes
contaminées est en-deça de la réalité. On ajoute un paramètre
pour tenir compte dela avec le modèle :class:`CovidSIRDc
<aftercovid.models.CovidSIRDc>`. Le modèle suppose
en outre que la contagion est la même tout au long de la
période d'étude alors que les mesures de confinement, le port du
masque impacte significativement le coefficient de propagation.
L'épidémie peut sans doute être modélisée avec un modèle SIRD
mais sur une courte période.

.. contents::
    :local:

Récupération des données
++++++++++++++++++++++++
"""
from datetime import timedelta
import warnings
import numpy
import pandas
from pandas import to_datetime
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from aftercovid.data import extract_hopkins_data, preprocess_hopkins_data
from aftercovid.models import CovidSIRDc, rolling_estimation


data = extract_hopkins_data()
diff, df = preprocess_hopkins_data(data)
df.tail()

##########################################
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

############################################
# .. _l-sliding-window-sir:
#
# Estimation d'un modèle
# ++++++++++++++++++++++
#
# L'approche sur une fenêtre glissante suggère que le modèle
# n'est pas bon pour approcher les données sur toute une période,
# mais que sur une période courte, le vrai modèle peut être
# approché par un modèle plus simple. On note :math:`W^*(t)`
# les paramètres optimaux au temps *t*, on étudie les courbes
# :math:`t \rightarrow W^*(t)` pour voir comment évolue
# ces paramètres.


model = CovidSIRDc()
print(model.quantity_names)

data = df[['safe', 'infected', 'recovered',
           'deaths']].values.astype(numpy.float32)
print(data[:5])

X = data[:-1]
y = data[1:] - data[:-1]
dates = df.index[:-1]

Nlast = 150
X = X[-Nlast:]
y = y[-Nlast:]
dates = dates[-Nlast:]

#########################################
# Estimation.


# 3 semaines car les séries sont cycliques
dfcoef, model = rolling_estimation(
    X, y, delay=21, dates=dates, verbose=1, model_name='SIRDc')
dfcoef.head(n=10)

#####################################
# Saving the results.

df.index = to_datetime(df.index)
dfcoef.index = to_datetime(dfcoef.index)
df.to_csv("plot_covid_france_sird_cst.data.csv", index=True)
dfcoef.to_csv("plot_covid_france_sird_cst.model.csv", index=True)

#####################################
# Fin de la période.

dfcoef.tail(n=10)

#############################################
# Statistiques.

dfcoef.describe()

#####################################
# Fin de la période.

df.tail(n=10)

#############################################
# Statistiques.

df.describe()

#############################################
# Graphe.

dfcoef['R0=1'] = 1
dfcoef['a_'] = -dfcoef['a']
dfcoef['cn'] = -dfcoef['correctness']

fig, ax = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoef[["loss_l1"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0", "R0=1"]].plot(ax=ax[0, 2])
    dfcoef[["a_", "b", "c", "cn"]].plot(ax=ax[1, 2])
    ax[0, 2].set_ylim(0, 5)
    df.iloc[-Nlast:].drop('safe', axis=1).plot(ax=ax[1, 1], logy=True)
    for i in range(ax.shape[1]):
        for tick in ax[1, i].get_xticklabels():
            tick.set_rotation(30)
    fig.suptitle('Estimation de R0 tout au long de la période', fontsize=12)
plt.show()

#############################################
# Graphe sur les dernières valeurs.

dfcoeflast = dfcoef.iloc[-30:, :]
dflast = df.iloc[-30:, :]

fig, ax = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    dfcoeflast[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoeflast[["beta"]].plot(ax=ax[0, 1], logy=True)
    dfcoeflast[["loss_l1"]].plot(ax=ax[1, 0], logy=True)
    dfcoeflast[["R0", "R0=1"]].plot(ax=ax[0, 2])
    ax[0, 2].set_ylim(0, 5)
    dfcoeflast[["a_", "b", "c", "cn"]].plot(ax=ax[1, 2])
    dflast.drop('safe', axis=1).plot(ax=ax[1, 1], logy=True)
    for i in range(ax.shape[1]):
        for tick in ax[1, i].get_xticklabels():
            tick.set_rotation(30)
    fig.suptitle('Estimation de R0 sur la fin de la période', fontsize=12)
plt.show()


#################################################
# Prédictions
# +++++++++++
#
# Ce ne sont pas seulement des prédictions
# mais des prédictions en boucle à partir d'un état
# initial. On utilise la méthode :meth:`simulate
# <aftercovid.models.EpidemicRegessor.simulate>`.

predictions = model.simulate(X[-30:], 21)
df = pandas.DataFrame(
    predictions[9], columns='Spred Ipred Rpred Dpred'.split()).set_index(
        pandas.to_datetime(dates[-21:]))
df.tail()

############################################
# On représente les prédictions.
# *Ipred* est la valeur prédite, *Itrue*
# est la valeur connue. Idem pour *Dpred* et *Dtrue*.

fig, ax = plt.subplots(2, 3, figsize=(14, 6))

df = pandas.DataFrame(
    predictions[9], columns='S Ipred R Dpred'.split()).set_index(
        pandas.to_datetime(dates[-21:]))
df[["Ipred"]].plot(ax=ax[0, 0])
df[["Dpred"]].plot(ax=ax[1, 0])
df = pandas.DataFrame(X[-21:], columns='S Itrue R Dtrue'.split()
                      ).set_index(pandas.to_datetime(dates[-21:]))
df[["Itrue"]].plot(ax=ax[0, 0])
df[["Dtrue"]].plot(ax=ax[1, 0])
ax[0, 0].set_title(f"Prediction à partir de {dates[-21]}")

dt = pandas.to_datetime(dates[-7])
dates2 = pandas.to_datetime([dt + timedelta(i) for i in range(21)])
df = pandas.DataFrame(
    predictions[-7], columns='S Ipred R Dpred'.split()).set_index(dates2)
df[["Ipred"]].plot(ax=ax[0, 1])
df[["Dpred"]].plot(ax=ax[1, 1])
df = pandas.DataFrame(X[-7:], columns='S Itrue R Dtrue'.split()
                      ).set_index(pandas.to_datetime(dates[-7:]))
df[["Itrue"]].plot(ax=ax[0, 1])
df[["Dtrue"]].plot(ax=ax[1, 1])
ax[0, 1].set_title(f"Prediction à partir de {dates[-7]}")

dt = pandas.to_datetime(dates[-1])
dates2 = pandas.to_datetime([dt + timedelta(i) for i in range(21)])
df = pandas.DataFrame(
    predictions[-7], columns='S Ipred R Dpred'.split()).set_index(dates2)
df[["Ipred"]].plot(ax=ax[0, 2])
df[["Dpred"]].plot(ax=ax[1, 2])
ax[0, 1].set_title(f"Prediction à partir de {dates[-1]}")
plt.show()

#################################################
# Ces prédictions varient beaucoup car une petite imprécision
# sur l'estimation a beaucoup d'impact à moyen terme.
