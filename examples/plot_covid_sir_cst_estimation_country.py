# coding: utf-8
"""
Estimation des paramètres d'un modèle SIR étendu pour la France
===============================================================

Le modèle proposé dens l'exemple :ref:`l-sir-france-example`
ne fonctionne pas très bien. Les données collectées sont erronées
pour le recensement des personnes infectées. Comme les malades n'étaient
pas testées au début de l'épidémie, le nombre officiel de personnes
contaminées est en-deça de la réalité. On ajoute un paramètre
pour tenir compte dela avec le modèle :class:`CovidSIRC
<aftercovid.models.CovidSIRC>`. Le modèle suppose
en outre que la contagion est la même tout au long de la
période d'étude alors que les mesures de confinement, le port du
masque impacte significativement le coefficient de propagation.
L'épidémie peut sans doute être modélisée avec un modèle SIR
mais sur une courte période.

.. contents::
    :local:

Récupération des données
++++++++++++++++++++++++
"""
from aftercovid.models import CovidSIRC, EpidemicRegressor
import numpy
import warnings
import matplotlib.pyplot as plt
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
import pandas


population = {
    'Belgium': 11.5e6,
    'France': 67e6,
    'Germany': 83e6,
    'Spain': 47e6,
    'Italy': 60e6,
    'UK': 67e6,
}


def extract_data(kind='deaths', country='France'):
    url = (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
        "master/csse_covid_19_data/"
        "csse_covid_19_time_series/time_series_covid19_%s_global.csv" %
        kind)
    df = pandas.read_csv(url)
    eur = df[df['Country/Region'].isin([country])
             & df['Province/State'].isna()]
    tf = eur.T.iloc[4:]
    tf.columns = [kind]
    return tf


def extract_whole_data(kind=['deaths', 'confirmed', 'recovered'],
                       country='France'):
    total = population[country]
    dfs = []
    for k in kind:
        df = extract_data(k, country)
        dfs.append(df)
    conc = pandas.concat(dfs, axis=1)
    conc['total'] = total - conc.sum(axis=1)
    return conc


df = extract_whole_data()
df.tail()

#########################################
# Graphes.

df.plot(logy=True, title="Données COVID")

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


model = CovidSIRC()
print(model.quantity_names)

data = df[['total', 'confirmed', 'recovered',
           'deaths']].values.astype(numpy.float32)
print(data[:5])

X = data[:-1]
y = data[1:] - data[:-1]
dates = df.index[:-1]

#########################################
# Estimation.


def find_best_model(Xt, yt, lrs, th):
    best_est, best_loss, best_lr = None, None, None
    for lr in lrs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EpidemicRegressor(
                'SIRC',
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
            best_lr = lr
        if best_loss < th:
            return best_est, best_loss, best_lr
    return best_est, best_loss, best_lr


def estimation(X, y, delay):
    coefs = []
    for k in range(0, X.shape[0] - delay + 1, 2):
        end = min(k + delay, X.shape[0])
        Xt, yt = X[k:end], y[k:end]
        m, loss, lr = find_best_model(
            Xt, yt, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 10)
        loss = m.score(Xt, yt)
        print("k={} iter={} loss={:1.3f} coef={} R0={} lr={}".format(
            k, m.iter_, loss, m.model_._val_p, m.model_.R0(), lr))
        obs = dict(k=k, loss=loss, it=m.iter_,
                   R0=m.model_.R0(), lr=lr, date=dates[end - 1])
        obs.update({k: v for k, v in zip(
            m.model_.param_names, m.model_._val_p)})
        coefs.append(obs)
    dfcoef = pandas.DataFrame(coefs)
    dfcoef = dfcoef.set_index("date")
    return dfcoef


# 3 semaines car les séries sont cycliques
dfcoef = estimation(X, y, 21)
dfcoef.head(n=10)

#####################################
#

dfcoef.tail(n=10)

#####################################
#

df['cache'] = dfcoef['cst'] * df['total'] * 1e-5
df.tail(n=10)

#############################################
# Graphe.

dfcoef['R0=1'] = 1

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    fig, ax = plt.subplots(2, 3, figsize=(14, 6))
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1])
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0", "R0=1"]].plot(ax=ax[0, 2])
    dfcoef[["cst"]].plot(ax=ax[1, 2])
    ax[0, 2].set_ylim(0, 5)
    df.drop('total', axis=1).plot(ax=ax[1, 1])
    fig.suptitle('Estimation de R0 tout au long de la période', fontsize=12)


#############################################
# Graphe sur le dernier mois.

dfcoeflast = dfcoef.iloc[-30:, :]
dflast = df.iloc[-30:, :]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    fig, ax = plt.subplots(2, 3, figsize=(14, 6))
    dfcoeflast[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoeflast[["beta"]].plot(ax=ax[0, 1])
    dfcoeflast[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoeflast[["R0", "R0=1"]].plot(ax=ax[0, 2])
    ax[0, 2].set_ylim(0, 5)
    dfcoeflast[["cst"]].plot(ax=ax[1, 2])
    dflast.drop('total', axis=1).plot(ax=ax[1, 1])
    fig.suptitle('Estimation de R0 sur la fin de la période', fontsize=12)


plt.show()
