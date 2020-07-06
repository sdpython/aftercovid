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
    conc['infected'] = conc['confirmed'] - (conc['deaths'] + conc['recovered'])
    conc['safe'] = total - conc.drop('confirmed', axis=1).sum(axis=1)
    return conc


df = extract_whole_data()
df.tail()

#########################################
# Graphes.

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
df.plot(logy=True, title="Données COVID", ax=ax[0])
df[['recovered', 'confirmed']].diff().plot(title="Différences", ax=ax[1])
df[['deaths']].diff().plot(title="Différences", ax=ax[2])

#########################################
# On lisse car les séries sont très agitées
# et empêchent les modèles de bien converger.

df = df.rolling(7, center=True).mean()
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
df.plot(logy=True, title="Données COVID lissées", ax=ax[0])
df[['recovered', 'confirmed']].diff().plot(title="Différences", ax=ax[1])
df[['deaths']].diff().plot(title="Différences", ax=ax[2])

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

data = df[['safe', 'infected', 'recovered',
           'deaths']].values.astype(numpy.float32)
print(data[:5])

X = data[:-1]
y = data[1:] - data[:-1]
dates = df.index[:-1]

#########################################
# Estimation.


def find_best_model(Xt, yt, lrs, th, verbose=0, init=None):
    best_est, best_loss, best_lr = None, None, None
    m = None
    for ilr, lr in enumerate(lrs):
        if verbose:
            print("--- TRY {}/{}: {}".format(ilr + 1, len(lrs), lr))
        tries = [None]
        if best_est is not None:
            tries.append(best_est)
        for init_m in tries:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = EpidemicRegressor(
                    'SIRC', learning_rate_init=lr, max_iter=500,
                    early_th=1, verbose=verbose, init=init_m)
                try:
                    m.fit(Xt, yt)
                except RuntimeError as e:
                    if verbose:
                        print('ERROR: {}'.format(e))
                    continue
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
    m = None
    for k in range(0, X.shape[0] - delay + 1, 2):
        end = min(k + delay, X.shape[0])
        Xt, yt = X[k:end], y[k:end]
        if any(numpy.isnan(Xt.ravel())) or any(numpy.isnan(yt.ravel())):
            continue
        m, loss, lr = find_best_model(
            Xt, yt, [1e8, 1e6, 1e4, 1e2, 1,
                     1e-2, 1e-4, 1e-6], 10,
            init=m, verbose=0)
        if m is None:
            print("k={} loss=nan".format(k))
            find_best_model(
                Xt, yt, [1e8, 1e6, 1e4, 1e2, 1,
                         1e-2, 1e-4, 1e-6], 10, verbose=True)
            continue
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
# Fin de la période.

dfcoef.tail(n=10)

#############################################
# Statistiques.

dfcoef.describe()

#####################################
# Fin de la période.

df['cacheR'] = (dfcoef['cR'] * df['N'] * 1e-5).fillna(method='bfill')
df['cacheS'] = (dfcoef['cS'] * df['N'] * 1e-5).fillna(method='bfill')
df.tail(n=10)

#############################################
# Statistiques.

df.describe()

#############################################
# Graphe.

dfcoef['R0=1'] = 1

fig, ax = plt.subplots(2, 3, figsize=(14, 6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    dfcoef[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoef[["beta"]].plot(ax=ax[0, 1])
    dfcoef[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoef[["R0", "R0=1"]].plot(ax=ax[0, 2])
    dfcoef[["cS", "cR"]].plot(ax=ax[1, 2])
    ax[0, 2].set_ylim(0, 5)
    df.drop('safe', axis=1).plot(ax=ax[1, 1])
    fig.suptitle('Estimation de R0 tout au long de la période', fontsize=12)
plt.show()

#############################################
# Graphe sur le dernier mois.

dfcoeflast = dfcoef.iloc[-30:, :]
dflast = df.iloc[-30:, :]

fig, ax = plt.subplots(2, 3, figsize=(14, 6))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    dfcoeflast[["mu", "nu"]].plot(ax=ax[0, 0], logy=True)
    dfcoeflast[["beta"]].plot(ax=ax[0, 1])
    dfcoeflast[["loss"]].plot(ax=ax[1, 0], logy=True)
    dfcoeflast[["R0", "R0=1"]].plot(ax=ax[0, 2])
    ax[0, 2].set_ylim(0, 5)
    dfcoeflast[["cS", "cR"]].plot(ax=ax[1, 2])
    dflast.drop('safe', axis=1).plot(ax=ax[1, 1])
    fig.suptitle('Estimation de R0 sur la fin de la période', fontsize=12)
plt.show()
