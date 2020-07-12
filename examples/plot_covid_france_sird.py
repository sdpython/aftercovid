# coding: utf-8
"""
.. _l-sir-france-example:

Estimation des paramètres d'un modèle SIRD pour la France
=========================================================

On récupère les données réelles pour un pays
et on cherche à estimer un modèle
:class:`CovidSIRD <aftercovid.models.CovidSIRD>`.

.. contents::
    :local:

Récupération des données
++++++++++++++++++++++++
"""
from aftercovid.models import CovidSIRD, EpidemicRegressor
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
df[['recovered', 'confirmed', 'infected']].diff().plot(
    title="Différences", ax=ax[1])
df[['deaths']].diff().plot(title="Différences", ax=ax[2])

#########################################
# On lisse car les séries sont très agitées
# et empêchent les modèles de bien converger.

df = df.rolling(7, center=True).mean()
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
df.plot(logy=True, title="Données COVID lissées", ax=ax[0])
df[['recovered', 'confirmed', 'infected']].diff().plot(
    title="Différences", ax=ax[1])
df[['deaths']].diff().plot(title="Différences", ax=ax[2])

################################################
# On voit qu'en France, les données sont difficilement
# exploitables en l'état. Et on sait qu'en France
# la pénurie de tests implique une sous-estimation
# du nombre de cas positifs. L'estimation du modèle
# est très compromise.

############################################
# Estimation d'un modèle
# ++++++++++++++++++++++


model = CovidSIRD()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = EpidemicRegressor(
                'SIR', learning_rate_init=lr, max_iter=500,
                early_th=1, verbose=verbose, init=m)
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
    for k in range(0, X.shape[0] - delay + 1, 7):
        end = min(k + delay, X.shape[0])
        Xt, yt = X[k:end], y[k:end]
        if any(numpy.isnan(Xt.ravel())) or any(numpy.isnan(yt.ravel())):
            continue
        m, loss, lr = find_best_model(
            Xt, yt, [1e8, 1e6, 1e4, 1e2, 1,
                     1e-2, 1e-4, 1e-6], 10,
            init=m)
        if m is None:
            print("k={} loss=nan".format(k))
            find_best_model(
                Xt, yt, [1e8, 1e6, 1e4, 1e2, 1,
                         1e-2, 1e-4, 1e-6], 10,
                init=m, verbose=True)
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
# Graphe sur le dernier mois.

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

dfcoef = estimation(X, y, 7)
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

dfcoef = estimation(X, y, 14)
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

dfcoef = estimation(X, y, 28)
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
