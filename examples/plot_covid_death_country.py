# coding: utf-8
"""
Visualisation des données par pays
==================================

Ces séries représentent celles des décès. Le nombre de cas positifs
est différent selon les pays de par leur capacité de tests et
au nombreux cas asymptomatiques.

.. contents::
    :local:

Récupération des données
++++++++++++++++++++++++
"""

from aftercovid.preprocess import ts_normalise_negative_values
import matplotlib.pyplot as plt
import pandas

url = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
       "master/csse_covid_19_data/"
       "csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
df = pandas.read_csv(url)
df.head()

#######################################
#

df.tail()


#################################
#  Tous les pays

print(" --- ".join(sorted(set(df['Country/Region']))))


##########################################
# On en sélectionne quelques-uns.


keep = ['France', 'Belgium', 'Italy', 'United Kingdom',
        'US', 'Spain', 'Germany']
eur = df[df['Country/Region'].isin(keep) & df['Province/State'].isna()]
eur

#######################################
# En colonne.

cols = list(eur['Country/Region'])
tf = eur.T.iloc[4:]
tf.columns = cols
tf.tail()


########################################
# Nombre de décès par pays
# ++++++++++++++++++++++++

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
tf.plot(logy=False, lw=3, title="Nombre de décès COVID", ax=ax[0])
tf.plot(logy=True, lw=3, title="Nombre de décès COVID", ax=ax[1])


#########################################
#

cols = list(eur['Country/Region'])
tf = eur.T.iloc[4:]
tf.columns = cols
tf.tail()


#########################################
# On lisse sur une semaine.

tdroll = tf.rolling(7, center=True, win_type='triang').mean()
tdroll.tail()

##################################
# Séries lissées.


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
tdroll.plot(logy=False, lw=3, ax=ax[0],
            title="Nombre de décès COVID lissé sur une semaine")
tdroll.plot(logy=True, lw=3, ax=ax[1],
            title="Nombre de décès COVID lissé sur une semaine")


################################
# Séries décalées
# +++++++++++++++
#
# On compare les séries en prenant comme point de
# départ la date qui correspond au 20ième décès.


def find_day(ts, th):
    tsth = ts[ts >= th]
    return tsth.index[0]


def delag(ts, th=21, begin=-2):
    index = find_day(ts, th)
    loc = ts.index.get_loc(index)
    values = ts.reset_index(drop=True)
    return values[loc + begin:].reset_index(drop=True)


print(find_day(tdroll['France'], 25), delag(tdroll['France'])[:15])


####################################
# On décale pour chaque pays.


data = {}
for c in tdroll.columns:
    data[c] = delag(tdroll[c], 21)
dl = pandas.DataFrame(data)
dl.head()

####################################
#

dl.tail()

####################################
# Graphes.


fig, ax = plt.subplots(1, 2, figsize=(14, 8))
dl.plot(logy=False, lw=3, ax=ax[0])
dl.plot(logy=True, lw=3, ax=ax[1])
ax[0].set_title(
    "Nombre de décès après N jours\ndepuis le début de l'épidémie")

##########################################
# Le fléchissement indique que la propagation n'est plus logarithmique
# après la fin du confinement.
#
# Séries différentielles
# ++++++++++++++++++++++
#
# C'est surtout celle-ci qu'on regarde pour contrôler
# l'évolution de l'épidémie. Certaines valeurs
# sont négatives laissant penser que la façon
# de reporter les décès a évolué au cours du temps.
# C'est problématique lorsqu'on souhaite caler un modèle.


neg = tf.diff()
neg[neg['Spain'] < 0]

#############################################
# Et pour la France.

neg[neg['France'] < 0]

#############################################
# On continue néanmoins mais en corrigeant ces séries
# qu'il n'y ait plus aucune valeur négative.


tfdiff = ts_normalise_negative_values(tf.diff()).rolling(
    7, center=False, win_type='triang').mean()

########################################

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
tfdiff.plot(
    logy=False, lw=3, ax=ax[0],
    title="Nombre de décès COVID par jour lissé par semaine")
ax[0].set_ylim(0)
tfdiff.plot(
    logy=True, lw=3, ax=ax[1],
    title="Nombre de décès COVID par jour lissé par semaine")

#############################################
# Les mêmes chiffres en recalant les séries
# au jour du 20ième décès.


dldiff = ts_normalise_negative_values(dl.diff()).rolling(
    7, center=False, win_type='triang').mean()

print(",".join(map(str, dl.diff()['Spain'])))

################################


fig, ax = plt.subplots(1, 2, figsize=(14, 8))
dldiff.plot(logy=False, lw=3, ax=ax[0])
dldiff.plot(logy=True, lw=3, ax=ax[1])
ax[0].set_ylim(0)
ax[0].set_title(
    "Nombre de décès lissé sur 7 jours\npar jour après N jours"
    "\ndepuis le début de l'épidémie")

################################

tfdiff = ts_normalise_negative_values(tf.diff(), extreme=2).rolling(
    7, center=False, win_type='triang').mean()


fig, ax = plt.subplots(1, 2, figsize=(14, 8))
tfdiff.plot(logy=False, lw=3, ax=ax[0])
tfdiff.plot(logy=True, lw=3, ax=ax[1])
ax[0].set_ylim(0)
ax[0].set_title(
    "Nombre de décès lissé sur 7 jours")

plt.show()
