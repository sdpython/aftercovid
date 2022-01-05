# coding: utf-8
"""
Dashboard COVID France
======================

Graphe simple fait à partir des données sur open.data.gouv.fr.

.. contents::
    :local:

Récupération des données
++++++++++++++++++++++++
"""
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime

now = datetime.now()
name = "covid-%d-%02d-%02d.csv" % (now.year, now.month, now.day)
if os.path.exists(name):
    covid = read_csv(name)
else:
    url = (
        "https://www.data.gouv.fr/en/datasets/r/"
        "d3a98a30-893f-47f7-96c5-2f4bcaaa0d71")
    covid = read_csv(url, sep=",")
    covid.to_csv(name, index=False)

covid['date'] = to_datetime(covid['date'])
now_4 = now - timedelta(120)
covid = covid[covid.date >= now_4]
covid = covid.set_index('date')
covid.tail()

##################################################
# Cas positifs et décès
# +++++++++++++++++++++

diff = covid.diff()
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
diff['cas_positifs'] = diff['total_cas_confirmes'].rolling(
    7, center=False).mean()
diff['deces'] = diff['total_deces_hopital'].rolling(7, center=False).mean()
diff[["total_cas_confirmes", "cas_positifs"]].plot(
    ax=ax[0], title="Cas positifs")
diff[["total_deces_hopital", "deces"]].plot(ax=ax[1], title="Décès")

#################################################
# Hôpital
# +++++++

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
covid['weeklyh'] = covid['nouveaux_patients_hospitalises'].rolling(
    7, center=False).mean()
covid['weeklyr'] = covid['nouveaux_patients_reanimation'].rolling(
    7, center=False).mean()
covid[["nouveaux_patients_hospitalises", "weeklyh"]].plot(
    ax=ax[0, 0], title="Hospitalisations")
covid[["nouveaux_patients_reanimation", "weeklyr"]].plot(
    ax=ax[1, 0], title="Réanimations")

diff['weeklyh'] = diff['patients_hospitalises'].rolling(7, center=False).mean()
diff['weeklyg'] = diff['total_patients_gueris'].rolling(7, center=False).mean()
diff['weeklyr'] = diff['patients_reanimation'].rolling(7, center=False).mean()
diff[["patients_hospitalises", "weeklyh", "total_patients_gueris",
      "weeklyg"]].plot(ax=ax[0, 1], title="Delta Hospitalisations")
diff[["patients_reanimation", "weeklyr"]].plot(
    ax=ax[1, 1], title="Delta Réanimations")

covid['weeklyph'] = covid['patients_hospitalises'].rolling(
    7, center=False).mean()
covid['weeklypr'] = covid['patients_reanimation'].rolling(
    7, center=False).mean()
covid[["patients_hospitalises", "weeklyph"]].plot(
    ax=ax[2, 0], title="Lits Hospitalisations")
covid[["patients_reanimation", "weeklypr"]].plot(
    ax=ax[2, 1], title="Lits Réanimations")

plt.show()
