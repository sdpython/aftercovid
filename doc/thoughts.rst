
Pensées
=======

.. contents::
    :local:

Sur le SIR
++++++++++

Je me demandais pourquoi un chercheur comme Didier Raoult disait
qu'il n'y aurait pas de seconde vague alors qu'il connaît
probablement les modèles SIR. A priori, on n'observe pas
vraiment de rebond en Chine non plus. Donc je me suis posé
la question d'un modèle qui pourrait éviter de prédire une seconde vague.
Je pense que le modèle SIR suppose en quelque sorte une répartition
homogène des contaminés au sein de la population, donc une diffusion
beaucoup plus rapide que la contamination. C'est peut-être vrai dans
le monde animal ou au début de l'épidémie mais ensuite on confine.
On sait que ça fait baisser le beta dans le modèle SIR mais ça implique
de réestimer le modèle. Ma question était donc de savoir s'il existe
un modèle pour lequel on n'a pas besoin de réestimer et qui serait bon
sur le long terme. Si je pousse le raisonnement plus loin, comme les
gens évitent les contacts, appliquent les gestes barrières, je me demande
s'il ne serait pas plus juste de modéliser l'épidémie comme un liquide
visqueux et mélanger l'aspect contamination et l'aspect visquosité.
Ca penche en faveur des modèles à base de simulations de personnes.
J'ai trouvé des modèles type Spatio-Temporal Point Processes :

* `Basics and recent developments on spatio-temporal point processes
  <https://informatique-mia.inra.fr/resste/sites/informatique-mia.inra.fr.resste/files/Gabriel_RESSTE2017.pdf>`_
* `On spatial and spatio-temporal multi-structure point process models
  <https://arxiv.org/abs/2003.01962>`_
* `Spatio-Temporal Point Processes
  <https://web.stanford.edu/class/stats253/lectures_2014/lect10.pdf>`_

Et d'autres choses :

* `surveillance: Temporal and Spatio-Temporal Modeling and Monitoring of Epidemic Phenomena
  <https://cran.r-project.org/web/packages/surveillance/index.html>`_
* `Population dynamics and epidemic processes on a trade network
  <https://tel.archives-ouvertes.fr/tel-02272853/>`_
* `Real-time forecasting of epidemic trajectories using computational dynamic ensembles
  <https://www.sciencedirect.com/science/article/pii/S1755436519301112>`_

Je pense qu'on pourrait approcher cela par des petits SIR localisés qui
communiquent entre eux, SIR à l'intérieur et contamination non visqueuse,
contamination visqueuse entre mini SIR et là on peut prendre en
compte la densité de la population.

Deux ou trois petites digressions
+++++++++++++++++++++++++++++++++

* `Spatio-temporal propagation of COVID-19 pandemics
  <https://www.medrxiv.org/content/10.1101/2020.03.23.20041517v2>`_
* `Spatial dependence in (origin-destination) air passenger flows
  <https://www.tse-fr.eu/articles/spatial-dependence-origin-destination-air-passenger-flows>`_
* `Interpreting of explanatory variables impacts in compositional regression models
  <https://www.tse-fr.eu/articles/interpreting-explanatory-variables-impacts-compositional-regression-models>`_
* `Processus ponctuels spatiaux pour l'analyse du positionnement optimal et de la concentration
  <https://tel.archives-ouvertes.fr/tel-00465270>`_
* `Convex Parameter Recovery for Interacting Marked Processes
  <https://arxiv.org/abs/2003.12935>`_
* `Population dynamics and epidemic processes on a trade network
  <https://tel.archives-ouvertes.fr/tel-02272853/>`_
* `About predictions in spatial autoregressive models
  <https://www.tse-fr.eu/articles/about-predictions-spatial-autoregressive-models>`_
* `Prédiction de l’usage des sols sur un zonage régulier à différentes résolutions et à partir de covariables facilement accessibles
  <https://www.tse-fr.eu/articles/prediction-de-lusage-des-sols-sur-un-zonage-regulier-differentes-resolutions-et-partir-de>`_
