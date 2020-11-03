"""
Shortcuts to *data*.

Source:

* `temperature_france.xlsx`:
  `meteociel <https://www.meteociel.fr/climatologie/obs_villes.php?
  code2=75107005&mois=11&annee=2020>`_
"""

from .data_hopkins import (  # noqa
    download_hopkins_data, extract_hopkins_data, preprocess_hopkins_data)
from .temperatures import load_temperatures  # noqa
from .image_grid import load_grid_image  # noqa
