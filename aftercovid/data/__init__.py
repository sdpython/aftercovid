"""
Shortcuts to *data*.
"""

from .data_hopkins import (  # noqa
    download_hopkins_data, extract_hopkins_data, preprocess_hopkins_data)
from .data_insee import (  # noqa
    data_covid_france_departments_hospitals,
    data_covid_france_departments_tests,
    data_france_departments)
from .temperatures import load_temperatures  # noqa
from .image_grid import load_grid_image  # noqa
