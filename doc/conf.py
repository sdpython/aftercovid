#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))
from aftercovid import __version__  # noqa


def snow():
    now = datetime.now()
    return "%02d-%02d-%04d" % (now.day, now.month, now.year)


extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_gallery.gen_gallery',
    'alabaster',
    'matplotlib.sphinxext.plot_directive',
    'pyquickhelper.sphinxext.sphinx_cmdref_extension',
    'pyquickhelper.sphinxext.sphinx_collapse_extension',
    'pyquickhelper.sphinxext.sphinx_docassert_extension',
    'pyquickhelper.sphinxext.sphinx_epkg_extension',
    'pyquickhelper.sphinxext.sphinx_exref_extension',
    'pyquickhelper.sphinxext.sphinx_faqref_extension',
    'pyquickhelper.sphinxext.sphinx_gdot_extension',
    'pyquickhelper.sphinxext.sphinx_runpython_extension',
]

templates_path = ['_templates']
html_logo = '_static/logo.png'
source_suffix = '.rst'
master_doc = 'index'
project = 'aftercovid'
copyright = 'Xavier Dupré - ' + snow()
author = 'Xavier Dupré'
version = __version__
release = __version__
language = 'en'
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = True

import alabaster  # noqa
html_theme = "alabaster"
html_theme_path = [alabaster.get_path()]

html_theme_options = {}
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

htmlhelp_basename = 'aftercovid'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc,
     'aftercovid.tex',
     'Documentation',
     'Xavier Dupré',
     'manual'),
]

texinfo_documents = [
    (master_doc, 'aftercovid', 'aftercovid Documentation',
     author, 'aftercovid', 'One line description of project.',
     'Miscellaneous'),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(
        sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None)
}

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': os.path.join(os.path.dirname(__file__), '../examples'),
    # path where to save gallery generated examples
    'gallery_dirs': 'auto_examples',
    'capture_repr': ('_repr_html_', '__repr__'),
    'ignore_repr_types': r'matplotlib.text|matplotlib.axes',
}

epkg_dictionary = {
    'C': 'https://en.wikipedia.org/wiki/C_(programming_language)',
    'C++': 'https://en.wikipedia.org/wiki/C%2B%2B',
    'COVID': 'https://en.wikipedia.org/wiki/Coronavirus_disease_2019',
    'covidtracker': 'https://covidtracker.fr/covidtracker-france/',
    'CSSE Johns Hopkins': 'https://github.com/CSSEGISandData/COVID-19',
    'cython': 'https://cython.org/',
    'DOT': 'https://www.graphviz.org/doc/info/lang.html',
    'epyestim': 'https://github.com/lo-hfk/epyestim',
    'INSEE': 'https://www.insee.fr/fr/accueil',
    'pyepydemic': 'https://pyepydemic.readthedocs.io/en/latest/index.html',
    'numpy': 'https://numpy.org/',
    'pyinstrument': 'https://github.com/joerick/pyinstrument',
    'python': 'https://www.python.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'SEIR': 'http://www.public.asu.edu/~hnesse/classes/seir.html',
    'SEIHFR': 'https://freakonometrics.hypotheses.org/60514',
    'SIR': ('https://fr.wikipedia.org/wiki/Mod%C3%A8les_compartimentaux_en'
            '_%C3%A9pid%C3%A9miologie'),
    'sphinx-gallery': 'https://github.com/sphinx-gallery/sphinx-gallery',
    'sympy': 'https://www.sympy.org/en/index.html',
}
