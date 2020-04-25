
.. image:: https://circleci.com/gh/sdpython/aftercovid/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/aftercovid/tree/master

.. image:: https://travis-ci.org/sdpython/aftercovid.svg?branch=master
    :target: https://travis-ci.org/sdpython/aftercovid
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/ffne8nhh96jdqo4p?svg=true
    :target: https://ci.appveyor.com/project/sdpython/aftercovid
    :alt: Build Status Windows

.. image:: https://codecov.io/gh/sdpython/aftercovid/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/sdpython/aftercovid

.. image:: https://badge.fury.io/py/aftercovid.svg
    :target: http://badge.fury.io/py/aftercovid

.. image:: http://img.shields.io/github/issues/sdpython/aftercovid.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/aftercovid/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

aftercovid: epidemics simulation
==============================

.. image:: https://raw.githubusercontent.com/sdpython/aftercovid/master/doc/_static/logo.png
    :width: 50

`documentation <http://www.xavierdupre.fr/app/aftercovid/helpsphinx/index.html>`_

Tools, tries about COVID epidemics.
The module must be compiled to be used inplace:

::

    python setup.py build_ext --inplace

Generate the setup in subfolder ``dist``:

::

    python setup.py sdist

Generate the documentation in folder ``dist/html``:

::

    python -m sphinx -T -b html doc dist/html

Run the unit tests:

::

    python -m unittest discover tests

Or:

::

    python -m pytest
    
To check style:

::

    python -m flake8 aftercovid tests examples

The function *check* or the command line ``python -m aftercovid check``
checks the module is properly installed and returns processing
time for a couple of functions or simply:

::

    import aftercovid
    aftercovid.check()
