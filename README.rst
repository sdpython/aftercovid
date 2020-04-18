
.. image:: https://circleci.com/gh/sdpython/covidsim/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/covidsim/tree/master

.. image:: https://travis-ci.org/sdpython/covidsim.svg?branch=master
    :target: https://travis-ci.org/sdpython/covidsim
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/wvo6ovlaxi8ypua4?svg=true
    :target: https://ci.appveyor.com/project/sdpython/td3a-cpp
    :alt: Build Status Windows

.. image:: https://dev.azure.com/xavierdupre3/covidsim/_apis/build/status/sdpython.covidsim
    :target: https://dev.azure.com/xavierdupre3/covidsim/

.. image:: https://badge.fury.io/py/covidsim.svg
    :target: http://badge.fury.io/py/covidsim

.. image:: http://img.shields.io/github/issues/sdpython/covidsim.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/covidsim/issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

covidsim: epidemics simulation
==============================

.. image:: https://raw.githubusercontent.com/sdpython/covidsim/master/doc/_static/logo.png
    :width: 50

`documentation <http://www.xavierdupre.fr/app/covidsim/helpsphinx/index.html>`_

Tools, tries about :epkg:`COVID` epidemics.
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

    python -m flake8 covidsim tests examples

The function *check* or the command line ``python -m covidsim check``
checks the module is properly installed and returns processing
time for a couple of functions or simply:

::

    import covidsim
    covidsim.check()
