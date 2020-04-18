
Developers' corner
==================

.. contents::
    :local:

Build, documentation, unittests
+++++++++++++++++++++++++++++++

Build the module inplace:

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

Setup
+++++

Instructions can be found in CI files:

* `Windows <https://github.com/sdpython/aftercovid/blob/master/appveyor.yml>`_
* `Linux (Debian) <https://github.com/sdpython/aftercovid/blob/master/.circleci/config.yml>`_
* `Linux (Ubuntu) <https://github.com/sdpython/aftercovid/blob/master/.travis.yml>`_
* `Mac OSX <https://github.com/sdpython/aftercovid/blob/master/azure-pipelines.yml#L50>`_
