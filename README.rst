PyDSTool
========

|buildstatus|_ |coverage|_

*This is a beta release version.*

PyDSTool is a sophisticated & integrated simulation and analysis environment
for dynamical systems models of physical systems (ODEs, DAEs, maps, and hybrid
systems).

PyDSTool is platform independent, written primarily in Python with some
underlying C and Fortran legacy code for fast solving. It makes extensive use
of the numpy and scipy libraries. PyDSTool supports symbolic math,
optimization, phase plane analysis, continuation and bifurcation analysis, data
analysis, and other tools for modeling -- particularly for biological
applications.

The project is fully open source with a BSD license, and welcomes contributions
from the community.

See more at `pydstool.sourceforge.net <http://pydstool.sourceforge.net>`__.

--------------

Requirements
~~~~~~~~~~~~

*  `Python <http://www.python.org>`__ 2.7 or 3.3+;
*  `numpy <http://www.numpy.org>`__;
*  `scipy <http://www.scipy.org>`__.

Dopri/Radau and AUTO interface requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  `swig <http://www.swig.org>`__ > 2.0;
*  C compiler (e.g, gcc or clang);
*  GNU Fortran compiler (Radau only).

Optional requirements
^^^^^^^^^^^^^^^^^^^^^

*  `matplotlib <http://www.matplotlib.org>`__
   Matplotlib is needed for plotting functionality and running many of the examples.

Recommended
^^^^^^^^^^^

*  `ipython <http://www.ipython.org>`__

Installation
~~~~~~~~~~~~

Debian/Ubuntu
^^^^^^^^^^^^^

*  install necessary packages:

   ::

           sudo apt-get update
           sudo apt-get install -qq gfortran swig
           sudo apt-get install -qq python-numpy python-scipy python-matplotlib

*  install ``PyDSTool``:

   ::

           sudo python setup.py install

   or

   ::

           sudo python setup.py develop

   Run without ``sudo`` and add flag ``--user`` to install for local
   user only.

Getting started and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the `online documentation <http://pydstool.sourceforge.net>`__,
particularly the GettingStarted and Tutorials pages! Please report bugs
and suggestions using the user forum linked to there.

Tests and examples
~~~~~~~~~~~~~~~~~~

Running examples
^^^^^^^^^^^^^^^^

Examples can be found in the ``examples`` directory. Some examples can
only be run once, others have produced their output, for instance
HH\_loaded.py, HH\_loaded\_dopri.py. Several of the examples require an
external compiler. An easy way to run all the examples in an appropriate
order is to run the script 'run\_all\_tests.py':

::

        cd examples
        python run_all_tests.py

There is a simple option in that file that you may need to edit in order
to select whether the external compiler tests should be run (in case you
do not have ``gcc`` and ``gfortran`` working on your system).

Note that on some platforms you will see an error report when the script
tries to automatically close matplotlib graph windows, to the effect of:

::

    "Fatal Python error: PyEval_RestoreThread: NULL tstate"

This error can be ignored. You may also have to close the plot windows
yourself before the script can continue. This will depend on your
platform and settings.

Running test suite
^^^^^^^^^^^^^^^^^^

To run test suite, install `py.test <http://www.pytest.org>`__ and
`mock <http://www.voidspace.org.uk/python/mock/>`__, using ``pip``:

::

        sudo pip install py.test mock

or package manager:

::

        sudo apt-get install python-pytest python-mock

Then run:

::

        python setup.py test


Getting coverage report
^^^^^^^^^^^^^^^^^^^^^^^

*  install ``pytest-cov``

   ::

           sudo pip install pytest-cov

*  run ``py.test``

   ::

           py.test --cov PyDSTool --cov-report html --cov-config .coveragerc

*  open file ``htmlcov/index.html`` in your browser

Credits
~~~~~~~

Coding and design by Robert Clewley, Erik Sherwood, Drew LaMar, Vladimir
Zakharov, and John Guckenheimer, except where otherwise stated in the
code or documentation. (Several other open source codes have been
redistributed here under the compatible licenses.)

--------------



.. |buildstatus| image:: https://travis-ci.org/robclewley/pydstool.svg?branch=master
.. _buildstatus: https://travis-ci.org/robclewley/pydstool

.. |coverage| image:: https://coveralls.io/repos/robclewley/pydstool/badge.png?branch=master
.. _coverage: https://coveralls.io/r/robclewley/pydstool?branch=master
