PyDSTool v0.88
==============

[![Build Status](https://travis-ci.org/robclewley/pydstool.svg?branch=master)](https://travis-ci.org/robclewley/pydstool) [![Coverage Status](https://coveralls.io/repos/robclewley/pydstool/badge.png?branch=master)](https://coveralls.io/r/robclewley/pydstool?branch=master)

Dec 2012. *This is a beta release version.*

* * * * *

### Requirements
* [Python](http://www.python.org) 2.6 or 2.7;
* [numpy](http://www.numpy.org) > 1.6;
* [scipy](http://www.scipy.org) > 0.9.

#### Optional requirements
* [matplotlib](http://www.matplotlib.org)
    Matplotlib is needed for plotting functionality and running many of the 
    examples.

* [swig](http://www.swig.org) > 2.0
    SWIG is needed to compile Radau and Dopri ODE generators.

#### Recommended
* [ipython](http://www.ipython.org)

### Installation
#### Debian/Ubuntu
* install necessary packages:
```
    sudo apt-get update
    sudo apt-get install -qq gfortran swig
    sudo apt-get install -qq python-numpy python-scipy python-matplotlib
```

* install `PyDSTool`:
```
    sudo python setup.py install
```
    or
```
    sudo python setup.py develop
```

    Run without `sudo` and add flag `--user` to install for local user only.

### Getting started and documentation

See the [online documentation](http://pydstool.sourceforge.net), particularly the GettingStarted and Tutorials pages! Please report bugs and suggestions using the user forum linked to there.

Full API documentation can be found locally [here](./html/index.html). It is auto-generated from the source code using Epydoc and is meant as a reference for functions and classes only.

### Tests and examples
#### Running examples
Examples can be found in the `examples` directory. Some examples can only be run once, others have produced their output, for instance HH\_loaded.py, HH\_loaded\_dopri.py. Several of the examples require an external compiler. An easy way to run all the examples in an appropriate order is to run the script
'run\_all\_tests.py': 
```
    cd examples
    python run_all_tests.py
```

There is a simple option in that file that you may need to edit in order to select whether the external compiler tests should be run (in case you do not have `gcc` and `gfortran` working on your system).

Note that on some platforms you will see an error report when the script tries to automatically close matplotlib graph windows, to the effect of:
```
"Fatal Python error: PyEval_RestoreThread: NULL tstate"
```
This error can be ignored. You may also have to close the plot windows yourself before the script can continue. This will depend on your platform and settings.

#### Running test suite
```
    python setup.py test
```

This requires [py.test](http://www.pytest.org), install it using `pip`:
```
    sudo pip install py.test
```
or with package manager:
```
    sudo apt-get install python-pytest
```

#### Getting coverage report
- install `pytest-cov`
```
    sudo pip install pytest-cov
```

- run `py.test`
```
    py.test --cov PyDSTool --cov-report html --cov-config .coveragerc
```

- open file `htmlcov/index.html` in your browser


### Change histories

Version change histories appear in the [bzr](http://pydstool.bzr.sourceforge.net/bzr/pydstool/changes) repository browser, and some older information is in the headers of each source file. An overview of the changes in a new release can be found in the SourceForge release notes.

* * * * *

Credits:
--------

Coding and design by Robert Clewley, Erik Sherwood, Drew LaMar, and John Guckenheimer, except where otherwise stated in the code or documentation. (Several other open source codes have been redistributed here under the compatible licenses.)
