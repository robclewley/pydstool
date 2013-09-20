#!/usr/bin/env python
"""
Setup script for PyDSTool

This uses Distutils, the standard Python mechanism for installing packages.
For the easiest installation just type::

python setup.py install

(root privileges probably required). If you'd like to install in a custom
directory, such as your home directoy, type the following to install
PyDSTool under `<dir>/lib/python`::

python setup.py install --home=<dir>

In addition, there are some other commands::

python setup.py clean - Clean all trash (*.pyc, emacs backups, etc.)
python setup.py test  - Run test suite

"""

import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup, os, find_packages
from setuptools.command.test import test as TestCommand
from setuptools import Command
import sys, sysconfig

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def check_dependency_versions():
    try:
	assert(int(sys.version_info.major) < 3)
    except AssertionError:
	raise AssertionError("PyDSTool only work with Python<3")

class clean(Command):
    description = 'Remove build and trash files'
    user_options = [("all", "a", "the same")]
    def initialize_options(self):
        self.all = None
    def finalize_options(self):
        pass
    def run(self):
        import os
        os.system("rm -fr ./*.pyc ./*~ ./*/*.pyc ./*/*~ ./*/*/*.pyc ./*/*/*~ ./*/*/*.so ./PyDSTool/tests/auto_temp ./PyDSTool/tests/dopri853_temp ./PyDSTool/tests/radau5_temp ./PyDSTool/tests/dop853* ./PyDSTool/tests/radau5* ./PyDSTool/tests/*.pkl ./PyDSTool/tests/fort.9")
        os.system("rm -fr build")
        os.system("rm -fr dist")
#        os.system("rm -fr doc/_build")

""" This will use the new test suite structure in PyDSTool/tests
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)
"""

""" This runs the original PyDSTool test scripts in PyDSTool/PyDSTool/tests
"""
class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'run_all_tests.py'], cwd='PyDSTool/tests/')
        raise SystemExit(errno)

check_dependency_versions()
setup(
    name = "PyDSTool",
    version = "0.88-20130406",
    packages = find_packages(),
    install_requires = [
			"matplotlib",
			"scipy>=0.9",
			"numpy"
			],
    tests_require = ['pytest'],
    cmdclass = {'test': PyTest,
		'clean': clean
		},
    author = "Rob Clewley; W. Erik Sherwood; M. Drew Lamar",
    maintainer = "Rob Clewley",
    maintainer_email = "rclewley@gsu.edu",
    description = ("Python dynamical systems simulation and modeling"),
    long_description = read('README.txt'),
    license = "BSD",
    keywords = "dynamical systems, bioinformatics, modeling, bifurcation analysis",
    url = "http://pydstool.sourceforge.net",
    include_package_data=True,
    platforms = ["any"],
    package_data = { 
			'': ['*.txt', '*.rst'], 

		   },
    classifiers = [
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Linux",
        ],

    )

