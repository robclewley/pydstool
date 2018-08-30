#!/usr/bin/env python
"""
Setup script for PyDSTool

This uses Distutils, the standard Python mechanism for installing packages.
For the easiest installation just type::

    python setup.py install

(root privileges probably required). If you'd like to install only for local
user, type the following to install PyDSTool::

    python setup.py install --user

In addition, there are some other commands::

python setup.py clean - Clean all trash (*.pyc, emacs backups, etc.)
python setup.py test  - Run test suite

"""


from setuptools import setup, os, find_packages
from setuptools.command.test import test as TestCommand
from setuptools import Command
import sys

MAJOR = 0
MINOR = 90
MICRO = 2
__version__ = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[0:2] < (3, 3):
    raise RuntimeError("Python version 2.7 or >= 3.3 required.")


class clean(Command):
    description = 'Remove build and trash files'
    user_options = [("all", "a", "the same")]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        os.system(
            "rm -fr ./*.pyc ./*~ ./*/*.pyc ./*/*~ ./*/*/*.pyc ./*/*/*~ "
            "./*/*/*.so ./PyDSTool/tests/auto_temp "
            "./PyDSTool/tests/dopri853_temp ./PyDSTool/tests/radau5_temp "
            "./PyDSTool/tests/dop853* ./PyDSTool/tests/radau5* "
            "./PyDSTool/tests/*.pkl ./PyDSTool/tests/fort.9")
        os.system("rm -rf tests/radau5_temp tests/dopri853_temp radau5_temp "
                  "dopri853_temp")
        os.system("rm -fr build")
        os.system("rm -fr dist")
        # os.system("rm -fr doc/_build")


class PyTest(TestCommand):

    def finalize_options(self):
        self.test_suite = 'tests'
        TestCommand.finalize_options(self)

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        os.system("rm -rf dop853_temp radau5_temp auto_temp")
        sys.exit(errno)


def get_datafiles():
    source_dirs = ['examples', 'tests']
    datafiles = []
    for s in source_dirs:
        for d, _, files in os.walk(s):
            datafiles.append((d, [os.path.join(d, f) for f in files]))
    return datafiles


needs_pytest = {'test', 'pytest', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []


setup(
    name="PyDSTool",
    version=__version__,
    packages=find_packages(),
    setup_requires=pytest_runner,
    install_requires=[
        "six",
        "scipy>=1.0,<2.0",
        "numpy>=1.6"
    ],
    tests_require=['pytest', 'pytest-mock', 'pytest-xdist'],
    cmdclass={
        'test': PyTest,
        'clean': clean
    },
    author="Rob Clewley; W. Erik Sherwood; M. Drew Lamar; Vladimir Zakharov",
    author_email="rob.clewley@gmail.com",
    maintainer="Rob Clewley",
    maintainer_email="rob.clewley@gmail.com",
    description="Python dynamical systems simulation and modeling",
    long_description=read('README.rst') + '\n\n' + read('WHATS_NEW.txt'),
    license="BSD",
    keywords="dynamical systems, bioinformatics, modeling, bifurcation analysis",
    url="http://pydstool.sourceforge.net/",
    download_url="https://github.com/robclewley/pydstool/tarball/v%s" % __version__,
    include_package_data=True,
    platforms=["any"],
    package_data={
        '': ['*.txt', '*.rst'],
    },
    data_files=get_datafiles(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: BSD :: FreeBSD",
        "Operating System :: POSIX :: Linux",
    ],
)
