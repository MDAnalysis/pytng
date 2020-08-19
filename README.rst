===========================================
pytng - A python library to read TNG files!
===========================================

.. image:: https://travis-ci.org/MDAnalysis/pytng.svg?branch=master
   :target: https://travis-ci.org/MDAnalysis/pytng
.. image:: https://codecov.io/gh/MDAnalysis/pytng/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/MDAnalysis/pytng


.. Warning::

   This package is under development. It is not ready for general use.


This package provides the ``TNGFile`` object to allow simple Pythonic access to data
contained within TNG files:

.. code-block:: python

  import pytng

  with pytng.TNGFile('traj.tng', 'r') as f:
      for ts in f:
          time = ts.time
          coordinates = ts.positions

This package contains Python bindings to libtng_ for TNG file format[1_] [2_].
This is used by molecular simulation programs such as Gromacs_ for storing the
topology and results from molecular dynamics simulations.

.. _libtng: https://gitlab.com/gromacs/tng
.. _1: http://link.springer.com/article/10.1007%2Fs00894-010-0948-5
.. _2: http://onlinelibrary.wiley.com/doi/10.1002/jcc.23495/abstract
.. _Gromacs: http://manual.gromacs.org/


Installation
============

To install using pip, simply run

.. code-block:: sh

  pip install pytng

To install the latest development version from source, run

.. code-block:: sh

  git clone git@github.com:MDAnalysis/pytng.git
  cd pytng
  python setup.py install


Getting help
============

For help using this library, please drop by the `GitHub issue tracker`_.

.. _GitHub issue tracker: https://github.com/MDAnalysis/pytng/issues

