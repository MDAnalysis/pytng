===========================================
pytng - A python library to read TNG files!
===========================================

.. image:: https://travis-ci.com/MDAnalysis/pytng.svg?branch=master
  :target: https://travis-ci.com/MDAnalysis/pytng
.. image:: https://codecov.io/gh/MDAnalysis/pytng/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/MDAnalysis/pytng
.. image:: https://zenodo.org/badge/92180386.svg
   :target: https://zenodo.org/badge/latestdoi/92180386


This package provides the ``TNGFileIterator`` object to allow simple Pythonic
access to data contained within TNG files.

.. code-block:: python

  import pytng
  import numpy as np

  with pytng.TNGFileIterator('traj.tng', 'r') as tng:

    positions = np.empty(shape=(tng.n_atoms,3), dtype=np.float32)

    for ts in tng:
      time = ts.get_time()
      positions = ts.get_positions(positions)

This package contains Python bindings to libtng_ for TNG file format[1_] [2_].
This is used by molecular simulation programs such as Gromacs_ for storing the
topology and results from molecular dynamics simulations.

.. Warning::

   This package is under active development. The API is liable to change
   between release versions.

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

