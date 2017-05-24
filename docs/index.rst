.. pytng documentation master file, created by
   sphinx-quickstart on Tue May 23 16:00:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pytng's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Usage example for TNGFile
-------------------------

TNG files can be read using the TNGFile class as a file handle.
The TNGFile returns one frame at a time, which each frame being returned as a
namedtuple.

For example, the coordinate information can be accessed via the `.xyz` attribute
of the returned frame object.

.. code-block:: python

  import pytng

  with pytng.TNGFile('traj.tng', 'r') as f:
      for ts in f:
          ts.xyz




API for the TNGFile class
-------------------------

.. autoclass:: pytng.pytng.TNGFile
  :members:

