.. pytng documentation master file, created by
   sphinx-quickstart on Tue May 23 16:00:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. include:: ../README.rst


Usage example for TNGFile
=========================

TNG files can be read using the TNGFile class as a file handle,
which supports use as a context manager.
The TNGFile returns one frame at a time, which each frame being returned as a
namedtuple with the following attributes:

============== ======== =======================================
attribute      type     description
============== ======== =======================================
``positions``  float32  coordinates for each atom in the frame
``time``       float    current system time
``step``       int      frame index
``box``        float32  3x3 matrix of the system volume
============== ======== =======================================

For example, the coordinate information can be accessed via the `.positions` 
attribute of the returned object.

.. code-block:: python

  import pytng

  with pytng.TNGFile('traj.tng', 'r') as f:
      for ts in f:
          ts.positions

It is also possible to slice and index the file object to select particular
frames

.. code-block:: python

  import pytng

  with pytng.TNGFile('traj.tng', 'r') as f:
      first_frame = f[0]
      last_frame = f[-1]

      every_other_frame = [ts for ts in f[::2]]


API for the TNGFile class
=========================

.. autoclass:: pytng.pytng.TNGFile
  :members:

