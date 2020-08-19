.. pytng documentation master file, created by
   sphinx-quickstart on Tue May 23 16:00:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. include:: ../README.rst


Usage example for TNGFileIterator
=================================

TNG files can be read using the TNGFileIterator class as a file handle,
which supports use as a context manager.

The TNGFileIterator has attributes related to the trajectory metadata, such as
the number of integrator steps, the number of steps with data, the block_ids 
available at each step, and the stride at which each block is written.

The TNGFileIterator returns one frame at a time, which is accessed from the
`.current_integrator_step` attribute. A NumPy array of the right size must
be provided to a getter attribute for the data to be read into.

An example of how to read positions from a TNG file is shown below.


.. code-block:: python

   import pytng
   import numpy as np

   with pytng.TNGFileIterator("tng.tng", 'r') as tng:

      # make a numpy array to hold the data using helper function
      positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
      
      # stride over the whole trajectory for the frames that have data
      for ts in range(0, len(tng), tng.block_strides["TNG_TRAJ_POSITIONS"]):
         # read the integrator timestep
         tng.read_step(ts)
         # get the data from the requested block by supplying NumPy array
         tng.current_integrator_step.get_pos(positions)

           

It is also possible to slice and index the file object to select particular
frames

.. code-block:: python

  import pytng
  import numpy as np

  with pytng.TNGFileIterator('traj.tng', 'r') as tng:
      tng[0].current_integrator_step.get_pos(positions)
      tng[0,100,10].get_pos(positions)


API for the TNGFile class
=========================

.. autoclass:: pytng.pytng.TNGFileIterator
  :members:

.. autoclass:: pytng.pytng.TNGCurrentIntegratorStep
  :members:

