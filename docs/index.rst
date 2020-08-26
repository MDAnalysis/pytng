.. pytng documentation master file, created by
   sphinx-quickstart on Tue May 23 16:00:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. include:: ../README.rst


Glossary of key terms
=====================

Some key terms used are defined below. These definitions are specific to
PyTNG. For more information on the original TNG API, see the following papers
[1_] [2_].

* step : an integrator timestep (one MD step).
* frame : an integrator timestep with data associated.
* stride : the number of *steps* between writes of data to the trajectory.
* block :  a data element in a TNG trajectory containing a specific type of data eg. positions, velocities or box vectors.
* block id : an integer (a long long) that indicates the type of data contained in a block.
* block name : a name that matches a specific block id, normally starts with the TNG prefix.
* particle dependency : indicates whether the data in a block is dependent on the particles in the simulation, (eg positions) or is not (eg box vectors).

Notes on the TNG format and PyTNG
=================================

While the TNG format supports storage of simulations conducted in the
grand canonical ensemble, PyTNG does not currently support this. Additonally,
the TNG format includes a TNG_MOLECULES block that contains the simulation
topology. PyTNG does not currently make use of this information.


Usage example for TNGFileIterator
=================================

TNG files can be read using the TNGFileIterator class as a file handle,
which supports use as a context manager.

The TNGFileIterator has attributes related to the trajectory metadata, such as
the number of integrator steps, the number of steps with data, the block_ids 
available at each step, and the stride at which each block is written.

The TNGFileIterator returns one frame at a time, which is accessed from the
:attr:`TNGFileIterator.current_integrator_step` attribute. A NumPy array of the
right size must be provided to a getter method for the data to be read into.

An example of how to read positions from a TNG file is shown below.


.. code-block:: python

   import pytng
   import numpy as np

   with pytng.TNGFileIterator("traj.tng", 'r') as tng:

      # make a numpy array to hold the data using helper function
      # this array will then be updated in-place 
      
      positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
      
      # the TNG API uses regular strides for data deposition, here we stride
      # over the whole trajectory for the frames that have position data
      # where len(tng) is the total number of steps in the file.

      for ts in range(0, len(tng), tng.block_strides["TNG_TRAJ_POSITIONS"]):
         # read the integrator timestep, modifying the current_integrator_step
         tng.read_step(ts)
         # get the data from the requested block by supplying NumPy array which
         # is updated in-place
         tng.current_integrator_step.get_pos(positions)

           

It is also possible to slice and index the file object to select particular
frames

.. code-block:: python

  import pytng
  import numpy as np

  with pytng.TNGFileIterator('traj.tng', 'r') as tng:
      tng[100].current_integrator_step.get_pos(positions)


Error handling
==============

PyTNG is designed so that Python level exceptions can (and should) be caught by
the calling code. However, as the trajectory reading itself is done 
by Cython at the C level with the GIL released, low level trajectory reading
errors are caught by more general exceptions at the Python level.


API for the TNGFile class
=========================

.. autoclass:: pytng.pytng.TNGFileIterator
  :members:

.. autoclass:: pytng.pytng.TNGCurrentIntegratorStep
  :members:

