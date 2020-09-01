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
* block :  a data element in a TNG trajectory containing a specific type of data e.g. positions, velocities or box vectors.
* block id : an integer (a long long) that indicates the type of data contained in a block.
* block name : a name that matches a specific block id, normally starts with the TNG prefix.
* particle dependency : indicates whether the data in a block is dependent on the particles in the simulation, (e.g. positions) or is not (e.g. box vectors).

Notes on the TNG format and PyTNG
=================================

While the TNG format supports storage of simulations conducted in the
grand canonical ensemble, PyTNG does not currently support this. Additionally,
the TNG format includes a TNG_MOLECULES block that contains the simulation
topology. PyTNG does not currently make use of this information.


Usage example for TNGFileIterator
=================================

TNG files can be read using the :class:`TNGFileIterator` class as a file handle,
which supports use as a context manager.

The TNGFileIterator has attributes related to the trajectory metadata, such as
the number of integrator steps, the number of steps with data, the block_ids 
available at each step, and the stride at which each block is written.

The TNGFileIterator returns one frame at a time, which is accessed from the
:attr:`TNGFileIterator.current_integrator_step` attribute or as part of any
slicing or indexing operation. A NumPy array of the right size must then be
provided to a getter method for the data to be read into.

An example of how to read positions and box vectors from a TNG file is
shown below.


.. code-block:: python

   import pytng
   import numpy as np

   with pytng.TNGFileIterator("traj.tng", 'r') as tng:

      # make a numpy array to hold the data using helper function
      # this array will then be updated in-place 
      
      positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
      box_vec = tng.make_ndarray_for_block_from_name("TNG_TRAJ_BOX_SHAPE")

      # the TNG API uses regular strides for data deposition, here we check
      # that the strides for positions and box_vectors are the same
      # and then iterate over all timesteps with this data
      # len(tng) is the total number of steps in the file

      assert (tng.block_strides["TNG_TRAJ_POSITIONS"] == tng.block_strides["TNG_TRAJ_BOX_SHAPE"])      

      for ts in tng[0:len(tng):tng.block_strides["TNG_TRAJ_POSITIONS"]]:
         # read the integrator timestep, modifying the current_integrator_step
         # which is returned as ts

         # get the data from the requested block by supplying NumPy array which
         # is updated in-place or returned 
         
         # update in place
         ts.get_positions(positions)
         # positions = ts.get_positions(positions) is equivalent

         # or return by value
         box_vec = ts.get_box(box_vec)
         # ts.get_box(box_vec) is equivalent

           

It is also possible to slice and index the file object to select particular
frames individually:

.. code-block:: python

  import pytng
  import numpy as np 

  with pytng.TNGFileIterator('traj.tng', 'r') as tng:
      positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
      box_vec = tng.make_ndarray_for_block_from_name("TNG_TRAJ_BOX_SHAPE")
      positions = tng[100].get_positions(positions)
      box_vec = tng[200].get_box(box_vec)


Available data blocks are listed at the end of this documentation. Common
blocks for which there are getter methods include:

* positions : :attr:`TNGFileIterator.get_positions`
* box vectors : :attr:`TNGFileIterator.get_box`
* forces : :attr:`TNGFileIterator.get_forces`
* velocities : :attr:`TNGFileIterator.get_velocities`

Other blocks can be accessed using the :attr:`TNGCurrentIntegratorStep.get_blockid`
method, where the block id needs to be supplied and can be accessed from the
:attr:`TNGFileIterator.block_ids` attribute.

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

