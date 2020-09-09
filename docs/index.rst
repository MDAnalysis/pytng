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
* GCD : greatest common denominator.
* blank read : an attempt to read a step where there is no data present

Notes on the TNG format and PyTNG
=================================

The TNG format is highly flexible, allowing the storage of almost any datatype
from any point in a simulation. This information is written at certain strides,
i.e at every *N* steps. Under most circumstances, strides are of a similar
magnitude or share a large *greatest common divisor (GCD)*. However this is not
always the case and introduces additional complexity in iterating through the
file effectively.  The main challenge is if you want to retrieve multiple datatypes in a single
pass that do not share a large GCD in their strides, necessitating lots of
blank reads. Avoiding this is still a work in progress for PyTNG.

While the TNG format supports storage of simulations conducted in the
grand canonical ensemble, PyTNG does not currently support this or any other type of simulation where the number of particles varies between frames.
Additionally, the TNG format includes a TNG_MOLECULES block that contains the simulation
topology. PyTNG does not currently make use of this information, but will in the future.


Usage example for TNGFileIterator
=================================

TNG files can be read using the :class:`TNGFileIterator` class as a file handle,
which supports use as a context manager.

The TNGFileIterator has attributes related to the trajectory metadata, such as
the number of integrator steps, the number of steps with data, the block_ids 
available at each step, and the stride at which each block is written.

The TNGFileIterator returns one frame at a time, which is accessed from the
:attr:`TNGFileIterator.current_integrator_step` attribute or as part of any
slicing or indexing operation. A NumPy array of the right size and datatype must then be
provided to a getter method for the data to be read into. The required datatype is dictated by what type of block is being read.
Supported datatypes are:

* TNG_INT_DATA : np.int64
* TNG_FLOAT_DATA : np.float32
* TNG_DOUBLE_DATA : np.float64

Helper methods are provided to create :class:`np.ndarray` instances of the right shape
and datatype for a particular block in :attr:`TNGFileIterator.make_ndarray_for_block_from_name` and :attr:`TNGFileIterator.make_ndarray_for_block_from_name`.

An example of how to read positions and box vectors from a TNG file is
shown below:


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
         
         # update in place or return by value
         ts.get_positions(positions)
         # positions = ts.get_positions(positions) is equivalent

         ts.get_box(box_vec)
         # box_vec = ts.get_box(box_vec) is equivalent

           

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


If the step to read is not on the stride of the requested datatype, the NumPy
array will be returned filled with `np.nan`. A contrived example of this is given below:

.. code-block:: python

   import pytng
   import numpy as np

   with pytng.TNGFileIterator("traj.tng", 'r') as tng:
      # make array for positions
      positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
      
      # choose a step
      step = 42

      # check that we are off stride (stride%step != 0)
      assert(tng.block_strides["TNG_TRAJ_POSITIONS"]%step != 0 )

      # get the data, which will be returned full of np.nan
      tng[step].get_positions(positions)

      # check that the read was blank
      is_blank_read = np.all(np.isnan(positions)) # this will be true
      if is_blank_read:
         print("This is a blank read")


Available data blocks are listed at the end of this documentation. Common
blocks for which there are getter methods include:

* positions : :attr:`TNGFileIterator.get_positions`
* box vectors : :attr:`TNGFileIterator.get_box`
* forces : :attr:`TNGFileIterator.get_forces`
* velocities : :attr:`TNGFileIterator.get_velocities`

Other blocks can be accessed using the :attr:`TNGCurrentIntegratorStep.get_blockid`
method, where the block id needs to be supplied and can be accessed from the
:attr:`TNGFileIterator.block_ids` attribute. An example of this is shown below:

.. code-block:: python

   import pytng
   import numpy as np

   with pytng.TNGFileIterator("traj.tng", 'r') as tng:

      # make array for the GMX potential energy block
      Epot = tng.make_ndarray_for_block_from_name("TNG_GMX_ENERGY_POTENTIAL")

      # get the block id for the GMX potential energy block
      Epot_block_id = tng.block_ids["TNG_GMX_ENERGY_POTENTIAL"]

      # get the block data for frame 0 with get_blockid
      tng[0].get_blockid[Epot_block_id]


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

