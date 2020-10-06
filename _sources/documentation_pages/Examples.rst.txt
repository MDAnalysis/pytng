Examples
========

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

         # you can check if the last read was successful (contained data) easily
         if not ts.read_success:
            raise IOError("No position data at this timestep")

         ts.get_box(box_vec)
         # box_vec = ts.get_box(box_vec) is equivalent

         # you can check if the last read was successful (contained data) easily
         if not ts.read_success:
            raise IOError("No box data at this timestep")

           

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

      # slice a single timestep
      ts = tng[step]

      # get the data, which will be returned full of np.nan
      ts.get_positions(positions)

      # the read_success property will indicate that there was no data found
      # when the getter was called
      assert(ts.read_success == False)

      #  but we can also double check that the read was blank
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
