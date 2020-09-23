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


Error handling
==============

PyTNG is designed so that Python level exceptions can (and should) be caught by
the calling code. However, as the trajectory reading itself is done 
by Cython at the C level with the GIL released, low level trajectory reading
errors are caught by more general exceptions at the Python level.

The success of the last call to ``get_blockid()``, ``get_positions()``, ``get_box()`` and
related methods can be checked using the :class:`TNGCurrentIntegratorStep.read_success`
property, which returns ``True`` if the attempt to read was successful or ``False`` if not.
Success (``True``) indicates that there was data of the requested block type present at the current step,
while failure (``False``) indicates either no data present, file corruption or a file reading exception not caught by PyTNG.
These types of data reading failures are treated on an equal footing as they are in the TNG library.


API for the TNGFile class
=========================

.. autoclass:: pytng.pytng.TNGFileIterator
  :members:

.. autoclass:: pytng.pytng.TNGCurrentIntegratorStep
  :members:

List of Available Blocks
========================

Platform independent definitions for the block ids can be found below.
You should **NOT** use these directly. Instead use the :attr:`TNGFileIterator.block_ids`
attribute to see the blocks and block ids available in your TNG file.

.. code-block:: python

  # GROUP 1 Standard non-trajectory blocks
  # Block IDs of standard non-trajectory blocks.
  
  TNG_GENERAL_INFO = 0x0000000000000000LL
  TNG_MOLECULES = 0x0000000000000001LL
  TNG_TRAJECTORY_FRAME_SET = 0x0000000000000002LL
  TNG_PARTICLE_MAPPING = 0x0000000000000003LL
  
  # GROUP 2 Standard trajectory blocks
  # Block IDs of standard trajectory blocks. Box shape and partial charges can
  # be either trajectory blocks or non-trajectory blocks
  
  TNG_TRAJ_BOX_SHAPE = 0x0000000010000000LL
  TNG_TRAJ_POSITIONS = 0x0000000010000001LL
  TNG_TRAJ_VELOCITIES = 0x0000000010000002LL
  TNG_TRAJ_FORCES = 0x0000000010000003LL
  TNG_TRAJ_PARTIAL_CHARGES = 0x0000000010000004LL
  TNG_TRAJ_FORMAL_CHARGES = 0x0000000010000005LL
  TNG_TRAJ_B_FACTORS = 0x0000000010000006LL
  TNG_TRAJ_ANISOTROPIC_B_FACTORS = 0x0000000010000007LL
  TNG_TRAJ_OCCUPANCY = 0x0000000010000008LL
  TNG_TRAJ_GENERAL_COMMENTS = 0x0000000010000009LL
  TNG_TRAJ_MASSES = 0x0000000010000010LL
  
  # GROUP 3 GROMACS data block IDs
  # Block IDs of data blocks specific to GROMACS.
  
  TNG_GMX_LAMBDA = 0x1000000010000000LL
  TNG_GMX_ENERGY_ANGLE = 0x1000000010000001LL
  TNG_GMX_ENERGY_RYCKAERT_BELL = 0x1000000010000002LL
  TNG_GMX_ENERGY_LJ_14 = 0x1000000010000003LL
  TNG_GMX_ENERGY_COULOMB_14 = 0x1000000010000004LL
  # NOTE changed from  TNG_GMX_ENERGY_LJ_(SR)
  DEF TNG_GMX_ENERGY_LJ_SR = 0x1000000010000005LL
  # NOTE changed from  TNG_GMX_ENERGY_COULOMB_(SR)
  TNG_GMX_ENERGY_COULOMB_SR = 0x1000000010000006LL
  TNG_GMX_ENERGY_COUL_RECIP = 0x1000000010000007LL
  TNG_GMX_ENERGY_POTENTIAL = 0x1000000010000008LL
  TNG_GMX_ENERGY_KINETIC_EN = 0x1000000010000009LL
  TNG_GMX_ENERGY_TOTAL_ENERGY = 0x1000000010000010LL
  TNG_GMX_ENERGY_TEMPERATURE = 0x1000000010000011LL
  TNG_GMX_ENERGY_PRESSURE = 0x1000000010000012LL
  TNG_GMX_ENERGY_CONSTR_RMSD = 0x1000000010000013LL
  TNG_GMX_ENERGY_CONSTR2_RMSD = 0x1000000010000014LL
  TNG_GMX_ENERGY_BOX_X = 0x1000000010000015LL
  TNG_GMX_ENERGY_BOX_Y = 0x1000000010000016LL
  TNG_GMX_ENERGY_BOX_Z = 0x1000000010000017LL
  TNG_GMX_ENERGY_BOXXX = 0x1000000010000018LL
  TNG_GMX_ENERGY_BOXYY = 0x1000000010000019LL
  TNG_GMX_ENERGY_BOXZZ = 0x1000000010000020LL
  TNG_GMX_ENERGY_BOXYX = 0x1000000010000021LL
  TNG_GMX_ENERGY_BOXZX = 0x1000000010000022LL
  TNG_GMX_ENERGY_BOXZY = 0x1000000010000023LL
  TNG_GMX_ENERGY_BOXVELXX = 0x1000000010000024LL
  TNG_GMX_ENERGY_BOXVELYY = 0x1000000010000025LL
  TNG_GMX_ENERGY_BOXVELZZ = 0x1000000010000026LL
  TNG_GMX_ENERGY_BOXVELYX = 0x1000000010000027LL
  TNG_GMX_ENERGY_BOXVELZX = 0x1000000010000028LL
  TNG_GMX_ENERGY_BOXVELZY = 0x1000000010000029LL
  TNG_GMX_ENERGY_VOLUME = 0x1000000010000030LL
  TNG_GMX_ENERGY_DENSITY = 0x1000000010000031LL
  TNG_GMX_ENERGY_PV = 0x1000000010000032LL
  TNG_GMX_ENERGY_ENTHALPY = 0x1000000010000033LL
  TNG_GMX_ENERGY_VIR_XX = 0x1000000010000034LL
  TNG_GMX_ENERGY_VIR_XY = 0x1000000010000035LL
  TNG_GMX_ENERGY_VIR_XZ = 0x1000000010000036LL
  TNG_GMX_ENERGY_VIR_YX = 0x1000000010000037LL
  TNG_GMX_ENERGY_VIR_YY = 0x1000000010000038LL
  TNG_GMX_ENERGY_VIR_YZ = 0x1000000010000039LL
  TNG_GMX_ENERGY_VIR_ZX = 0x1000000010000040LL
  TNG_GMX_ENERGY_VIR_ZY = 0x1000000010000041LL
  TNG_GMX_ENERGY_VIR_ZZ = 0x1000000010000042LL
  TNG_GMX_ENERGY_SHAKEVIR_XX = 0x1000000010000043LL
  TNG_GMX_ENERGY_SHAKEVIR_XY = 0x1000000010000044LL
  TNG_GMX_ENERGY_SHAKEVIR_XZ = 0x1000000010000045LL
  TNG_GMX_ENERGY_SHAKEVIR_YX = 0x1000000010000046LL
  TNG_GMX_ENERGY_SHAKEVIR_YY = 0x1000000010000047LL
  TNG_GMX_ENERGY_SHAKEVIR_YZ = 0x1000000010000048LL
  TNG_GMX_ENERGY_SHAKEVIR_ZX = 0x1000000010000049LL
  TNG_GMX_ENERGY_SHAKEVIR_ZY = 0x1000000010000050LL
  TNG_GMX_ENERGY_SHAKEVIR_ZZ = 0x1000000010000051LL
  TNG_GMX_ENERGY_FORCEVIR_XX = 0x1000000010000052LL
  TNG_GMX_ENERGY_FORCEVIR_XY = 0x1000000010000053LL
  TNG_GMX_ENERGY_FORCEVIR_XZ = 0x1000000010000054LL
  TNG_GMX_ENERGY_FORCEVIR_YX = 0x1000000010000055LL
  TNG_GMX_ENERGY_FORCEVIR_YY = 0x1000000010000056LL
  TNG_GMX_ENERGY_FORCEVIR_YZ = 0x1000000010000057LL
  TNG_GMX_ENERGY_FORCEVIR_ZX = 0x1000000010000058LL
  TNG_GMX_ENERGY_FORCEVIR_ZY = 0x1000000010000059LL
  TNG_GMX_ENERGY_FORCEVIR_ZZ = 0x1000000010000060LL
  TNG_GMX_ENERGY_PRES_XX = 0x1000000010000061LL
  TNG_GMX_ENERGY_PRES_XY = 0x1000000010000062LL
  TNG_GMX_ENERGY_PRES_XZ = 0x1000000010000063LL
  TNG_GMX_ENERGY_PRES_YX = 0x1000000010000064LL
  TNG_GMX_ENERGY_PRES_YY = 0x1000000010000065LL
  TNG_GMX_ENERGY_PRES_YZ = 0x1000000010000066LL
  TNG_GMX_ENERGY_PRES_ZX = 0x1000000010000067LL
  TNG_GMX_ENERGY_PRES_ZY = 0x1000000010000068LL
  TNG_GMX_ENERGY_PRES_ZZ = 0x1000000010000069LL
  TNG_GMX_ENERGY_SURFXSURFTEN = 0x1000000010000070LL
  TNG_GMX_ENERGY_MUX = 0x1000000010000071LL
  TNG_GMX_ENERGY_MUY = 0x1000000010000072LL
  TNG_GMX_ENERGY_MUZ = 0x1000000010000073LL
  TNG_GMX_ENERGY_VCOS = 0x1000000010000074LL
  TNG_GMX_ENERGY_VISC = 0x1000000010000075LL
  TNG_GMX_ENERGY_BAROSTAT = 0x1000000010000076LL
  TNG_GMX_ENERGY_T_SYSTEM = 0x1000000010000077LL
  TNG_GMX_ENERGY_LAMB_SYSTEM = 0x1000000010000078LL
  TNG_GMX_SELECTION_GROUP_NAMES = 0x1000000010000079LL
  TNG_GMX_ATOM_SELECTION_GROUP = 0x1000000010000080LL