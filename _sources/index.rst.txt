.. pytng documentation master file, created by
   sphinx-quickstart on Tue May 23 16:00:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 3
   :caption: Overview:

   ./documentation_pages/Examples
   ./documentation_pages/Errors
   ./documentation_pages/API
   ./documentation_pages/Blocks

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


