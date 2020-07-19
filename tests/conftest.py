from collections import namedtuple
import numpy as np
import os
import pytest

HERE = os.path.dirname(__file__)


@pytest.fixture()
def CORRUPT_FILEPATH():
    # actually just an ascii file
    return os.path.join(HERE, 'reference_files', 'badtngfile.tng')


@pytest.fixture()
def MISSING_FILEPATH():
    # return a file that doesn't exist
    return 'nonexistant.tng'


@pytest.fixture()
def TNG_EXAMPLE():
    return os.path.join(HERE, 'reference_files', 'tng_example.tng')

@pytest.fixture
def TNG_EXAMPLE_DATA():
    # reference data for GMX_REF from Gromacs/tng library
    TNG = namedtuple(
        'TNGData',
        ['length', 'natoms', 'first_frame', 'last_frame', 'time', 'box'])

    # reference data determined via `gmx dump`
    # 5 water molecules, chain W resname WAT
    # names O, HO1, HO2

    time = [None] * 10

    first_frame = np.array(
        [[1.00000e+00, 1.00000e+00, 1.00000e+00],
         [2.00000e+00, 2.00000e+00, 2.00000e+00],
         [3.00000e+00, 3.00000e+00, 3.00000e+00],
         [1.10000e+01, 1.10000e+01, 1.10000e+01],
         [1.20000e+01, 1.20000e+01, 1.20000e+01],
         [1.30000e+01, 1.30000e+01, 1.30000e+01],
         [2.10000e+01, 2.10000e+01, 2.10000e+01],
         [2.20000e+01, 2.20000e+01, 2.20000e+01],
         [2.30000e+01, 2.30000e+01, 2.30000e+01],
         [8.25000e+00, 3.30000e+01, 3.30000e+01],
         [8.25000e+00, 3.40000e+01, 3.30000e+01],
         [8.50000e+00, 3.30000e+01, 3.40000e+01],
         [5.00000e+01, 5.00000e+01, 5.00000e+01],
         [5.10000e+01, 5.10000e+01, 5.10000e+01],
         [1.00000e+02, 1.00000e+02, 1.00000e+02]],
        dtype=np.float64)

    last_frame = np.array(
        [[1.015625e+00, 1.015625e+00, 1.015625e+00],
         [2.00000e+00, 2.00000e+00, 2.00000e+00],
         [3.00000e+00, 3.00000e+00, 3.00000e+00],
         [1.10000e+01, 1.10000e+01, 1.10000e+01],
         [1.20000e+01, 1.20000e+01, 1.20000e+01],
         [1.30000e+01, 1.30000e+01, 1.30000e+01],
         [2.10000e+01, 2.10000e+01, 2.10000e+01],
         [2.20000e+01, 2.20000e+01, 2.20000e+01],
         [2.30000e+01, 2.30000e+01, 2.30000e+01],
         [8.25000e+00, 3.30000e+01, 3.30000e+01],
         [8.25000e+00, 3.40000e+01, 3.30000e+01],
         [8.50000e+00, 3.30000e+01, 3.40000e+01],
         [5.00000e+01, 5.00000e+01, 5.00000e+01],
         [5.10000e+01, 5.10000e+01, 5.10000e+01],
         [1.00000e+02, 1.00000e+02, 1.00000e+02]],
        dtype=np.float64)

    return TNG(
        length=10,  # number of frames
        natoms=15,
        first_frame=first_frame,
        last_frame=last_frame,
        time=time,
        box=np.eye(3) * 50, )



@pytest.fixture()
def ARGON_NPT_COMPRESSED():
    return os.path.join(HERE, 'reference_files', 'argon_npt_compressed.tng')


@pytest.fixture
def ARGON_NPT_COMPRESSED_DATA():
    # reference data for Argon NPT COMPRESSED
    TNG = namedtuple(
        'TNGData',
        ['length', 'natoms', 'first_frame', 'last_frame', 'time', 'box'])


    time = [None] * 10

    first_frame = np.array(
        [[1.00000e+00, 1.00000e+00, 1.00000e+00],
         [2.00000e+00, 2.00000e+00, 2.00000e+00],
         [3.00000e+00, 3.00000e+00, 3.00000e+00],
         [1.10000e+01, 1.10000e+01, 1.10000e+01],
         [1.20000e+01, 1.20000e+01, 1.20000e+01],
         [1.30000e+01, 1.30000e+01, 1.30000e+01],
         [2.10000e+01, 2.10000e+01, 2.10000e+01],
         [2.20000e+01, 2.20000e+01, 2.20000e+01],
         [2.30000e+01, 2.30000e+01, 2.30000e+01],
         [8.25000e+00, 3.30000e+01, 3.30000e+01],
         [8.25000e+00, 3.40000e+01, 3.30000e+01],
         [8.50000e+00, 3.30000e+01, 3.40000e+01],
         [5.00000e+01, 5.00000e+01, 5.00000e+01],
         [5.10000e+01, 5.10000e+01, 5.10000e+01],
         [1.00000e+02, 1.00000e+02, 1.00000e+02]],
        dtype=np.float64)

    last_frame = np.array(
        [[1.015625e+00, 1.015625e+00, 1.015625e+00],
         [2.00000e+00, 2.00000e+00, 2.00000e+00],
         [3.00000e+00, 3.00000e+00, 3.00000e+00],
         [1.10000e+01, 1.10000e+01, 1.10000e+01],
         [1.20000e+01, 1.20000e+01, 1.20000e+01],
         [1.30000e+01, 1.30000e+01, 1.30000e+01],
         [2.10000e+01, 2.10000e+01, 2.10000e+01],
         [2.20000e+01, 2.20000e+01, 2.20000e+01],
         [2.30000e+01, 2.30000e+01, 2.30000e+01],
         [8.25000e+00, 3.30000e+01, 3.30000e+01],
         [8.25000e+00, 3.40000e+01, 3.30000e+01],
         [8.50000e+00, 3.30000e+01, 3.40000e+01],
         [5.00000e+01, 5.00000e+01, 5.00000e+01],
         [5.10000e+01, 5.10000e+01, 5.10000e+01],
         [1.00000e+02, 1.00000e+02, 1.00000e+02]],
        dtype=np.float64)

    return TNG(
        length=500001,  # number of frames
        natoms=1000,
        first_frame=first_frame,
        last_frame=last_frame,
        time=time,
        box=np.eye(3) * 50, )

@pytest.fixture()
def WATER_NPT_COMPRESSED_TRJCONV():
    return os.path.join(HERE, 'reference_files', 'water_npt_compressed_trjconv.tng')


@pytest.fixture()
def WATER_NPT_COMPRESSED_TRJCONV():
    return os.path.join(HERE, 'reference_files', 'water_uncompressed_vels_forces.tng')
