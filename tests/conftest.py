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
        ['length', 'natoms', 'first_frame_first_10', 'last_frame_last_10', 'first_box', 'last_box', 'time'])

    time = [None] * 10

    first_frame_first_10 = np.array(
        [[2.53300e+00,  1.24400e+00,  3.50600e+00],
         [8.30000e-01,  2.54400e+00,  3.44800e+00],
         [1.09100e+00,  1.10000e-01,  3.12900e+00],
         [2.45500e+00,  5.00000e-03,  3.01200e+00],
         [2.71400e+00,  1.35300e+00,  5.53000e-01],
         [3.05100e+00,  2.89300e+00,  2.69100e+00],
         [1.42200e+00,  2.77000e+00,  1.46000e-01],
         [2.22300e+00,  1.21100e+00,  3.26800e+00],
         [2.81100e+00,  2.78900e+00,  2.38500e+00],
         [4.87000e-01,  1.15900e+00,  1.17100e+00]],
        dtype=np.float64)

    last_frame_last_10 = np.array(
        [[7.76000e-01,  1.19600e+00,  7.73000e-01],
         [6.27000e-01,  3.34000e-01,  2.04900e+00],
         [6.09000e-01,  3.46300e+00,  2.57000e-01],
         [3.02000e+00,  3.18400e+00,  2.97600e+00],
         [2.64700e+00,  7.74000e-01,  1.81500e+00],
         [1.56000e-01,  1.28300e+00,  3.28100e+00],
         [6.58000e-01,  3.03300e+00,  2.90800e+00],
         [2.08500e+00,  3.55100e+00,  1.43600e+00],
         [1.56000e-01,  3.50200e+00,  3.14000e-01],
         [1.28900e+00,  9.98000e-01,  1.64500e+00]],
        dtype=np.float64)

    first_box = np.array([[3.60140, 0.00000, 0.000000], [0.000000,
                          3.60140, 0.000000], [0.000000, 0.000000, 3.60140]])

    last_box = np.array([[3.589650, 0.000000, 0.000000], [0.000000,
                         3.589650, 0.000000], [0.000000, 0.000000, 3.589650]])

    return TNG(
        length=500001,  # number of frames
        natoms=1000,
        first_frame_first_10=first_frame_first_10,
        last_frame_last_10=last_frame_last_10,
        first_box = first_box,
        last_box = last_box,
        time=time )


@pytest.fixture()
def WATER_NPT_COMPRESSED_TRJCONV():
    return os.path.join(HERE, 'reference_files', 'water_npt_compressed_trjconv.tng')


@pytest.fixture()
def WATER_NPT_UNCOMPRESSED_VELS_FORCES():
    return os.path.join(HERE, 'reference_files', 'water_uncompressed_vels_forces.tng')
