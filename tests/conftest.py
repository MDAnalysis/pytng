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


@pytest.fixture()
def ARGON_NPT_COMPRESSED():
    return os.path.join(HERE, 'reference_files', 'argon_npt_compressed.tng')


@pytest.fixture()
def WATER_NPT_COMPRESSED_TRJCONV():
    return os.path.join(, 'reference_files', 'water_npt_compressed_trjconv.tng')


@pytest.fixture()
def WATER_NPT_COMPRESSED_TRJCONV():
    return os.path.join(, 'reference_files', 'water_uncompressed_vels_forces.tng')
