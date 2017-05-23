#import pytng

from collections import namedtuple
import numpy as np
import pytest

@pytest.fixture
def TNG_REF():
    TNG = namedtuple('TNGData',
                     ['length', 'natoms', 'first_frame', 'last_frame'])

    # reference data determined via `gmx dump`
    # 5 water molecules, chain W resname WAT
    # names O, HO1, HO2

    first_frame = np.array([
        [ 1.00000e+00,  1.00000e+00,  1.00000e+00],
        [ 2.00000e+00,  2.00000e+00,  2.00000e+00],
        [ 3.00000e+00,  3.00000e+00,  3.00000e+00],
        [ 1.10000e+01,  1.10000e+01,  1.10000e+01],
        [ 1.20000e+01,  1.20000e+01,  1.20000e+01],
        [ 1.30000e+01,  1.30000e+01,  1.30000e+01],
        [ 2.10000e+01,  2.10000e+01,  2.10000e+01],
        [ 2.20000e+01,  2.20000e+01,  2.20000e+01],
        [ 2.30000e+01,  2.30000e+01,  2.30000e+01],
        [ 8.25000e+00,  3.30000e+01,  3.30000e+01],
        [ 8.25000e+00,  3.40000e+01,  3.30000e+01],
        [ 8.50000e+00,  3.30000e+01,  3.40000e+01],
        [ 5.00000e+01,  5.00000e+01,  5.00000e+01],
        [ 5.10000e+01,  5.10000e+01,  5.10000e+01],
        [ 1.00000e+02,  1.00000e+02,  1.00000e+02]
    ], dtype=np.float64)

    last_frame = np.array([
        [ 1.01562e+00,  1.01562e+00,  1.01562e+00],
        [ 2.00000e+00,  2.00000e+00,  2.00000e+00],
        [ 3.00000e+00,  3.00000e+00,  3.00000e+00],
        [ 1.10000e+01,  1.10000e+01,  1.10000e+01],
        [ 1.20000e+01,  1.20000e+01,  1.20000e+01],
        [ 1.30000e+01,  1.30000e+01,  1.30000e+01],
        [ 2.10000e+01,  2.10000e+01,  2.10000e+01],
        [ 2.20000e+01,  2.20000e+01,  2.20000e+01],
        [ 2.30000e+01,  2.30000e+01,  2.30000e+01],
        [ 8.25000e+00,  3.30000e+01,  3.30000e+01],
        [ 8.25000e+00,  3.40000e+01,  3.30000e+01],
        [ 8.50000e+00,  3.30000e+01,  3.40000e+01],
        [ 5.00000e+01,  5.00000e+01,  5.00000e+01],
        [ 5.10000e+01,  5.10000e+01,  5.10000e+01],
        [ 1.00000e+02,  1.00000e+02,  1.00000e+02]
    ], dtype=np.float64)

    return TNG(
        length=10,  # number of frames
        natoms=15,
        first_frame=first_frame,
        last_frame=last_frame,
    )


def test_len(TNG_REF):
    assert TNG_REF.length == 10


def test_natoms(TNG_REF):
    assert TNG_REF.natoms == 15


def test_first_positions(TNG_REF):
    assert np.array_equal(TNG_REF.first_frame, TNG_REF.first_frame)


def test_last_positions(TNG_REF):
    assert TNG_REF.last_frame[0, 0] == 1.01562
