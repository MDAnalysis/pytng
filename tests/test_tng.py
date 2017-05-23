#import pytng

import numpy as np
import pytest


def test_len(TNG_REF_DATA):
    assert TNG_REF_DATA.length == 10


def test_natoms(TNG_REF_DATA):
    assert TNG_REF_DATA.natoms == 15


def test_first_positions(TNG_REF_DATA):
    assert np.array_equal(TNG_REF_DATA.first_frame, TNG_REF_DATA.first_frame)


def test_last_positions(TNG_REF_DATA):
    assert TNG_REF_DATA.last_frame[0, 0] == 1.01562

def test_path(TNG_REF_FILEPATH):
    assert isinstance(TNG_REF_FILEPATH, str)
