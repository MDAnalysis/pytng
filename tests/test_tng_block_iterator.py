import pytng
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_array_almost_equal,
)
import pytest

T, F = True, False


def test_file_open(TNG_EXAMPLE):
    file_iterator = pytng.TNGFileIterator(TNG_EXAMPLE, mode="r")


def test_next_block_call(ARGON_NPT_COMPRESSED):
    file_iterator = pytng.TNGFileIterator(ARGON_NPT_COMPRESSED, mode="r")
    file_iterator.spool()
    #raise Exception

def test_next_block_types(ARGON_NPT_COMPRESSED):
    file_iterator = pytng.TNGFileIterator(ARGON_NPT_COMPRESSED, mode="r")
    file_iterator.get_block_types_of_next_frame()
    #raise Exception

# def test_next_block_call_ex(WATER_NPT_UNCOMPRESSED_VELS_FORCES):
#     file_iterator = pytng.TNGFileIterator(WATER_NPT_UNCOMPRESSED_VELS_FORCES, mode="r")
#     file_iterator.spool()
#     raise Exception

