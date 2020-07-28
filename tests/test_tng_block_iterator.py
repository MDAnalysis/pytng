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


def test_spool(ARGON_NPT_COMPRESSED):
    file_iterator = pytng.TNGFileIterator(ARGON_NPT_COMPRESSED, mode="r")
    file_iterator.spool()
    raise Exception


def test_spool2(ARGON_NPT_COMPRESSED):
    file_iterator = pytng.TNGFileIterator(ARGON_NPT_COMPRESSED, mode="r")
    file_iterator.spool2()
    raise Exception

