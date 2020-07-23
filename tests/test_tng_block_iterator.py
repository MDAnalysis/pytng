import pytng
import numpy as np
from numpy.testing import (
    assert_almost_equal, assert_equal, assert_array_almost_equal)
import pytest

T, F = True, False

def test_file_open(TNG_EXAMPLE):
    file_iterator = TNGFileIterator(TNG_EXAMPLE, mode='r')
