import pytng

import numpy as np
import pytest


def test_load_bad_file(TNG_BAD_FILEPATH):
    with pytest.raises(IOError):
        with pytng.TNGFile(TNG_BAD_FILEPATH) as tng:
            tng.read()

def test_load_missing_file(TNG_MISSING_FILEPATH):
    with pytest.raises(IOError):
        with pytng.TNGFile(TNG_MISSING_FILEPATH) as tng:
            tng.read()

def test_len(TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        assert TNG_REF_DATA.length == tng.n_frames
        assert TNG_REF_DATA.length == len(tng)

def test_iter(TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        for i, ts in enumerate(tng):
            assert i == ts.step

@pytest.mark.parametrize('slice_idx', [
    (None, None, None),
    (2, None, None),
    (None, 2, None),
    (None, None, 3),
])
def test_sliced_iteration(slice_idx, TNG_REF_DATA, TNG_REF_FILEPATH):
    start, stop, step = slice_idx
    ref_steps = np.arange(0, TNG_REF_DATA.length)[start:stop:step]

    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        for ref_ts, ts in zip(ref_steps, tng[start:stop:step]):
            assert ref_ts == ts.step

@pytest.mark.parametrize('idx', [0, 4, 9])
def test_getitem_int(idx, TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        ts = tng.read()
        assert idx == ts.step

def test_natoms(TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        assert TNG_REF_DATA.natoms == tng.n_atoms


def test_first_positions(TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        first_frame = tng.read().xyz
        assert np.array_equal(TNG_REF_DATA.first_frame, first_frame)


def test_last_positions(TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        tng.seek(tng.n_frames - 1)
        last_frame = tng.read().xyz
        assert np.array_equal(TNG_REF_DATA.last_frame, last_frame)

def test_time(TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        for ref_time, ts in zip(TNG_REF_DATA.time, tng):
            assert ref_time == ts.time


def test_box(TNG_REF_DATA, TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        frame = tng.read()
        assert np.array_equal(TNG_REF_DATA.box, frame.box)

def test_double_iteration(TNG_REF_FILEPATH):
    with pytng.TNGFile(TNG_REF_FILEPATH) as tng:
        for i, frame in enumerate(tng):
            assert i == frame.step

        for i, frame in enumerate(tng):
            assert i == frame.step


def test_path(TNG_REF_FILEPATH):
    assert isinstance(TNG_REF_FILEPATH, str)
