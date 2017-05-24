import pytng

import numpy as np
import pytest


def test_load_bad_file(CORRUPT_FILEPATH):
    with pytest.raises(IOError):
        with pytng.TNGFile(CORRUPT_FILEPATH) as tng:
            tng.read()


def test_load_missing_file(MISSING_FILEPATH):
    with pytest.raises(IOError):
        with pytng.TNGFile(MISSING_FILEPATH) as tng:
            tng.read()


def test_len(GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert GMX_REF_DATA.length == tng.n_frames
        assert GMX_REF_DATA.length == len(tng)


def test_iter(GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        for i, ts in enumerate(tng):
            assert i == ts.step


@pytest.mark.parametrize('slice_idx', [
    (None, None, None),
    (2, None, None),
    (None, 2, None),
    (None, None, 3),
])
def test_sliced_iteration(slice_idx, GMX_REF_DATA, GMX_REF_FILEPATH):
    start, stop, step = slice_idx
    ref_steps = np.arange(0, GMX_REF_DATA.length)[start:stop:step]

    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        for ref_ts, ts in zip(ref_steps, tng[start:stop:step]):
            assert ref_ts == ts.step


@pytest.mark.parametrize('indices', (
    [0, 1, 2],
    [5, 3, 1],
    [1, 1, 1]
))
@pytest.mark.parametrize('cls', [list, np.array])
def test_getitem_multipl_ints(indices, cls, GMX_REF_DATA, GMX_REF_FILEPATH):
    indices = cls(indices)
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        for ref_step, ts in zip(indices, tng[indices]):
            assert ref_step == ts.step


@pytest.mark.parametrize('idx', [0, 4, 9])
def test_getitem_int(idx, GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        ts = tng[idx]
        assert idx == ts.step

T, F = True, False
@pytest.mark.parametrize('arr', (
    [T] * 10,
    [F] * 10,
))
@pytest.mark.parametrize('cls', [list, np.array])
def test_getitem_bool(arr, cls, GMX_REF_DATA, GMX_REF_FILEPATH):
    slx = cls(arr)
    ref = np.arange(GMX_REF_DATA.length)[slx]

    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        for ref_ts, ts in zip(ref, tng[slx]):
            assert ref_ts == ts.step


@pytest.mark.parametrize('cls', [list, np.array])
def test_getitem_bool_TypeError(cls, GMX_REF_FILEPATH):
    slx = cls([True, False, True])

    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        with pytest.raises(TypeError):
            for ts in tng[slx]:
                ts.step


def test_natoms(GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert GMX_REF_DATA.natoms == tng.n_atoms


def test_first_positions(GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        first_frame = tng.read().xyz
        assert np.array_equal(GMX_REF_DATA.first_frame, first_frame)


def test_last_positions(GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        tng.seek(tng.n_frames - 1)
        last_frame = tng.read().xyz
        assert np.array_equal(GMX_REF_DATA.last_frame, last_frame)


def test_time(GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        for ref_time, ts in zip(GMX_REF_DATA.time, tng):
            assert ref_time == ts.time


def test_box(GMX_REF_DATA, GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        frame = tng.read()
        assert np.array_equal(GMX_REF_DATA.box, frame.box)


def test_double_iteration(GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        for i, frame in enumerate(tng):
            assert i == frame.step

        for i, frame in enumerate(tng):
            assert i == frame.step
