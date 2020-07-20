import pytng
import numpy as np
from numpy.testing import (
    assert_almost_equal, assert_equal, assert_array_almost_equal)
import pytest

T, F = True, False


def test_tng_example_load_bad_file(CORRUPT_FILEPATH):
    with pytest.raises(IOError):
        with pytng.TNGFile(CORRUPT_FILEPATH) as tng:
            tng.read()


def test_tng_example_open_missing_file_mode_r(MISSING_FILEPATH):
    with pytest.raises(IOError) as excinfo:
        with pytng.TNGFile(MISSING_FILEPATH, mode='r') as tng:
            tng.read()
        assert 'does not exist' in str(excinfo.value)


def test_tng_example_open_mode_w(MISSING_FILEPATH):
    with pytest.raises(NotImplementedError):
        pytng.TNGFile(MISSING_FILEPATH, mode='w')


def test_tng_example_open_invalide_mode(TNG_EXAMPLE):
    with pytest.raises(IOError) as excinfo:
        pytng.TNGFile(TNG_EXAMPLE, mode='invalid')
    assert 'mode must be one of "r" or "w"' in str(excinfo.value)


def test_tng_example_len(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        assert_equal(tng.n_frames, 10)
        assert_equal(len(tng), 10)


def test_tng_example_iter(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for i, ts in enumerate(tng):
            assert i == ts.step


@pytest.mark.parametrize('slice_idx', [
    (None, None, None),
    (2, None, None),
    (None, 2, None),
    (None, None, 3),
    (1, 3, None),
    (None, None, -1),
    (0, -1, None),
    (None, 99, None),  # Out of bound
])
def test_tng_example_sliced_iteration(slice_idx, TNG_EXAMPLE):
    start, stop, step = slice_idx
    ref_steps = np.arange(0, 10)[start:stop:step]

    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for ref_ts, ts in zip(ref_steps, tng[start:stop:step]):
            assert ref_ts == ts.step


@pytest.mark.parametrize('slx', (
    [0, 1, 2],
    [5, 3, 1],
    [1, 1, 1],
    [0, -1, 0],
    [-2, -3, -4],
))
@pytest.mark.parametrize('cls', [list, np.array])
def test_tng_example_getitem_multipl_ints(slx, cls, TNG_EXAMPLE, TNG_EXAMPLE_DATA):
    slx = cls(slx)
    indices = np.arange(TNG_EXAMPLE_DATA.length)
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for ref_step, ts in zip(indices[slx], tng[slx]):
            assert ref_step == ts.step


@pytest.mark.parametrize('idx', [0, 4, 9, -1, -2])
def test_tng_example_getitem_int(idx, TNG_EXAMPLE, TNG_EXAMPLE_DATA):
    indices = np.arange(TNG_EXAMPLE_DATA.length)
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        ts = tng[idx]
        assert ts.step == indices[idx]


@pytest.mark.parametrize('idx', [
    'a',
    'invalid',
    (0, 1),
    lambda x: x
])
def test_tng_example_getitem_single_invalid(idx, TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        with pytest.raises(TypeError) as excinfo:
            tng[idx]
    message = ("Trajectories must be an indexed using an integer,"
               " slice or list of indices")
    assert message in str(excinfo.value)


@pytest.mark.parametrize('arr', (
    [T] * 10,
    [F] * 10,
    [T, F, T, F, T, F, T, F, T, F]
))
@pytest.mark.parametrize('cls', [list, np.array])
def test_tng_example_getitem_bool(arr, cls, TNG_EXAMPLE, TNG_EXAMPLE_DATA):
    slx = cls(arr)
    ref = np.arange(TNG_EXAMPLE_DATA.length)[slx]

    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for ref_ts, ts in zip(ref, tng[slx]):
            assert ref_ts == ts.step


@pytest.mark.parametrize('cls', [list, np.array])
def test_tng_example_getitem_bool_TypeError(cls, TNG_EXAMPLE):
    slx = cls([True, False, True])
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        with pytest.raises(TypeError):
            for ts in tng[slx]:
                ts.step


def test_tng_example_natoms(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        assert TNG_EXAMPLE_DATA.natoms == tng.n_atoms


def test_tng_example_tng_example_first_positions(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        first_frame = tng.read().positions
        assert np.array_equal(TNG_EXAMPLE_DATA.first_frame, first_frame)


def test_tng_example_tng_example_last_positions(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        tng.seek(tng.n_frames - 1)
        last_frame = tng.read().positions
        assert np.array_equal(TNG_EXAMPLE_DATA.last_frame, last_frame)


@pytest.mark.parametrize('idx', [-11, -12, 10, 11])
def test_tng_example_seek_IndexError(idx, TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE, 'r') as tng:
        with pytest.raises(IndexError):
            tng[idx]


@pytest.mark.skip(reason="Write mode not implemented yet.")
def test_tng_example_seek_write(MISSING_FILEPATH):
    with pytng.TNGFile(MISSING_FILEPATH, mode='w') as tng:
        with pytest.raises(IOError) as excinfo:
            tng.seek(0)
        assert "seek not allowed in write mode" in str(excinfo.value)


def test_tng_example_seek_not_open(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        pass
    with pytest.raises(IOError) as excinfo:
        tng.seek(0)
    assert 'No file currently opened' in str(excinfo.value)


def test_tng_example_time(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for ref_time, ts in zip(TNG_EXAMPLE_DATA.time, tng):
            assert ref_time == ts.time


def test_tng_example_double_iteration(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for i, frame in enumerate(tng):
            assert i == frame.step

        for i, frame in enumerate(tng):
            assert i == frame.step


@pytest.mark.parametrize('prop', ('n_frames', 'n_atoms'))
def test_tng_example_property_not_open(prop, TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        pass
    with pytest.raises(IOError) as excinfo:
        getattr(tng, prop)
    assert 'No file currently opened' in str(excinfo.value)


def test_tng_example_tell(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for step, frame in enumerate(tng, start=1):
            assert step == tng.tell()


def test_tng_example_read_not_open(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        pass
    with pytest.raises(IOError) as excinfo:
        tng.read()
    assert 'No file opened' in str(excinfo.value)


@pytest.mark.skip(reason="Write mode not implemented yet.")
def test_tng_example_read_not_mode_r(MISSING_FILEPATH):
    with pytest.raises(IOError) as excinfo:
        with pytng.TNGFile(MISSING_FILEPATH, mode='w') as tng:
            tng.read()
    assert 'Reading only allow in mode "r"' in str(excinfo.value)


def test_tng_example_seek_reset_eof(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for ts in tng:
            pass
        tng.seek(0)
        next(tng)


def test_tng_example_reached_eof(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        # test with iter protocol
        for ts in tng:
            pass
        with pytest.raises(StopIteration):
            next(tng)
        # test __getitem__
        tng.seek(0)
        for ts in tng[:]:
            pass
        with pytest.raises(StopIteration):
            next(tng)


def test_argon_npt_compressed_open(ARGON_NPT_COMPRESSED):
    with pytng.TNGFile(ARGON_NPT_COMPRESSED) as tng:
        pass

def test_argon_npt_compressed_stride_setup(ARGON_NPT_COMPRESSED):
    with pytng.TNGFile(ARGON_NPT_COMPRESSED) as tng:
        assert tng._pos == 1
        assert tng._box == 1
        assert tng._vel == 0
        assert tng._frc == 0


def test_argon_npt_compressed_len(ARGON_NPT_COMPRESSED):
    with pytng.TNGFile(ARGON_NPT_COMPRESSED) as tng:
        assert tng.n_frames == 500001
        assert len(tng) == 500001


def test_argon_npt_compressed_n_particles(ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA):
    with pytng.TNGFile(ARGON_NPT_COMPRESSED) as tng:
        assert ARGON_NPT_COMPRESSED_DATA.natoms == tng.n_atoms


def test_argon_npt_compressed_first_positions(ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA):
    with pytng.TNGFile(ARGON_NPT_COMPRESSED) as tng:
        first_frame_first_10 = tng.read().positions[:10, :]
        assert_array_almost_equal(
            ARGON_NPT_COMPRESSED_DATA.first_frame_first_10, first_frame_first_10)


def test_argon_npt_compressed_last_positions(ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA):
    with pytng.TNGFile(ARGON_NPT_COMPRESSED) as tng:
        tng.seek(tng.n_frames-1)
        last_frame = tng.read().positions
        last_frame_last_10 = last_frame[990:1000, :]
        assert_array_almost_equal(
            ARGON_NPT_COMPRESSED_DATA.last_frame_last_10, last_frame_last_10)


def test_argon_npt_compressed_first_box(ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA):
        with pytng.TNGFile(ARGON_NPT_COMPRESSED) as tng:
            first_box = tng.read().box
            assert_array_almost_equal(
            ARGON_NPT_COMPRESSED_DATA.first_box, first_box)






def test_water_npt_uncompressed_vels_forces_open(WATER_NPT_UNCOMPRESSED_VELS_FORCES):
    with pytng.TNGFile(WATER_NPT_UNCOMPRESSED_VELS_FORCES) as tng:
        pass

def test_water_npt_uncompressed_vels_forces_stride_setup(WATER_NPT_UNCOMPRESSED_VELS_FORCES):
    with pytng.TNGFile(WATER_NPT_UNCOMPRESSED_VELS_FORCES) as tng:
        assert tng._pos == 1
        assert tng._box == 1
        assert tng._vel == 1
        assert tng._frc == 1

