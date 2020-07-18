import pytng
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal)
import pytest

T, F = True, False


def test_load_bad_file(CORRUPT_FILEPATH):
    with pytest.raises(IOError):
        with pytng.TNGFile(CORRUPT_FILEPATH) as tng:
            tng.read()


def test_open_missing_file_mode_r(MISSING_FILEPATH):
    with pytest.raises(IOError) as excinfo:
        with pytng.TNGFile(MISSING_FILEPATH, mode='r') as tng:
            tng.read()
        assert 'does not exist' in str(excinfo.value)


def test_open_mode_w(MISSING_FILEPATH):
    with pytest.raises(NotImplementedError):
        pytng.TNGFile(MISSING_FILEPATH, mode='w')


def test_open_invalide_mode(TNG_EXAMPLE):
    with pytest.raises(IOError) as excinfo:
        pytng.TNGFile(TNG_EXAMPLE, mode='invalid')
    assert 'mode must be one of "r" or "w"' in str(excinfo.value)


def test_len(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        assert_equal(tng.n_frames, 10)
        assert_equal(len(tng), 10)



def test_iter(TNG_EXAMPLE):
    with pytng.TNGFile(TNG_EXAMPLE) as tng:
        for i, ts in enumerate(tng):
            assert i == ts.step


# @pytest.mark.parametrize('slice_idx', [
#     (None, None, None),
#     (2, None, None),
#     (None, 2, None),
#     (None, None, 3),
#     (1, 3, None),
#     (None, None, -1),
#     (0, -1, None),
#     (None, 99, None),  # Out of bound
# ])
# def test_sliced_iteration(slice_idx, GMX_REF_DATA, GMX_REF_FILEPATH):
#     start, stop, step = slice_idx
#     ref_steps = np.arange(0, GMX_REF_DATA.length)[start:stop:step]

#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         for ref_ts, ts in zip(ref_steps, tng[start:stop:step]):
#             assert ref_ts == ts.step


# @pytest.mark.parametrize('slx', (
#     [0, 1, 2],
#     [5, 3, 1],
#     [1, 1, 1],
#     [0, -1, 0],
#     [-2, -3, -4],
# ))
# @pytest.mark.parametrize('cls', [list, np.array])
# def test_getitem_multipl_ints(slx, cls, GMX_REF_DATA, GMX_REF_FILEPATH):
#     slx = cls(slx)
#     indices = np.arange(GMX_REF_DATA.length)
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         for ref_step, ts in zip(indices[slx], tng[slx]):
#             assert ref_step == ts.step


# @pytest.mark.parametrize('idx', [0, 4, 9, -1, -2])
# def test_getitem_int(idx, GMX_REF_DATA, GMX_REF_FILEPATH):
#     indices = np.arange(GMX_REF_DATA.length)
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         ts = tng[idx]
#         assert ts.step == indices[idx]


# @pytest.mark.parametrize('idx', [
#     'a',
#     'invalid',
#     (0, 1),
#     lambda x: x
# ])
# def test_getitem_single_invalid(idx, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         with pytest.raises(TypeError) as excinfo:
#             tng[idx]
#     message = ("Trajectories must be an indexed using an integer,"
#                " slice or list of indices")
#     assert message in str(excinfo.value)


# @pytest.mark.parametrize('arr', (
#     [T] * 10,
#     [F] * 10,
#     [T, F, T, F, T, F, T, F, T, F]
# ))
# @pytest.mark.parametrize('cls', [list, np.array])
# def test_getitem_bool(arr, cls, GMX_REF_DATA, GMX_REF_FILEPATH):
#     slx = cls(arr)
#     ref = np.arange(GMX_REF_DATA.length)[slx]

#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         for ref_ts, ts in zip(ref, tng[slx]):
#             assert ref_ts == ts.step


# @pytest.mark.parametrize('cls', [list, np.array])
# def test_getitem_bool_TypeError(cls, GMX_REF_FILEPATH):
#     slx = cls([True, False, True])

#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         with pytest.raises(TypeError):
#             for ts in tng[slx]:
#                 ts.step


# def test_natoms(GMX_REF_DATA, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         assert GMX_REF_DATA.natoms == tng.n_atoms


# def test_first_positions(GMX_REF_DATA, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         first_frame = tng.read().positions
#         assert np.array_equal(GMX_REF_DATA.first_frame, first_frame)


# def test_last_positions(GMX_REF_DATA, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         tng.seek(tng.n_frames - 1)
#         last_frame = tng.read().positions
#         assert np.array_equal(GMX_REF_DATA.last_frame, last_frame)


# @pytest.mark.parametrize('idx', [-11, -12, 10, 11])
# def test_seek_IndexError(idx, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH, 'r') as tng:
#         with pytest.raises(IndexError):
#             tng[idx]


# @pytest.mark.skip(reason="Write mode not implemented yet.")
# def test_seek_write(MISSING_FILEPATH):
#     with pytng.TNGFile(MISSING_FILEPATH, mode='w') as tng:
#         with pytest.raises(IOError) as excinfo:
#             tng.seek(0)
#         assert "seek not allowed in write mode" in str(excinfo.value)


# def test_seek_not_open(GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         pass
#     with pytest.raises(IOError) as excinfo:
#         tng.seek(0)
#     assert 'No file currently opened' in str(excinfo.value)


# def test_time(GMX_REF_DATA, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         for ref_time, ts in zip(GMX_REF_DATA.time, tng):
#             assert ref_time == ts.time


# def test_box(GMX_REF_DATA, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         frame = tng.read()
#         assert np.array_equal(GMX_REF_DATA.box, frame.box)


# def test_double_iteration(GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         for i, frame in enumerate(tng):
#             assert i == frame.step

#         for i, frame in enumerate(tng):
#             assert i == frame.step


# @pytest.mark.parametrize('prop', ('n_frames', 'n_atoms'))
# def test_property_not_open(prop, GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         pass
#     with pytest.raises(IOError) as excinfo:
#         getattr(tng, prop)
#     assert 'No file currently opened' in str(excinfo.value)


# def test_tell(GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         for step, frame in enumerate(tng, start=1):
#             assert step == tng.tell()


# def test_read_not_open(GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         pass
#     with pytest.raises(IOError) as excinfo:
#         tng.read()
#     assert 'No file opened' in str(excinfo.value)


# @pytest.mark.skip(reason="Write mode not implemented yet.")
# def test_read_not_mode_r(MISSING_FILEPATH):
#     with pytest.raises(IOError) as excinfo:
#         with pytng.TNGFile(MISSING_FILEPATH, mode='w') as tng:
#             tng.read()
#     assert 'Reading only allow in mode "r"' in str(excinfo.value)


# def test_seek_reset_eof(GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         for ts in tng:
#             pass
#         tng.seek(0)
#         next(tng)


# def test_reached_eof(GMX_REF_FILEPATH):
#     with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
#         # test with iter protocol
#         for ts in tng:
#             pass
#         with pytest.raises(StopIteration):
#             next(tng)
#         # test __getitem__
#         tng.seek(0)
#         for ts in tng[:]:
#             pass
#         with pytest.raises(StopIteration):
#             next(tng)
