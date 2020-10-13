import pytng
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_array_almost_equal,
)
import pytest

T, F = True, False


def test_tng_example_load_bad_file(CORRUPT_FILEPATH):
    with pytest.raises(IOError):
        with pytng.TNGFileIterator(CORRUPT_FILEPATH) as tng:
            tng.read_step(0)

def test_tng_example_load_utf8_special_char(TNG_UTF8_EXAMPLE):
    with pytng.TNGFileIterator(TNG_UTF8_EXAMPLE) as tng:
        tng.read_step(0)


def test_tng_example_open_missing_file_mode_r(MISSING_FILEPATH):
    with pytest.raises(IOError) as excinfo:
        with pytng.TNGFileIterator(MISSING_FILEPATH, mode="r") as tng:
            tng.read_step(0)
        assert "does not exist" in str(excinfo.value)


def test_tng_example_open_mode_w(MISSING_FILEPATH):
    with pytest.raises(NotImplementedError):
        pytng.TNGFileIterator(MISSING_FILEPATH, mode="w")


def test_tng_example_open_invalide_mode(TNG_EXAMPLE):
    with pytest.raises(ValueError) as excinfo:
        pytng.TNGFileIterator(TNG_EXAMPLE, mode="invalid")
    assert 'mode must be one of' in str(excinfo.value)


def test_tng_example_len(TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        assert_equal(len(tng), 10)


def test_tng_example_iter(TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        for i, dummy in enumerate(tng):
            tng.read_step(i)


@pytest.mark.parametrize(
    "slice_idx",
    [
        (None, None, None),
        (2, None, None),
        (None, 2, None),
        (None, None, 3),
        (1, 3, None),
        (None, None, -1),
        (0, -1, None),
        (None, 99, None),  # Out of bound
    ],
)
def test_tng_example_sliced_iteration(slice_idx, TNG_EXAMPLE):
    start, stop, step = slice_idx
    ref_steps = np.arange(0, 10)[start:stop:step]

    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        for ref_ts, ts in zip(ref_steps, tng[start:stop:step]):
            assert ref_ts == ts.step


@pytest.mark.parametrize(
    "slx", ([[0, 1, 2], [5, 3, 1], [1, 1, 1], [0, 1, 0]])

)
@pytest.mark.parametrize("cls", [list, np.array])
def test_tng_example_getitem_multipl_ints(
    slx, cls, TNG_EXAMPLE, TNG_EXAMPLE_DATA
):
    slx = cls(slx)
    indices = np.arange(TNG_EXAMPLE_DATA.length)
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        for ref_step, ts in zip(indices[slx], tng[slx]):
            assert ref_step == ts.step


@pytest.mark.parametrize("idx", [0, 4, 9])
def test_tng_example_getitem_int(idx, TNG_EXAMPLE, TNG_EXAMPLE_DATA):
    indices = np.arange(TNG_EXAMPLE_DATA.length)
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        ts = tng[idx]
        assert ts.step == indices[idx]


@pytest.mark.parametrize("idx", ["a", "invalid", (0, 1), lambda x: x])
def test_tng_example_getitem_single_invalid(idx, TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        with pytest.raises(TypeError) as excinfo:
            tng[idx]
    message = (
        "Trajectories must be an indexed using an integer,"
        " slice or list of indices"
    )
    assert message in str(excinfo.value)


@pytest.mark.parametrize(
    "arr", ([T] * 10, [F] * 10, [T, F, T, F, T, F, T, F, T, F])
)
@pytest.mark.parametrize("cls", [list, np.array])
def test_tng_example_getitem_bool(arr, cls, TNG_EXAMPLE, TNG_EXAMPLE_DATA):
    slx = cls(arr)
    ref = np.arange(TNG_EXAMPLE_DATA.length)[slx]

    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        for ref_ts, ts in zip(ref, tng[slx]):
            assert ref_ts == ts.step


@pytest.mark.parametrize("cls", [list, np.array])
def test_tng_example_getitem_bool_TypeError(cls, TNG_EXAMPLE):
    slx = cls([True, False, True])
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        with pytest.raises(TypeError):
            for ts in tng[slx]:
                ts.step


def test_tng_example_natoms(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        assert TNG_EXAMPLE_DATA.natoms == tng.n_atoms


def test_README_example(TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE, 'r') as tng:
        positions = np.empty((tng.n_atoms, 3), dtype=np.float32)
        for ts in tng:
            time = ts.get_time()
            ts.get_positions(positions)
            if not ts.read_success:
                raise IOError

def test_DOCS_example(TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE, 'r') as tng:
        positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
        for ts in tng[0:len(tng):tng.block_strides["TNG_TRAJ_POSITIONS"]]:
            positions = ts.get_positions(positions)
            if not ts.read_success:
                raise IOError

@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.uint32, np.uint64, np.complex64, np.complex128])
def test_bad_dtype(TNG_EXAMPLE, dtype):
    with pytng.TNGFileIterator(TNG_EXAMPLE, 'r') as tng:
        positions = np.empty(shape=(tng.n_atoms, 3), dtype=dtype)
        with pytest.raises(TypeError) as excinfo:
            positions = tng[0].get_positions(positions)
        message = "PYTNG ERROR: datatype of numpy array not supported"
        assert message in str(excinfo.value)

@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_supported_dtype_wrong_block(TNG_EXAMPLE, dtype):
    with pytng.TNGFileIterator(TNG_EXAMPLE, 'r') as tng:
        positions = np.empty(shape=(tng.n_atoms, 3), dtype=dtype)
        with pytest.raises(TypeError) as excinfo:
            positions = tng[0].get_positions(positions)
        message = "does not match TNG dtype float"
        assert message in str(excinfo.value)

def test_tng_example_tng_example_first_positions(
    TNG_EXAMPLE_DATA, TNG_EXAMPLE
):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pos = np.empty(shape=(15, 3), dtype=np.float32)
        pos = tng[0].get_positions(pos)
        assert np.array_equal(TNG_EXAMPLE_DATA.first_frame, pos)


def test_tng_example_tng_example_last_positions(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pos = np.empty(shape=(15, 3), dtype=np.float32)
        pos = tng[9].get_positions(pos)
        assert np.array_equal(TNG_EXAMPLE_DATA.last_frame, pos)

def test_tng_example_first_positions_neg_index(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pos = np.empty(shape=(15, 3), dtype=np.float32)
        pos = tng[-10].get_positions(pos)
        assert np.array_equal(TNG_EXAMPLE_DATA.first_frame, pos)

def test_tng_example_last_positions_neg_index(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pos = np.empty(shape=(15, 3), dtype=np.float32)
        pos = tng[-1].get_positions(pos)
        assert np.array_equal(TNG_EXAMPLE_DATA.last_frame, pos)

def test_tng_example_first_and_last_seq_positions(TNG_EXAMPLE_DATA,
 TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pos = np.empty(shape=(15, 3), dtype=np.float32)
        pos = tng[0].get_positions(pos)
        assert np.array_equal(TNG_EXAMPLE_DATA.first_frame, pos)
        pos = tng[9].get_positions(pos)
        assert np.array_equal(TNG_EXAMPLE_DATA.last_frame, pos)


def test_tng_example_tng_example_pos_through_read_step(
    TNG_EXAMPLE_DATA, TNG_EXAMPLE
):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pos = np.empty(shape=(15, 3), dtype=np.float32)
        tngstep = tng.read_step(0)
        pos = tngstep.get_positions(pos)
        assert np.array_equal(TNG_EXAMPLE_DATA.first_frame, pos)



@pytest.mark.skip(reason="Write mode not implemented yet.")
def test_tng_example_read_step_write(MISSING_FILEPATH):
    with pytng.TNGFileIterator(MISSING_FILEPATH, mode="w") as tng:
        with pytest.raises(IOError) as excinfo:
            tng.read_step(0)
        assert "read_step not allowed in write mode" in str(excinfo.value)


def test_tng_example_time(TNG_EXAMPLE_DATA, TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        for ref_time, ts in zip(TNG_EXAMPLE_DATA.time, tng):
            assert ref_time == tng.current_integrator_step.get_time()


@pytest.mark.parametrize("prop", ("n_steps", "n_atoms"))
def test_tng_example_property_not_open(prop, TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pass
    with pytest.raises(IOError) as excinfo:
        getattr(tng, prop)
    assert "File is not yet open" in str(excinfo.value)


def test_tng_example_read_not_open(TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        pass
    with pytest.raises(IOError) as excinfo:
        tng.read_step(0)
    assert "File is not yet open" in str(excinfo.value)


@pytest.mark.skip(reason="Write mode not implemented yet.")
def test_tng_example_read_not_mode_r(MISSING_FILEPATH):
    with pytest.raises(IOError) as excinfo:
        with pytng.TNGFileIterator(MISSING_FILEPATH, mode="w") as tng:
            tng.read_step(0)
    assert 'Reading only allow in mode "r"' in str(excinfo.value)


def test_tng_example_reached_eof(TNG_EXAMPLE):
    with pytng.TNGFileIterator(TNG_EXAMPLE) as tng:
        # test with iter protocol
        for ts in tng:
            pass
        with pytest.raises(StopIteration):
            next(tng)
        # test __getitem__
        tng.read_step(0)
        for ts in tng[:]:
            pass
        with pytest.raises(StopIteration):
            next(tng)


def test_argon_npt_compressed_open(ARGON_NPT_COMPRESSED):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        pass


def test_argon_npt_compressed_len(ARGON_NPT_COMPRESSED):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        assert tng.n_steps == 500001
        assert len(tng) == 500001

def test_argon_npt_compressed_off_stride_is_nan(ARGON_NPT_COMPRESSED):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
        step = 42
        assert(tng.block_strides["TNG_TRAJ_POSITIONS"]%step != 0 )
        ts = tng[step]
        ts.get_positions(positions)
        assert(ts.read_success == False)
        assert(np.all(np.isnan(positions)))

def test_argon_npt_compressed_off_stride_reverse_idx_is_nan(ARGON_NPT_COMPRESSED):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        positions = tng.make_ndarray_for_block_from_name("TNG_TRAJ_POSITIONS")
        step = -42
        assert(tng.block_strides["TNG_TRAJ_POSITIONS"]%step != 0 )
        ts = tng[step]
        ts.get_positions(positions)
        assert(ts.read_success == False)
        assert(np.all(np.isnan(positions)))


def test_argon_npt_compressed_n_particles(
    ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA
):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        assert ARGON_NPT_COMPRESSED_DATA.natoms == tng.n_atoms


def test_argon_npt_compressed_first_positions(
    ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA
):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        pos = np.empty(shape=(1000, 3), dtype=np.float32)
        pos = tng[0].get_positions(pos)
        assert_array_almost_equal(
            ARGON_NPT_COMPRESSED_DATA.first_frame_first_10_pos,
            pos[:10, :],
        )


def test_argon_npt_compressed_last_positions(
    ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA
):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        pos = np.empty(shape=(1000, 3), dtype=np.float32)
        pos = tng[len(tng)-1].get_positions(pos)
        assert_array_almost_equal(
            ARGON_NPT_COMPRESSED_DATA.last_frame_last_10_pos,
            pos[990:1000, :]
        )


def test_argon_npt_compressed_first_box(
    ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA
):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        box = np.empty(shape=(1, 9), dtype=np.float32)
        box = tng[0].get_box(box)
        assert_array_almost_equal(
            ARGON_NPT_COMPRESSED_DATA.first_box, box.reshape((3, 3))
        )


def test_argon_npt_compressed_last_box(
    ARGON_NPT_COMPRESSED, ARGON_NPT_COMPRESSED_DATA
):
    with pytng.TNGFileIterator(ARGON_NPT_COMPRESSED) as tng:
        box = np.empty(shape=(1, 9), dtype=np.float32)
        box = tng[len(tng)-1].get_box(box)
        assert_array_almost_equal(
            ARGON_NPT_COMPRESSED_DATA.last_box, box.reshape(3, 3))


def test_water_npt_uncompressed_vels_forces_open(
    WATER_NPT_UNCOMPRESSED_VELS_FORCES,
):
    with pytng.TNGFileIterator(WATER_NPT_UNCOMPRESSED_VELS_FORCES) as tng:
        pass


def test_water_npt_uncompressed_vels_forces_first_vels(
    WATER_NPT_UNCOMPRESSED_VELS_FORCES, WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA
):
    with pytng.TNGFileIterator(WATER_NPT_UNCOMPRESSED_VELS_FORCES) as tng:
        vel = np.empty(shape=(2700, 3), dtype=np.float32)
        vel = tng[0].get_velocities(vel)
        assert_array_almost_equal(
            WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA.first_frame_first_10_vels,
            vel[:10, :],
            decimal=2,
        )  # decimal = 2 really slack


def test_water_npt_uncompressed_vels_forces_last_vels(
    WATER_NPT_UNCOMPRESSED_VELS_FORCES, WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA
):
    with pytng.TNGFileIterator(WATER_NPT_UNCOMPRESSED_VELS_FORCES) as tng:
        vel = np.empty(shape=(2700, 3), dtype=np.float32)
        vel = tng[len(tng)-1].get_velocities(vel)
        assert_array_almost_equal(
            WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA.last_frame_last_10_vels,
            vel[2690:2700, :],
            decimal=2
        )  # decimal = 2 really slack


def test_water_npt_uncompressed_vels_forces_first_frc(
    WATER_NPT_UNCOMPRESSED_VELS_FORCES, WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA
):
    with pytng.TNGFileIterator(WATER_NPT_UNCOMPRESSED_VELS_FORCES) as tng:
        frc = np.empty(shape=(2700, 3), dtype=np.float32)
        frc = tng[0].get_forces(frc)  # todo forces
        assert_array_almost_equal(
            WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA.first_frame_first_10_frc,
            frc[:10, :],
            decimal=2,
        )  # decimal = 2 really slack


def test_water_npt_uncompressed_vels_forces_last_frc(
    WATER_NPT_UNCOMPRESSED_VELS_FORCES, WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA
):
    with pytng.TNGFileIterator(WATER_NPT_UNCOMPRESSED_VELS_FORCES) as tng:
        frc = np.empty(shape=(2700, 3), dtype=np.float32)
        frc = tng[len(tng)-1].get_forces(frc)
        assert_array_almost_equal(
            WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA.last_frame_last_10_frc,
            frc[2690:2700, :],
            decimal=2,
        )  # decimal = 2 really slack
