from collections import namedtuple
import numpy as np
import os
import pytest

HERE = os.path.dirname(__file__)


@pytest.fixture()
def CORRUPT_FILEPATH():
    # actually just an ascii file
    return os.path.join(HERE, "reference_files", "badtngfile.tng")


@pytest.fixture()
def MISSING_FILEPATH():
    # return a file that doesn't exist
    return "nonexistant.tng"


@pytest.fixture()
def TNG_EXAMPLE():
    return os.path.join(HERE, "reference_files", "tng_example.tng")


@pytest.fixture()
def TNG_UTF8_EXAMPLE():
    return os.path.join(HERE, "reference_files", "tng_😝_emoji.tng")


@pytest.fixture
def TNG_EXAMPLE_DATA():
    # reference data for GMX_REF from Gromacs/tng library
    TNG = namedtuple(
        "TNGData",
        ["length", "natoms", "first_frame", "last_frame", "time", "box"],
    )

    # reference data determined via `gmx dump`
    # 5 water molecules, chain W resname WAT
    # names O, HO1, HO2

    time = [None] * 10

    first_frame = np.array(
        [
            [1.00000e00, 1.00000e00, 1.00000e00],
            [2.00000e00, 2.00000e00, 2.00000e00],
            [3.00000e00, 3.00000e00, 3.00000e00],
            [1.10000e01, 1.10000e01, 1.10000e01],
            [1.20000e01, 1.20000e01, 1.20000e01],
            [1.30000e01, 1.30000e01, 1.30000e01],
            [2.10000e01, 2.10000e01, 2.10000e01],
            [2.20000e01, 2.20000e01, 2.20000e01],
            [2.30000e01, 2.30000e01, 2.30000e01],
            [8.25000e00, 3.30000e01, 3.30000e01],
            [8.25000e00, 3.40000e01, 3.30000e01],
            [8.50000e00, 3.30000e01, 3.40000e01],
            [5.00000e01, 5.00000e01, 5.00000e01],
            [5.10000e01, 5.10000e01, 5.10000e01],
            [1.00000e02, 1.00000e02, 1.00000e02],
        ],
        dtype=np.float64,
    )

    last_frame = np.array(
        [
            [1.015625e00, 1.015625e00, 1.015625e00],
            [2.00000e00, 2.00000e00, 2.00000e00],
            [3.00000e00, 3.00000e00, 3.00000e00],
            [1.10000e01, 1.10000e01, 1.10000e01],
            [1.20000e01, 1.20000e01, 1.20000e01],
            [1.30000e01, 1.30000e01, 1.30000e01],
            [2.10000e01, 2.10000e01, 2.10000e01],
            [2.20000e01, 2.20000e01, 2.20000e01],
            [2.30000e01, 2.30000e01, 2.30000e01],
            [8.25000e00, 3.30000e01, 3.30000e01],
            [8.25000e00, 3.40000e01, 3.30000e01],
            [8.50000e00, 3.30000e01, 3.40000e01],
            [5.00000e01, 5.00000e01, 5.00000e01],
            [5.10000e01, 5.10000e01, 5.10000e01],
            [1.00000e02, 1.00000e02, 1.00000e02],
        ],
        dtype=np.float64,
    )

    return TNG(
        length=10,  # number of frames
        natoms=15,
        first_frame=first_frame,
        last_frame=last_frame,
        time=time,
        box=np.eye(3) * 50,
    )


@pytest.fixture()
def ARGON_NPT_COMPRESSED():
    return os.path.join(HERE, "reference_files", "argon_npt_compressed.tng")


@pytest.fixture
def ARGON_NPT_COMPRESSED_DATA():
    # reference data for Argon NPT COMPRESSED
    TNG = namedtuple(
        "TNGData",
        [
            "length",
            "natoms",
            "first_frame_first_10_pos",
            "last_frame_last_10_pos",
            "first_box",
            "last_box",
            "time",
        ],
    )

    time = [None] * 10

    first_frame_first_10_pos = np.array(
        [
            [2.53300e00, 1.24400e00, 3.50600e00],
            [8.30000e-01, 2.54400e00, 3.44800e00],
            [1.09100e00, 1.10000e-01, 3.12900e00],
            [2.45500e00, 5.00000e-03, 3.01200e00],
            [2.71400e00, 1.35300e00, 5.53000e-01],
            [3.05100e00, 2.89300e00, 2.69100e00],
            [1.42200e00, 2.77000e00, 1.46000e-01],
            [2.22300e00, 1.21100e00, 3.26800e00],
            [2.81100e00, 2.78900e00, 2.38500e00],
            [4.87000e-01, 1.15900e00, 1.17100e00],
        ],
        dtype=np.float64,
    )

    last_frame_last_10_pos = np.array(
        [
            [7.76000e-01, 1.19600e00, 7.73000e-01],
            [6.27000e-01, 3.34000e-01, 2.04900e00],
            [6.09000e-01, 3.46300e00, 2.57000e-01],
            [3.02000e00, 3.18400e00, 2.97600e00],
            [2.64700e00, 7.74000e-01, 1.81500e00],
            [1.56000e-01, 1.28300e00, 3.28100e00],
            [6.58000e-01, 3.03300e00, 2.90800e00],
            [2.08500e00, 3.55100e00, 1.43600e00],
            [1.56000e-01, 3.50200e00, 3.14000e-01],
            [1.28900e00, 9.98000e-01, 1.64500e00],
        ],
        dtype=np.float64,
    )

    first_box = np.array(
        [
            [3.60140, 0.00000, 0.000000],
            [0.000000, 3.60140, 0.000000],
            [0.000000, 0.000000, 3.60140],
        ]
    )

    last_box = np.array(
        [
            [3.589650, 0.000000, 0.000000],
            [0.000000, 3.589650, 0.000000],
            [0.000000, 0.000000, 3.589650],
        ]
    )

    return TNG(
        length=500001,  # number of frames
        natoms=1000,
        first_frame_first_10_pos=first_frame_first_10_pos,
        last_frame_last_10_pos=last_frame_last_10_pos,
        first_box=first_box,
        last_box=last_box,
        time=time,
    )


@pytest.fixture()
def WATER_NPT_COMPRESSED_TRJCONV():
    return os.path.join(
        HERE, "reference_files", "water_npt_compressed_trjconv.tng"
    )


@pytest.fixture()
def WATER_NPT_UNCOMPRESSED_VELS_FORCES():
    return os.path.join(
        HERE, "reference_files", "water_uncompressed_vels_forces.tng"
    )


@pytest.fixture
def WATER_NPT_UNCOMPRESSED_VELS_FORCES_DATA():
    # reference data for Argon NPT COMPRESSED
    TNG = namedtuple(
        "TNGData",
        [
            "length",
            "natoms",
            "first_frame_first_10_pos",
            "last_frame_last_10_pos",
            "first_box",
            "last_box",
            "first_frame_first_10_vels",
            "last_frame_last_10_vels",
            "first_frame_first_10_frc",
            "last_frame_last_10_frc",
            "time",
        ],
    )

    time = [None] * 10

    first_frame_first_10_pos = np.array(
        [
            [2.52700e00, 2.61101e00, 2.45398e00, ],
            [2.50319e00, 2.59390e00, 2.54510e00, ],
            [2.61687e00, 2.57898e00, 2.44623e00, ],
            [1.09097e00, 1.27301e00, 1.99202e00, ],
            [1.01457e00, 1.23310e00, 2.03366e00, ],
            [1.13694e00, 1.19976e00, 1.95100e00, ],
            [2.20399e00, 1.37297e00, 8.83017e-01, ],
            [2.13535e00, 1.38523e00, 9.48592e-01, ],
            [2.21780e00, 1.46022e00, 8.46139e-01, ],
            [1.10605e00, 2.11799e00, 5.61040e-01],
        ],
        dtype=np.float64,
    )

    last_frame_last_10_pos = np.array(
        [
            [7.98970e-01, 2.15481e00, 2.75854e00, ],
            [6.32804e-01, 6.59262e-01, 1.12701e00, ],
            [5.47739e-01, 6.89158e-01, 1.09488e00, ],
            [6.16521e-01, 5.70554e-01, 1.15907e00, ],
            [5.33961e-01, 2.20212e00, 6.22357e-02, ],
            [4.79836e-01, 2.17921e00, 1.37788e-01, ],
            [4.79169e-01, 2.18181e00, 2.88140e00, ],
            [5.76261e-01, 1.85258e00, 1.69974e00, ],
            [6.60233e-01, 1.87443e00, 1.74016e00, ],
            [5.79366e-01, 1.75766e00, 1.68776e00, ],
        ],
        dtype=np.float64,
    )

    first_box = np.array(
        [
            [2.87951e00, 0.00000e00, 0.00000e00],
            [0.00000e00, 2.87951e00, 0.00000e00],
            [0.00000e00, 0.00000e00, 2.87951e00],
        ]
    )

    last_box = np.array(
        [
            [2.89497e00, 0.00000e00, 0.00000e00],
            [0.00000e00, 2.89497e00, 0.00000e00],
            [0.00000e00, 0.00000e00, 2.89497e00],
        ]
    )

    first_frame_first_10_vels = np.array(
        [
            [3.51496e-01, 7.29674e-01, -5.33343e-02, ],
            [5.97873e-02, -1.00359e00, -4.19582e-01, ],
            [2.56209e-01, 5.52850e-01, -4.53435e-01, ],
            [-1.09184e-02, 3.66412e-01, -4.85018e-01, ],
            [9.26847e-01, -6.03737e-01, 3.67032e-01, ],
            [-9.85010e-02, 1.09447e00, -1.94833e00, ],
            [-4.60571e-02, 3.64507e-01, -2.01200e-01, ],
            [-1.23912e00, -3.46699e-01, -1.27041e00, ],
            [6.12738e-01, 7.64292e-01, 9.39986e-01, ],
            [-6.34257e-02, -3.96772e-02, -4.55601e-01, ],
        ],
        dtype=np.float64,
    )

    last_frame_last_10_vels = np.array(
        [
            [-1.29712e00, 1.89736e-01, -4.58020e-01, ],
            [-2.24550e-01, 1.98991e-01, -7.18228e-01, ],
            [9.92350e-02, 1.55654e-01, -1.64584e00, ],
            [-6.58128e-01, 4.26997e-01, -2.94439e-01, ],
            [-2.47945e-01, -4.03298e-01, 2.42530e-01, ],
            [3.88940e-01, 2.55276e-01, 9.15576e-01, ],
            [-1.57709e00, 5.61387e-01, 9.03308e-01, ],
            [-5.50578e-01, -3.38237e-01, -9.82961e-02, ],
            [4.52938e-01, -7.97070e-01, -1.83071e00, ],
            [-7.36810e-01, -2.02619e-01, -1.35719e00, ],
        ],
        dtype=np.float64,
    )

    first_frame_first_10_frc = np.array(
        [
            [-4.35261e02, 3.36017e02, -9.38570e02, ],
            [-1.75984e01, -2.44064e02, 1.25406e03, ],
            [6.57882e02, -2.07715e02, 2.72886e02, ],
            [1.75474e01, 1.57273e03, 2.80544e01, ],
            [-5.30602e02, -8.79351e02, 2.76766e02, ],
            [7.45154e01, -5.15662e02, -3.61260e02, ],
            [4.70405e02, -1.26065e03, -2.68651e02, ],
            [-5.15954e02, 5.19739e02, 2.85984e02, ],
            [-3.90010e02, 4.82308e02, 2.96046e00, ],
            [1.23199e03, -7.51883e02, -6.58181e02, ],
        ],
        dtype=np.float64,
    )

    last_frame_last_10_frc = np.array(
        [
            [-4.49360e02, -5.46652e02, 5.24477e02, ],
            [1.27648e03, 8.27699e02, 2.98916e01, ],
            [-9.49143e02, -3.13201e02, -3.78830e02, ],
            [-5.04814e02, -5.57331e02, -6.48604e01, ],
            [1.24046e03, 1.05411e03, 4.06005e02, ],
            [-3.61442e02, -5.29395e02, 1.26982e02, ],
            [-4.76165e02, -5.24370e02, -3.48132e02, ],
            [-7.41153e02, 1.19924e01, -7.19316e02, ],
            [5.67011e02, 6.64948e01, 2.13465e02, ],
            [2.43871e02, -4.09309e02, 4.87609e01, ],
        ],
        dtype=np.float64,
    )

    return TNG(
        length=500001,  # number of frames
        natoms=1000,
        first_frame_first_10_pos=first_frame_first_10_pos,
        last_frame_last_10_pos=last_frame_last_10_pos,
        first_box=first_box,
        last_box=last_box,
        first_frame_first_10_vels=first_frame_first_10_vels,
        last_frame_last_10_vels=last_frame_last_10_vels,
        first_frame_first_10_frc=first_frame_first_10_frc,
        last_frame_last_10_frc=last_frame_last_10_frc,
        time=time,
    )
