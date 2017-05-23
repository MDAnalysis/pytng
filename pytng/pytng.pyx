from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL

status_error_message = ['OK', 'Failure', 'Critical']

cdef extern from "tng/tng_io.h":
    ctypedef struct tng_trajectory_t:
        pass

    tng_function_status tng_util_trajectory_open(
        const char *filename,
        const char mode,
        tng_trajectory_t *tng_data_p)
    tng_function_status tng_util_trajectory_close(
        tng_trajectory_t *tng_data_p)

    tng_function_status tng_num_frames_get(
        const tng_trajectory_t tng_data,
        int64_t *n)

    tng_function_status tng_num_particles_get(
        const tng_trajectory_t tng_data,
        int64_t *n)

    tng_function_status tng_distance_unit_exponential_get(
        const tng_trajectory_t tng_data,
        int64_t *exp);

    tng_function_status tng_util_pos_read_range(
        const tng_trajectory_t tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float **positions,
        int64_t *stride_length)


cdef class TNGFile:
    cdef tng_trajectory_t _traj
    cdef readonly fname
    cdef str mode
    cdef int is_open
    cdef int64_t _n_frames
    cdef int64_t _n_atoms
    cdef int64_t pos
    cdef float distance_scale

    def __cinit__(self, fname, mode='r'):
        self.fname = fname
        self._n_frames = -1
        self.open(self.fname, mode)

    def __dealloc__(self):
        self.close()

    def open(self, fname, mode):
        self.mode = mode

        cdef char _mode
        if self.mode == 'r':
            _mode = 'r'
        elif self.mode == 'w':
            _mode = 'w'
        else:
            raise IOError('mode must be one of "r" or "w", you '
                          'supplied {}'.format(mode))

        cdef int64_t exponent, ok

        ok = tng_util_trajectory_open(fname, _mode, & self._traj)
        if ok != TNG_SUCCESS:
            raise IOError("An error ocurred opening the file. {}".format(status_error_message[ok]))

        if self.mode == 'r':
            ok = tng_num_frames_get(self._traj, &self._n_frames)
            if ok != TNG_SUCCESS:
                raise IOError("An error ocurred reading n_frames. {}".format(status_error_message[ok]))

            ok = tng_num_particles_get(self._traj, & self._n_atoms)
            if ok != TNG_SUCCESS:
                raise IOError("An error ocurred reading n_atoms. {}".format(status_error_message[ok]))

            ok = tng_distance_unit_exponential_get(self._traj, &exponent)
            if ok != TNG_SUCCESS:
                raise IOError("An error ocurred reading distance unit exponent. {}".format(status_error_message[ok]))
            self.distance_scale = 10.0**(exponent+9)

        self.is_open = True
        self.pos = 0

    def close(self):
        if self.is_open:
            tng_util_trajectory_close(&self._traj)
            self.is_open = False
            self._n_frames = -1

    def __enter__(self):
        """Support context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager"""
        self.close()
        # always propagate exceptions forward
        return False

    @property
    def n_frames(self):
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._n_frames

    @property
    def n_atoms(self):
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._n_atoms

    def __len__(self):
        return self.n_frames

    def read(self):
        if self.pos >= self._n_frames:
            raise EOFError

        cdef np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] xyz = np.empty((self.n_atoms, 3), dtype=np.float32)
        cdef float* positions = NULL
        cdef int64_t stride_length, ok, i

        ok = tng_util_pos_read_range(self._traj, self.pos, self.pos, &positions, &stride_length)
        if ok != TNG_SUCCESS:
            if positions != NULL:
                free(positions)
            raise IOError("error reading frame")

        for i in range(self._n_atoms):
            for j in range(3):
                xyz[i, j] = positions[i*3 + j]
        xyz *= self.distance_scale

        self.pos += 1
        return xyz
