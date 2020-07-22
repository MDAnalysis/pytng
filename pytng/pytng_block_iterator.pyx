# cython: linetrace=True
# cython: embedsignature=True
# distutils: define_macros=CYTHON_TRACE=1
from numpy cimport(PyArray_SimpleNewFromData,
                   PyArray_SetBaseObject,
                   NPY_FLOAT,
                   Py_INCREF,
                   npy_intp,
                   )

from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

from collections import namedtuple
import os
import numbers
import numpy as np

cimport numpy as np
np.import_array()


ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL
ctypedef enum tng_data_type: TNG_CHAR_DATA, TNG_INT_DATA, TNG_FLOAT_DATA, TNG_DOUBLE_DATA
ctypedef enum tng_hash_mode: TNG_SKIP_HASH, TNG_USE_HASH


status_error_message = ['OK', 'Failure', 'Critical']

cdef extern from "tng/tng_io.h":
    ctypedef struct tng_trajectory_t:
        pass

    tng_function_status tng_util_trajectory_open(
        const char * filename,
        const char mode,
        tng_trajectory_t * tng_data_p)

    tng_function_status tng_util_trajectory_close(
        tng_trajectory_t * tng_data_p)

    tng_function_status tng_num_frames_get(
        const tng_trajectory_t tng_data,
        int64_t * n)

    tng_function_status tng_num_particles_get(
        const tng_trajectory_t tng_data,
        int64_t * n)

    tng_function_status tng_distance_unit_exponential_get(
        const tng_trajectory_t tng_data,
        int64_t * exp)

    tng_function_status tng_util_pos_read_range(
        const tng_trajectory_t tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** positions,
        int64_t * stride_length)

    tng_function_status tng_util_box_shape_read_range(
        const tng_trajectory_t tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** box_shape,
        int64_t * stride_length)

    tng_function_status tng_util_vel_read_range(
        const tng_trajectory_t tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** velocities,
        int64_t * stride_length)

    tng_function_status tng_util_force_read_range(
        const tng_trajectory_t tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** forces,
        int64_t * stride_length)

    tng_function_status tng_util_pos_read(
        const tng_trajectory_t tng_data,
        float ** positions,
        int64_t * stride_length)

    tng_function_status tng_util_box_shape_read(
        const tng_trajectory_t tng_data,
        float ** box_shape,
        int64_t * stride_length)

    tng_function_status tng_util_vel_read(
        const tng_trajectory_t tng_data,
        float ** velocities,
        int64_t * stride_length)

    tng_function_status tng_util_force_read(
        const tng_trajectory_t tng_data,
        float ** forces,
        int64_t * stride_length)

    tng_function_status tng_util_time_of_frame_get(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        double * time)

    tng_function_status tng_data_vector_interval_get(
        const tng_trajectory_t tng_data,
        const int64_t block_id,
        const int64_t start_frame_nr,
        const int64_t end_frame_nr,
        const char hash_mode,
        void ** values,
        int64_t * stride_length,
        int64_t * n_values_per_frame,
        char * type)

TNGFrame = namedtuple("TNGFrame", "positions velocities forces time step box ")


cdef class MemoryWrapper:
    # holds a pointer to C allocated memory, deals with malloc&free
    # based on:
    # https://gist.github.com/GaelVaroquaux/1249305/ac4f4190c26110fe2791a1e7a6bed9c733b3413f
    cdef void * ptr  # TODO do we want to use std::unique_ptr?

    def __cinit__(MemoryWrapper self, int size):
        # malloc not PyMem_Malloc as gmx later does realloc
        self.ptr = malloc(size)
        if self.ptr is NULL:
            raise MemoryError

    def __dealloc__(MemoryWrapper self):
        if self.ptr != NULL:
            free(self.ptr)


cdef class TNGFileIterator:
    """File handle object for TNG files

    Supports use as a context manager ("with" blocks).
    """
    cdef tng_trajectory_t _traj
    cdef int64_t _n_frames

    def __cinit__(self, fname, mode='r'):
        self.fname = fname
        self._n_frames = 0
        self._open(self.fname, mode)
        
    def open(self, fname, mode):
        """Open a file handle

        Parameters
        ----------
        fname : str
           path to the file
        mode : str
           mode to open the file in, 'r' for read, 'w' for write
        """
        self.mode = mode

        cdef char _mode
        if self.mode == 'r':
            _mode = 'r'
        elif self.mode == 'w':
            _mode = 'w'
            raise NotImplementedError('Writing is not implemented yet.')
        elif self.mode == 'a':
            _mode = 'a'
            raise NotImplementedError('Appending is not implemented yet')
        else:
            raise ValueError('mode must be one of "r", "w", or "a" you '
                          'supplied {}'.format(mode))


        # handle file not existing at python level,
        # C level is nasty and causes crash
        if self.mode == 'r' and not os.path.isfile(fname):
            raise IOError("File '{}' does not exist".format(fname))

        cdef tng_function_status stat
        fname_bytes = fname.encode('UTF-8')
        stat = tng_util_trajectory_open(fname_bytes, _mode, & self._traj)
        if stat != TNG_SUCCESS:
            raise IOError("File '{}' cannot be opened".format(fname))
        
        #python level
        self.is_open = True
        self.step = 0
        self.reached_eof = False


