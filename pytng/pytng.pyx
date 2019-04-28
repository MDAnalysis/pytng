# cython: linetrace=True
# cython: embedsignature=True
# distutils: define_macros=CYTHON_TRACE=1
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

from collections import namedtuple
import os
import numbers
import numpy as np

cimport numpy as np
np.import_array()
from numpy cimport (PyArray_SimpleNewFromData,
                    PyArray_SetBaseObject,
                    NPY_FLOAT,
                    Py_INCREF,
                    npy_intp,
)


ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL
ctypedef enum tng_data_type: TNG_CHAR_DATA, TNG_INT_DATA, TNG_FLOAT_DATA, TNG_DOUBLE_DATA
ctypedef enum tng_hash_mode: TNG_SKIP_HASH, TNG_USE_HASH

cdef long long TNG_TRAJ_BOX_SHAPE = 0x0000000010000000LL

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

    tng_function_status tng_util_time_of_frame_get(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        double *time)

    tng_function_status tng_data_vector_interval_get(
        const tng_trajectory_t tng_data,
        const int64_t block_id,
        const int64_t start_frame_nr,
        const int64_t end_frame_nr,
        const char hash_mode,
        void **values,
        int64_t *stride_length,
        int64_t *n_values_per_frame,
        char *type)

TNGFrame = namedtuple("TNGFrame", "positions time step box")


cdef class MemoryWrapper:
    # holds a pointer to C allocated memory
    cdef void* ptr
    cdef int size

    def __cinit__(MemoryWrapper self):
        self.size = 0
        self.ptr = NULL

    def __dealloc__(MemoryWrapper self):
        if self.ptr != NULL:
            free(self.ptr)

    cdef void set_data(MemoryWrapper self, void* ptr, int size):
        self.ptr = ptr
        self.size = size



cdef class TNGFile:
    """File handle object for TNG files

    Supports use as a context manager ("with" blocks).
    """
    cdef tng_trajectory_t _traj
    cdef readonly fname
    cdef str mode
    cdef int is_open
    cdef int reached_eof
    cdef int64_t _n_frames
    cdef int64_t _n_atoms
    cdef int64_t step
    cdef float distance_scale

    def __cinit__(self, fname, mode='r'):
        self.fname = fname
        self._n_frames = -1
        self.open(self.fname, mode)

    def __dealloc__(self):
        self.close()

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
        else:
            raise IOError('mode must be one of "r" or "w", you '
                          'supplied {}'.format(mode))

        cdef int64_t exponent, ok

        # handle file not existing at python level,
        # C level is nasty and causes crash
        if self.mode == 'r' and not os.path.isfile(fname):
            raise IOError("File '{}' does not exist".format(fname))

        fname_bytes = fname.encode('UTF-8')
        ok = tng_util_trajectory_open(fname_bytes, _mode, & self._traj)
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
        self.step = 0
        self.reached_eof = False

    def close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            tng_util_trajectory_close(&self._traj)
            self.is_open = False
            self._n_frames = -1

    def __enter__(self):
        # Support context manager
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # always propagate exceptions forward
        return False

    def __iter__(self):
        self.close()
        self.open(self.fname, self.mode)
        return self

    def __next__(self):
        if self.reached_eof:
            raise StopIteration
        return self.read()

    @property
    def n_frames(self):
        """Number of frames in the trajectory"""
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._n_frames

    @property
    def n_atoms(self):
        """Number of atoms in each frame"""
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._n_atoms

    def __len__(self):
        return self.n_frames

    def tell(self):
        """Get current frame"""
        return self.step

    def read(self):
        """Read the next frame

        Returns
        -------
        frame : namedtuple
          contains all data from this frame
        """
        if self.reached_eof:
            raise IOError('Reached last frame in TNG, seek to 0')
        if not self.is_open:
            raise IOError('No file opened')
        if self.mode != 'r':
            raise IOError('File opened in mode: {}. Reading only allow '
                          'in mode "r"'.format('self.mode'))
        if self.step >= self.n_frames:
            self.reached_eof = True
            raise StopIteration("Reached EOF in read")

        cdef MemoryWrapper wrap
        cdef float* positions
        # malloc not PyMem_Malloc as gmx later does realloc call on this
        # freeing is done in Wrapper object __dealloc__
        positions = <float*> malloc(3 * self.n_atoms * sizeof(float))
        wrap = MemoryWrapper()
        wrap.set_data(<void*> positions, 3 * self.n_atoms)

        cdef np.ndarray xyz
        cdef int64_t stride_length, ok, i, n_values_per_frame

        ok = tng_util_pos_read_range(self._traj, self.step, self.step, &positions, &stride_length)
        if ok != TNG_SUCCESS:
            raise IOError("error reading frame")

        # move C data to numpy array
        cdef npy_intp dims[2]
        cdef int err
        cdef int nd = 2

        dims[0] = self.n_atoms
        dims[1] = 3
        xyz = PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, <void*>positions)
        Py_INCREF(wrap)
        err = PyArray_SetBaseObject(xyz, wrap)
        if err:
            raise ValueError('failed to create positions array')
        xyz *= self.distance_scale

        cdef double frame_time
        ok = tng_util_time_of_frame_get(self._traj, self.step, &frame_time)
        if ok != TNG_SUCCESS:
            # No time available
            time = None
        else:
            time = frame_time * 1e12

        cdef np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] box = np.empty((3, 3), dtype=np.float32)
        cdef char data_type
        cdef void* box_shape = NULL
        cdef float* float_box
        cdef double* double_box

        try:
            ok = tng_data_vector_interval_get(self._traj, TNG_TRAJ_BOX_SHAPE, self.step, self.step, TNG_USE_HASH,
                                              &box_shape, &stride_length, &n_values_per_frame, &data_type)
            if ok != TNG_SUCCESS:
                raise IOError("error reading box shape")

            if data_type == TNG_DOUBLE_DATA:
                double_box = <double*>box_shape
                for j in range(3):
                    for k in range(3):
                        box[j, k] = double_box[j*3 + k]
            else:
                float_box = <float*>box_shape
                for j in range(3):
                    for k in range(3):
                        box[j, k] = float_box[j*3 + k]
            box *= self.distance_scale
        finally:
            if box_shape != NULL:
                free(box_shape)

        self.step += 1
        return TNGFrame(xyz.reshape(self.n_atoms, 3), time, self.step - 1, box)

    def seek(self, step):
        """Move the file handle to a particular frame number

        Parameters
        ----------
        step : int
           desired frame number
        """
        if self.mode == 'w':
            raise IOError("seek not allowed in write mode")
        if self.is_open:
            if step < 0:
                step += len(self)
            if (step < 0) or (step >= len(self)):
                raise IndexError("Seek index out of bounds")
            self.step = step
            self.reached_eof = False
        else:
            raise IOError('No file currently opened')

    def __getitem__(self, frame):
        cdef int64_t start, stop, step, i

        if isinstance(frame, numbers.Integral):
            self.seek(frame)
            return self.read()
        elif isinstance(frame, (list, np.ndarray)):
            if isinstance(frame[0], (bool, np.bool_)):
                if not (len(frame) == len(self)):
                    raise TypeError("Boolean index must match length of trajectory")

                # Avoid having list of bools
                frame = np.asarray(frame, dtype=np.bool)
                # Convert bool array to int array
                frame = np.arange(len(self))[frame]

            def listiter(frames):
                for f in frames:
                    if not isinstance(f, numbers.Integral):
                        raise TypeError("Frames indices must be integers")
                    self.seek(f)
                    yield self.read()
            return listiter(frame)
        elif isinstance(frame, slice):
            start = frame.start if frame.start is not None else 0
            stop = frame.stop if frame.stop is not None else self.n_frames
            step = frame.step if frame.step is not None else 1
            def sliceiter(start, stop, step):
                for i in range(start, stop, step):
                    self.seek(i)
                    yield self.read()
            return sliceiter(start, stop, step)
        else:
            raise TypeError("Trajectories must be an indexed using an integer,"
                            " slice or list of indices")
