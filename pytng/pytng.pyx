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
cdef long long TNG_TRAJ_POSITIONS = 0x0000000010000001LL
cdef long long TNG_TRAJ_VELOCITIES = 0x0000000010000002LL
cdef long long TNG_TRAJ_FORCES = 0x0000000010000003LL

status_error_message = ['OK', 'Failure', 'Critical']

cdef extern from "tng/tng_io.h":
    ctypedef struct tng_trajectory_t:
        pass

    tng_function_status tng_num_frame_sets_get(
        const tng_trajectory_t tng_data,
        int64_t *n)

    tng_function_status tng_first_frame_nr_of_next_frame_set_get(
        const tng_trajectory_t tng_data,
        int64_t *frame)

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

    tng_function_status tng_num_frame_sets_get(
        const tng_trajectory_t tng_data,
        int64_t* n)

    tng_function_status tng_util_trajectory_next_frame_present_data_blocks_find(
        const tng_trajectory_t tng_data,
        int64_t current_frame,
        const int64_t n_requested_data_block_ids,
        const int64_t *requested_data_block_ids,
        int64_t *next_frame,
        int64_t *n_data_blocks_in_next_frame,
        int64_t **data_block_ids_in_next_frame)


TNGFrame = namedtuple("TNGFrame", "positions time step box")


cdef class MemoryWrapper:
    # holds a pointer to C allocated memory, deals with malloc&free
    # based on:
    # https://gist.github.com/GaelVaroquaux/1249305/ac4f4190c26110fe2791a1e7a6bed9c733b3413f
    cdef void* ptr

    def __cinit__(MemoryWrapper self, int size):
        # malloc not PyMem_Malloc as gmx later does realloc
        self.ptr = malloc(size)
        if self.ptr is NULL:
            raise MemoryError

    def __dealloc__(MemoryWrapper self):
        if self.ptr != NULL:
            free(self.ptr)

    cdef np.ndarray to_array(MemoryWrapper self, int nd, npy_intp* dims, int dtype):
        # convert wrapped memory to numpy array
        cdef np.ndarray arr
        cdef int err

        arr = PyArray_SimpleNewFromData(nd, dims, dtype, self.ptr)
        Py_INCREF(self)
        err = PyArray_SetBaseObject(arr, self)

        if err:
            raise ValueError("Failed to convert to array")
        return arr




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
    cdef int64_t blocks
    cdef int64_t* block_ids
    cdef int64_t last_frame

    def __cinit__(self, fname, mode='r'):
        self.fname = fname
        self._n_frames = -1
        self.block_ids = NULL
        self.open(self.fname, mode)

    def __dealloc__(self):
        if not self.block_ids == NULL:
            free(self.block_ids)
        self.close()

    cdef int64_t _get_nframes(self):
        # figure out how many frames of interest there are
        cdef int64_t i, next_id

        i = 0
        next_id = self.find_next_frame_id(-1)
        while next_id >= 0:
            i += 1
            next_id = self.find_next_frame_id(next_id)

        return i

    cpdef int64_t find_next_frame_id(self, int64_t current_frame):
        # return next frame id or -1 for EOF
        cdef tng_function_status status
        cdef int64_t next_id, num_blocks
        cdef int64_t* block_ids

        cdef MemoryWrapper wrap
        wrap = MemoryWrapper(sizeof(int64_t) * 1)
        block_ids = <int64_t*>malloc(sizeof(int64_t) * 1)

        status = tng_util_trajectory_next_frame_present_data_blocks_find(
            self._traj,
            current_frame,
            self.blocks, &self.block_ids[0],
            &next_id,
            &num_blocks,
            &block_ids)

        free(block_ids)

        # if we errored or no blocks found
        if status or (num_blocks == 0):
            return -1
        else:
            return next_id

        #print("Next frame id: ", next_id)
        #print("Blocks found: ", num_blocks)
        #print("Blocktypes: ")
        #print("Box ", req_blocks[0])
        #print("Positions ", req_blocks[1])
        #print("Velocities ", req_blocks[2])
        #print("Forces ", req_blocks[3])
        #print("Found: ")
        #for i in range(num_blocks):
        #    print(block_ids[i])
        #free(block_ids)

        return next_id

    def open(self, fname, mode):
        """Open a file handle

        Parameters
        ----------
        fname : str
           path to the file
        mode : str
           mode to open the file in, 'r' for read, 'w' for write
        """
        self.last_frame = -1
        # initialise what blocks we care about
        # TODO: One day make this customisable
        self.blocks = 2
        self.block_ids = <int64_t*>malloc(sizeof(int64_t) * 2)
        self.block_ids[0] = TNG_TRAJ_BOX_SHAPE
        self.block_ids[1] = TNG_TRAJ_POSITIONS
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
            self._n_frames = self._get_nframes()

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

        # get frame index of next frame
        cdef tng_function_status status
        cdef int64_t next_frame

        next_frame = self.find_next_frame_id(self.last_frame)

        cdef MemoryWrapper wrap
        cdef float* positions
        wrap = MemoryWrapper(3 * self.n_atoms * sizeof(float))
        positions = <float*> wrap.ptr

        cdef int64_t stride_length, n_values_per_frame
        status = tng_util_pos_read_range(self._traj, next_frame, next_frame, &positions, &stride_length)
        if status != TNG_SUCCESS:
            raise IOError("error reading frame")

        # move C data to numpy array
        cdef np.ndarray xyz
        cdef npy_intp dims[2]
        cdef int err
        cdef int nd = 2

        dims[0] = self.n_atoms
        dims[1] = 3

        xyz = wrap.to_array(nd, &dims[0], NPY_FLOAT)

        if err:
            raise ValueError('failed to create positions array')
        xyz *= self.distance_scale

        cdef double frame_time
        ok = tng_util_time_of_frame_get(self._traj, next_frame, &frame_time)
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
            ok = tng_data_vector_interval_get(self._traj, TNG_TRAJ_BOX_SHAPE, next_frame, next_frame, TNG_USE_HASH,
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
        self.last_frame = next_frame
        return TNGFrame(xyz, time, self.step - 1, box)

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
