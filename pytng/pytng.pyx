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

    # note that the _t suffix is a typedef mangle for a pointer to the base struct

    ctypedef struct tng_trajectory_t:
        pass

    ctypedef struct tng_gen_block_t:
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

    tng_function_status  tng_block_read_next(tng_trajectory_t tng_data,
                                             tng_gen_block_t  block_data,
                                             char             hash_mode)

    tng_function_status tng_block_init(tng_gen_block_t* block_p)


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
    cdef readonly fname
    cdef str mode 
    cdef int is_open
    cdef int reached_eof
    cdef int64_t step
 


    def __cinit__(self, fname, mode='r'):
        self.fname = fname
        self._n_frames = 0
        self._open(self.fname, mode)
    
    def __dealloc__(self):
        self.close()

    def _open(self, fname, mode):
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

        stat = tng_num_frames_get(self._traj, & self._n_frames)
        if stat != TNG_SUCCESS:
            raise IOError("Number of frames cannot be read")

        # python level
        self.is_open = True
        self.step = 0
        self.reached_eof = False

    def read_next_block(self):
        cdef tng_function_status stat
        cdef tng_gen_block_t block
        stat = tng_block_init( & block)
        if stat != TNG_SUCCESS:
            raise ValueError("failed to init block")
        stat = tng_block_read_next(self._traj, block, TNG_SKIP_HASH)
        if stat != TNG_SUCCESS:
            raise ValueError("failed to read subsequent block")




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
    cdef bint _pos
    cdef bint _box
    cdef bint _frc
    cdef bint _vel
    cdef int64_t _pos_stride
    cdef int64_t _box_stride
    cdef int64_t _frc_stride
    cdef int64_t _vel_stride
    cdef int64_t step
    cdef float distance_scale

    def __cinit__(self, fname, mode='r'):
        self.fname = fname
        self._n_frames = -1
        self._pos = 0
        self._frc = 0
        self._vel = 0
        self._pos_stride = 0
        self._box_stride = 0
        self._frc_stride = 0
        self._vel_stride = 0
        self.open(self.fname, mode)
        self.get_strides()

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
            raise IOError("An error ocurred opening the file. {}".format(
                status_error_message[ok]))

        if self.mode == 'r':
            ok = tng_num_frames_get(self._traj, & self._n_frames)
            if ok != TNG_SUCCESS:
                raise IOError("An error ocurred reading n_frames. {}".format(
                    status_error_message[ok]))

            ok = tng_num_particles_get(self._traj, & self._n_atoms)
            if ok != TNG_SUCCESS:
                raise IOError("An error ocurred reading n_atoms. {}".format(
                    status_error_message[ok]))

            ok = tng_distance_unit_exponential_get(self._traj, & exponent)
            if ok != TNG_SUCCESS:
                raise IOError("An error ocurred reading distance unit exponent. {}".format(
                    status_error_message[ok]))
            self.distance_scale = 10.0**(exponent+9)

        self.is_open = True
        self.step = 0
        self.reached_eof = False

    def get_strides(self):
        # ugly, relies on there being data of the right type at the first frame
        cdef float * pos_ptr = NULL
        cdef float * box_ptr = NULL
        cdef float * vel_ptr = NULL
        cdef float * frc_ptr = NULL
        cdef int64_t stride_length = 0
        cdef tng_function_status ok

        ok = tng_util_pos_read_range(self._traj, 0, 0, & pos_ptr, & stride_length)
        if (ok == TNG_SUCCESS) and pos_ptr:
            self._pos = 1  # true
            self._pos_stride = stride_length

        ok = tng_util_box_shape_read_range(self._traj, 0, 0, & box_ptr, & stride_length)
        if (ok == TNG_SUCCESS) and box_ptr:
            self._box = 1  # true
            self._box_stride = stride_length

        ok = tng_util_vel_read_range(self._traj, 0, 0, & vel_ptr, & stride_length)
        if (ok == TNG_SUCCESS) and vel_ptr:
            self._vel = 1  # true
            self._vel_stride = stride_length

        ok = tng_util_force_read_range(self._traj, 0, 0, & frc_ptr, & stride_length)
        if (ok == TNG_SUCCESS) and frc_ptr:
            self._frc = 1  # true
            self._frc_stride = stride_length

    def close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            tng_util_trajectory_close( & self._traj)
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

    @property
    def _pos(self):
        """has positions"""
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._pos

    @property
    def _box(self):
        """has box data"""
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._box

    @property
    def _vel(self):
        """has vel data"""
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._vel

    @property
    def _frc(self):
        """has force data"""
        if not self.is_open:
            raise IOError('No file currently opened')
        return self._frc

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

        # TODO this seem wasteful but can't cdef inside a conditional?
        cdef MemoryWrapper wrap_pos
        cdef int64_t stride_length, ok
        cdef np.ndarray xyz
        cdef npy_intp dims[2]
        cdef int err
        cdef int nd = 2

        xyz = None

        if self._pos:
            if (self.step % self._pos_stride == 0):
                wrap_pos = MemoryWrapper(3 * self.n_atoms * sizeof(float))
                positions = <float*> wrap_pos.ptr
                # TODO this will break when using frames spaced more than 1 apart
                ok = tng_util_pos_read_range(self._traj, self.step, self.step, & positions, & stride_length)
                if ok != TNG_SUCCESS:
                    raise IOError("error reading frame")

                dims[0] = self.n_atoms
                dims[1] = 3
                xyz = PyArray_SimpleNewFromData(
                    nd, dims, NPY_FLOAT, wrap_pos.ptr)
                Py_INCREF(wrap_pos)
                err = PyArray_SetBaseObject(xyz, wrap_pos)
                if err:
                    raise ValueError('failed to create positions array')
                xyz *= self.distance_scale
        else:
            xyz = None

        # TODO this seem wasteful but can't cdef inside a conditional?
        cdef MemoryWrapper wrap_box
        cdef np.ndarray[ndim = 2, dtype = np.float32_t, mode = 'c'] box = np.empty((3, 3), dtype=np.float32)

        if self._box:
            if (self.step % self._box_stride == 0):
                # BOX SHAPE
                # cdef float* box_s
                wrap_box = MemoryWrapper(3 * 3 * sizeof(float))
                box_shape = <float*> wrap_box.ptr
                # TODO this will break when using frames spaced more than 1 apart
                ok = tng_util_box_shape_read_range(self._traj, self.step, self.step, & box_shape, & stride_length)
                if ok != TNG_SUCCESS:
                    raise IOError("error reading box shape")
                # populate box, can this be done the same way as positions above? #TODO is there a canonical way to convert to numpy array
                for i in range(3):
                    for j in range(3):
                        box[i, j] = box_shape[3*i + j]
        else:
            box = None

        cdef MemoryWrapper wrap_vel
        cdef np.ndarray vels

        if self._vel:
            if (self.step % self._vel_stride == 0):
                wrap_vel = MemoryWrapper(3 * self.n_atoms * sizeof(float))
                velocities = <float*> wrap_vel.ptr
                ok = tng_util_vel_read_range(self._traj, self.step, self.step, & velocities, & stride_length)
                if ok != TNG_SUCCESS:
                    raise IOError("error reading velocities")
                
                dims[0] = self.n_atoms
                dims[1] = 3
                vels = PyArray_SimpleNewFromData(
                    nd, dims, NPY_FLOAT, wrap_vel.ptr)
                Py_INCREF(wrap_vel)
                err = PyArray_SetBaseObject(vels, wrap_vel)
                if err:
                    raise ValueError('failed to create positions array')
        else:
            vels = None
                
        cdef MemoryWrapper wrap_frc
        cdef np.ndarray frc

        if self._frc:
            if (self.step % self._frc_stride == 0):
                wrap_frc = MemoryWrapper(3 * self.n_atoms * sizeof(float))
                forces = <float*> wrap_frc.ptr
                ok = tng_util_force_read_range(self._traj, self.step, self.step, & forces, & stride_length)
                if ok != TNG_SUCCESS:
                    raise IOError("error reading forces")
                
                dims[0] = self.n_atoms
                dims[1] = 3
                frc = PyArray_SimpleNewFromData(
                    nd, dims, NPY_FLOAT, wrap_frc.ptr)
                Py_INCREF(wrap_frc)
                err = PyArray_SetBaseObject(frc, wrap_frc)
        else:
            frc = None

        # FRAME
        cdef double frame_time
        ok = tng_util_time_of_frame_get(self._traj, self.step, & frame_time)
        if ok != TNG_SUCCESS:
            # No time available
            time = None
        else:
            time = frame_time * 1e12

        # return frame_data
        self.step += 1


        return TNGFrame(xyz, vels, frc, time, self.step - 1, box)

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
                    raise TypeError(
                        "Boolean index must match length of trajectory")

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
