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

ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL
ctypedef enum tng_data_type: TNG_CHAR_DATA, TNG_INT_DATA, TNG_FLOAT_DATA, TNG_DOUBLE_DATA
ctypedef enum tng_hash_mode: TNG_SKIP_HASH, TNG_USE_HASH
ctypedef enum tng_block_type: TNG_NON_TRAJECTORY_BLOCK, TNG_TRAJECTORY_BLOCK
ctypedef enum tng_compression: TNG_UNCOMPRESSED, TNG_XTC_COMPRESSION, TNG_TNG_COMPRESSION, TNG_GZIP_COMPRESSION
ctypedef enum tng_particle_dependency: TNG_NON_PARTICLE_BLOCK_DATA, TNG_PARTICLE_BLOCK_DATA

cdef long long TNG_TRAJ_BOX_SHAPE = 0x0000000010000000LL
cdef long long TNG_TRAJ_POSITIONS = 0x0000000010000001LL

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

    tng_function_status tng_util_box_shape_write_interval_set(
        const tng_trajectory_t tng_data,
        const int64_t interval)

    tng_function_status tng_util_pos_write_interval_set(
        const tng_trajectory_t tng_data,
        const int64_t interval)

    tng_function_status tng_util_vel_write_interval_set(
        const tng_trajectory_t tng_data,
        const int64_t interval)

    tng_function_status tng_util_force_write_interval_set(
        const tng_trajectory_t tng_data,
        const int64_t interval)

    tng_function_status tng_util_box_shape_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const float *box_shape)

    tng_function_status tng_util_box_shape_with_time_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const double time,
        const float *box_shape)

    tng_function_status tng_util_pos_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const float *positions)

    tng_function_status tng_util_pos_with_time_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const double time,
        const float *positions)

    tng_function_status tng_util_vel_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const float *velocities)

    tng_function_status tng_util_vel_with_time_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const double time,
        const float *velocities)

    tng_function_status tng_util_force_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const float *forces)

    tng_function_status tng_util_force_with_time_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const double time,
        const float *forces)

    tng_function_status tng_implicit_num_particles_set(
        const tng_trajectory_t tng_data,
        const int64_t n)

    tng_function_status tng_num_frames_per_frame_set_set(
        const tng_trajectory_t tng_data,
        const int64_t n)

    tng_function_status tng_util_generic_with_time_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const double time,
        const float *values,
        const int64_t n_values_per_frame,
        const int64_t block_id,
        const char *block_name,
        const char particle_dependency,
        const char compression)

    tng_function_status tng_time_per_frame_set(
        const tng_trajectory_t tng_data,
        const double time)

    tng_function_status tng_util_generic_with_time_write(
        const tng_trajectory_t tng_data,
        const int64_t frame_nr,
        const double time,
        const float *values,
        const int64_t n_values_per_frame,
        const int64_t block_id,
        const char *block_name,
        const char particle_dependency,
        const char compression)

    tng_function_status tng_util_generic_write_interval_set(
        const tng_trajectory_t tng_data,
        const int64_t i,
        const int64_t n_values_per_frame,
        const int64_t block_id,
        const char *block_name,
        const char particle_dependency,
        const char compression)


    tng_function_status tng_frame_set_write(
        const tng_trajectory_t tng_data,
        const char hash_mode)

TNGFrame = namedtuple("TNGFrame", "positions time step box")

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
        elif self.mode == 'w':
            self._n_frames = 0  # No frame were written yet
            # self._n_atoms ?

        self.is_open = True
        self.step = 0
        self.reached_eof = False

    def close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            ok = tng_util_trajectory_close(&self._traj)
            if_not_ok(ok, "couldn't close")
            self.is_open = False
            self._n_frames = -1
            print("closed file")

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
            raise IOError('File opened in mode: {}. Reading only allowed '
                          'in mode "r"'.format(self.mode))
        if self.step >= self.n_frames:
            self.reached_eof = True
            raise StopIteration("Reached EOF in read")

        cdef np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] xyz = np.empty((self.n_atoms, 3), dtype=np.float32)
        cdef float* positions = NULL
        cdef int64_t stride_length, ok, i, n_values_per_frame

        try:
            ok = tng_util_pos_read_range(self._traj, self.step, self.step, &positions, &stride_length)
            if ok != TNG_SUCCESS:
                raise IOError("error reading frame")

            for i in range(self._n_atoms):
                for j in range(3):
                    xyz[i, j] = positions[i*3 + j]
            xyz *= self.distance_scale
        finally:
            if positions != NULL:
                free(positions)

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
            # DO NOT FREE float_box or double_box here. They point to the same
            # memory as box_shape

        self.step += 1
        return TNGFrame(xyz, time, self.step - 1, box)

    def write(self,
              np.ndarray[np.float32_t, ndim=2, mode='c'] positions,
              np.ndarray[np.float32_t, ndim=2, mode='c'] box,
              time=None):
        if self.mode != 'w':
            raise IOError('File opened in mode: {}. Writing only allowed '
                          'in mode "w"'.format(self.mode))
        if not self.is_open:
            raise IOError('No file currently opened')

        cdef int64_t ok
        cdef np.ndarray[float, ndim=2, mode='c'] xyz
        cdef np.ndarray[float, ndim=2, mode='c'] box_contiguous
        cdef double dt

        if self._n_frames == 0:
            # TODO: The number of frames per frame set should be tunable.
            ok = tng_num_frames_per_frame_set_set(self._traj, 1)
            if_not_ok(ok, 'Could not set the number of frames per frame set')
            # The number of atoms must be set either with a full description
            # of the system content (topology), or with just the number of
            # particles. We should fall back on the latter, but being
            # able to write the topology would be a nice addition
            # in the future.
            self._n_atoms = positions.shape[0]
            ok = tng_implicit_num_particles_set(self._traj, self.n_atoms)
            if_not_ok(ok, 'Could not set the number of particles')
            # Set the writing interval to 1 for all blocks.
            ok = tng_util_pos_write_interval_set(self._traj, 1)
            if_not_ok(ok, 'Could not set the writing interval for positions')
            # When we use the "tn_util_box_shape_*" functions to write the box
            # shape, gromacs tools fail to uncompress the data block. Instead of
            # using the default gzip to compress the box, we do not compress it.
            # ok = tng_util_box_shape_write_interval_set(self._traj, 1)
            ok = tng_util_generic_write_interval_set(
                self._traj, 1, 9,
                TNG_TRAJ_BOX_SHAPE,
                "BOX SHAPE",
                TNG_NON_PARTICLE_BLOCK_DATA,
                TNG_UNCOMPRESSED
            )
            if_not_ok(ok, 'Could not set the writing interval for the box shape')
        elif self.n_atoms != positions.shape[0]:
            message = ('Only fixed number of particles is currently supported. '
                       'Cannot write {} particles instead of {}.'
                       .format(positions.shape[0], self.n_atoms))
            raise NotImplementedError(message)

        if time is not None:
            try:
                time = float(time)  # Make sure time is a real
                # Time is provided to this function in picoseconds,
                # but functions from tng_io expect seconds.
                time *= 1e-12
            except ValueError:
                raise ValueError('time must be a real number or None')
            # The time per frame has to be set for the time to be written in
            # the frames.
            # To be able to set an arbitrary time, we need to set the time per
            # frame to 0 and to use one frame per frame set. Using the actual
            # time difference between consecutive frames can cause issues if
            # the difference is negative, or if the difference is 0 and the
            # frame is not the first of the frame set.
            ok = tng_time_per_frame_set(self._traj, 0)
            if_not_ok(ok, 'Could not set the time per frame')

        box_contiguous = np.ascontiguousarray(box, dtype=np.float32)
        if time is None:
            ok = tng_util_box_shape_write(self._traj, self.step,
                                          &box_contiguous[0, 0])
        else:
            #ok = tng_util_box_shape_with_time_write(self._traj,
            #                                        self.step,
            #                                        time,
            #                                        &box_contiguous[0, 0])
            ok = tng_util_generic_with_time_write(
                self._traj, self.step, time,
                &box[0, 0],
                9, TNG_TRAJ_BOX_SHAPE, "BOX SHAPE",
                TNG_NON_PARTICLE_BLOCK_DATA,
                TNG_UNCOMPRESSED
            )
        if_not_ok(ok, 'Could not write box shape')

        xyz = np.ascontiguousarray(positions, dtype=np.float32)
        if time is None:
            ok = tng_util_pos_write(self._traj, self.step, &xyz[0, 0])
        else:
            ok = tng_util_pos_with_time_write(self._traj, self.step,
                                              time, &xyz[0, 0])
        if_not_ok(ok, 'Could not write positions')

        # finish frame set to write step, hashing should be configurable
        tng_frame_set_write(self._traj, TNG_USE_HASH)

        self.step += 1
        self._n_frames += 1

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


def if_not_ok(ok, message, exception=IOError):
    if ok != TNG_SUCCESS:
        raise exception(message)
