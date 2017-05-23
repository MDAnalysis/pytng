from libc.stdint cimport int64_t

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

cdef class TNGFile:
    cdef tng_trajectory_t _traj
    cdef readonly fname
    cdef str mode
    cdef int is_open
    cdef readonly int64_t _n_frames

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

        cdef int ok
        ok = tng_util_trajectory_open(fname, _mode, & self._traj)
        if ok != TNG_SUCCESS:
            raise IOError("An error ocurred opening the file. {}".format(status_error_message[ok]))

        ok = tng_num_frames_get(self._traj, &self._n_frames)
        if ok != TNG_SUCCESS:
            raise IOError("An error ocurred reading n_frames. {}".format(status_error_message[ok]))

        self.is_open = True

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

    def __len__(self):
        return self.n_frames
