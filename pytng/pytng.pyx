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

cdef class TNGFile:
    cdef tng_trajectory_t _traj
    cdef readonly fname
    cdef str mode
    cdef int is_open

    def __cinit__(self, fname, mode='r'):
        self.fname = fname
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
        self.is_open = True

    def close(self):
        if self.is_open:
            tng_util_trajectory_close(&self._traj)
            self.is_open = False

    def __enter__(self):
        """Support context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support context manager"""
        self.close()
        # always propagate exceptions forward
        return False
