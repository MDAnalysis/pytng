# cython: linetrace=True
# cython: embedsignature=True
# cython: profile=True
# distutils: define_macros=CYTHON_TRACE=1
from numpy cimport(PyArray_SimpleNewFromData,
                   PyArray_SetBaseObject,
                   NPY_FLOAT,
                   NPY_DOUBLE,
                   Py_INCREF,
                   npy_intp,
                   )

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.stdint cimport int64_t, uint64_t, int32_t, uint32_t
from libc.stdlib cimport malloc, free, realloc
from libc.stdio cimport printf, FILE, SEEK_SET, SEEK_CUR, SEEK_END
from libc.string cimport memcpy

#from libcpp.cast cimport reinterpret_cast

from posix.types cimport off_t


from collections import namedtuple
import os
import numbers
import numpy as np

cimport numpy as np
np.import_array()

from  cython.operator cimport dereference


ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL
ctypedef enum tng_hash_mode: TNG_SKIP_HASH, TNG_USE_HASH
ctypedef enum tng_datatypes: TNG_CHAR_DATA, TNG_INT_DATA, TNG_FLOAT_DATA, TNG_DOUBLE_DATA
ctypedef enum  tng_particle_dependency: TNG_NON_PARTICLE_BLOCK_DATA, TNG_PARTICLE_BLOCK_DATA
ctypedef enum tng_compression: TNG_UNCOMPRESSED, TNG_XTC_COMPRESSION, TNG_TNG_COMPRESSION, TNG_GZIP_COMPRESSION


status_error_message = ['OK', 'Failure', 'Critical']


cdef extern from "string.h":
    size_t strlen(char *s)

cdef extern from "<stdio.h>" nogil:
    # Seek and tell with off_t
    int fseeko(FILE *, off_t, int)
    off_t ftello(FILE *)


cdef extern from "tng/tng_io.h":

    cdef enum:
        TNG_MAX_STR_LEN
        TNG_MD5_HASH_LEN

    

    # note that the _t suffix is a typedef mangle for a pointer to the base struct
    ctypedef struct tng_molecule_t:
        pass


    struct tng_particle_mapping:

        #/** The index number of the first particle in this mapping block */
        int64_t num_first_particle
        #/** The number of particles list in this mapping block */
        int64_t n_particles
        #/** the mapping of index numbers to the real particle numbers in the
        #* trajectory. real_particle_numbers[0] is the real particle number
        #* (as it is numbered in the molecular system) of the first particle
        #* in the data blocks covered by this particle mapping block */
        int64_t* real_particle_numbers


    struct tng_trajectory_frame_set:

        #/** The number of different particle mapping blocks present. */
        int64_t n_mapping_blocks
        #/** The atom mappings of this frame set */
        tng_particle_mapping* mappings
        #/** The first frame of this frame set */
        int64_t first_frame
        #/** The number of frames in this frame set */
        int64_t n_frames
        #/** The number of written frames in this frame set (used when writing o
        #* frame at a time). */
        int64_t n_written_frames
        #/** The number of frames not yet written to file in this frame set
        #* (used from the utility functions to finish the writing properly. */
        int64_t n_unwritten_frames


        #/** A list of the number of each molecule type - only used when using
        #* variable number of atoms */
        int64_t* molecule_cnt_list
        #/** The number of particles/atoms - only used when using variable numbe
        #* of atoms */
        int64_t n_particles
        #/** The file position of the next frame set */
        int64_t next_frame_set_file_pos
        #/** The file position of the previous frame set */
        int64_t prev_frame_set_file_pos
        #/** The file position of the frame set one long stride step ahead */
        int64_t medium_stride_next_frame_set_file_pos
        #/** The file position of the frame set one long stride step behind */
        int64_t medium_stride_prev_frame_set_file_pos
        #/** The file position of the frame set one long stride step ahead */
        int64_t long_stride_next_frame_set_file_pos
        #/** The file position of the frame set one long stride step behind */
        int64_t long_stride_prev_frame_set_file_pos
        #/** Time stamp (in seconds) of first frame in frame set */
        double first_frame_time

        #/* The data blocks in a frame set are trajectory data blocks */
        #/** The number of trajectory data blocks of particle dependent data */
        int n_particle_data_blocks
        #/** A list of data blocks containing particle dependent data */
        tng_data* tr_particle_data
        #/** The number of trajectory data blocks independent of particles */
        int n_data_blocks
        #/** A list of data blocks containing particle indepdendent data */
        tng_data* tr_data


    
    #/* FIXME: Should there be a pointer to a tng_gen_block from each data block? */
    struct tng_data:
        #/** The block ID of the data block containing this particle data.
        # *  This is used to determine the kind of data that is stored */
        int64_t block_id
        #/** The name of the data block. This is used to determine the kind of
        # *  data that is stored */
        char* block_name
        #/** The type of data stored. */
        char datatype
        #/** A flag to indicate if this data block contains frame and/or particle dependent
        # * data */
        char dependency
        #/** The frame number of the first data value */
        int64_t first_frame_with_data
        #/** The number of frames in this frame set */
        int64_t n_frames
        #/** The number of values stored per frame */
        int64_t n_values_per_frame
        #/** The number of frames between each data point - e.g. when
        # *  storing sparse data. */
        int64_t stride_length
        #/** ID of the CODEC used for compression 0 == no compression. */
        int64_t codec_id
        #/** If reading one frame at a time this is the last read frame */
        int64_t last_retrieved_frame
        #/** The multiplier used for getting integer values for compression */
        double compression_multiplier
        #/** A 1-dimensional array of values of length
        # *  [sizeof (datatype)] * n_frames * n_particles * n_values_per_frame */
        void* values
        #/** If storing character data store it in a 3-dimensional array */
        char**** strings

    
    struct tng_trajectory:
        #/** The path of the input trajectory file */
        char* input_file_path
        #/** A handle to the input file */
        FILE* input_file
        #/** The length of the input file */
        int64_t input_file_len
        #/** The path of the output trajectory file */
        char* output_file_path
        #/** A handle to the output file */
        FILE* output_file
        #/** Function to swap 32 bit values to and from the endianness of the
        #* input file */
        tng_function_status (*input_endianness_swap_func_32)(const tng_trajectory*, uint32_t*);
        #/** Function to swap 64 bit values to and from the endianness of the
        #* input file */
        tng_function_status (*input_endianness_swap_func_64)(const  tng_trajectory*, uint64_t*);
        #/** Function to swap 32 bit values to and from the endianness of the
        #* input file */
        tng_function_status (*output_endianness_swap_func_32)(const  tng_trajectory*, uint32_t*);
        #/** Function to swap 64 bit values to and from the endianness of the
        #* input file */
        tng_function_status (*output_endianness_swap_func_64)(const  tng_trajectory*, uint64_t*);
        #/** The endianness of 32 bit values of the current computer */
        char endianness_32
        #/** The endianness of 64 bit values of the current computer */
        char endianness_64

        #/** The name of the program producing this trajectory */
        char* first_program_name
        #/** The forcefield used in the simulations */
        char* forcefield_name
        #/** The name of the user running the simulations */
        char* first_user_name
        #/** The name of the computer on which the simulations were performed */
        char* first_computer_name
        #/** The PGP signature of the user creating the file. */
        char* first_pgp_signature
        #/** The name of the program used when making last modifications to the
        #*  file */
        char* last_program_name
        #/** The name of the user making the last modifications to the file */
        char* last_user_name
        #/** The name of the computer on which the last modifications were made */
        char* last_computer_name
        #/** The PGP signature of the user making the last modifications to the
        # *  file. */
        char* last_pgp_signature
        #/** The time (n seconds since 1970) when the file was created */
        int64_t time
        #/** The exponential of the value of the distance unit used. The default
        #* distance unit is nm (1e-9), i.e. distance_unit_exponential = -9. If
        #* the measurements are in Å the distance_unit_exponential = -10. */
        int64_t distance_unit_exponential

        #/** A flag indicating if the number of atoms can vary throughout the
        # *  simulation, e.g. using a grand canonical ensemble */
        char var_num_atoms_flag
        #/** The number of frames in a frame set. It is allowed to have frame sets
        #*  with fewer frames, but this will help searching for specific frames */
        int64_t frame_set_n_frames
        #/** The number of frame sets in a medium stride step */
        int64_t medium_stride_length
        #/** The number of frame sets in a long stride step */
        int64_t long_stride_length
        #/** The current (can change from one frame set to another) time length
        #*  (in seconds) of one frame */
        double time_per_frame

        #/** The number of different kinds of molecules in the trajectory */
        int64_t n_molecules
        #/** A list of molecules in the trajectory */
        tng_molecule_t molecules;
        #/** A list of the count of each molecule - if using variable number of
        #*  particles this will be specified in each frame set */
        int64_t* molecule_cnt_list
        #/** The total number of particles/atoms. If using variable number of
        #*  particles this will be specified in each frame set */
        int64_t n_particles

        #/** The pos in the src file of the first frame set */
        int64_t first_trajectory_frame_set_input_file_pos
        #/** The pos in the dest file of the first frame set */
        int64_t first_trajectory_frame_set_output_file_pos
        #/** The pos in the src file of the last frame set */
        int64_t last_trajectory_frame_set_input_file_pos
        #/** The pos in the dest file of the last frame set */
        int64_t last_trajectory_frame_set_output_file_pos
        #/** The currently active frame set */
        tng_trajectory_frame_set current_trajectory_frame_set
        #/** The pos in the src file of the current frame set */
        int64_t current_trajectory_frame_set_input_file_pos
        #/** The pos in the dest file of the current frame set */
        int64_t current_trajectory_frame_set_output_file_pos
        #/** The number of frame sets in the trajectory N.B. Not saved in file and
        #*  cannot be trusted to be up-to-date */
        int64_t n_trajectory_frame_sets

        #/* These data blocks are non-trajectory data blocks */
        #/** The number of non-frame dependent particle dependent data blocks */
        int n_particle_data_blocks
        #/** A list of data blocks containing particle dependent data */
        tng_data* non_tr_particle_data

        #/** The number of frame and particle independent data blocks */
        int n_data_blocks
        #/** A list of frame and particle indepdendent data blocks */
        tng_data* non_tr_data

        #/** TNG compression algorithm for compressing positions */
        int* compress_algo_pos
        #/** TNG compression algorithm for compressing velocities */
        int* compress_algo_vel
        #/** The precision used for lossy compression */
        double compression_precision

    struct tng_gen_block:
        #The size of the block header in bytes */
        int64_t header_contents_size
        #The size of the block contents in bytes */
        int64_t block_contents_size
        # The ID of the block to determine its type */
        int64_t id
        #The MD5 hash of the block to verify integrity */
        char md5_hash[16] #TNG_MD5_HASH_LEN == 16
        #The name of the block */
        char *name
        #The library version used to write the block */
        int64_t block_version
        int64_t alt_hash_type
        int64_t alt_hash_len
        char *alt_hash
        int64_t signature_type
        int64_t signature_len
        char *signature
        # The full block header contents */
        char *header_contents
        # The full block contents */
        char *block_contents


    tng_function_status tng_util_trajectory_open(
        const char * filename,
        const char mode,
        tng_trajectory* * tng_data_p)

    tng_function_status tng_util_trajectory_close(
        tng_trajectory* * tng_data_p)

    tng_function_status tng_num_frames_get(
        const tng_trajectory* tng_data,
        int64_t * n)

    tng_function_status tng_num_particles_get(
        const tng_trajectory* tng_data,
        int64_t * n)

    tng_function_status tng_distance_unit_exponential_get(
        const tng_trajectory* tng_data,
        int64_t * exp)

    tng_function_status tng_util_pos_read_range(
        const tng_trajectory* tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** positions,
        int64_t * stride_length)

    tng_function_status tng_util_box_shape_read_range(
        const tng_trajectory* tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** box_shape,
        int64_t * stride_length)

    tng_function_status tng_util_vel_read_range(
        const tng_trajectory* tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** velocities,
        int64_t * stride_length)

    tng_function_status tng_util_force_read_range(
        const tng_trajectory* tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** forces,
        int64_t * stride_length)

    tng_function_status tng_util_pos_read(
        const tng_trajectory* tng_data,
        float ** positions,
        int64_t * stride_length)

    tng_function_status tng_util_box_shape_read(
        const tng_trajectory* tng_data,
        float ** box_shape,
        int64_t * stride_length)

    tng_function_status tng_util_vel_read(
        const tng_trajectory* tng_data,
        float ** velocities,
        int64_t * stride_length)

    tng_function_status tng_util_force_read(
        const tng_trajectory* tng_data,
        float ** forces,
        int64_t * stride_length)

    tng_function_status tng_util_time_of_frame_get(
        const tng_trajectory* tng_data,
        const int64_t frame_nr,
        double * time)

    tng_function_status tng_data_vector_interval_get(
        const tng_trajectory* tng_data,
        const int64_t block_id,
        const int64_t start_frame_nr,
        const int64_t end_frame_nr,
        const char hash_mode,
        void ** values,
        int64_t * stride_length,
        int64_t * n_values_per_frame,
        char * type)

    tng_function_status  tng_block_read_next(tng_trajectory* tng_data,
                                             tng_gen_block*  block_data,
                                             char             hash_mode)

    tng_function_status tng_block_init(tng_gen_block** block_p)

    tng_function_status tng_block_header_read(tng_trajectory* tng_data, tng_gen_block* block)

    tng_function_status tng_num_frame_sets_get(tng_trajectory* tng_data, int64_t * n)

    tng_function_status tng_block_destroy(tng_gen_block** block_p)

    tng_function_status tng_data_get_stride_length( tng_trajectory* tng_data, int64_t block_id, int64_t frame, int64_t* stride_length)

    tng_function_status tng_util_trajectory_next_frame_present_data_blocks_find(tng_trajectory *tng_data, int64_t current_frame, int64_t n_requested_data_block_ids, int64_t *requested_data_block_ids, int64_t *next_frame, int64_t *n_data_blocks_in_next_frame, int64_t **data_block_ids_in_next_frame)

    tng_function_status  tng_data_block_name_get(tng_trajectory* tng_data, const int64_t block_id, char*    name, const int   max_len)

    tng_function_status tng_data_block_dependency_get( tng_trajectory *tng_data, const int64_t block_id, int *block_dependency)

    tng_function_status  tng_util_particle_data_next_frame_read( tng_trajectory* tng_data, const int64_t block_id, void**  values, char* data_type, int64_t* retrieved_frame_number, double* retrieved_time)

    tng_function_status  tng_util_non_particle_data_next_frame_read( tng_trajectory* tng_data, const int64_t block_id, void** values, char*  data_type, int64_t* retrieved_frame_number, double* retrieved_time)

    tng_function_status tng_data_block_num_values_per_frame_get(tng_trajectory* tng_data,  int64_t block_id, int64_t *n_values_per_frame)

    tng_function_status  tng_util_frame_current_compression_get(tng_trajectory* tng_data, int64_t block_id, int64_t *codec_id, double *factor)

TNGFrame = namedtuple("TNGFrame", "positions velocities forces time step box ")


cdef class MemoryWrapper:
    # holds a pointer to C allocated memory, deals with malloc&free based on:
    # https://gist.github.com/GaelVaroquaux/
    # 1249305/ac4f4190c26110fe2791a1e7a6bed9c733b3413f
    cdef void * ptr 

    def __cinit__(MemoryWrapper self, int size):
        # malloc not PyMem_Malloc as gmx later does realloc
        self.ptr = malloc(size)
        if self.ptr is NULL:
            raise MemoryError

    def __dealloc__(MemoryWrapper self):
        if self.ptr != NULL:
            free(self.ptr)


cdef class TrajectoryWrapper:
    """A wrapper class for a tng_trajectory"""
    cdef tng_trajectory *_ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            free(self._ptr)
            self._ptr = NULL

    # Extension class properties
    # @property
    # def a(self):
    #     return self._ptr.a if self._ptr is not NULL else None

    # @property
    # def b(self):
    #     return self._ptr.b if self._ptr is not NULL else None

    @staticmethod
    cdef TrajectoryWrapper from_ptr(tng_trajectory *_ptr, bint owner=False):
        """Factory function to create WrapperClass objects from
        given tng_trajectory pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Call to __new__ bypasses __init__ constructor
        cdef TrajectoryWrapper wrapper = TrajectoryWrapper.__new__(TrajectoryWrapper)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef TrajectoryWrapper new_struct():
        """Factory function to create WrapperClass objects with
        newly allocated my_c_struct"""
        cdef tng_trajectory *_ptr = <tng_trajectory *>malloc(sizeof(tng_trajectory))
        if _ptr is NULL:
            raise MemoryError
        # _ptr.a = 0
        # _ptr.b = 0
        return TrajectoryWrapper.from_ptr(_ptr, owner=True)

cdef class TNGFileIterator:
    """File handle object for TNG files

    Supports use as a context manager ("with" blocks).
    
    """


    cdef tng_trajectory* _traj_p   #TODO should we make this a TrajectoryWrapper also
    cdef TrajectoryWrapper _traj
    cdef readonly fname
    cdef str mode
    cdef int is_open
    cdef int reached_eof
    cdef int64_t step

    cdef int64_t _n_frames
    cdef int64_t _n_particles
    cdef int64_t _n_frame_sets
    cdef float _distance_scale

    cdef int64_t _current_frame
    cdef int64_t _current_frame_set

    def __cinit__(self, fname, mode='r'):

        self._traj = TrajectoryWrapper.from_ptr(self._traj_p, owner=True)
        self.fname = fname
        self._n_frames = -1
        self._n_particles = -1
        self._n_frame_sets = -1
        self._distance_scale = 0.0
        self._current_frame = -1
        self._current_frame_set = -1

        self._open(self.fname, mode)

    def __dealloc__(self):
        self._close()

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
        stat = tng_util_trajectory_open(fname_bytes, _mode, & self._traj._ptr)
        if stat != TNG_SUCCESS:
            raise IOError("File '{}' cannot be opened".format(fname))

        # TODO propagate errmsg upwards in all the below calls
        stat = tng_num_frames_get(self._traj._ptr, & self._n_frames)
        if stat != TNG_SUCCESS:
            raise IOError("Number of frames cannot be read")

        stat = tng_num_particles_get(self._traj._ptr, & self._n_particles)
        if stat != TNG_SUCCESS:
            raise IOError("Number of particles cannot be read")

        # NOTE can we just loop over this directly in some way?
        stat = tng_num_frame_sets_get(self._traj._ptr, & self._n_frame_sets)
        # TODO can this be read straight from the struct as self._traj->n_frame_sets?
        # they note that this is not always updated

        cdef int64_t exponent
        stat = tng_distance_unit_exponential_get(self._traj._ptr, & exponent)
        if stat != TNG_SUCCESS:
            raise IOError("Distance exponent cannot be read")

        self._distance_scale = 10.0**(exponent+9)
        self.is_open = True
        self.step = 0
        self.reached_eof = False
    

    def _close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            tng_util_trajectory_close(& self._traj._ptr)
            self.is_open = False
            self._n_frames = -1

    def read_all_frames(self):
        self._spool()
   
    cdef _spool(self):
        # outer decl
        cdef int64_t step, n_blocks
        cdef int64_t nframe = 0
        cdef int64_t *block_ids = NULL
        cdef tng_function_status stat  = tng_util_trajectory_next_frame_present_data_blocks_find(self._traj._ptr, -1, 0, NULL, &step, &n_blocks, &block_ids)
        if stat !=TNG_SUCCESS:
            raise Exception("cannot find the number of blocks")

        cdef int64_t block_counter = 0
        cdef tng_function_status read_stat = TNG_SUCCESS
        cdef block = TNGDataBlock(self._traj, debug=True)

        while (read_stat == TNG_SUCCESS):
            for i in range(n_blocks):
                block_counter += 1
                block.block_read(block_ids[i])
                printf("block_count %ld \n", block_counter)

            nframe +=1
            read_stat = tng_util_trajectory_next_frame_present_data_blocks_find(self._traj._ptr, step, 0, NULL, &step, &n_blocks, &block_ids);
            printf("loop status %d \n", read_stat)
            printf("nframe  %ld \n\n", nframe)  

    cdef _block_interpret(self):
        pass
        #logic to interpret the block types heretng_data_get_stride_length

    cdef _block_numpy_cast(self):
        pass
        #logic to cast data blocks to numpy arrays here.
    
    cdef seek(self):
        pass
        #logic to move the file handle pointers here
        # must be done at both python and C level ?

    


cdef class TNGDataBlock:

    cdef tng_trajectory* _traj
    cdef int64_t block_id 
    cdef bint debug 

    cdef int64_t step
    cdef double frame_time
    cdef double precision
    cdef int64_t n_values_per_frame, n_atoms
    cdef char* block_name
    cdef double* _values
    cdef MemoryWrapper _wrapper
    cdef tng_function_status read_stat
    cdef np.ndarray values


    def __cinit__(self, TrajectoryWrapper traj, bint debug=False):

        self._traj = traj._ptr
        self.debug = debug
        self.block_id = -1

        self.step = -1
        self.frame_time = -1
        self.precision = -1
        self.n_values_per_frame = -1
        self.n_atoms = -1 
        self.block_name = <char*> malloc(TNG_MAX_STR_LEN * sizeof(char))
        self._values = NULL

  
    def __dealloc__(self):
        self._close()
    
    def block_read(self, id): #NOTE does this have to be python
        self._block_read(id)
        self._block_2d_numpy_cast(self.n_values_per_frame, self.n_atoms)
    
    cdef _close(self):
        if self._values != NULL:
            free(self._values)
        free(self.block_name)

    cdef void _block_2d_numpy_cast(self, int64_t n_values_per_frame, int64_t n_atoms):
        if self.debug:
            printf("CREATING NUMPY_ARRAY \n")
        if n_values_per_frame == -1 or n_atoms == -1:
            raise ValueError("array dimensions are not correct")
        cdef int nd = 2
        cdef int err
        cdef npy_intp dims[2]
        dims[0] = n_values_per_frame
        dims[1] = n_atoms
        # self.values = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, self._wrapper.ptr)
        # Py_INCREF(self._wrapper)
        # err = PyArray_SetBaseObject(self.values, self._wrapper)
        # if err:
        #     raise ValueError("failed to create value array")

    cdef void _block_read(self, int64_t id):
        self.block_id = id
        read_stat = self.get_data_next_frame(self.block_id, &self._values, &self.step, &self.frame_time, &self.n_values_per_frame, &self.n_atoms, &self.precision, self.block_name, self.debug)

        if self.debug:
            printf("block id %ld \n", self.block_id)
            printf("data block name %s \n", self.block_name)
            printf("n_values_per_frame %ld \n", self.n_values_per_frame)
            printf("n_atoms  %ld \n", self.n_atoms)
            for j in range(self.n_values_per_frame * self.n_atoms):
                printf(" %f ", self._values[j])

    cdef tng_function_status get_data_next_frame(self, int64_t block_id, double** values, int64_t* step, double* frame_time, int64_t* n_values_per_frame, int64_t* n_atoms, double* prec, char* block_name, bint debug):
    
        cdef tng_function_status stat
        cdef char                datatype = -1
        cdef int64_t             codec_id;
        cdef int                 block_dependency
        cdef void*               data = NULL
        cdef double              local_prec

        #Flag to indicate frame dependent data. */
        cdef int TNG_FRAME_DEPENDENT = 1
        #Flag to indicate particle dependent data. */
        cdef int  TNG_PARTICLE_DEPENDENT = 2

        stat = tng_data_block_name_get(self._traj, block_id, block_name, TNG_MAX_STR_LEN)
        if stat != TNG_SUCCESS:
            raise Exception("cannot get block_name")

        stat = tng_data_block_dependency_get(self._traj, block_id, &block_dependency)
        if stat != TNG_SUCCESS:
            raise Exception("cannot get block_dependency")
        
        if block_dependency.__and__(TNG_PARTICLE_DEPENDENT): # bitwise & due to enum defs
            if debug:
                printf("reading particle data \n")
            tng_num_particles_get(self._traj, n_atoms)
            stat = tng_util_particle_data_next_frame_read(self._traj, block_id, &data, &datatype, step, frame_time)
        else:
            if debug:
                printf("reading NON particle data \n")
            n_atoms[0] = 1 # still used for some allocs
            stat = tng_util_non_particle_data_next_frame_read(self._traj, block_id, &data, &datatype, step, frame_time)

        if stat == TNG_CRITICAL:
            #raise Exception("critical data reading failure")
            return TNG_CRITICAL

        stat = tng_data_block_num_values_per_frame_get(self._traj, block_id, n_values_per_frame)
        if stat == TNG_CRITICAL:
            #raise Exception("critical data reading failure")
            return TNG_CRITICAL

        values[0] = <double*> realloc(values[0], sizeof(double)* n_values_per_frame[0] * n_atoms[0]) # renew to be right size
        
        if self.debug:
            printf("realloc values array to be %ld  doubles and %ld bits long \n", n_values_per_frame[0] * n_atoms[0], n_values_per_frame[0] * n_atoms[0]*sizeof(double))

        self.convert_to_double_arr(data, values[0], n_atoms[0], n_values_per_frame[0], datatype, debug)

        tng_util_frame_current_compression_get(self._traj, block_id, &codec_id, &local_prec)

        if codec_id != TNG_TNG_COMPRESSION:

            prec[0] = -1.0
        else:
            prec[0] = local_prec
        
        # free local data
        free(data)
        return TNG_SUCCESS
    
    cdef void convert_to_double_arr(self, void* source, double* to, const int n_atoms, const int n_vals, const char datatype, bint debug):
        # do we need to account for changes in the decl of double etc ie is this likely to be portable?.
        # a lot of this is a bit redundant but could be used to differntiate casts to numpy arrays etc in the future
        cdef int i, j

        if datatype == TNG_FLOAT_DATA:
            for i in range(n_atoms):
                for j in range(n_vals):
                    to[i*n_vals +j ] = (<float*>source)[i *n_vals +j] #NOTE do we explicitly need to use reinterpret_cast ??
            #memcpy(to,  source, n_vals * sizeof(float) * n_atoms)

        elif datatype == TNG_INT_DATA:
            for i in range(n_atoms):
                for j in range(n_vals):
                    to[i*n_vals +j ] = (<int64_t*>source)[i *n_vals +j] # redundant but could be changed later
            #memcpy(to, source, n_vals * sizeof(int64_t) * n_atoms)

        elif datatype == TNG_DOUBLE_DATA:
            for i in range(n_atoms):
                for j in range(n_vals):
                    to[i*n_vals +j ] = (<double*>source)[i *n_vals +j] # should probs use memcpy
            #memcpy(to, source, n_vals * sizeof(double) * n_atoms)

        elif datatype == TNG_CHAR_DATA:
            raise NotImplementedError("char data reading is not implemented")

        else: # the default is meant to be double
            printf(" WARNING type %d not understood \n", datatype) #TODO currently non particle block data isnt working



cdef class TNGFile:
    """File handle object for TNG files

    Supports use as a context manager ("with" blocks).
    """
    cdef tng_trajectory* _traj
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

        # handle file not existing at python level, C level is nasty and causes
        # crash
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
                raise IOError("""An error ocurred reading distance unit
                 exponent. {}""".format(
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

        ok = tng_util_pos_read_range(self._traj, 0, 0, & pos_ptr,
                                     & stride_length)
        if (ok == TNG_SUCCESS) and pos_ptr:
            self._pos = 1  # true
            self._pos_stride = stride_length

        ok = tng_util_box_shape_read_range(self._traj, 0, 0, & box_ptr,
                                           & stride_length)
        if (ok == TNG_SUCCESS) and box_ptr:
            self._box = 1  # true
            self._box_stride = stride_length

        ok = tng_util_vel_read_range(self._traj, 0, 0, & vel_ptr,
                                     & stride_length)
        if (ok == TNG_SUCCESS) and vel_ptr:
            self._vel = 1  # true
            self._vel_stride = stride_length

        ok = tng_util_force_read_range(self._traj, 0, 0, & frc_ptr,
                                       & stride_length)
        if (ok == TNG_SUCCESS) and frc_ptr:
            self._frc = 1  # true
            self._frc_stride = stride_length

    def close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            tng_util_trajectory_close(& self._traj)
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
                # TODO this will break when using frames spaced more than 1
                # apart
                ok = tng_util_pos_read_range(self._traj, self.step, self.step,
                                             & positions, & stride_length)
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
        cdef np.ndarray[ndim= 2, dtype = np.float32_t, mode = 'c'] box = \
            np.empty((3, 3), dtype=np.float32)

        if self._box:
            if (self.step % self._box_stride == 0):
                # BOX SHAPE cdef float* box_s
                wrap_box = MemoryWrapper(3 * 3 * sizeof(float))
                box_shape = <float*> wrap_box.ptr
                # TODO this will break when using frames spaced more than 1
                # apart
                ok = tng_util_box_shape_read_range(self._traj, self.step,
                                                   self.step, & box_shape,
                                                   & stride_length)
                if ok != TNG_SUCCESS:
                    raise IOError("error reading box shape")
                # populate box, can this be done the same way as positions
                # above? #TODO is there a canonical way to convert to numpy
                # array
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
                ok = tng_util_vel_read_range(self._traj, self.step, self.step,
                                             & velocities, & stride_length)
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
                ok = tng_util_force_read_range(self._traj, self.step,
                                               self.step, & forces,
                                               & stride_length)
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
