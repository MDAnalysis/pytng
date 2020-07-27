# cython: linetrace=True
# cython: embedsignature=True
# distutils: define_macros=CYTHON_TRACE=1
from numpy cimport(PyArray_SimpleNewFromData,
                   PyArray_SetBaseObject,
                   NPY_FLOAT,
                   Py_INCREF,
                   npy_intp,
                   )

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.stdint cimport int64_t, uint64_t, int32_t, uint32_t
from libc.stdlib cimport malloc, free, realloc
from libc.stdio cimport printf, FILE, SEEK_SET, SEEK_CUR, SEEK_END

from posix.types cimport off_t


from collections import namedtuple
import os
import numbers
import numpy as np

cimport numpy as np
np.import_array()

from  cython.operator cimport dereference


ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL
ctypedef enum tng_data_type: TNG_CHAR_DATA, TNG_INT_DATA, TNG_FLOAT_DATA, \
    TNG_DOUBLE_DATA
ctypedef enum tng_hash_mode: TNG_SKIP_HASH, TNG_USE_HASH


status_error_message = ['OK', 'Failure', 'Critical']


cdef extern from "string.h":
    size_t strlen(char *s)

cdef extern from "<stdio.h>" nogil:
    # Seek and tell with off_t
    int fseeko(FILE *, off_t, int)
    off_t ftello(FILE *)


cdef extern from "tng/tng_io.h":

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


TNGFrame = namedtuple("TNGFrame", "positions velocities forces time step box ")


cdef class MemoryWrapper:
    # holds a pointer to C allocated memory, deals with malloc&free based on:
    # https://gist.github.com/GaelVaroquaux/
    # 1249305/ac4f4190c26110fe2791a1e7a6bed9c733b3413f
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
    cdef tng_trajectory* _traj
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
        stat = tng_util_trajectory_open(fname_bytes, _mode, & self._traj)
        if stat != TNG_SUCCESS:
            raise IOError("File '{}' cannot be opened".format(fname))

        # TODO propagate errmsg upwards in all the below calls
        stat = tng_num_frames_get(self._traj, & self._n_frames)
        if stat != TNG_SUCCESS:
            raise IOError("Number of frames cannot be read")

        stat = tng_num_particles_get(self._traj, & self._n_particles)
        if stat != TNG_SUCCESS:
            raise IOError("Number of particles cannot be read")

        # NOTE can we just loop over this directly in some way?
        stat = tng_num_frame_sets_get(self._traj, & self._n_frame_sets)
        # TODO can this be read straight from the struct as self._traj->n_frame_sets?
        # they note that this is not always updated

        cdef int64_t exponent
        stat = tng_distance_unit_exponential_get(self._traj, & exponent)
        if stat != TNG_SUCCESS:
            raise IOError("Distance exponent cannot be read")

        self._distance_scale = 10.0**(exponent+9)

        self.is_open = True
        self.step = 0

        self.reached_eof = False
    

    def _close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            tng_util_trajectory_close(& self._traj)
            self.is_open = False
            self._n_frames = -1

    def spool(self): # this reads sucessfully to the end of the file
        cdef tng_function_status stat = TNG_SUCCESS
        cdef int64_t block_count = 0
        printf("total file_length %ld \n", self._traj.input_file_len)
        #fseeko(self._traj.input_file, 0, SEEK_SET) # after init is called, the file SEEK is at the start of the first frame set, so we need to reset to the start block
        #comment this ^ back in to read from start
        cdef off_t offset
        while stat != TNG_CRITICAL: #TODO replace with a file len check
            print("block read called {} \n".format(block_count))
            printf("NEW BLOCK \n")
            offset = ftello(self._traj.input_file)
            printf("file position %ld \n",offset)
            stat = self._read_next_block()
            block_count += 1

    # NOTE this looks simple and may work ? may be too low level as it all hinges on whether the file positions are kept up to date
    # can we just fseek to the first TFS?
    cdef tng_function_status _read_next_block(self):
        cdef tng_function_status stat
        cdef tng_gen_block* block
        cdef int64_t block_id = -1
        stat = tng_block_init(& block)
        if stat != TNG_SUCCESS:
            return TNG_CRITICAL
        
        stat = tng_block_header_read(self._traj, block)
        if stat != TNG_SUCCESS:
            tng_block_destroy(&block)
            return TNG_CRITICAL
            
        stat = tng_block_read_next(self._traj, block, TNG_SKIP_HASH)
        if stat != TNG_SUCCESS:
            tng_block_destroy(& block)
            return TNG_CRITICAL
        
        block_id = block.id
        printf("block id %ld \n", block_id)
        cdef name_len = strlen(block.name) +1
        cdef bname = <char*> malloc(name_len * sizeof(char)) # TNG_MAX_STR_LEN = 1024
        bname = block.name
        printf("block name %s \n",block.name)

        #cdef int64_t stride = self.get_traj_strides(block_id)
        #printf("\n %ld \n",stride)

        return TNG_SUCCESS

    cdef get_block_types_of_next_frame(self):
        cdef int64_t step, nBlocks
        cdef int64_t *block_ids = NULL
        stat = tng_util_trajectory_next_frame_present_data_blocks_find(self._traj, -1, 0, NULL, &step, &nBlocks, &block_ids);
        printf("step = %d \n", step)
        printf("nblocks = %d \n", nBlocks)
    
    cdef spool2(self):
        # outer decl
        cdef int64_t step, nBlocks
        cdef int64_t *block_ids = NULL
        cdef tng_function_status stat  = tng_util_trajectory_next_frame_present_data_blocks_find(self._traj, -1, 0, NULL, &step, &nBlocks, &block_ids);
        
        #inner loop decls
        cdef double frame_time
        cdef double precision
        cdef int64_t n_values_per_frame, n_atoms
        #char block_name[1024]

        while  stat == TNG_SUCCESS:
            for i in range(nBlocks):
                stat = tng_util_trajectory_next_frame_present_data_blocks_find(self._traj, -1, 0, NULL, &step, &nBlocks, &block_ids);


    # cdef get_data_next_frame()



    cdef get_traj_strides(self, block_id): # TODO BROKEN this hangs and looks like it reads the same block over and over again forever
        cdef int64_t stride_length
        tng_data_get_stride_length(self._traj, block_id, 1, &stride_length)
        return stride_length




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

    


    # #SKELETON to read whole file ?
    # for i in range self._n_frame_sets:
    #     tng_frame_set_read() ? tng_frame_set_read_current_only_data_from_block_id()?


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
