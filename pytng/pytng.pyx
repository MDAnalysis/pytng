# cython: linetrace=True
# cython: embedsignature=True
# cython: profile=True
# cython: binding=True
# distutils: define_macros=[CYTHON_TRACE=1, CYTHON_TRACE_NOGIL=1]

import numpy as np
import numbers
import os
from libc.stdio cimport printf, FILE, SEEK_SET, SEEK_CUR, SEEK_END
from libc.stdlib cimport malloc, free, realloc
from libc.stdint cimport int64_t, uint64_t, int32_t, uint32_t
cimport cython
cimport numpy as np
np.import_array()

# Marks operation success
ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL
# Marks use of hash checking in file read
ctypedef enum tng_hash_mode: TNG_SKIP_HASH, TNG_USE_HASH
# Datatypes that can be read off disk
ctypedef enum tng_datatypes: TNG_CHAR_DATA, TNG_INT_DATA, TNG_FLOAT_DATA, \
    TNG_DOUBLE_DATA
# Marks particle or non-particle data
ctypedef enum tng_particle_dependency: TNG_NON_PARTICLE_BLOCK_DATA, \
    TNG_PARTICLE_BLOCK_DATA
# Indicates compression type
ctypedef enum tng_compression: TNG_UNCOMPRESSED, TNG_XTC_COMPRESSION, \
    TNG_TNG_COMPRESSION, TNG_GZIP_COMPRESSION
# TNG alias for T/F
ctypedef enum tng_bool: TNG_FALSE, TNG_TRUE
# Indicates variable number of atoms
ctypedef enum tng_variable_n_atoms_flag:   TNG_CONSTANT_N_ATOMS, \
    TNG_VARIABLE_N_ATOMS

# Flag to indicate frame dependent data.
DEF TNG_FRAME_DEPENDENT = 1
# Flag to indicate particle dependent data.
DEF TNG_PARTICLE_DEPENDENT = 2


# GROUP 1 Standard non-trajectory blocks
# Block IDs of standard non-trajectory blocks.

DEF TNG_GENERAL_INFO = 0x0000000000000000LL
DEF TNG_MOLECULES = 0x0000000000000001LL
DEF TNG_TRAJECTORY_FRAME_SET = 0x0000000000000002LL
DEF TNG_PARTICLE_MAPPING = 0x0000000000000003LL

# GROUP 2 Standard trajectory blocks
# Block IDs of standard trajectory blocks. Box shape and partial charges can
# be either trajectory blocks or non-trajectory blocks

DEF TNG_TRAJ_BOX_SHAPE = 0x0000000010000000LL
DEF TNG_TRAJ_POSITIONS = 0x0000000010000001LL
DEF TNG_TRAJ_VELOCITIES = 0x0000000010000002LL
DEF TNG_TRAJ_FORCES = 0x0000000010000003LL
DEF TNG_TRAJ_PARTIAL_CHARGES = 0x0000000010000004LL
DEF TNG_TRAJ_FORMAL_CHARGES = 0x0000000010000005LL
DEF TNG_TRAJ_B_FACTORS = 0x0000000010000006LL
DEF TNG_TRAJ_ANISOTROPIC_B_FACTORS = 0x0000000010000007LL
DEF TNG_TRAJ_OCCUPANCY = 0x0000000010000008LL
DEF TNG_TRAJ_GENERAL_COMMENTS = 0x0000000010000009LL
DEF TNG_TRAJ_MASSES = 0x0000000010000010LL

# GROUP 3 GROMACS data block IDs
# Block IDs of data blocks specific to GROMACS.

DEF TNG_GMX_LAMBDA = 0x1000000010000000LL
DEF TNG_GMX_ENERGY_ANGLE = 0x1000000010000001LL
DEF TNG_GMX_ENERGY_RYCKAERT_BELL = 0x1000000010000002LL
DEF TNG_GMX_ENERGY_LJ_14 = 0x1000000010000003LL
DEF TNG_GMX_ENERGY_COULOMB_14 = 0x1000000010000004LL
# NOTE changed from  TNG_GMX_ENERGY_LJ_(SR)
DEF TNG_GMX_ENERGY_LJ_SR = 0x1000000010000005LL
# NOTE changed from  TNG_GMX_ENERGY_COULOMB_(SR)
DEF TNG_GMX_ENERGY_COULOMB_SR = 0x1000000010000006LL
DEF TNG_GMX_ENERGY_COUL_RECIP = 0x1000000010000007LL
DEF TNG_GMX_ENERGY_POTENTIAL = 0x1000000010000008LL
DEF TNG_GMX_ENERGY_KINETIC_EN = 0x1000000010000009LL
DEF TNG_GMX_ENERGY_TOTAL_ENERGY = 0x1000000010000010LL
DEF TNG_GMX_ENERGY_TEMPERATURE = 0x1000000010000011LL
DEF TNG_GMX_ENERGY_PRESSURE = 0x1000000010000012LL
DEF TNG_GMX_ENERGY_CONSTR_RMSD = 0x1000000010000013LL
DEF TNG_GMX_ENERGY_CONSTR2_RMSD = 0x1000000010000014LL
DEF TNG_GMX_ENERGY_BOX_X = 0x1000000010000015LL
DEF TNG_GMX_ENERGY_BOX_Y = 0x1000000010000016LL
DEF TNG_GMX_ENERGY_BOX_Z = 0x1000000010000017LL
DEF TNG_GMX_ENERGY_BOXXX = 0x1000000010000018LL
DEF TNG_GMX_ENERGY_BOXYY = 0x1000000010000019LL
DEF TNG_GMX_ENERGY_BOXZZ = 0x1000000010000020LL
DEF TNG_GMX_ENERGY_BOXYX = 0x1000000010000021LL
DEF TNG_GMX_ENERGY_BOXZX = 0x1000000010000022LL
DEF TNG_GMX_ENERGY_BOXZY = 0x1000000010000023LL
DEF TNG_GMX_ENERGY_BOXVELXX = 0x1000000010000024LL
DEF TNG_GMX_ENERGY_BOXVELYY = 0x1000000010000025LL
DEF TNG_GMX_ENERGY_BOXVELZZ = 0x1000000010000026LL
DEF TNG_GMX_ENERGY_BOXVELYX = 0x1000000010000027LL
DEF TNG_GMX_ENERGY_BOXVELZX = 0x1000000010000028LL
DEF TNG_GMX_ENERGY_BOXVELZY = 0x1000000010000029LL
DEF TNG_GMX_ENERGY_VOLUME = 0x1000000010000030LL
DEF TNG_GMX_ENERGY_DENSITY = 0x1000000010000031LL
DEF TNG_GMX_ENERGY_PV = 0x1000000010000032LL
DEF TNG_GMX_ENERGY_ENTHALPY = 0x1000000010000033LL
DEF TNG_GMX_ENERGY_VIR_XX = 0x1000000010000034LL
DEF TNG_GMX_ENERGY_VIR_XY = 0x1000000010000035LL
DEF TNG_GMX_ENERGY_VIR_XZ = 0x1000000010000036LL
DEF TNG_GMX_ENERGY_VIR_YX = 0x1000000010000037LL
DEF TNG_GMX_ENERGY_VIR_YY = 0x1000000010000038LL
DEF TNG_GMX_ENERGY_VIR_YZ = 0x1000000010000039LL
DEF TNG_GMX_ENERGY_VIR_ZX = 0x1000000010000040LL
DEF TNG_GMX_ENERGY_VIR_ZY = 0x1000000010000041LL
DEF TNG_GMX_ENERGY_VIR_ZZ = 0x1000000010000042LL
DEF TNG_GMX_ENERGY_SHAKEVIR_XX = 0x1000000010000043LL
DEF TNG_GMX_ENERGY_SHAKEVIR_XY = 0x1000000010000044LL
DEF TNG_GMX_ENERGY_SHAKEVIR_XZ = 0x1000000010000045LL
DEF TNG_GMX_ENERGY_SHAKEVIR_YX = 0x1000000010000046LL
DEF TNG_GMX_ENERGY_SHAKEVIR_YY = 0x1000000010000047LL
DEF TNG_GMX_ENERGY_SHAKEVIR_YZ = 0x1000000010000048LL
DEF TNG_GMX_ENERGY_SHAKEVIR_ZX = 0x1000000010000049LL
DEF TNG_GMX_ENERGY_SHAKEVIR_ZY = 0x1000000010000050LL
DEF TNG_GMX_ENERGY_SHAKEVIR_ZZ = 0x1000000010000051LL
DEF TNG_GMX_ENERGY_FORCEVIR_XX = 0x1000000010000052LL
DEF TNG_GMX_ENERGY_FORCEVIR_XY = 0x1000000010000053LL
DEF TNG_GMX_ENERGY_FORCEVIR_XZ = 0x1000000010000054LL
DEF TNG_GMX_ENERGY_FORCEVIR_YX = 0x1000000010000055LL
DEF TNG_GMX_ENERGY_FORCEVIR_YY = 0x1000000010000056LL
DEF TNG_GMX_ENERGY_FORCEVIR_YZ = 0x1000000010000057LL
DEF TNG_GMX_ENERGY_FORCEVIR_ZX = 0x1000000010000058LL
DEF TNG_GMX_ENERGY_FORCEVIR_ZY = 0x1000000010000059LL
DEF TNG_GMX_ENERGY_FORCEVIR_ZZ = 0x1000000010000060LL
DEF TNG_GMX_ENERGY_PRES_XX = 0x1000000010000061LL
DEF TNG_GMX_ENERGY_PRES_XY = 0x1000000010000062LL
DEF TNG_GMX_ENERGY_PRES_XZ = 0x1000000010000063LL
DEF TNG_GMX_ENERGY_PRES_YX = 0x1000000010000064LL
DEF TNG_GMX_ENERGY_PRES_YY = 0x1000000010000065LL
DEF TNG_GMX_ENERGY_PRES_YZ = 0x1000000010000066LL
DEF TNG_GMX_ENERGY_PRES_ZX = 0x1000000010000067LL
DEF TNG_GMX_ENERGY_PRES_ZY = 0x1000000010000068LL
DEF TNG_GMX_ENERGY_PRES_ZZ = 0x1000000010000069LL
DEF TNG_GMX_ENERGY_SURFXSURFTEN = 0x1000000010000070LL
DEF TNG_GMX_ENERGY_MUX = 0x1000000010000071LL
DEF TNG_GMX_ENERGY_MUY = 0x1000000010000072LL
DEF TNG_GMX_ENERGY_MUZ = 0x1000000010000073LL
DEF TNG_GMX_ENERGY_VCOS = 0x1000000010000074LL
DEF TNG_GMX_ENERGY_VISC = 0x1000000010000075LL
DEF TNG_GMX_ENERGY_BAROSTAT = 0x1000000010000076LL
DEF TNG_GMX_ENERGY_T_SYSTEM = 0x1000000010000077LL
DEF TNG_GMX_ENERGY_LAMB_SYSTEM = 0x1000000010000078LL
DEF TNG_GMX_SELECTION_GROUP_NAMES = 0x1000000010000079LL
DEF TNG_GMX_ATOM_SELECTION_GROUP = 0x1000000010000080LL


cdef extern from "tng/tng_io.h":

    cdef enum:
        TNG_MAX_STR_LEN
        TNG_MD5_HASH_LEN

    # note that the _t suffix is a typedef mangle for a pointer to the struct
    ctypedef struct tng_molecule_t:
        pass

    struct tng_particle_mapping:

        # The index number of the first particle in this mapping block
        int64_t num_first_particle
        # The number of particles list in this mapping block
        int64_t n_particles
        # the mapping of index numbers to the real particle numbers in the
        # trajectory. real_particle_numbers[0] is the real particle number
        # (as it is numbered in the molecular system) of the first particle
        # in the data blocks covered by this particle mapping block
        int64_t * real_particle_numbers

    struct tng_trajectory_frame_set:

        # The number of different particle mapping blocks present.
        int64_t n_mapping_blocks
        # The atom mappings of this frame set
        tng_particle_mapping * mappings
        # The first frame of this frame set
        int64_t first_frame
        # The number of frames in this frame set
        int64_t n_frames
        # The number of written frames in this frame set (used when writing
        # *one frame at a time).
        int64_t n_written_frames
        # The number of frames not yet written to file in this frame set
        # (used from the utility functions to finish the writing properly.
        int64_t n_unwritten_frames

        # A list of the number of each molecule type - only used when using
        #  variable number of atoms
        int64_t * molecule_cnt_list
        #  The number of particles/atoms - only used when using variable
        #  number of atoms
        int64_t n_particles
        # The file position of the next frame set
        int64_t next_frame_set_file_pos
        # The file position of the previous frame set
        int64_t prev_frame_set_file_pos
        # The file position of the frame set one long stride step ahead
        int64_t medium_stride_next_frame_set_file_pos
        # The file position of the frame set one long stride step behind
        int64_t medium_stride_prev_frame_set_file_pos
        # The file position of the frame set one long stride step ahead
        int64_t long_stride_next_frame_set_file_pos
        # The file position of the frame set one long stride step behind
        int64_t long_stride_prev_frame_set_file_pos
        #  Time stamp (in seconds) of first frame in frame set
        double first_frame_time

        # The data blocks in a frame set are trajectory data blocks
        # The number of trajectory data blocks of particle dependent data
        int n_particle_data_blocks
        # A list of data blocks containing particle dependent data
        tng_data * tr_particle_data
        #  The number of trajectory data blocks independent of particles
        int n_data_blocks
        # A list of data blocks containing particle indepdendent data
        tng_data * tr_data

    struct tng_data:
        # The block ID of the data block containing this particle data.
        # This is used to determine the kind of data that is stored
        int64_t block_id
        # The name of the data block. This is used to determine the kind of
        # data that is stored
        char * block_name
        # The type of data stored.
        char datatype
        # A flag to indicate if this data block contains frame
        # and/or particle dependent data

        char dependency
        # The frame number of the first data value
        int64_t first_frame_with_data
        # The number of frames in this frame set
        int64_t n_frames
        # The number of values stored per frame
        int64_t n_values_per_frame
        # The number of frames between each data point - e.g. when
        # storing sparse data.
        int64_t stride_length
        # ID of the CODEC used for compression 0 == no compression.
        int64_t codec_id
        # If reading one frame at a time this is the last read frame
        int64_t last_retrieved_frame
        # The multiplier used for getting integer values for compression
        double compression_multiplier
        # A 1-dimensional array of values of length
        # [sizeof (datatype)] * n_frames * n_particles*n_values_per_frame
        void * values
        # If storing character data store it in a 3-dimensional array
        char**** strings

    struct tng_trajectory:
        # The path of the input trajectory file
        char * input_file_path
        # A handle to the input file
        FILE * input_file
        # The length of the input file
        int64_t input_file_len
        # The path of the output trajectory file
        char * output_file_path
        # A handle to the output file
        FILE * output_file
        # Function to swap 32 bit values to and from the endianness of the
        # * input file
        tng_function_status(*input_endianness_swap_func_32)(const tng_trajectory*, uint32_t*)
        # Function to swap 64 bit values to and from the endianness of the
        # input file
        tng_function_status(*input_endianness_swap_func_64)(const  tng_trajectory*, uint64_t*)
        # Function to swap 32 bit values to and from the endianness of the
        # input file
        tng_function_status(*output_endianness_swap_func_32)(const  tng_trajectory*, uint32_t*)
        # Function to swap 64 bit values to and from the endianness of the
        # input file
        tng_function_status(*output_endianness_swap_func_64)(const  tng_trajectory*, uint64_t*)
        # The endianness of 32 bit values of the current computer
        char endianness_32
        # The endianness of 64 bit values of the current computer
        char endianness_64

        # The name of the program producing this trajectory
        char * first_program_name
        # The forcefield used in the simulations
        char * forcefield_name
        # The name of the user running the simulations
        char * first_user_name
        # The name of the computer on which the simulations were performed
        char * first_computer_name
        # The PGP signature of the user creating the file.
        char * first_pgp_signature
        # The name of the program used when making last modifications
        # to the file
        char * last_program_name
        # The name of the user making the last modifications to the file
        char * last_user_name
        # The name of the computer on which the last modifications were made
        char * last_computer_name
        # The PGP signature of the user making the last modifications to the
        # file.
        char * last_pgp_signature
        # The time (n seconds since 1970) when the file was created
        int64_t time
        # The exponential of the value of the distance unit used. The default
        # distance unit is nm (1e-9), i.e. distance_unit_exponential = -9. If
        # the measurements are in Å the distance_unit_exponential = -10.
        int64_t distance_unit_exponential

        # A flag indicating if the number of atoms can vary throughout the
        # simulation, e.g. using a grand canonical ensemble.
        char var_num_atoms_flag
        # The number of frames in a frame set. It is allowed to have frame sets
        # with fewer frames, but this will help searching for specific frames
        int64_t frame_set_n_frames
        # The number of frame sets in a medium stride step
        int64_t medium_stride_length
        # The number of frame sets in a long stride step
        int64_t long_stride_length
        # The current (can change from one frame set to another) time length
        # (in seconds) of one frame.
        double time_per_frame

        # The number of different kinds of molecules in the trajectory
        int64_t n_molecules
        # A list of molecules in the trajectory
        tng_molecule_t molecules
        # A list of the count of each molecule - if using variable number of
        # particles this will be specified in each frame set
        int64_t * molecule_cnt_list
        # The total number of particles/atoms. If using variable number of
        # particles this will be specified in each frame set
        int64_t n_particles

        # The pos in the src file of the first frame set
        int64_t first_trajectory_frame_set_input_file_pos
        # The pos in the dest file of the first frame set
        int64_t first_trajectory_frame_set_output_file_pos
        # The pos in the src file of the last frame set
        int64_t last_trajectory_frame_set_input_file_pos
        # The pos in the dest file of the last frame set
        int64_t last_trajectory_frame_set_output_file_pos
        # The currently active frame set
        tng_trajectory_frame_set current_trajectory_frame_set
        # The pos in the src file of the current frame set
        int64_t current_trajectory_frame_set_input_file_pos
        # The pos in the dest file of the current frame set
        int64_t current_trajectory_frame_set_output_file_pos
        # The number of frame sets in the trajectory N.B. Not saved in file and
        # cannot be trusted to be up-to-date
        int64_t n_trajectory_frame_sets

        # These data blocks are non-trajectory data blocks
        # The number of non-frame dependent particle dependent data blocks
        int n_particle_data_blocks
        # A list of data blocks containing particle dependent data
        tng_data * non_tr_particle_data

        # The number of frame and particle independent data blocks
        int n_data_blocks
        # A list of frame and particle indepdendent data blocks
        tng_data * non_tr_data

        # TNG compression algorithm for compressing positions
        int * compress_algo_pos
        # TNG compression algorithm for compressing velocities
        int * compress_algo_vel
        # The precision used for lossy compression
        double compression_precision

    struct tng_gen_block:
        # The size of the block header in bytes
        int64_t header_contents_size
        # The size of the block contents in bytes
        int64_t block_contents_size
        # The ID of the block to determine its type
        int64_t id
        # The MD5 hash of the block to verify integrity
        char md5_hash[16]  # TNG_MD5_HASH_LEN == 16
        # The name of the block
        char * name
        # The library version used to write the block
        int64_t block_version
        int64_t alt_hash_type
        int64_t alt_hash_len
        char * alt_hash
        int64_t signature_type
        int64_t signature_len
        char * signature
        # The full block header contents
        char * header_contents
        # The full block contents
        char * block_contents

    tng_function_status tng_util_trajectory_open(
        const char * filename,
        const char mode,
        tng_trajectory * * tng_data_p) nogil

    tng_function_status tng_util_trajectory_close(
        tng_trajectory * * tng_data_p) nogil

    tng_function_status tng_num_frames_get(
        const tng_trajectory * tng_data,
        int64_t * n) nogil

    tng_function_status tng_num_particles_get(
        const tng_trajectory * tng_data,
        int64_t * n) nogil

    tng_function_status tng_distance_unit_exponential_get(
        const tng_trajectory * tng_data,
        int64_t * exp) nogil

    tng_function_status tng_util_time_of_frame_get(
        const tng_trajectory * tng_data,
        const int64_t frame_nr,
        double * time) nogil

    tng_function_status  tng_block_read_next(
        tng_trajectory * tng_data,
        tng_gen_block * block_data,
        char             hash_mode) nogil

    tng_function_status tng_block_init(
        tng_gen_block ** block_p) nogil

    tng_function_status tng_block_header_read(
        tng_trajectory * tng_data,
        tng_gen_block * block) nogil

    tng_function_status tng_num_frame_sets_get(
        tng_trajectory * tng_data,
        int64_t * n) nogil

    tng_function_status tng_block_destroy(
        tng_gen_block ** block_p) nogil

    tng_function_status tng_data_get_stride_length(
        tng_trajectory * tng_data,
        int64_t block_id,
        int64_t frame,
        int64_t * stride_length) nogil

    tng_function_status tng_util_trajectory_next_frame_present_data_blocks_find(
        tng_trajectory * tng_data,
        int64_t current_frame,
        int64_t n_requested_data_block_ids,
        int64_t * requested_data_block_ids,
        int64_t * next_frame,
        int64_t * n_data_blocks_in_next_frame,
        int64_t ** data_block_ids_in_next_frame) nogil

    tng_function_status tng_data_block_name_get(
        tng_trajectory * tng_data,
        const int64_t block_id,
        char * name,
        const int max_len) nogil

    tng_function_status tng_data_block_dependency_get(
        tng_trajectory * tng_data,
        const int64_t block_id,
        int * block_dependency) nogil

    tng_function_status tng_util_particle_data_next_frame_read(
        tng_trajectory * tng_data,
        const int64_t block_id,
        void ** values,
        char * data_type,
        int64_t * retrieved_frame_number,
        double * retrieved_time) nogil

    tng_function_status tng_util_non_particle_data_next_frame_read(
        tng_trajectory * tng_data,
        const int64_t block_id,
        void ** values,
        char * data_type,
        int64_t * retrieved_frame_number,
        double * retrieved_time) nogil

    tng_function_status tng_data_block_num_values_per_frame_get(
        tng_trajectory * tng_data,
        int64_t block_id,
        int64_t * n_values_per_frame) nogil

    tng_function_status tng_util_frame_current_compression_get(
        tng_trajectory * tng_data,
        int64_t block_id,
        int64_t * codec_id,
        double * factor) nogil

    tng_function_status tng_util_num_frames_with_data_of_block_id_get(
        tng_trajectory * tng_data,
        int64_t block_id,
        int64_t * n_frames) nogil

    tng_function_status tng_gen_data_vector_interval_get(
        tng_trajectory * tng_data,
        const int64_t  block_id,
        const tng_bool is_particle_data,
        const int64_t  start_frame_nr,
        const int64_t end_frame_nr,
        const char  hash_mode,
        void ** values,
        int64_t * n_particles,
        int64_t * stride_length,
        int64_t * n_values_per_frame,
        char * type) nogil

    tng_function_status tng_num_particles_variable_get(
        tng_trajectory * tng_data, char * variable) nogil

cdef int64_t gcd(int64_t a, int64_t b):
    cdef int64_t temp
    while b < 0:
        temp = b
        b = a % b
        a = temp
    return a

cdef int64_t gcd_list(list a):
    cdef int size = len(a)
    cdef int i
    cdef int64_t result = a[0]
    for i in range(1, size):
        result = gcd(result, a[i])
    return result

cdef class TrajectoryWrapper:
    """A wrapper class for a tng_trajectory"""
    cdef tng_trajectory * _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            free(self._ptr)
            self._ptr = NULL

    @staticmethod
    cdef TrajectoryWrapper from_ptr(tng_trajectory * _ptr, bint owner=False):
        """Factory function to create WrapperClass objects from
        given tng_trajectory pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Call to __new__ bypasses __init__ constructor
        cdef TrajectoryWrapper wrapper = \
            TrajectoryWrapper.__new__(TrajectoryWrapper)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef TrajectoryWrapper new_struct():
        """Factory function to create WrapperClass objects with
        newly allocated tng_trajectory"""
        cdef tng_trajectory * _ptr = \
            <tng_trajectory * >malloc(sizeof(tng_trajectory))
        if _ptr is NULL:
            raise MemoryError
        return TrajectoryWrapper.from_ptr(_ptr, owner=True)


cdef class TNGFileIterator:
    """File handle object for TNG files

    Supports use as a context manager ("with" blocks).

    """
    # tng_trajectory pointer
    cdef tng_trajectory * _traj_p
    # trajectory wrapper
    cdef TrajectoryWrapper _traj
    # filename
    cdef readonly fname
    # mode (r,w,a)
    cdef str mode
    # enable/disable debug output
    cdef bint debug
    # mark trajectory to be closed/open
    cdef int is_open
    # have we reached the end of the file
    cdef int reached_eof

    # integrator timestep
    cdef int64_t step
    # number of integrator timesteps
    cdef int64_t _n_steps
    # number of particles
    cdef int64_t _n_particles
    # distance unit
    cdef float _distance_scale

    # stride at which each block is written
    cdef dict   _frame_strides
    # number of actual frames with data for each block
    cdef dict   _n_data_frames
    # the number of values per frame for each data block
    cdef dict   _values_per_frame
    # particle dependencies for each data block
    cdef dict _particle_dependencies

    # greatest common divisor of data strides
    cdef int64_t _gcd

    # holds data at the current trajectory timestep
    cdef TNGCurrentIntegratorStep current_step

    def __cinit__(self, fname, mode='r', debug=False):

        self._traj = TrajectoryWrapper.from_ptr(self._traj_p, owner=True)
        self.fname = fname
        self.debug = debug
        self.step = 0

        self._n_steps = -1
        self._n_particles = -1
        self._distance_scale = 0.0

        self._frame_strides = {}
        self._n_data_frames = {}
        self._values_per_frame = {}
        self._particle_dependencies = {}
        self._gcd = -1

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
        cdef char var_natoms_flag

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
        if self.mode == 'r' and not os.path.isfile(fname):
            raise IOError("File '{}' does not exist".format(fname))

        cdef tng_function_status stat

        fname_bytes = fname.encode('UTF-8')
        # open the trajectory
        stat = tng_util_trajectory_open(fname_bytes, _mode, & self._traj._ptr)
        if stat != TNG_SUCCESS:
            raise IOError("File '{}' cannot be opened".format(fname))

        # check if the number of particles can vary
        stat = tng_num_particles_variable_get(self._traj._ptr, & var_natoms_flag)
        if stat != TNG_SUCCESS:
            raise IOError("Particle variability cannot be read")
        if var_natoms_flag != TNG_CONSTANT_N_ATOMS:
            raise IOError("Variable numbers of particles not supported")

        # get the number of integrator timesteps
        stat = tng_num_frames_get(self._traj._ptr, & self._n_steps)
        if stat != TNG_SUCCESS:
            raise IOError("Number of frames cannot be read")
        # get the number of particles
        stat = tng_num_particles_get(self._traj._ptr, & self._n_particles)
        if stat != TNG_SUCCESS:
            raise IOError("Number of particles cannot be read")
        # get the unit scale
        cdef int64_t exponent
        stat = tng_distance_unit_exponential_get(self._traj._ptr, & exponent)
        if stat != TNG_SUCCESS:
            raise IOError("Distance exponent cannot be read")
        # fill out dictionaries of block metadata
        stat = self._get_block_metadata()
        if stat != TNG_SUCCESS:
            raise IOError("Strides for each data block cannot be read")

        self._distance_scale = 10.0**(exponent+9)
        self.is_open = True
        self.reached_eof = False

    # close the file
    def _close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            tng_util_trajectory_close(& self._traj._ptr)
            self.is_open = False
            self.reached_eof = True
            self._n_steps = -1

    @property
    def n_steps(self):
        """The number of integrator steps in the TNG file

        Returns
        -------
        n_steps : int
            number of integrator steps
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self._n_steps

    @property
    def n_atoms(self):
        """The number of atoms in the TNG file

        Returns
        -------
        n_atoms : int
            number of atoms
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self._n_particles

    @property
    def block_strides(self):
        """Dictionary of block names and the strides (in integrator steps)
        at which they are written in the TNG file

        Returns
        -------
        block_strides : dict
            dictionary of block names (keys) and strides of each block (values)
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self._frame_strides

    @property
    def block_ids(self):
        """Dictionary of block names and block ids (long longs) in the
        TNG file

        Returns
        -------
        block_ids : dict
            dictionary of block names (keys) and block ids (values)
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        block_id_dict = {}
        for k in self._frame_strides.keys():
            block_id_dict[k] = block_id_dictionary[k]
        return block_id_dict

    @property
    def n_data_frames(self):
        """Dictionary of block names and the number of actual steps with data
        for that block in the TNG file

        Returns
        -------
        n_data_frames : dict
            dictionary of block names (keys) and number of steps with data
            (values)
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self._n_data_frames

    @property
    def values_per_frame(self):
        """Dictionary of block names and the number of values per frame for the
        block

        Returns
        -------
        values_per_frame : dict
            dictionary of block names (keys) and number of values per frame
            (values)
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self._values_per_frame

    @property
    def particle_dependencies(self):
        """Dictionary of block names and whether the block is particle
        dependent

        Returns
        -------
        particle_dependencies : dict
            dictionary of block names (keys) and particle dependencies (values)
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self._particle_dependencies

    cpdef np.ndarray make_ndarray_for_block_from_name(self, str block_name):
        """Make a NumPy array that can hold a specified block from the block
        name

        Parameters
        ----------
        block_name : str
           a block name

        Returns
        -------
        target : :class:`np.ndarray`
            A NumPy array that can hold the data values for a specified block

        See Also
        --------
        block_ids : dict
            dictionary of block names (keys) and block ids (values) available
            in this TNG file
        """
        if block_name not in block_dictionary.values():
            raise ValueError("Block name {} not recognised".format(block_name))

        if self._particle_dependencies[block_name]:
            ax0 = self._n_particles
        else:
            ax0 = 1
        ax1 = self._values_per_frame[block_name]
        target = np.ndarray(shape=(ax0, ax1), dtype=np.float32, order='C')
        return target

    cpdef np.ndarray make_ndarray_for_block_from_id(self, int64_t block_id):
        """Make a NumPy array that can hold a specified block from the block id

        Parameters
        ----------
        block_id : int64_t
           a block id

        Returns
        -------
        target : :class:`np.ndarray`
            A NumPy array that can hold the data values for a specified block

        See Also
        --------
        block_ids : dict
            dictionary of block names (keys) and block ids (values) available
            in this TNG file
        """
        return self.make_ndarray_for_block_from_name(block_id_dictionary[block_id])

    @property
    def step(self):
        """The current integrator step being read

        Returns
        -------
        step : int
            the current step in the TNG file
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self.step

    @property
    def current_integrator_step(self):
        """Class that retrieves data from the file at the current integrator
           step

        Returns
        -------
        current_integrator_step : :class:`TNGCurrentIntegratorStep`
            The data accessor at the current integrator step
        """
        if not self.is_open:
            raise IOError("File is not yet open")
        return self.current_step

    cpdef TNGCurrentIntegratorStep  read_step(self, step):
        """Read a step (integrator step) from the file

        Parameters
        ----------
        step : int
           step to read from the file

        Returns
        -------
        current_integrator_step : :class:`TNGCurrentIntegratorStep`
            The data accessor at the current integrator step

        Raises
        ------
        ValueError
            attempt to read a negative step or step number greater than that in
            the input file
        """
        if not self.is_open:
            raise IOError('File is not yet open')

        if step >= self._n_steps:
            raise ValueError("""frame specified is greater than number of steps
            in input file {}""".format(self._n_steps))

        if step < 0:
            step = self._n_steps - np.abs(step)

        self.step = step
        self.current_step = TNGCurrentIntegratorStep(
            self._traj, step, debug=self.debug)

        return self.current_step

    # NOTE here we assume that the first frame has all the blocks
    #  that are present in the whole traj

    cdef tng_function_status _get_block_metadata(self):
        """Gets the ids, strides and number of frames with
           actual data from the trajectory"""
        cdef int64_t step, n_blocks
        cdef int64_t nframes, stride_length, n_values_per_frame
        cdef int64_t block_counter = 0
        cdef int64_t * block_ids = NULL
        cdef int     block_dependency
        cdef bint particle_dependent

        cdef tng_function_status read_stat = \
            tng_util_trajectory_next_frame_present_data_blocks_find(
                self._traj._ptr, -1, 0, NULL, & step, & n_blocks, & block_ids)

        for i in range(n_blocks):
            read_stat = tng_data_get_stride_length(
                self._traj._ptr, block_ids[i], -1, & stride_length)
            if read_stat != TNG_SUCCESS:
                return TNG_CRITICAL
            read_stat = tng_util_num_frames_with_data_of_block_id_get(
                self._traj._ptr, block_ids[i], & nframes)
            if read_stat != TNG_SUCCESS:
                return TNG_CRITICAL
            read_stat = tng_data_block_num_values_per_frame_get(
                self._traj._ptr, block_ids[i], & n_values_per_frame)
            read_stat = tng_data_block_dependency_get(self._traj._ptr,
                                                      block_ids[i],
                                                      & block_dependency)
            if read_stat != TNG_SUCCESS:
                return TNG_CRITICAL
            if block_dependency & TNG_PARTICLE_DEPENDENT:
                particle_dependent = True
            else:
                particle_dependent = False

            # stride length for the block
            self._frame_strides[block_dictionary[block_ids[i]]] = stride_length
            # number of actual data frames for the block
            self._n_data_frames[block_dictionary[block_ids[i]]] = nframes
            # number of values per frame
            self._values_per_frame[block_dictionary[block_ids[i]]
                                   ] = n_values_per_frame

            self._particle_dependencies[block_dictionary[block_ids[i]]
                                        ] = particle_dependent

        # TODO we will use this if we want to instead iterate
        # over the greatest common divisor of the data strides
        self._gcd = gcd_list(list(self._frame_strides.values()))
        if self.debug:
            printf("PYTNG INFO: gcd of strides %ld \n", self._gcd)

        return TNG_SUCCESS

    def __enter__(self):
        # Support context manager
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        # always propagate exceptions forward
        return False

    def __len__(self):
        return self._n_steps

    def __iter__(self):
        self._close()
        self._open(self.fname, self.mode)
        self.read_step(self.step)

        return self

    def __next__(self):
        if self.step == self._n_steps - 1:
            raise StopIteration
        self.read_step(self.step)
        self.step += 1
        return self.current_integrator_step

    def __getitem__(self, frame):
        cdef int64_t start, stop, step, i

        if isinstance(frame, numbers.Integral):
            if self.debug:
                print("slice is a number")
            self.read_step(frame)
            return self.current_integrator_step

        elif isinstance(frame, (list, np.ndarray)):
            if self.debug:
                print("slice is a list or array")
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
                    self.read_step(f)
                    yield self.current_integrator_step
            return listiter(frame)
        elif isinstance(frame, slice):
            start = frame.start if frame.start is not None else 0
            stop = frame.stop if frame.stop is not None else self._n_steps
            step = frame.step if frame.step is not None else 1

            def sliceiter(start, stop, step):
                for i in range(start, stop, step):
                    self.read_step(i)
                    yield self.current_integrator_step
            return sliceiter(start, stop, step)
        else:
            raise TypeError("Trajectories must be an indexed using an integer,"
                            " slice or list of indices")


cdef class TNGCurrentIntegratorStep:
    """Retrieves data at the curent trajectory step"""

    cdef bint debug
    cdef int64_t _n_blocks

    cdef tng_trajectory * _traj
    cdef int64_t step
    cdef bint read_success

    def __cinit__(self, TrajectoryWrapper traj, int64_t step,
                  bint debug=False):
        self.debug = debug

        self._traj = traj._ptr
        self.step = step
        self.read_success = False

    def __dealloc__(self):
        pass

    @property
    def step(self):
        """The current integrator step being read

        Returns
        -------
        step : int
            the current step in the TNG file
        """
        return self.step

    @property
    def read_success(self):
        """Indicates whether the last attempt to read data was successful

        Returns
        -------
        read_success : bool
            Whether the last attempt to read data was successful
        """
        return self.read_success

    cpdef  get_time(self):
        """Get the time of the current integrator step being read from the file

        Returns
        -------
        time : int
            the time of the current step
        """
        cdef tng_function_status read_stat
        cdef double _step_time
        read_stat = self._get_step_time(& _step_time)
        if read_stat != TNG_SUCCESS:
            return None
        else:
            return _step_time

    cpdef np.ndarray get_positions(self, np.ndarray data):
        """Get the positions present at the current step and read them into a
        NumPy array

        Parameters
        ----------
        data : np.ndarray
           NumPy array to read the data into. As this is a particle dependent
           block, the shape should be (n_atoms, n_values_per_frame)
           ie (n_atoms, 3).
        """
        self.get_blockid(TNG_TRAJ_POSITIONS, data)
        return data

    cpdef np.ndarray get_box(self, np.ndarray data):
        """Get the box vectors present at the current step and read them into a
        NumPy array. The box vectors are a (3,3) matrix comprised of 3
        three-dimensional basis vectors for the coordinate system of the
        simulation. The vectors can be accessed in their proper shape by
        reshaping the resulting (1,9) matrix to be (3,3) with
        ndarray.reshape(3,3).

        Parameters
        ----------
        data : np.ndarray
           NumPy array to read the data into. As this is NOT a particle 
           dependent block, the shape should be
           (1, n_values_per_frame) ie (1,9)
        """
        self.get_blockid(TNG_TRAJ_BOX_SHAPE, data)
        return data

    cpdef np.ndarray get_velocities(self, np.ndarray data):
        """Get the velocities present at the current step and read them into a
        NumPy array

        Parameters
        ----------
        data : np.ndarray
           NumPy array to read the data into. As this is a particle dependent
           block, the shape should be (n_atoms, n_values_per_frame)
           ie (n_atoms, 3).
        """
        self.get_blockid(TNG_TRAJ_VELOCITIES, data)
        return data

    cpdef np.ndarray get_forces(self, np.ndarray data):
        """Get the forces present at the current step and read them into a
        NumPy array

        Parameters
        ----------
        data : np.ndarray
           NumPy array to read the data into. As this is a particle dependent
           block, the shape should be (n_atoms, n_values_per_frame)
           ie (n_atoms, 3).
        """
        self.get_blockid(TNG_TRAJ_FORCES, data)
        return data

    cpdef get_blockid(self, int64_t block_id, np.ndarray data):
        """Get a block ID present at the current step and read it into a
        NumPy array

        Parameters
        ----------
        block_id : int64_t
           TNG block id to read from the current step
        data : np.ndarray
           NumPy array to read the data into, the required shape is determined
           by the block dependency and the number of values per frame.

        Raises
        ------
        TypeError
            The dtype of the numpy array provided is not supported by TNG
            datatypes or does not match the underlying datatype.
        IOError
            The block data type cannot be understood.
        IndexError
            The shape of the numpy array provided does not match the shape of
            the data to be read from disk.
        """
        shape = data.shape
        dtype = data.dtype

        if dtype not in [np.int64, np.float32, np.float64]:
            raise TypeError(
                "PYTNG ERROR: datatype of numpy array not supported\n")

        cdef void * values = NULL
        cdef int64_t n_values_per_frame = -1
        cdef int64_t n_atoms = -1
        cdef double precision = -1
        cdef char datatype = -1
        cdef tng_function_status read_stat = TNG_CRITICAL

        cdef np.float32_t[::1] _float_view
        cdef int64_t[::1] _int64_t_view
        cdef double[::1] _double_view

        cdef int i, j

        with nogil:
            read_stat = self._get_data_current_step(block_id,
                                                    self.step,
                                                    & values,
                                                    & n_values_per_frame,
                                                    & n_atoms,
                                                    & precision,
                                                    & datatype,
                                                    self.debug)

        if read_stat != TNG_SUCCESS:
            self.read_success = False
            # NOTE I think we should still nan fill on blank read
            data[:, :] = np.nan
            return data
        else:
            self.read_success = True

        if data.ndim > 2:
            raise IndexError(
                """PYTNG ERROR: Numpy array must be 2 dimensional,
                 you supplied a {} dimensional ndarray""".format(data.ndim))

        if shape[0] != n_atoms:
            raise IndexError(
                """PYTNG ERROR: First axis must be n_atoms long,
                you supplied {}""".format(shape[0]))

        if shape[1] != n_values_per_frame:
            raise IndexError(
                """PYTNG ERROR: Second axis must be n_values_per_frame long,
                 you supplied {}""".format(shape[1]))

        cdef int64_t n_vals = n_atoms * n_values_per_frame

        if datatype == TNG_FLOAT_DATA:
            if dtype != np.float32:
                raise TypeError(
                    "PYTNG ERROR: dtype of array {} does not match TNG dtype float".format(dtype))
            _float_view = <np.float32_t[:n_vals] > ( < float*> values)
            data[:, :] = np.asarray(_float_view, dtype=np.float32).reshape(
                n_atoms, n_values_per_frame)

        elif datatype == TNG_INT_DATA:
            if dtype != np.int64:
                raise TypeError(
                    "PYTNG ERROR: dtype of array {} does not match TNG dtype int64_t".format(dtype))
            _int64_t_view = <int64_t[:n_vals] > ( < int64_t*> values)
            data[:, :] = np.asarray(_int64_t_view, dtype=np.int64).reshape(
                n_atoms, n_values_per_frame)

        elif datatype == TNG_DOUBLE_DATA:
            if dtype != np.float64:
                raise TypeError(
                    "PYTNG ERROR: dtype of array {} does not match TNG dtype double".format(dtype))
            _double_view = <double[:n_vals] > ( < double*> values)
            data[:, :] = np.asarray(_double_view, dtype=np.float64).reshape(
                n_atoms, n_values_per_frame)

        else:
            raise IOError("PYTNG ERROR: block datatype not understood")

        return data

    cdef tng_function_status _get_data_current_step(self, int64_t block_id,
                                                    int64_t step,
                                                    void ** values,
                                                    int64_t * n_values_per_frame,
                                                    int64_t * n_atoms,
                                                    double * prec,
                                                    char * datatype,
                                                    bint debug) nogil:
        """Gets the frame data off disk and into C level arrays

        Parameters
        ----------
        block_id : int64_t
            block id to read
        step : int64_t
            integrator step to read
        values : void **
            NULL void pointer to hook the data onto
        n_values_per_frame : int64_t *
            set to the number of values per frame for the block
        n_atoms : int64_t *
            set to the number of atoms or 1 if particle dependent
        prec : double *
            set to the precision of the block
        datatype : char *
            set to the datatype of the block
        debug : bint
            debug the block read

        Notes
        -----
        This function is private. Additionally, this function is  marked nogil 
        and called without the GIL so cannot contain python or python
        exceptions. Instead failure is marked by returning
        :data:`TNG_CRITICAL`. Success is indicated by returning
        :data:`TNG_SUCCESS`. Cleanup is then be done by the calling
        code in :method:`get_blockid` so the user should not have to deal with
        C level exceptions
        """
        cdef tng_function_status stat = TNG_CRITICAL
        cdef int64_t             codec_id
        cdef int             block_dependency
        cdef void * data = NULL
        cdef double              local_prec
        cdef int64_t             stride_length

        # is this a particle dependent block?
        stat = tng_data_block_dependency_get(self._traj, block_id,
                                             & block_dependency)
        if stat != TNG_SUCCESS:
            return TNG_CRITICAL

        if block_dependency & TNG_PARTICLE_DEPENDENT:  # bitwise & due to enums
            tng_num_particles_get(self._traj, n_atoms)
            # read particle data off disk with hash checking
            stat = tng_gen_data_vector_interval_get(self._traj,
                                                    block_id,
                                                    TNG_TRUE,
                                                    self.step,
                                                    self.step,
                                                    TNG_USE_HASH,
                                                    values,
                                                    n_atoms,
                                                    & stride_length,
                                                    n_values_per_frame,
                                                    datatype)

        else:
            n_atoms[0] = 1  # still used for some allocs
            # read non particle data off disk with hash checking
            stat = tng_gen_data_vector_interval_get(self._traj,
                                                    block_id,
                                                    TNG_FALSE,
                                                    self.step,
                                                    self.step,
                                                    TNG_USE_HASH,
                                                    values,
                                                    NULL,
                                                    & stride_length,
                                                    n_values_per_frame,
                                                    datatype)
        if stat != TNG_SUCCESS:
            return TNG_CRITICAL

        if self.step % stride_length != 0:
            return TNG_CRITICAL

        # get the compression of the current frame
        stat = tng_util_frame_current_compression_get(self._traj,
                                                      block_id,
                                                      & codec_id,
                                                      & local_prec)
        if stat != TNG_SUCCESS:
            return TNG_CRITICAL

        if codec_id != TNG_TNG_COMPRESSION:
            prec[0] = -1.0
        else:
            prec[0] = local_prec

        return TNG_SUCCESS

    cdef tng_function_status _get_step_time(self, double * step_time):
        stat = tng_util_time_of_frame_get(self._traj, self.step, step_time)
        if stat != TNG_SUCCESS:
            return TNG_CRITICAL


block_dictionary = {}

# group 1
block_dictionary[TNG_GENERAL_INFO] = "TNG_GENERAL_INFO"
block_dictionary[TNG_MOLECULES] = "TNG_MOLECULES"
block_dictionary[TNG_TRAJECTORY_FRAME_SET] = "TNG_TRAJECTORY_FRAME_SET"
block_dictionary[TNG_PARTICLE_MAPPING] = "TNG_PARTICLE_MAPPING"

# group 2
block_dictionary[TNG_TRAJ_BOX_SHAPE] = "TNG_TRAJ_BOX_SHAPE"
block_dictionary[TNG_TRAJ_POSITIONS] = "TNG_TRAJ_POSITIONS"
block_dictionary[TNG_TRAJ_VELOCITIES] = "TNG_TRAJ_VELOCITIES"
block_dictionary[TNG_TRAJ_FORCES] = "TNG_TRAJ_FORCES"
block_dictionary[TNG_TRAJ_PARTIAL_CHARGES] = "TNG_TRAJ_PARTIAL_CHARGES"
block_dictionary[TNG_TRAJ_FORMAL_CHARGES] = "TNG_TRAJ_FORMAL_CHARGES"
block_dictionary[TNG_TRAJ_B_FACTORS] = "TNG_TRAJ_B_FACTORS"
block_dictionary[TNG_TRAJ_ANISOTROPIC_B_FACTORS] = "TNG_TRAJ_ANISOTROPIC_B_FACTORS"
block_dictionary[TNG_TRAJ_OCCUPANCY] = "TNG_TRAJ_OCCUPANCY"
block_dictionary[TNG_TRAJ_GENERAL_COMMENTS] = "TNG_TRAJ_GENERAL_COMMENTS"
block_dictionary[TNG_TRAJ_MASSES] = "TNG_TRAJ_MASSES"

# group 3
block_dictionary[TNG_GMX_LAMBDA] = "TNG_GMX_LAMBDA"
block_dictionary[TNG_GMX_ENERGY_ANGLE] = "TNG_GMX_ENERGY_ANGLE"
block_dictionary[TNG_GMX_ENERGY_RYCKAERT_BELL] = "TNG_GMX_ENERGY_RYCKAERT_BELL"
block_dictionary[TNG_GMX_ENERGY_LJ_14] = "TNG_GMX_ENERGY_LJ_14"
block_dictionary[TNG_GMX_ENERGY_COULOMB_14] = "TNG_GMX_ENERGY_COULOMB_14"
block_dictionary[TNG_GMX_ENERGY_LJ_SR] = "TNG_GMX_ENERGY_LJ_SR"
block_dictionary[TNG_GMX_ENERGY_COULOMB_SR] = "TNG_GMX_ENERGY_COULOMB_SR"
block_dictionary[TNG_GMX_ENERGY_COUL_RECIP] = "TNG_GMX_ENERGY_COUL_RECIP"
block_dictionary[TNG_GMX_ENERGY_POTENTIAL] = "TNG_GMX_ENERGY_POTENTIAL"
block_dictionary[TNG_GMX_ENERGY_KINETIC_EN] = "TNG_GMX_ENERGY_KINETIC_EN"
block_dictionary[TNG_GMX_ENERGY_TOTAL_ENERGY] = "TNG_GMX_ENERGY_TOTAL_ENERGY"
block_dictionary[TNG_GMX_ENERGY_TEMPERATURE] = "TNG_GMX_ENERGY_TEMPERATURE"
block_dictionary[TNG_GMX_ENERGY_PRESSURE] = "TNG_GMX_ENERGY_PRESSURE"
block_dictionary[TNG_GMX_ENERGY_CONSTR_RMSD] = "TNG_GMX_ENERGY_CONSTR_RMSD"
block_dictionary[TNG_GMX_ENERGY_CONSTR2_RMSD] = "TNG_GMX_ENERGY_CONSTR2_RMSD"
block_dictionary[TNG_GMX_ENERGY_BOX_X] = "TNG_GMX_ENERGY_BOX_X"
block_dictionary[TNG_GMX_ENERGY_BOX_Y] = "TNG_GMX_ENERGY_BOX_Y"
block_dictionary[TNG_GMX_ENERGY_BOX_Z] = "TNG_GMX_ENERGY_BOX_Z"
block_dictionary[TNG_GMX_ENERGY_BOXXX] = "TNG_GMX_ENERGY_BOXXX"
block_dictionary[TNG_GMX_ENERGY_BOXYY] = "TNG_GMX_ENERGY_BOXYY"
block_dictionary[TNG_GMX_ENERGY_BOXZZ] = "TNG_GMX_ENERGY_BOXZZ"
block_dictionary[TNG_GMX_ENERGY_BOXYX] = "TNG_GMX_ENERGY_BOXYX"
block_dictionary[TNG_GMX_ENERGY_BOXZX] = "TNG_GMX_ENERGY_BOXZX"
block_dictionary[TNG_GMX_ENERGY_BOXZY] = "TNG_GMX_ENERGY_BOXZY"
block_dictionary[TNG_GMX_ENERGY_BOXVELXX] = "TNG_GMX_ENERGY_BOXVELXX"
block_dictionary[TNG_GMX_ENERGY_BOXVELYY] = "TNG_GMX_ENERGY_BOXVELYY"
block_dictionary[TNG_GMX_ENERGY_BOXVELZZ] = "TNG_GMX_ENERGY_BOXVELZZ"
block_dictionary[TNG_GMX_ENERGY_BOXVELYX] = "TNG_GMX_ENERGY_BOXVELYX"
block_dictionary[TNG_GMX_ENERGY_BOXVELZX] = "TNG_GMX_ENERGY_BOXVELZX"
block_dictionary[TNG_GMX_ENERGY_BOXVELZY] = "TNG_GMX_ENERGY_BOXVELZY"
block_dictionary[TNG_GMX_ENERGY_VOLUME] = "TNG_GMX_ENERGY_VOLUME"
block_dictionary[TNG_GMX_ENERGY_DENSITY] = "TNG_GMX_ENERGY_DENSITY"
block_dictionary[TNG_GMX_ENERGY_PV] = "TNG_GMX_ENERGY_PV"
block_dictionary[TNG_GMX_ENERGY_ENTHALPY] = "TNG_GMX_ENERGY_ENTHALPY"
block_dictionary[TNG_GMX_ENERGY_VIR_XX] = "TNG_GMX_ENERGY_VIR_XX"
block_dictionary[TNG_GMX_ENERGY_VIR_XY] = "TNG_GMX_ENERGY_VIR_XY"
block_dictionary[TNG_GMX_ENERGY_VIR_XZ] = "TNG_GMX_ENERGY_VIR_XZ"
block_dictionary[TNG_GMX_ENERGY_VIR_YX] = "TNG_GMX_ENERGY_VIR_YX"
block_dictionary[TNG_GMX_ENERGY_VIR_YY] = "TNG_GMX_ENERGY_VIR_YY"
block_dictionary[TNG_GMX_ENERGY_VIR_YZ] = "TNG_GMX_ENERGY_VIR_YZ"
block_dictionary[TNG_GMX_ENERGY_VIR_ZX] = "TNG_GMX_ENERGY_VIR_ZX"
block_dictionary[TNG_GMX_ENERGY_VIR_ZY] = "TNG_GMX_ENERGY_VIR_ZY"
block_dictionary[TNG_GMX_ENERGY_VIR_ZZ] = "TNG_GMX_ENERGY_VIR_ZZ"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_XX] = "TNG_GMX_ENERGY_SHAKEVIR_XX"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_XY] = "TNG_GMX_ENERGY_SHAKEVIR_XY"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_XZ] = "TNG_GMX_ENERGY_SHAKEVIR_XZ"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_YX] = "TNG_GMX_ENERGY_SHAKEVIR_YX"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_YY] = "TNG_GMX_ENERGY_SHAKEVIR_YY"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_YZ] = "TNG_GMX_ENERGY_SHAKEVIR_YZ"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_ZX] = "TNG_GMX_ENERGY_SHAKEVIR_ZX"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_ZY] = "TNG_GMX_ENERGY_SHAKEVIR_ZY"
block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_ZZ] = "TNG_GMX_ENERGY_SHAKEVIR_ZZ"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_XX] = "TNG_GMX_ENERGY_FORCEVIR_XX"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_XY] = "TNG_GMX_ENERGY_FORCEVIR_XY"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_XZ] = "TNG_GMX_ENERGY_FORCEVIR_XZ"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_YX] = "TNG_GMX_ENERGY_FORCEVIR_YX"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_YY] = "TNG_GMX_ENERGY_FORCEVIR_YY"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_YZ] = "TNG_GMX_ENERGY_FORCEVIR_YZ"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_ZX] = "TNG_GMX_ENERGY_FORCEVIR_ZX"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_ZY] = "TNG_GMX_ENERGY_FORCEVIR_ZY"
block_dictionary[TNG_GMX_ENERGY_FORCEVIR_ZZ] = "TNG_GMX_ENERGY_FORCEVIR_ZZ"
block_dictionary[TNG_GMX_ENERGY_PRES_XX] = "TNG_GMX_ENERGY_PRES_XX"
block_dictionary[TNG_GMX_ENERGY_PRES_XY] = "TNG_GMX_ENERGY_PRES_XY"
block_dictionary[TNG_GMX_ENERGY_PRES_XZ] = "TNG_GMX_ENERGY_PRES_XZ"
block_dictionary[TNG_GMX_ENERGY_PRES_YX] = "TNG_GMX_ENERGY_PRES_YX"
block_dictionary[TNG_GMX_ENERGY_PRES_YY] = "TNG_GMX_ENERGY_PRES_YY"
block_dictionary[TNG_GMX_ENERGY_PRES_YZ] = "TNG_GMX_ENERGY_PRES_YZ"
block_dictionary[TNG_GMX_ENERGY_PRES_ZX] = "TNG_GMX_ENERGY_PRES_ZX"
block_dictionary[TNG_GMX_ENERGY_PRES_ZY] = "TNG_GMX_ENERGY_PRES_ZY"
block_dictionary[TNG_GMX_ENERGY_PRES_ZZ] = "TNG_GMX_ENERGY_PRES_ZZ"
block_dictionary[TNG_GMX_ENERGY_SURFXSURFTEN] = "TNG_GMX_ENERGY_SURFXSURFTEN"
block_dictionary[TNG_GMX_ENERGY_MUX] = "TNG_GMX_ENERGY_MUX"
block_dictionary[TNG_GMX_ENERGY_MUY] = "TNG_GMX_ENERGY_MUY"
block_dictionary[TNG_GMX_ENERGY_MUZ] = "TNG_GMX_ENERGY_MUZ"
block_dictionary[TNG_GMX_ENERGY_VCOS] = "TNG_GMX_ENERGY_VCOS"
block_dictionary[TNG_GMX_ENERGY_VISC] = "TNG_GMX_ENERGY_VISC"
block_dictionary[TNG_GMX_ENERGY_BAROSTAT] = "TNG_GMX_ENERGY_BAROSTAT"
block_dictionary[TNG_GMX_ENERGY_T_SYSTEM] = "TNG_GMX_ENERGY_T_SYSTEM"
block_dictionary[TNG_GMX_ENERGY_LAMB_SYSTEM] = "TNG_GMX_ENERGY_LAMB_SYSTEM"
block_dictionary[TNG_GMX_SELECTION_GROUP_NAMES] = "TNG_GMX_SELECTION_GROUP_NAMES"
block_dictionary[TNG_GMX_ATOM_SELECTION_GROUP] = "TNG_GMX_ATOM_SELECTION_GROUP"

# reverse the mapping
block_id_dictionary = {v: k for k, v in block_dictionary.items()}
