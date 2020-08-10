# cython: linetrace=True
# cython: embedsignature=True
# cython: profile=True
# cython: binding=True
# distutils: define_macros=[CYTHON_TRACE=1, CYTHON_TRACE_NOGIL=1]

import numpy as np
import numbers
import os
from collections import namedtuple
from posix.types cimport off_t
from libc.string cimport memcpy
from libc.stdio cimport printf, FILE, SEEK_SET, SEEK_CUR, SEEK_END
from libc.stdlib cimport malloc, free, realloc
from libc.stdint cimport int64_t, uint64_t, int32_t, uint32_t
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from numpy cimport(PyArray_SimpleNewFromData,
                   PyArray_SetBaseObject,
                   NPY_FLOAT,
                   NPY_DOUBLE,
                   Py_INCREF,
                   npy_intp,
                   )
cimport cython


cimport numpy as np
np.import_array()


ctypedef enum tng_function_status: TNG_SUCCESS, TNG_FAILURE, TNG_CRITICAL
ctypedef enum tng_hash_mode: TNG_SKIP_HASH, TNG_USE_HASH
ctypedef enum tng_datatypes: TNG_CHAR_DATA, TNG_INT_DATA, TNG_FLOAT_DATA, \
    TNG_DOUBLE_DATA
ctypedef enum tng_particle_dependency: TNG_NON_PARTICLE_BLOCK_DATA, \
    TNG_PARTICLE_BLOCK_DATA
ctypedef enum tng_compression: TNG_UNCOMPRESSED, TNG_XTC_COMPRESSION, \
    TNG_TNG_COMPRESSION, TNG_GZIP_COMPRESSION
ctypedef enum tng_bool: TNG_FALSE, TNG_TRUE

cdef union data_values:
    double d
    float f
    int64_t i
    char * c

status_error_message = ['OK', 'Failure', 'Critical']


cdef extern from "string.h":
    size_t strlen(char * s)

cdef extern from "<stdio.h>" nogil:
    # Seek and tell with off_t
    int fseeko(FILE * , off_t, int)
    off_t ftello(FILE * )


# /** @defgroup def1 Standard non-trajectory blocks
#  *  Block IDs of standard non-trajectory blocks.
#  * @{
#  */
DEF TNG_GENERAL_INFO = 0x0000000000000000LL
DEF TNG_MOLECULES = 0x0000000000000001LL
DEF TNG_TRAJECTORY_FRAME_SET = 0x0000000000000002LL
DEF TNG_PARTICLE_MAPPING = 0x0000000000000003LL
# /** @} */

# /** @defgroup def2 Standard trajectory blocks
#  * Block IDs of standard trajectory blocks. Box shape and partial charges can
#  * be either trajectory blocks or non-trajectory blocks
#  * @{
#  */
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
# /** @} */

# /** @defgroup def3 GROMACS data block IDs
#  *  Block IDs of data blocks specific to GROMACS.
#  * @{
#  */
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

    tng_function_status tng_util_pos_read_range(
        const tng_trajectory * tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** positions,
        int64_t * stride_length) nogil

    tng_function_status tng_util_box_shape_read_range(
        const tng_trajectory * tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** box_shape,
        int64_t * stride_length) nogil

    tng_function_status tng_util_vel_read_range(
        const tng_trajectory * tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** velocities,
        int64_t * stride_length) nogil

    tng_function_status tng_util_force_read_range(
        const tng_trajectory * tng_data,
        const int64_t first_frame,
        const int64_t last_frame,
        float ** forces,
        int64_t * stride_length) nogil

    tng_function_status tng_util_pos_read(
        const tng_trajectory * tng_data,
        float ** positions,
        int64_t * stride_length) nogil

    tng_function_status tng_util_box_shape_read(
        const tng_trajectory * tng_data,
        float ** box_shape,
        int64_t * stride_length) nogil

    tng_function_status tng_util_vel_read(
        const tng_trajectory * tng_data,
        float ** velocities,
        int64_t * stride_length) nogil

    tng_function_status tng_util_force_read(
        const tng_trajectory * tng_data,
        float ** forces,
        int64_t * stride_length) nogil

    tng_function_status tng_util_time_of_frame_get(
        const tng_trajectory * tng_data,
        const int64_t frame_nr,
        double * time) nogil

    tng_function_status tng_data_vector_interval_get(
        const tng_trajectory * tng_data,
        const int64_t block_id,
        const int64_t start_frame_nr,
        const int64_t end_frame_nr,
        const char hash_mode,
        void ** values,
        int64_t * stride_length,
        int64_t * n_values_per_frame,
        char * type) nogil

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

    tng_function_status tng_data_get_stride_length(
        tng_trajectory * tng_data,
        int64_t block_id,
        int64_t frame,
        int64_t * stride_length) nogil

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

TNGFrame = namedtuple("TNGFrame", "positions velocities forces time step box")


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


@cython.final
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

    cdef inline void renew(self, int size) nogil:
        self.ptr = realloc(self.ptr, size)

    def __dealloc__(MemoryWrapper self):
        if self.ptr != NULL:
            free(self.ptr)


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

    # Extension class properties
    # @property
    # def a(self):
    #     return self._ptr.a if self._ptr is not NULL else None

    # @property
    # def b(self):
    #     return self._ptr.b if self._ptr is not NULL else None

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
        # _ptr.a = 0
        # _ptr.b = 0
        return TrajectoryWrapper.from_ptr(_ptr, owner=True)


cdef class TNGFileIterator:
    """File handle object for TNG files

    Supports use as a context manager ("with" blocks).

    """

    cdef tng_trajectory * _traj_p
    cdef TrajectoryWrapper _traj
    cdef readonly fname
    cdef str mode
    cdef bint debug
    cdef int is_open
    cdef int reached_eof
    cdef int64_t step

    cdef int64_t _n_frames
    cdef int64_t _n_particles
    cdef int64_t _n_frame_sets
    cdef float _distance_scale

    cdef int64_t _current_frame
    cdef int64_t _current_frame_set
    cdef int64_t _gcd  # greatest common divisor of data strides

    cdef dict   _frame_strides
    cdef dict   _n_data_frames

    # holds the current blocks at a trajectory timestep
    cdef TNGDataBlockHolder block_holder
    cdef TNGBlockTypes BLOCK_TYPES

    def __cinit__(self, fname, mode='r', debug=False):

        self._traj = TrajectoryWrapper.from_ptr(self._traj_p, owner=True)
        self.fname = fname
        self.debug = debug
        self._n_frames = -1
        self._n_particles = -1
        self._n_frame_sets = -1
        self._distance_scale = 0.0
        self._current_frame = -1
        self._current_frame_set = -1
        self._gcd = -1
        self._frame_strides = {}
        self._n_data_frames = {}

        # the mappings of block ids to block names and vice versa
        self.BLOCK_TYPES = TNGBlockTypes()

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

        stat = tng_num_frame_sets_get(self._traj._ptr, & self._n_frame_sets)
        if stat != TNG_SUCCESS:
            raise IOError("Number of trajectory frame sets cannot be read")

        cdef int64_t exponent
        stat = tng_distance_unit_exponential_get(self._traj._ptr, & exponent)
        if stat != TNG_SUCCESS:
            raise IOError("Distance exponent cannot be read")

        stat = self._get_frame_indicies()
        if stat != TNG_SUCCESS:
            raise IOError("Strides for each data block cannot be read")

        self._distance_scale = 10.0**(exponent+9)
        self.is_open = True
        self.step = 0
        self.reached_eof = False

    def _close(self):
        """Make sure the file handle is closed"""
        if self.is_open:
            tng_util_trajectory_close( & self._traj._ptr)
            self.is_open = False
            self._n_frames = -1

    @property
    def block_set(self):  # NOTE perhaps we should not expose this
        """Dictionary where keys are available block id
        and values are TngDataBlock instance"""
        return self.block_holder.block_set

    @property
    def block_ids(self):  # NOTE perhaps we should not expose this
        """List of block ids available at the current frame"""
        return list(self.block_holder.block_set.keys())

    @property
    def block_names(self):
        """List of block names available at the current frame (unordered)"""
        block_ids = list(self.block_holder.block_set.keys())
        return [self.BLOCK_TYPES.block_dictionary[id] for id in block_ids]

    def get_block_by_name(self, name):
        if self.block_holder.block_set.get(
                self.BLOCK_TYPES.block_id_dictionary[name]) == None:
            return None
        else:
            return self.block_holder.block_set.get(
                self.BLOCK_TYPES.block_id_dictionary[name])

    def get_block_values_by_name(self, name):
        if self.block_holder.block_set.get(
                self.BLOCK_TYPES.block_id_dictionary[name]) == None:
            return None
        else:
            return self.block_holder.block_set.get(
                self.BLOCK_TYPES.block_id_dictionary[name]).values

    @property
    def block_strides(self):
        return [(self.BLOCK_TYPES.block_dictionary[k], v)
                for k, v in self._frame_strides.items()]

    @property
    def pos(self):
        if self.block_holder.block_set.get(TNG_TRAJ_POSITIONS) == None:
            return None
        else:
            return self.block_holder.block_set.get(TNG_TRAJ_POSITIONS).values

    @property
    def vel(self):
        if self.block_holder.block_set.get(TNG_TRAJ_VELOCITIES) == None:
            return None
        else:
            return self.block_holder.block_set.get(TNG_TRAJ_VELOCITIES).values

    @property
    def frc(self):
        if self.block_holder.block_set.get(TNG_TRAJ_FORCES) == None:
            return None
        else:
            return self.block_holder.block_set.get(TNG_TRAJ_FORCES).values

    @property
    def box(self):
        if self.block_holder.block_set.get(TNG_TRAJ_BOX_SHAPE) == None:
            return None
        else:
            return self.block_holder.block_set.get(TNG_TRAJ_BOX_SHAPE).values

    def read_frame(self, frame):
        """Read a frame (integrator step) from the file,
           modifies the state of self.block_holder to contain
           the current blocks"""
        self.block_holder = TNGDataBlockHolder(debug=self.debug)

        # TODO fix this to whatever kind of iteration we want

        for block, stride in self._frame_strides.items():
            if frame % stride == 0:
                # read the frame if we are on stride
                self._read_single_frame(frame, block, self.block_holder)

        if self.debug:
            print(self.block_holder.block_set)
            for k, v in self.block_holder.block_set.items():
                print(k)
                print(v.values)

    # NOTE here we assume that the first frame has all the blocks
    #  that are present in the whole traj

    cdef tng_function_status _get_frame_indicies(self):
        """Gets the ids, strides and number of frames with
           actual data from the trajectory"""
        cdef int64_t step, n_blocks
        cdef int64_t nframes, stride_length
        cdef int64_t block_counter = 0
        cdef int64_t * block_ids = NULL

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
            # stride length for the block
            self._frame_strides[block_ids[i]] = stride_length
            # number of actual data frames for the block
            self._n_data_frames[block_ids[i]] = nframes

        # TODO we will use this if we want to instead iterate
        # over the greatest common divisor of the data strides
        self._gcd = gcd_list(list(self._frame_strides.values()))
        if self.debug:
            printf("greatest common divisor of strides %ld \n", self._gcd)

        return TNG_SUCCESS

    cdef void _read_single_frame(self, int64_t frame, int64_t block_id,
                                 TNGDataBlockHolder block_holder):
        """Read the current block of a given block id into a
           TNGDataBlock instance and pops that instance to the block holder"""

        if self.debug:
            print("READING FRAME {}  \n".format(frame))
        cdef block = TNGDataBlock(self._traj, frame, debug=self.debug)
        block.block_read(block_id)  # read the actual block
        # add the block to the block holder
        block_holder.add_block(block_id, block)

    def __len__(self):
        return self._n_frames

    def __iter__(self):
        self.close()
        self.open(self.fname, self.mode)
        return self

    def __next__(self):
        if self.reached_eof:
            raise StopIteration
        # return self

    def __getitem__(self, frame):
        cdef int64_t start, stop, step, i

        if isinstance(frame, numbers.Integral):
            if self.debug:
                print("slice is a number")
            return self.read_frame(frame)
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
                    yield self.read_frame(f)
            return listiter(frame)
        elif isinstance(frame, slice):
            start = frame.start if frame.start is not None else 0
            stop = frame.stop if frame.stop is not None else self._n_frames
            step = frame.step if frame.step is not None else 1

            def sliceiter(start, stop, step):
                for i in range(start, stop, step):
                    yield self.read_frame(i)
            return sliceiter(start, stop, step)
        else:
            raise TypeError("Trajectories must be an indexed using an integer,"
                            " slice or list of indices")


cdef class TNGDataBlockHolder:
    """Holds data blocks at the curent trajectory step"""

    cdef bint debug
    cdef int64_t _n_blocks
    cdef dict blocks  # TODO should we use a fixed size container?

    def __cinit__(self, bint debug=False):
        self.debug = debug
        self.blocks = {}

    cdef inline void add_block(self, int64_t block_id, TNGDataBlock block):
        self.blocks[block_id] = block  # add the block to the block dictionary

    @property
    def block_set(self):
        return self.blocks  # expose blocks to python layer


cdef class TNGDataBlock:
    """Contains the actual data in a tng_data_block
       and exposes it as a Numpy array"""

    cdef tng_trajectory * _traj
    cdef int64_t _frame
    cdef int64_t block_id
    cdef bint debug

    cdef int64_t step
    cdef double frame_time
    cdef double precision
    cdef int64_t n_values_per_frame, n_atoms
    cdef char[TNG_MAX_STR_LEN] block_name
    cdef MemoryWrapper _wrapper  # manages the numpy array lifetime
    cdef tng_function_status read_stat
    cdef np.ndarray values  # the final values as a numpy array

    cdef bint block_is_read

    def __cinit__(self, TrajectoryWrapper traj, int64_t frame,
                  bint debug=False):

        self._traj = traj._ptr
        self._frame = frame
        self.debug = debug
        self.block_id = -1

        self.step = -1
        self.frame_time = -1
        self.precision = -1
        self.n_values_per_frame = -1
        self.n_atoms = -1
        self.block_is_read = False

        # TODO allocs a single byte,
        # this can be changed if the signature of MemoryWrapper is changed
        self._wrapper = MemoryWrapper(1)

    def __dealloc__(self):
        pass

    def block_read(self, id):
        """Reads the block off file"""
        cdef tng_function_status stat
        self._refresh()
        stat = self._block_read(id)
        if stat != TNG_SUCCESS:
            raise IOError("Critical failure: block cannot be read")
        self._block_2d_numpy_cast(self.n_values_per_frame, self.n_atoms)
        self.block_is_read = True

    cdef void _refresh(self):
        """Refreshes the TngDataBlock instance so that we can
           read different blocks into the same instance"""
        self.block_id = -1
        self.step = -1
        self.frame_time = -1
        self.precision = -1
        self.n_values_per_frame = -1
        self.n_atoms = -1
        self.block_is_read = False
        # TODO this can be changed if the signature of MemoryWrapper is changed
        self._wrapper = MemoryWrapper(1)

    @property
    def values(self):
        """the values of the block"""
        if not self.block_is_read:
            raise IOError(
                'Values not available until block_read() has been called')
        return self.values

    cdef void _block_2d_numpy_cast(self, int64_t n_values_per_frame,
                                   int64_t n_atoms):
        """Casts the block to a 2D Numpy array whose lifetime is managed by
           an associated MemoryWrapper instance"""

        if self.debug:
            printf("CREATING NUMPY_ARRAY \n")
        if n_values_per_frame == -1 or n_atoms == -1:
            raise ValueError(
                "Array dimensions for block casting are not set correctly")
        cdef int nd = 2
        cdef int err
        cdef npy_intp dims[2]
        dims[0] = n_atoms
        dims[1] = n_values_per_frame
        self.values = PyArray_SimpleNewFromData(
            2, dims, NPY_DOUBLE, self._wrapper.ptr)
        Py_INCREF(self._wrapper)
        err = PyArray_SetBaseObject(self.values, self._wrapper)
        if err:
            raise ValueError("Array object cannot be created")
        if self.debug:
            print(self.values)

    cdef tng_function_status  _block_read(self, int64_t id):
        """Does the actual block reading"""
        self.block_id = id
        cdef tng_function_status read_stat
        with nogil:
            read_stat = self._get_data_next_frame(self.block_id,  & self.step,
                                                  & self.frame_time,
                                                  & self.n_values_per_frame,
                                                  & self.n_atoms,
                                                  & self.precision,
                                                  self.block_name,
                                                  self.debug)

        if self.debug:
            printf("block id %ld \n", self.block_id)
            printf("data block name %s \n", self.block_name)
            printf("n_values_per_frame %ld \n", self.n_values_per_frame)
            printf("n_atoms  %ld \n", self.n_atoms)

        return read_stat

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef tng_function_status _get_data_next_frame(self, int64_t block_id,
                                                  int64_t * step,
                                                  double * frame_time,
                                                  int64_t * n_values_per_frame,
                                                  int64_t * n_atoms,
                                                  double * prec,
                                                  char * block_name,
                                                  bint debug) nogil:
        """Gets the frame data off disk and into C level arrays"""
        cdef tng_function_status stat
        cdef char                datatype = -1
        cdef int64_t             codec_id
        cdef int             block_dependency
        cdef void * data = NULL
        cdef double              local_prec
        cdef int64_t             n_particles
        cdef int64_t             stride_length

        # Flag to indicate frame dependent data.
        cdef int TNG_FRAME_DEPENDENT = 1
        # Flag to indicate particle dependent data.
        cdef int  TNG_PARTICLE_DEPENDENT = 2
        stat = tng_data_block_name_get(
            self._traj, block_id, block_name, TNG_MAX_STR_LEN)
        if stat != TNG_SUCCESS:
            return TNG_CRITICAL

        # is this a particle dependent block?
        stat = tng_data_block_dependency_get(self._traj, block_id,
                                             & block_dependency)
        if stat != TNG_SUCCESS:
            return TNG_CRITICAL
        if debug:
            printf("BLOCK DEPS %d \n", block_dependency)

        if block_dependency & TNG_PARTICLE_DEPENDENT:  # bitwise & due to enums
            if debug:
                printf("reading particle data \n")
            tng_num_particles_get(self._traj, n_atoms)
            # read particle data off disk with hash checking
            stat = tng_gen_data_vector_interval_get(self._traj,
                                                    block_id,
                                                    TNG_TRUE,
                                                    self._frame,
                                                    self._frame,
                                                    TNG_USE_HASH,
                                                    & data,
                                                    & n_particles,
                                                    & stride_length,
                                                    n_values_per_frame,
                                                    & datatype)

        else:
            if debug:
                printf("reading NON particle data \n")
            n_atoms[0] = 1  # still used for some allocs
            # read non particle data off disk with hash checking
            stat = tng_gen_data_vector_interval_get(self._traj,
                                                    block_id,
                                                    TNG_FALSE,
                                                    self._frame,
                                                    self._frame,
                                                    TNG_USE_HASH,
                                                    & data,
                                                    NULL,
                                                    & stride_length,
                                                    n_values_per_frame,
                                                    & datatype)

        if stat != TNG_SUCCESS:
            printf(
                "WARNING: critical failure in tng_gen_data_vector_get \n")
            return TNG_CRITICAL

        stat = tng_data_block_num_values_per_frame_get(
            self._traj, block_id, n_values_per_frame)
        if stat == TNG_CRITICAL:
            printf(
                """WARNING: critical failure in
                tng_data_block_num_values_per_frame_get \n""")
            return TNG_CRITICAL

        # renew the MemoryWrapper instance to have the right size
        # to hold the data that has come off disk and will be cast to double*
        self._wrapper.renew(
            sizeof(double) * n_values_per_frame[0] * n_atoms[0])

        _wrapped_values = <double*> self._wrapper.ptr

        if self.debug:
            printf("realloc values array to %ld doubles and %ld bits long \n",
                   n_values_per_frame[0] * n_atoms[0], n_values_per_frame[0]
                   * n_atoms[0]*sizeof(double))

        # convert the data that was read off disk into an array of doubles
        self.convert_to_double_arr(
            data, _wrapped_values, n_atoms[0], n_values_per_frame[0],
            datatype, debug)

        # get the compression of the current frame
        stat = tng_util_frame_current_compression_get(self._traj,
                                                      block_id,
                                                      & codec_id,
                                                      & local_prec)
        if stat == TNG_CRITICAL:
            printf(
                """WARNING: critical failure in
                 tng_util_frame_current_compression_get \n""")
            return TNG_CRITICAL

        if codec_id != TNG_TNG_COMPRESSION:
            prec[0] = -1.0
        else:
            prec[0] = local_prec

        # free local data
        free(data)
        return TNG_SUCCESS

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef void convert_to_double_arr(self,
                                    void * source,
                                    double * to,
                                    const int n_atoms,
                                    const int n_vals,
                                    const char datatype,
                                    bint debug) nogil:

        """ Convert a void array to an array of doubles

            NOTES:
            - this should use memcpy but doesn't yet
            - is this likely to be portable?.
            - a lot of this is a bit redundant but could be used to
              differntiate casts to numpy arrays etc in the future.

        """
        cdef int i, j

        if datatype == TNG_FLOAT_DATA:
            for i in range(n_atoms):
                for j in range(n_vals):
                    to[i*n_vals + j ] = ( < float*>source)[i * n_vals + j]  
            #memcpy(to,  source, n_vals * sizeof(float) * n_atoms)
            if debug:
                printf("TNG_FLOAT \n")

        elif datatype == TNG_INT_DATA:
            for i in range(n_atoms):
                for j in range(n_vals):
                    to[i*n_vals + j ] = ( < int64_t*>source)[i * n_vals + j] 
            #memcpy(to, source, n_vals * sizeof(int64_t) * n_atoms)
            if debug:
                printf("TNG_INT \n")

        elif datatype == TNG_DOUBLE_DATA:
            for i in range(n_atoms):
                for j in range(n_vals):
                    to[i*n_vals + j ] = ( < double*>source)[i * n_vals + j] 
            #memcpy(to, source, n_vals * sizeof(double) * n_atoms)
            if debug:
                printf("TNG_DOUBLE\n")

        elif datatype == TNG_CHAR_DATA:
            # NOTE not implemented in TNG library either
            printf("WARNING: char data reading is not implemented \n")

        else:
            printf("WARNING: block data type %d not understood \n", datatype)


cdef class TNGBlockTypes:
    """Block type dictionary

    """
    # KEY = block_id (int64_t) # VAL = TNG_BLOCK_NAME (str)
    cdef dict block_dictionary
    # KEY = TNG_BLOCK_NAME (str)  # VAL = block_id (int64_t)
    cdef dict block_id_dictionary

    def __cinit__(self):
        self.block_dictionary = {}
        self._open()

    cdef void _open(self):
        # group 1
        self.block_dictionary[TNG_GENERAL_INFO] = \
            "TNG_GENERAL_INFO"
        self.block_dictionary[TNG_MOLECULES] = \
            "TNG_MOLECULES"
        self.block_dictionary[TNG_TRAJECTORY_FRAME_SET] = \
            "TNG_TRAJECTORY_FRAME_SET"
        self.block_dictionary[TNG_PARTICLE_MAPPING] = \
            "TNG_PARTICLE_MAPPING"

        # group 2
        self.block_dictionary[TNG_TRAJ_BOX_SHAPE] = \
            "TNG_TRAJ_BOX_SHAPE"
        self.block_dictionary[TNG_TRAJ_POSITIONS] = \
            "TNG_TRAJ_POSITIONS"
        self.block_dictionary[TNG_TRAJ_VELOCITIES] = \
            "TNG_TRAJ_VELOCITIES"
        self.block_dictionary[TNG_TRAJ_FORCES] = \
            "TNG_TRAJ_FORCES"
        self.block_dictionary[TNG_TRAJ_PARTIAL_CHARGES] = \
            "TNG_TRAJ_PARTIAL_CHARGES"
        self.block_dictionary[TNG_TRAJ_FORMAL_CHARGES] = \
            "TNG_TRAJ_FORMAL_CHARGES"
        self.block_dictionary[TNG_TRAJ_B_FACTORS] = \
            "TNG_TRAJ_B_FACTORS"
        self.block_dictionary[TNG_TRAJ_ANISOTROPIC_B_FACTORS] = \
            "TNG_TRAJ_ANISOTROPIC_B_FACTORS"
        self.block_dictionary[TNG_TRAJ_OCCUPANCY] = \
            "TNG_TRAJ_OCCUPANCY"
        self.block_dictionary[TNG_TRAJ_GENERAL_COMMENTS] = \
            "TNG_TRAJ_GENERAL_COMMENTS"
        self.block_dictionary[TNG_TRAJ_MASSES] = \
            "TNG_TRAJ_MASSES"

        # group 3
        self.block_dictionary[TNG_GMX_LAMBDA] = \
            "TNG_GMX_LAMBDA"
        self.block_dictionary[TNG_GMX_ENERGY_ANGLE] = \
            "TNG_GMX_ENERGY_ANGLE"
        self.block_dictionary[TNG_GMX_ENERGY_RYCKAERT_BELL] = \
            "TNG_GMX_ENERGY_RYCKAERT_BELL"
        self.block_dictionary[TNG_GMX_ENERGY_LJ_14] = \
            "TNG_GMX_ENERGY_LJ_14"
        self.block_dictionary[TNG_GMX_ENERGY_COULOMB_14] = \
            "TNG_GMX_ENERGY_COULOMB_14"
        self.block_dictionary[TNG_GMX_ENERGY_LJ_SR] = \
            "TNG_GMX_ENERGY_LJ_SR"
        self.block_dictionary[TNG_GMX_ENERGY_COULOMB_SR] = \
            "TNG_GMX_ENERGY_COULOMB_SR"
        self.block_dictionary[TNG_GMX_ENERGY_COUL_RECIP] = \
            "TNG_GMX_ENERGY_COUL_RECIP"
        self.block_dictionary[TNG_GMX_ENERGY_POTENTIAL] = \
            "TNG_GMX_ENERGY_POTENTIAL"
        self.block_dictionary[TNG_GMX_ENERGY_KINETIC_EN] = \
            "TNG_GMX_ENERGY_KINETIC_EN"
        self.block_dictionary[TNG_GMX_ENERGY_TOTAL_ENERGY] = \
            "TNG_GMX_ENERGY_TOTAL_ENERGY"
        self.block_dictionary[TNG_GMX_ENERGY_TEMPERATURE] = \
            "TNG_GMX_ENERGY_TEMPERATURE"
        self.block_dictionary[TNG_GMX_ENERGY_PRESSURE] = \
            "TNG_GMX_ENERGY_PRESSURE"
        self.block_dictionary[TNG_GMX_ENERGY_CONSTR_RMSD] = \
            "TNG_GMX_ENERGY_CONSTR_RMSD"
        self.block_dictionary[TNG_GMX_ENERGY_CONSTR2_RMSD] = \
            "TNG_GMX_ENERGY_CONSTR2_RMSD"
        self.block_dictionary[TNG_GMX_ENERGY_BOX_X] = \
            "TNG_GMX_ENERGY_BOX_X"
        self.block_dictionary[TNG_GMX_ENERGY_BOX_Y] = \
            "TNG_GMX_ENERGY_BOX_Y"
        self.block_dictionary[TNG_GMX_ENERGY_BOX_Z] = \
            "TNG_GMX_ENERGY_BOX_Z"
        self.block_dictionary[TNG_GMX_ENERGY_BOXXX] = \
            "TNG_GMX_ENERGY_BOXXX"
        self.block_dictionary[TNG_GMX_ENERGY_BOXYY] = \
            "TNG_GMX_ENERGY_BOXYY"
        self.block_dictionary[TNG_GMX_ENERGY_BOXZZ] = \
            "TNG_GMX_ENERGY_BOXZZ"
        self.block_dictionary[TNG_GMX_ENERGY_BOXYX] = \
            "TNG_GMX_ENERGY_BOXYX"
        self.block_dictionary[TNG_GMX_ENERGY_BOXZX] = \
            "TNG_GMX_ENERGY_BOXZX"
        self.block_dictionary[TNG_GMX_ENERGY_BOXZY] = \
            "TNG_GMX_ENERGY_BOXZY"
        self.block_dictionary[TNG_GMX_ENERGY_BOXVELXX] = \
            "TNG_GMX_ENERGY_BOXVELXX"
        self.block_dictionary[TNG_GMX_ENERGY_BOXVELYY] = \
            "TNG_GMX_ENERGY_BOXVELYY"
        self.block_dictionary[TNG_GMX_ENERGY_BOXVELZZ] = \
            "TNG_GMX_ENERGY_BOXVELZZ"
        self.block_dictionary[TNG_GMX_ENERGY_BOXVELYX] = \
            "TNG_GMX_ENERGY_BOXVELYX"
        self.block_dictionary[TNG_GMX_ENERGY_BOXVELZX] = \
            "TNG_GMX_ENERGY_BOXVELZX"
        self.block_dictionary[TNG_GMX_ENERGY_BOXVELZY] = \
            "TNG_GMX_ENERGY_BOXVELZY"
        self.block_dictionary[TNG_GMX_ENERGY_VOLUME] = \
            "TNG_GMX_ENERGY_VOLUME"
        self.block_dictionary[TNG_GMX_ENERGY_DENSITY] = \
            "TNG_GMX_ENERGY_DENSITY"
        self.block_dictionary[TNG_GMX_ENERGY_PV] = \
            "TNG_GMX_ENERGY_PV"
        self.block_dictionary[TNG_GMX_ENERGY_ENTHALPY] = \
            "TNG_GMX_ENERGY_ENTHALPY"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_XX] = \
            "TNG_GMX_ENERGY_VIR_XX"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_XY] = \
            "TNG_GMX_ENERGY_VIR_XY"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_XZ] = \
            "TNG_GMX_ENERGY_VIR_XZ"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_YX] = \
            "TNG_GMX_ENERGY_VIR_YX"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_YY] = \
            "TNG_GMX_ENERGY_VIR_YY"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_YZ] = \
            "TNG_GMX_ENERGY_VIR_YZ"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_ZX] = \
            "TNG_GMX_ENERGY_VIR_ZX"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_ZY] = \
            "TNG_GMX_ENERGY_VIR_ZY"
        self.block_dictionary[TNG_GMX_ENERGY_VIR_ZZ] = \
            "TNG_GMX_ENERGY_VIR_ZZ"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_XX] = \
            "TNG_GMX_ENERGY_SHAKEVIR_XX"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_XY] = \
            "TNG_GMX_ENERGY_SHAKEVIR_XY"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_XZ] = \
            "TNG_GMX_ENERGY_SHAKEVIR_XZ"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_YX] = \
            "TNG_GMX_ENERGY_SHAKEVIR_YX"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_YY] = \
            "TNG_GMX_ENERGY_SHAKEVIR_YY"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_YZ] = \
            "TNG_GMX_ENERGY_SHAKEVIR_YZ"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_ZX] = \
            "TNG_GMX_ENERGY_SHAKEVIR_ZX"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_ZY] = \
            "TNG_GMX_ENERGY_SHAKEVIR_ZY"
        self.block_dictionary[TNG_GMX_ENERGY_SHAKEVIR_ZZ] = \
            "TNG_GMX_ENERGY_SHAKEVIR_ZZ"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_XX] = \
            "TNG_GMX_ENERGY_FORCEVIR_XX"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_XY] = \
            "TNG_GMX_ENERGY_FORCEVIR_XY"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_XZ] = \
            "TNG_GMX_ENERGY_FORCEVIR_XZ"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_YX] = \
            "TNG_GMX_ENERGY_FORCEVIR_YX"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_YY] = \
            "TNG_GMX_ENERGY_FORCEVIR_YY"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_YZ] = \
            "TNG_GMX_ENERGY_FORCEVIR_YZ"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_ZX] = \
            "TNG_GMX_ENERGY_FORCEVIR_ZX"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_ZY] = \
            "TNG_GMX_ENERGY_FORCEVIR_ZY"
        self.block_dictionary[TNG_GMX_ENERGY_FORCEVIR_ZZ] = \
            "TNG_GMX_ENERGY_FORCEVIR_ZZ"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_XX] = \
            "TNG_GMX_ENERGY_PRES_XX"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_XY] = \
            "TNG_GMX_ENERGY_PRES_XY"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_XZ] = \
            "TNG_GMX_ENERGY_PRES_XZ"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_YX] = \
            "TNG_GMX_ENERGY_PRES_YX"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_YY] = \
            "TNG_GMX_ENERGY_PRES_YY"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_YZ] = \
            "TNG_GMX_ENERGY_PRES_YZ"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_ZX] = \
            "TNG_GMX_ENERGY_PRES_ZX"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_ZY] = \
            "TNG_GMX_ENERGY_PRES_ZY"
        self.block_dictionary[TNG_GMX_ENERGY_PRES_ZZ] = \
            "TNG_GMX_ENERGY_PRES_ZZ"
        self.block_dictionary[TNG_GMX_ENERGY_SURFXSURFTEN] = \
            "TNG_GMX_ENERGY_SURFXSURFTEN"
        self.block_dictionary[TNG_GMX_ENERGY_MUX] = \
            "TNG_GMX_ENERGY_MUX"
        self.block_dictionary[TNG_GMX_ENERGY_MUY] = \
            "TNG_GMX_ENERGY_MUY"
        self.block_dictionary[TNG_GMX_ENERGY_MUZ] = \
            "TNG_GMX_ENERGY_MUZ"
        self.block_dictionary[TNG_GMX_ENERGY_VCOS] = \
            "TNG_GMX_ENERGY_VCOS"
        self.block_dictionary[TNG_GMX_ENERGY_VISC] = \
            "TNG_GMX_ENERGY_VISC"
        self.block_dictionary[TNG_GMX_ENERGY_BAROSTAT] = \
            "TNG_GMX_ENERGY_BAROSTAT"
        self.block_dictionary[TNG_GMX_ENERGY_T_SYSTEM] = \
            "TNG_GMX_ENERGY_T_SYSTEM"
        self.block_dictionary[TNG_GMX_ENERGY_LAMB_SYSTEM] = \
            "TNG_GMX_ENERGY_LAMB_SYSTEM"
        self.block_dictionary[TNG_GMX_SELECTION_GROUP_NAMES] = \
            "TNG_GMX_SELECTION_GROUP_NAMES"
        self.block_dictionary[TNG_GMX_ATOM_SELECTION_GROUP] = \
            "TNG_GMX_ATOM_SELECTION_GROUP"

        # reverse the mapping
        self.block_id_dictionary = {v: k for k,
                                    v in self.block_dictionary.items()}

    @property
    def block_dictionary(self):
        return self.block_dictionary

    @property
    def block_id_dictionary(self):
        return self.block_id_dictionary


cdef class TNGFile:
    """File handle object for TNG files

    Supports use as a context manager ("with" blocks).
    """
    cdef tng_trajectory * _traj
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
        cdef np.ndarray[ndim = 2, dtype = np.float32_t, mode = 'c'] box = \
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
