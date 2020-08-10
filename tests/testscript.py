import pytng

with pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r", debug=False) as tng:

    tng[0]
    print(tng.block_strides)
    tng[5000]

    l = [0, 1, 2, 3]

    for ts in tng[l]:
        print(tng.pos)

# for ts in file_iterator[0:100000:5000]:
#     print(file_iterator.pos)


# file_iterator.read_frame(10000) # this integrator timestep has data
# print(file_iterator.block_names)
# print(file_iterator.pos)
# print(file_iterator.box)
# print(file_iterator.vel)
# print(file_iterator.frc)
# print(file_iterator.get_block_values_by_name("TNG_GMX_LAMBDA")) # this retreives the block values 
# print(file_iterator.get_block_by_name("TNG_GMX_LAMBDA")) # this retreives the block object itself


# file_iterator.read_frame(100001) # integrator timestep does not have data, thereby returning empty
# print(file_iterator.block_names)
# print(file_iterator.pos)
# print(file_iterator.box)
# print(file_iterator.vel)
# print(file_iterator.frc)
# print(file_iterator.get_block_values_by_name("TNG_GMX_LAMBDA"))
# print(file_iterator.get_block_by_name("TNG_GMX_LAMBDA"))