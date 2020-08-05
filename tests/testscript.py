import pytng

file_iterator = pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r", debug=False)
file_iterator.read_frame(0)
file_iterator.read_frame(10000)
print(file_iterator.block_names)
print(file_iterator.block_set)
print(file_iterator.block_ids)
print(file_iterator.pos)
print(file_iterator.box)
print(file_iterator.get_block_by_name("TNG_GMX_LAMBDA"))
print(file_iterator.vel)
