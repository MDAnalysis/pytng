import pytng

file_iterator = pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r", debug=False)
file_iterator.read_frame(0)
file_iterator.read_frame(10000)
print(file_iterator.block_set)
print(file_iterator.block_ids)
print(file_iterator.block_set[268435457].values)
