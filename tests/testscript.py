import pytng

file_iterator = pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r")
file_iterator.read_frame_indicies()
file_iterator.read_all_frames()