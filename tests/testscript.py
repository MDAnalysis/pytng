import pytng

file_iterator = pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r")
file_iterator.spool2()