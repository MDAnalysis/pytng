import pytng

file_iterator = pytng.TNGFileIterator("./reference_files/water_uncompressed_vels_forces.tng", mode="r")
file_iterator.spool2()