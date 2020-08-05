import pytng

file_iterator = pytng.TNGFileIterator("./reference_files/water_uncompressed_vels_forces.tng", mode="r", debug=False)
file_iterator.read_frame(0)
file_iterator.read_frame(10000) # this integrator timestep has data
print(file_iterator.block_names)
print(file_iterator.pos)
print(file_iterator.box)
print(file_iterator.vel)
print(file_iterator.frc)
print(file_iterator.get_block_values_by_name("TNG_GMX_LAMBDA"))
print(file_iterator.get_block_by_name("TNG_GMX_LAMBDA"))


file_iterator.read_frame(100001) # this one does not, thereby returning empty
print(file_iterator.block_names)
print(file_iterator.pos)
print(file_iterator.box)
print(file_iterator.vel)
print(file_iterator.frc)
print(file_iterator.get_block_values_by_name("TNG_GMX_LAMBDA"))
print(file_iterator.get_block_by_name("TNG_GMX_LAMBDA"))