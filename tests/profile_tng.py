import pstats, cProfile

import pytng

ctx = """
file_iterator = pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r", debug=False)
for i in range(101):
    file_iterator.read_frame(5000*i)
"""


cProfile.runctx(ctx, globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()