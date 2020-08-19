import pstats
import cProfile
import numpy as np
import pytng

ctx = """
with pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r", debug=False) as tng:
    for i in range(0,500000,5000):
        tng.read_step(i)
        positions = np.zeros((1000,3), dtype=np.float32)
        box = np.zeros((1,9), dtype=np.float32)
        lmbda = np.zeros((1,1), dtype=np.float32)
        tng.current_integrator_step.get_blockid(268435457, positions)
        tng.current_integrator_step.get_blockid(1152921504875282432,lmbda)
        tng.current_integrator_step.get_blockid(268435456, box)

"""


cProfile.runctx(ctx, globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
