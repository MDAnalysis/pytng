import pytng
import numpy as np


positions = np.zeros((1000, 3), dtype=np.float32)
box = np.zeros((1, 9), dtype=np.float32)
lmbda = np.zeros((1, 1), dtype=np.float32)

with pytng.TNGFileIterator("./reference_files/argon_npt_compressed.tng", mode="r", debug=False) as tng:
    print(tng.block_ids)
    for i in range(0, len(tng), 5000):
        tng.read_step(i)
        tng.current_integrator_step.get_pos(positions)
        tng.current_integrator_step.get_blockid(1152921504875282432, lmbda)
        tng.current_integrator_step.get_box(box)
        time = tng.current_integrator_step.get_time()
        print(positions)
        print(lmbda)
        print(box)
        print(time)
