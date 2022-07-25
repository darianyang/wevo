"""
Check the total simulation time of a west.h5 file.
"""

import numpy as np
import h5py

def check_agg_time(h5, last_iter=None):
    f = h5py.File(h5, "r")
    particles = f["summary"]["n_particles"]
    if last_iter:
        print(np.sum(particles[:last_iter]))
    else:
        print(np.sum(particles))

check_agg_time("west_default100.h5", last_iter=63)
check_agg_time("21i/west.h5")