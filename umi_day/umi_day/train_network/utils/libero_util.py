import os
from umi_day.train_network.env_runner.libero_bddl_mapping import bddl_path

def hdf5_to_bddl(hdf5_path):
    libero_split = os.path.basename(os.path.dirname(hdf5_path))
    hdf5_name = os.path.basename(hdf5_path)
    bddl_name = hdf5_name.replace("_demo.hdf5", ".bddl")
    cur_bddl_path = os.path.join(bddl_path, libero_split, bddl_name)
    return cur_bddl_path
