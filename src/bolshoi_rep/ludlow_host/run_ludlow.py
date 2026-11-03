"""
Step 2 of the Ludlow host test (see master.py): same per-tree abundance
measurement and h5 layout as the fid.h5 run -- Tree_Reader(mass_threshold)
-> compute_concentration(rng, c_true) -> write_out_abundance(), one group per
tree_index -- on the Ludlow-arm evolved trees.

compute_concentration(c_true="zhao") uses the tree's own host c, which for
these trees is the Ludlow c, so ProcessH5(..., conctype=None) on the output
gives logc = log10(Ludlow host c). (The h5's "ludlow_c" field is Tree_Reader's
older z=0 Ludlow estimate and is NOT the c used in the evolution.)
"""
import numpy as np
import sys
import os
import time
import multiprocessing as mp
import h5py
import json

with open("config.json", "r") as f:
    config = json.load(f)

parentdir = os.path.join(config["satgen_dir"], "mcmc", "src")
datadir = config["out_dir"]
save_name = config["save_name"]

sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.join(config["satgen_dir"], "src"))
import jsm_stellarhalo

files = sorted(os.path.join(datadir, filename) for filename in os.listdir(datadir)
               if filename.startswith('tree') and filename.endswith('evo.npz'))

Ntrees = len(files)


def process_file(file_i):
    try:
        tree_i = jsm_stellarhalo.Tree_Reader(file=file_i, mass_threshold=config["mass_cut"], verbose=False)
        tree_i.compute_concentration(rng=np.random.default_rng(int(config["rng_seed"])), c_true=config["c_true"])
        return tree_i.write_out_abundance()

    except Exception as e:
        print(f"Error processing {file_i}: {e}", flush=True)
        return None


if __name__ == "__main__":

    print(f"processing {Ntrees} evolved trees in {datadir} -> {save_name}.h5", flush=True)
    t_start = time.time()
    results = []
    with mp.Pool(processes=config["N_cpus"]) as pool:
        for i, entry in enumerate(pool.imap_unordered(process_file, files), 1):
            results.append(entry)
            if i % 100 == 0 or i == Ntrees:
                print(f"--- progress: {i}/{Ntrees} ({(time.time() - t_start) / 60:.1f} min elapsed) ---", flush=True)

    # Filter out None results safely
    valid_entries = [entry for entry in results if isinstance(entry, dict) and "tree_index" in entry]
    print(f"{len(valid_entries)} processed, {Ntrees - len(valid_entries)} errors", flush=True)

    dict_dict = {
        entry["tree_index"]: {k: v for k, v in entry.items() if k != "tree_index"}
        for entry in valid_entries
    }

    with h5py.File(f"{save_name}.h5", "w") as f: # saving the dictionary for the surviving population!
        for sim_name, attributes in dict_dict.items():
            sim_group = f.create_group(sim_name)  # Create a group for each simulation

            for attr_name, data in attributes.items():
                if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
                    sim_group.create_dataset(attr_name, data=data)  # no compression
                else:
                    sim_group.create_dataset(attr_name, data=data, compression="gzip", compression_opts=9)
    print(f"saved {save_name}.h5")
