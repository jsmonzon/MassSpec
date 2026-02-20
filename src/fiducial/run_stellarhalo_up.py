import numpy as np
import sys
import os
import multiprocessing as mp
import h5py
import json

with open("config.json", "r") as f:
    config = json.load(f)

with open(config["datafiles"], "r") as f:
    files = f.read().splitlines()

# Define paths based on location
location = config["location"]
if location == "server":
    parentdir = "/home/jsm99/SatGen/mcmc/src/"

sys.path.insert(0, parentdir)
import jsm_visualize

hf_prefix = config["datafiles"].split(".txt")[0]
save_name = "/home/jsm99/data/mass_spec/DF_up/"+hf_prefix

Ntrees = len(files)
print("Reading in", Ntrees, "trees")
print("---------")
print("saving to", save_name)

def process_file(file_i):
    try:
        tree_i = jsm_visualize.Arborist(file=file_i, merger_crit=config["merger_crit"], fesc=config["fesc"], scatter=config["scatter"], verbose=False)
        tree_i.plant_roots()
        tree_i.water_roots()
        tree_i.dendrochronology(mass_threshold=config["mass_cut"])
        tree_i.canopy(mass_threshold=config["mass_cut"])
        return tree_i.write_out_abundance()
    
    except Exception as e:
        print(f"Error processing {file_i}: {e}")
        return None
    
if __name__ == "__main__":

    with mp.Pool(processes=config["N_cpus"]) as pool:
        results = pool.map(process_file, files) 

    # Filter out None results safely
    valid_entries = [entry for entry in results if isinstance(entry, dict) and "tree_index" in entry]

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