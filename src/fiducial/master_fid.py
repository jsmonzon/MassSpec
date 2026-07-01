import subprocess
import json
import numpy as np
import os

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "mass_cut": 7.75e10}

mass_bin_edges = np.arange(12.4, 14.2, 0.2) # the discrete mass bins
mass_bins = np.char.mod('%.1f', mass_bin_edges) # conver to strings
datadir = "/../../netb/vdbosch/jsm99/data/mass_spec_vdb/DF_fid/" # all the files are saved here!

h5_blocks = [] # a list of all the text files

for bin in mass_bins:
    file_list = bin+"_files.txt" # add the mass bin to the list
    h5_blocks.append(file_list)

    if os.path.exists(file_list):
        os.remove(file_list)  # clear any stale list from a previous run before appending fresh entries

    with open(file_list, "a") as f:
        for filename in os.listdir(datadir):
            if bin in filename and filename.endswith('evo.npz'): #save all of the files to this block
                writeout = datadir+filename
                f.write(writeout + "\n")

for h5 in h5_blocks:

    config["datafiles"] = h5 

    # Write the configuration to a JSON file
    with open("config.json", "w") as f:
        json.dump(config, f)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", "run_stellarhalo_fid.py"])