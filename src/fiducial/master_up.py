import subprocess
import json
import numpy as np
import os

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "merger_crit": -2,
    "fesc": 0.2,
    "scatter": True,
    "mass_cut": 6.75e9}

range = np.arange(11, 14.2, 0.2) # the discrete mass bins
mass_bins = np.char.mod('%.1f', range) # conver to strings
datadir = "/../../netb/vdbosch/jsm99/data/mass_spec/DF_up/" # all the files are saved here!

h5_blocks = [] # a list of all the text files

for bin in mass_bins:
    file_list = bin+"_files.txt" # add the mass bin to the list
    h5_blocks.append(file_list)

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

    subprocess.run(["python", "run_stellarhalo_up.py"])