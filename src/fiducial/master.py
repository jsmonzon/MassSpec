import subprocess
import json
import numpy as np
import os

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "mass_cut": 7.75e10}

mass_bin_edges = np.arange(12.6, 14.2, 0.2)  # the discrete mass bins
mass_bins = np.char.mod('%.1f', mass_bin_edges)  # convert to strings

DF_tags = ["DF_01", "DF_1", "DF_2", "DF_3", "DF_4", "DF_5", "DF_10"]

for DF_tag in DF_tags:
    datadir = f"/../../netb/vdbosch/jsm99/data/mass_spec_zhao/{DF_tag}/"  # per-DF_tag source dir
    config["DF_tag"] = DF_tag

    h5_blocks = []  # a list of all the text files for this DF_tag
    for bin in mass_bins:
        # include DF_tag in the filename so file lists don't collide across DF_tags
        file_list = f"{DF_tag}_{bin}_files.txt"
        h5_blocks.append(file_list)
        if os.path.exists(file_list):
            os.remove(file_list)  # clear any stale list from a previous run before appending fresh entries
        with open(file_list, "a") as f:
            for filename in os.listdir(datadir):
                if bin in filename and filename.endswith('evo.npz'):  # save all of the files to this block
                    writeout = datadir + filename
                    f.write(writeout + "\n")

    for h5 in h5_blocks:
        config["datafiles"] = h5
        # Write the configuration to a JSON file
        with open("config.json", "w") as f:
            json.dump(config, f)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"%%%  DF_tag = {DF_tag}  |  {h5}")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        subprocess.run(["python", "run_abundance.py"])