import subprocess
import json
import numpy as np
import os

mass_cut_configs = [
    {"mass_cut": 7.5e9,  "alias": "N50"},
    {"mass_cut": 7.5e10, "alias": "N500"},
    {"mass_cut": 1.5e11, "alias": "N1000"},
]

config = {
    "location": "server",
    "N_cpus": 16,
    "merger_crit": -2,
    "fesc": 0.2,
    "scatter": True,
    "mass_cut": None}  # will be set in loop

range = np.arange(11, 14.2, 0.2)
mass_bins = np.char.mod('%.1f', range)
datadir = "/../../netb/vdbosch/jsm99/data/mass_spec/DF_fid/"

# ---create h5 blocks once
h5_blocks = []
for bin in mass_bins:
    file_list = bin+"_files.txt"
    h5_blocks.append(file_list)

    with open(file_list, "a") as f:
        for filename in os.listdir(datadir):
            if bin in filename and filename.endswith('evo.npz'):
                writeout = datadir+filename
                f.write(writeout + "\n")

# ---now loop over mass cuts, reusing the same h5 blocks
for mc in mass_cut_configs:
    config["mass_cut"] = mc["mass_cut"]
    config["alias"]    = mc["alias"]

    for h5 in h5_blocks:
        config["datafiles"] = h5

        with open("config.json", "w") as f:
            json.dump(config, f)

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"alias = {mc['alias']} | mass_cut = {mc['mass_cut']:.2e} | h5 block = {h5}")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        subprocess.run(["python", "run_stellarhalo_fid.py"])