import subprocess
import json
import numpy as np

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "seed": 42,
    "fixed_c": 10,
    "mass_cut": 1e9}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

Nparticles = [1e4, 1e5, 1e6, 1e7]

for Npart in Nparticles:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Running with log Nparticles = {np.log10(Npart)}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    config["Nparticles"] = Npart

    with open("config.json", "w") as f:
        json.dump(config, f)

    subprocess.run(["python", "run_S0.py"])
