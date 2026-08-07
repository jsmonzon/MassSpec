import subprocess
import json
import numpy as np

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "seed": 42,
    "mass_cut": 7.75e10}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

cmodels = ["zhao", "ludlow"]

for c in cmodels:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Running on the "+c+" model")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    config["cmodel"] = c

    with open("config.json", "w") as f:
        json.dump(config, f)

    subprocess.run(["python", "run_sample.py"])
