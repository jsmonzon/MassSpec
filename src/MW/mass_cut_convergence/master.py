import subprocess
import json
import numpy as np

config = {
    "location": "server",
    "N_cpus": 16,
    "merger_crit": -2,
    "fesc": 0.2,
    "scatter": True,
}

# Define your mass cut grid
mass_cut_grid = np.logspace(-4, 0, 11)[0:-1] * 1e12

for mass_cut in mass_cut_grid:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Running with mass_cut = {mass_cut:.2e}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    config["mass_cut"] = mass_cut

    with open("config.json", "w") as f:
        json.dump(config, f)

    subprocess.run(["python", "run_worker.py"])