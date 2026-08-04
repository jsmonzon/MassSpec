import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "seed": 42,
    "Nparticles": 1e7}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

mass_cut = [1.00000000e+09, 3.16227766e+09, 1.00000000e+10, 3.16227766e+10, 1.00000000e+11]

for cut in mass_cut:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Running with log mass_cut = {cut}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    config["mass_cut"] = cut

    with open("config.json", "w") as f:
        json.dump(config, f)

    subprocess.run(["python", "run_S0.py"])
