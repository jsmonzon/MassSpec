import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "merger_crit": -2,
    "fesc": 0.2,
    "scatter": True,
    "mass_cut": 1.2*10**8}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

scripts = []

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", script])
