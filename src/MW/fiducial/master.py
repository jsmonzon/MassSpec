import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "merger_crit": -2,
    "fesc": 0.2,
    "scatter": True,
    "mass_cut": 1e9}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

scripts = ["run_S0.py", "run_S15.py", "run_S30.py"]

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", script])
