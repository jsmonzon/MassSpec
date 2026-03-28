import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "merger_crit": -2,
    "fesc": 0.2,
    "scatter": True,
    "mass_cut": 6.75e9}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

scripts = ["run_fid.py", "run_alpha_down.py", "run_alpha_up.py", "run_DF_up.py"]

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", script])
