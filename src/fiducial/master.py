import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 8,
    "merger_crit": -2,
    "fesc": 0.2,
    "scatter": True}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

scripts = ["run_s0_fid.py", "run_s0_down.py", "run_s0_up.py", "run_s0_off.py"]

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", script])
