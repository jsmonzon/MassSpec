import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 12,
    "merger_crit": -1,
    "fesc": 0.2,
    "scatter": True}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

scripts = ["identitical.py", "jiang.py", "zentner.py"]

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", script])
