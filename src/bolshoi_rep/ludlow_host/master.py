"""
Ludlow-vs-Zhao HOST concentration test on the Bolshoi-Rep S13.2 sample
(1517 trees, host logMvir in [13.1, 13.3]).

The existing fiducial run (bolshoi_rep/fid/, -> fid.h5) is the "zhao" arm.
This directory produces the "ludlow" arm: the SAME unevolved trees, with only
the host row of the concentration array replaced on the fly by the
Ludlow+2016 c(z) (full collapsed mass history, f=0.02, C=400; SatGen
ludlow.concentration_history_Ludlow2016), evolved with SatGen's own
jsm_SubEvo.evolve_tree at its fiducial settings. Subhalo concentrations stay
Zhao. No modified unevolved trees are written to disk.

Steps (run in order by this script):
  1. run_evolve_ludlow.py  unevolved tree -> Ludlow host c(z) -> evolve ->
                           <out_dir>/tree_<logM>_<i>_evo.npz
                           (+ a small c(z)/z_-2 sidecar per tree in
                           <out_dir>/ludlow_c_history/)
                           Resumable: trees whose _evo.npz already exists are
                           skipped, so just re-run master.py after a crash.
  2. run_ludlow.py         evolved trees -> Tree_Reader + compute_concentration
                           + write_out_abundance -> <save_name>.h5, same layout
                           and per-tree measurement as the fid.h5 runs, so
                           ProcessH5(..., conctype=None) gives a z0 CSV whose
                           logc is the Ludlow host c.

SatGen/src/config.py must have the BolshoiP cosmology block active (these
trees have 359 snapshots); step 1 refuses to start otherwise.

Usage (on the server, from this directory):
    python master.py
"""
import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "N_cpus": 16,
    "mass_cut": 6.75e9,     # same as the fid / DF / alpha runs
    "evo_seed": 42,         # np.random seed for SubEvo (per tree: evo_seed + tree index)
    "rng_seed": 42,         # compute_concentration rng (fixed-seed convention)
    "c_true": "zhao",       # compute_concentration: "zhao" = the tree's OWN host c, i.e. Ludlow here
    "f_CMH": 0.02,          # Ludlow+2016 Sec.4.2-4.3, App.B
    "A_L16": 400.,          # Ludlow+2016 Sec.4.3 footnote 7 (Parkinson+08-style trees)
}

if config["location"] == "server":
    config.update({
        "satgen_dir": "/home/jsm99/SatGen",
        "unevo_dir": "/netb/vdbosch/jsm99/data/bolshoi_rep/",
        "fid_dir": "/netb/vdbosch/jsm99/data/bolshoi_rep/fid/",
        "out_dir": "/netb/vdbosch/jsm99/data/bolshoi_rep/ludlow/",
        "save_name": "/home/jsm99/data/mass_spec/bolshoi_rep/ludlow",
    })

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f, indent=1)

scripts = ["run_evolve_ludlow.py", "run_ludlow.py"]

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running {script}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush=True)

    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"{script} failed (exit code {result.returncode}) -- stopping here.")
        break
