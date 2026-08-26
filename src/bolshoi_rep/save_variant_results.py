"""
save_variant_results.py

Reads the six bolshoi_rep evolved-merger-tree variants (each tree evolved
under slightly different sub-halo physics) and writes one summary .h5 file
per variant, using jsm_stellarhalo.Tree_Reader exactly as
bolshoi_rep/concentration_test/run_sample.py does for a single variant.

Input directory layout expected (unevolved trees generated from the Bolshoi
catalog, 1517 hosts in the 13.2 mass bin; each subdirectory holds that same
tree set run through a different evolution variant):

    /netb/vdbosch/jsm99/data/bolshoi_rep/
        fid/          -- fiducial evolution
        alpha_up/     -- tidal-track alpha parameter perturbed up
        alpha_down/   -- tidal-track alpha parameter perturbed down
        DF_up/        -- dynamical friction perturbed up
        DF_down/      -- dynamical friction perturbed down
        bound/        -- self-bound-mass tracking variant

Each subdirectory is expected to hold "tree*evo.npz" files, one per host,
same convention as concentration_test/run_sample.py and bolshoi_rep/DF/*.

For every tree this script:
  1. builds a jsm_stellarhalo.Tree_Reader (mass_threshold = CONFIG["mass_cut"]),
  2. runs the Monte Carlo concentration-measurement test via
     .compute_concentration(rng, c_true=CONFIG["cmodel"]) -- this is what
     populates c_measured_smooth/fixed_COM/shifted_COM, fsub_used, Nsub_used,
     and ludlow_c/ludlow_z2 in the output,
  3. writes out the full per-tree dict via .write_out_abundance() (host MAH/
     Rvir/c/Rmax/Vcirc/z10/z50/z90, the concentration measurements above, and
     the per-regime/per-order Nsub/Msub/fsub/SHMF matrices).

One .h5 file is written per variant into CONFIG["output_dir"], named
"<variant>.h5", with one HDF5 group per tree (keyed by tree_index) --
the same nesting/compression convention run_sample.py already uses, so
these files are drop-in readable with jsm_ancillary.load_sample /
load_massspec_z0 like every other .h5 in this repo.

Before running: set CONFIG["output_dir"] below to wherever you want the six
files to land, and double check CONFIG["N_cpus"] against what's actually
free on the machine you're running this on.
"""

import os
import sys
import multiprocessing as mp

import numpy as np
import h5py

# --------------------------------------------------------------------------
# Configuration -- edit these before running
# --------------------------------------------------------------------------

CONFIG = {
    # "server" -> reads/writes the /netb paths below (run this on the
    # remote machine); "local" -> swap in your own local SatGen checkout
    # if you ever want to test this against a small local tree sample.
    "location": "server",

    # where the six evolved-tree variant subdirectories live
    "input_base_dir": "/netb/vdbosch/jsm99/data/bolshoi_rep",

    # <<< SET THIS to wherever you want the six .h5 files written >>>
    "output_dir": "/CHANGE/ME/to/your/chosen/output/directory",

    # the six variant subdirectory names, each processed into its own .h5
    "variants": ["fid", "alpha_up", "alpha_down", "DF_up", "DF_down", "bound"],

    # Tree_Reader(mass_threshold=...): minimum resolved subhalo mass [Msun],
    # matching bolshoi_rep/concentration_test/master.py's convention
    "mass_cut": 7.75e10,

    # compute_concentration(c_true=...): ground-truth concentration model
    # seeding the Monte Carlo NFW+Plummer realization test
    "cmodel": "zhao",

    # compute_concentration(rng=np.random.default_rng(seed)): a fresh RNG
    # seeded identically for every tree, matching run_sample.py's convention
    # (common random numbers across trees, not shared across processes)
    "seed": 42,

    "N_cpus": 16,
}

# --------------------------------------------------------------------------
# Make jsm_stellarhalo importable, exactly as the sibling run_*.py scripts do
# --------------------------------------------------------------------------

if CONFIG["location"] == "server":
    _parentdir = "/home/jsm99/SatGen/mcmc/src/"
elif CONFIG["location"] == "local":
    _parentdir = "/Users/jsmonzon/Research/SatGen/mcmc/src/"
else:
    raise ValueError(f"unrecognized location: {CONFIG['location']!r}")

sys.path.insert(0, _parentdir)
import jsm_stellarhalo  # noqa: E402  (import after sys.path manipulation, by design)


# --------------------------------------------------------------------------
# Per-tree worker -- must be a module-level function so mp.Pool can pickle it
# --------------------------------------------------------------------------

def process_file(file_i):
    """
    Build a Tree_Reader for one evolved tree, run the concentration test,
    and return its write_out_abundance() dict -- or None on failure (the
    exception is printed so failures are visible in the driver's stdout,
    not silently dropped).
    """
    try:
        tree_i = jsm_stellarhalo.Tree_Reader(
            file=file_i,
            mass_threshold=CONFIG["mass_cut"],
            verbose=False,
        )
        tree_i.compute_concentration(
            rng=np.random.default_rng(CONFIG["seed"]),
            c_true=CONFIG["cmodel"],
        )
        return tree_i.write_out_abundance()

    except Exception as e:
        print(f"Error processing {file_i}: {e}")
        return None


# --------------------------------------------------------------------------
# Per-variant driver: find files, run the pool, write the .h5
# --------------------------------------------------------------------------

def process_variant(variant):
    datadir = os.path.join(CONFIG["input_base_dir"], variant)

    if not os.path.isdir(datadir):
        print(f"[{variant}] SKIPPED -- directory not found: {datadir}")
        return

    files = [
        os.path.join(datadir, filename)
        for filename in os.listdir(datadir)
        if filename.startswith("tree") and filename.endswith("evo.npz")
    ]

    if not files:
        print(f"[{variant}] SKIPPED -- no tree*evo.npz files found in {datadir}")
        return

    print(f"[{variant}] found {len(files)} tree files, processing with "
          f"{CONFIG['N_cpus']} workers...")

    with mp.Pool(processes=CONFIG["N_cpus"]) as pool:
        results = pool.map(process_file, files)

    valid_entries = [
        entry for entry in results
        if isinstance(entry, dict) and "tree_index" in entry
    ]
    n_failed = len(files) - len(valid_entries)

    if not valid_entries:
        print(f"[{variant}] SKIPPED writing .h5 -- 0/{len(files)} trees succeeded")
        return

    dict_dict = {
        entry["tree_index"]: {k: v for k, v in entry.items() if k != "tree_index"}
        for entry in valid_entries
    }

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["output_dir"], f"{variant}.h5")

    with h5py.File(save_path, "w") as f:
        for sim_name, attributes in dict_dict.items():
            sim_group = f.create_group(sim_name)

            for attr_name, data in attributes.items():
                if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
                    sim_group.create_dataset(attr_name, data=data)  # no compression
                else:
                    sim_group.create_dataset(attr_name, data=data, compression="gzip", compression_opts=9)

    print(f"[{variant}] wrote {len(valid_entries)}/{len(files)} trees "
          f"({n_failed} failed) -> {save_path}")


# --------------------------------------------------------------------------

if __name__ == "__main__":

    if CONFIG["output_dir"].startswith("/CHANGE/ME"):
        raise SystemExit(
            "Set CONFIG['output_dir'] to a real path before running this script."
        )

    for variant in CONFIG["variants"]:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"Processing variant: {variant}")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        process_variant(variant)
