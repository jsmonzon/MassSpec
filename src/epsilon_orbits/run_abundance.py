"""
Run abundance-matching measurements on the epsilon-scaled orbit trees, and
(as a comparison) on the fiducial (no-epsilon) trees for the same mass bin.

Modeled after MassSpec/src/fiducial/run_abundance.py and
MassSpec/src/N_particle/run_abundance.py, but simplified: no config.json /
file-list.txt indirection is needed since we're just pointing at directories
of evolved trees.

Two tree sets, same naming convention (tree_{mass}_{idx}_evo.npz), one
directory apart:
  - fiducial (no epsilon): /netb/vdbosch/jsm99/data/mass_spec_zhao/
      -> only the tree_13.0_*_evo.npz files, since epsilon was only run
         for that mass bin
  - epsilon-scaled orbits: /netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits/
      -> all evolved trees in that directory (already just the 13.0 bin)

Each tree is run through jsm_stellarhalo.Tree_Reader: first
compute_concentration(rng=..., c_true="zhao") -- this populates
self.without_sub / self.with_sub / self.N_subhalos_FORCE / etc, which
write_out_abundance() reads unconditionally, so it must be called first
(matching N_particle/run_abundance.py's convention; skipping it, as
fiducial/run_abundance.py does, raises
"'Tree_Reader' object has no attribute 'without_sub'") -- then
write_out_abundance(). Results are aggregated into a .h5 file (one group
per tree_index, matching the structure of the existing N1000 h5 files). By
default both directories are run, saving two separate h5 files into
/home/jsm99/data/epsilon/:
    fiducial.h5
    epsilon.h5

Usage (on the cluster):
    python run_abundance.py                    # runs both, saves both h5s
    python run_abundance.py --which fiducial    # just the fiducial comparison set
    python run_abundance.py --which epsilon     # just the epsilon-scaled set
    python run_abundance.py --ncores 16 --mass-cut 7.75e10
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import h5py

DEFAULT_FIDUCIAL_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/"
DEFAULT_EPSILON_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits/"
DEFAULT_SAVE_DIR = "/home/jsm99/data/epsilon/"
DEFAULT_PARENTDIR = "/home/jsm99/SatGen/mcmc/src/"
DEFAULT_MASS_CUT = 7.75e10
DEFAULT_TARGET_MASS = "13.0"  # only mass bin epsilon has been run for so far
DEFAULT_NCORES = 16
DEFAULT_C_TRUE = "zhao"     # matches N_particle/run_abundance.py's convention
DEFAULT_RNG_SEED = 42       # matches N_particle/run_abundance.py's fixed-seed convention


def find_evo_files(input_dir, target_mass=None):
    """List tree_{mass}_{idx}_evo.npz files in input_dir, optionally restricted
    to a single mass-bin token (e.g. "13.0"), matching the filename convention
    tree_{mass}_{idx}[_evo].npz used throughout this pipeline."""

    def mass_token(filename):
        return filename.split("_")[1]

    return sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.startswith("tree") and filename.endswith("evo.npz")
        and (target_mass is None or mass_token(filename) == target_mass)
    )


def _process_one(job):
    file_i, mass_cut, parentdir, c_true, rng_seed = job
    sys.path.insert(0, parentdir)
    import jsm_stellarhalo
    try:
        tree_i = jsm_stellarhalo.Tree_Reader(file=file_i, mass_threshold=mass_cut, verbose=False)
        # write_out_abundance() reads self.without_sub / self.with_sub / etc,
        # which are only populated by compute_concentration() -- it must be
        # called first (see N_particle/run_abundance.py's convention; the
        # fiducial/run_abundance.py template predates this and omits it,
        # which is why copying that one alone throws
        # "'Tree_Reader' object has no attribute 'without_sub'").
        tree_i.compute_concentration(rng=np.random.default_rng(rng_seed), c_true=c_true)
        return tree_i.write_out_abundance()
    except Exception as e:
        return {"tree_index": None, "error": f"{os.path.basename(file_i)}: {e}"}


def run_abundance(input_dir, save_path, label="", target_mass=None,
                   parentdir=DEFAULT_PARENTDIR, mass_cut=DEFAULT_MASS_CUT,
                   c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                   ncores=DEFAULT_NCORES, progress_every=100):

    files = find_evo_files(input_dir, target_mass=target_mass)
    Ntrees = len(files)
    tag = f"[{label}] " if label else ""
    print(f"{tag}found {Ntrees} evolved trees in {input_dir}"
          + (f" (mass bin {target_mass})" if target_mass else ""))
    print(f"{tag}saving to {save_path}")
    print("---------")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    jobs = [(file_i, mass_cut, parentdir, c_true, rng_seed) for file_i in files]

    valid_entries = []
    n_error = 0
    t0 = time.time()

    with mp.Pool(processes=ncores) as pool:
        for i, entry in enumerate(pool.imap_unordered(_process_one, jobs), 1):
            if isinstance(entry, dict) and entry.get("tree_index") is not None:
                valid_entries.append(entry)
            else:
                n_error += 1
                if isinstance(entry, dict) and "error" in entry:
                    print(f"  {tag}ERROR: {entry['error']}")

            if i % progress_every == 0 or i == Ntrees:
                elapsed = time.time() - t0
                print(f"--- {tag}progress: {i}/{Ntrees} ({elapsed:.0f}s elapsed) ---")

    dict_dict = {
        entry["tree_index"]: {k: v for k, v in entry.items() if k != "tree_index"}
        for entry in valid_entries
    }

    with h5py.File(save_path, "w") as f:
        for sim_name, attributes in dict_dict.items():
            sim_group = f.create_group(sim_name)
            for attr_name, data in attributes.items():
                if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
                    sim_group.create_dataset(attr_name, data=data)  # no compression
                else:
                    sim_group.create_dataset(attr_name, data=data, compression="gzip", compression_opts=9)

    print("---------")
    print(f"{tag}done: {len(valid_entries)} processed, {n_error} errors")

    return {"processed": len(valid_entries), "errors": n_error}


def run_comparison(which="both", fiducial_dir=DEFAULT_FIDUCIAL_DIR, epsilon_dir=DEFAULT_EPSILON_DIR,
                    save_dir=DEFAULT_SAVE_DIR, target_mass=DEFAULT_TARGET_MASS,
                    parentdir=DEFAULT_PARENTDIR, mass_cut=DEFAULT_MASS_CUT,
                    c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                    ncores=DEFAULT_NCORES, progress_every=100):

    results = {}

    if which in ("fiducial", "both"):
        results["fiducial"] = run_abundance(
            input_dir=fiducial_dir, save_path=os.path.join(save_dir, "fiducial.h5"),
            label="fiducial", target_mass=target_mass, parentdir=parentdir,
            mass_cut=mass_cut, c_true=c_true, rng_seed=rng_seed,
            ncores=ncores, progress_every=progress_every)

    if which in ("epsilon", "both"):
        results["epsilon"] = run_abundance(
            input_dir=epsilon_dir, save_path=os.path.join(save_dir, "epsilon.h5"),
            label="epsilon", target_mass=target_mass, parentdir=parentdir,
            mass_cut=mass_cut, c_true=c_true, rng_seed=rng_seed,
            ncores=ncores, progress_every=progress_every)

    return results


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["fiducial", "epsilon", "both"], default="both",
                   help="which tree set(s) to process (default: both)")
    p.add_argument("--fiducial-dir", default=DEFAULT_FIDUCIAL_DIR)
    p.add_argument("--epsilon-dir", default=DEFAULT_EPSILON_DIR)
    p.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    p.add_argument("--target-mass", default=DEFAULT_TARGET_MASS,
                   help="mass-bin token to restrict to, e.g. '13.0' (default: %(default)s); "
                        "pass '' to disable filtering")
    p.add_argument("--parentdir", default=DEFAULT_PARENTDIR)
    p.add_argument("--mass-cut", type=float, default=DEFAULT_MASS_CUT)
    p.add_argument("--c-true", default=DEFAULT_C_TRUE, choices=["zhao", "ludlow"],
                   help="concentration model fed to compute_concentration() (default: %(default)s)")
    p.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES)
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_comparison(which=args.which, fiducial_dir=args.fiducial_dir, epsilon_dir=args.epsilon_dir,
                   save_dir=args.save_dir, target_mass=(args.target_mass or None),
                   parentdir=args.parentdir, mass_cut=args.mass_cut,
                   c_true=args.c_true, rng_seed=args.rng_seed,
                   ncores=args.ncores, progress_every=args.progress_every)
