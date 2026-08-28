"""
Run the Tree_Reader_Light measurement over the symmetric-MW ("sym_MW")
evolved merger trees and save one .h5 file -- the light counterpart to
run_abundance.py in this same directory.

Modeled on run_abundance.py (same tree discovery, same multiprocessing
pattern, same one-group-per-tree_index h5 layout), but built on
jsm_stellarhalo.Tree_Reader_Light instead of Tree_Reader: no regime/order
Nsub/Msub/fsub bookkeeping, no z=0 SHMF, no particle-painted concentration
remeasurement. Just, per tree:

  host:      full MAH, full concentration history, full Rvir history,
             z10/z50/z90
  subhalos:  survives (bool, alive at z=0), z0_mass / z0_position (kept
             only where surviving + within the host's z=0 Rvir + above
             mass_cut, NaN otherwise), and zacc_mass / zacc_concentration
             / zacc_redshift at the redshift each subhalo's branch fell
             into the host's MAIN PROGENITOR (proper_acc_index).

  z=0 subhalo concentration is deliberately NOT saved here: SatGen's raw
  trees only ever populate the 'concentration' array for the host, never
  for subhalos at their z=0 snapshot (verified directly against this
  project's tree files) -- it would just be a column of NaN.

All five subhalo arrays are sliced to [1:] before saving, dropping the
host's own row 0 (Tree_Reader_Light keeps the same host-is-row-0
convention as Tree_Reader internally, but a "survives"/"z0_mass"/etc.
value for the host itself isn't meaningful).

COSMOLOGY -- this is the one thing that's actually EASIER here than in
run_abundance.py: Tree_Reader_Light never builds host_profiles or touches
cfg.Dvsample (that's exactly the machinery this light class doesn't
carry), so the Symphony-vs-zhao config.py cosmology mismatch documented
in run_abundance.py's docstring does not apply to this script -- no
_check_cosmology() call here.

Read the result back with a plain h5py.File(...); jsm_processh5.ProcessH5
is built around Tree_Reader's full (regime x order) field set and won't
recognize these field names.

Usage (on the cluster):
    python run_abundance_light.py
    python run_abundance_light.py --ncores 32
    python run_abundance_light.py --mass-cut 6.75e9
    python run_abundance_light.py --input-dir /netb/vdbosch/jsm99/data/sym_MW/DF_2/ \\
        --save-path /home/jsm99/data/sym_MW/DF_2_light.h5
    python run_abundance_light.py --target-mass 12.05     # one bin only
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import h5py

DEFAULT_INPUT_DIR = "/netb/vdbosch/jsm99/data/sym_MW/DF_1/"
DEFAULT_SAVE_PATH = "/home/jsm99/data/sym_MW/DF_1_light.h5"
DEFAULT_PARENTDIR = "/home/jsm99/SatGen/mcmc/src/"
DEFAULT_MASS_CUT = 1e8      # same resolution-limit cut as run_abundance.py
DEFAULT_TARGET_MASS = None  # sym_MW spans 12.05-12.15; take every bin
DEFAULT_NCORES = 16


def find_evo_files(input_dir, target_mass=None):
    """List tree_{mass}_{idx}_evo.npz files in input_dir, optionally restricted
    to a single mass-bin token (e.g. "12.05"). Identical to run_abundance.py's."""

    def mass_token(filename):
        return filename.split("_")[1]

    return sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.startswith("tree") and filename.endswith("evo.npz")
        and (target_mass is None or mass_token(filename) == target_mass)
    )


def _light_dict(tree_i):
    """Tree_Reader_Light has no write_out_*() method -- it's plain
    attributes by design -- so this is the equivalent for this script:
    pull the fields we want into the same {name: array/scalar} shape
    run_abundance.py's write_out_abundance() produces, ready for
    _write_h5(). Subhalo arrays are sliced to [1:] to drop the host's own
    row."""
    return {
        "tree_index":         tree_i.tree_index,
        "host_MAH":           tree_i.host_MAH,
        "host_concentration": tree_i.host_concentration,
        "host_Rvir":          tree_i.host_Rvir,
        "host_z10":           tree_i.host_z10,
        "host_z50":           tree_i.host_z50,
        "host_z90":           tree_i.host_z90,
        "survives":           tree_i.survives[1:],
        "z0_mass":            tree_i.z0_mass[1:],
        "z0_position":        tree_i.z0_position[1:],
        "zacc_mass":          tree_i.zacc_mass[1:],
        "zacc_concentration": tree_i.zacc_concentration[1:],
        "zacc_redshift":      tree_i.zacc_redshift[1:],
    }


def _process_one(job):
    file_i, mass_cut, parentdir = job
    sys.path.insert(0, parentdir)
    import jsm_stellarhalo
    try:
        tree_i = jsm_stellarhalo.Tree_Reader_Light(file=file_i, mass_threshold=mass_cut, verbose=False)
        return _light_dict(tree_i)
    except Exception as e:
        return {"tree_index": None, "error": f"{os.path.basename(file_i)}: {e}"}


def _run_directory(input_dir, label="", target_mass=None, parentdir=DEFAULT_PARENTDIR,
                   mass_cut=DEFAULT_MASS_CUT, ncores=DEFAULT_NCORES, progress_every=200):
    """Runs Tree_Reader_Light over every evolved tree in input_dir, in
    parallel. Returns {"entries": [...], "total": Ntrees}, same shape as
    run_abundance.py's _run_directory()."""
    files = find_evo_files(input_dir, target_mass=target_mass)
    Ntrees = len(files)
    tag = f"[{label}] " if label else ""
    print(f"{tag}found {Ntrees} evolved trees in {input_dir}"
          + (f" (mass bin {target_mass})" if target_mass else " (all mass bins)"))
    if Ntrees == 0:
        raise SystemExit(f"no tree_*_evo.npz files matched in {input_dir}")

    jobs = [(file_i, mass_cut, parentdir) for file_i in files]

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

    print(f"{tag}done: {len(valid_entries)} processed, {n_error} errors")
    return {"entries": valid_entries, "total": Ntrees}


def _write_h5(save_path, dict_dict):
    """dict_dict: {tree_index: {attr_name: data, ...}, ...} -> one HDF5
    group per tree_index. Identical convention to run_abundance.py's."""
    with h5py.File(save_path, "w") as f:
        for sim_name, attributes in dict_dict.items():
            sim_group = f.create_group(sim_name)
            for attr_name, data in attributes.items():
                if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
                    sim_group.create_dataset(attr_name, data=data)  # no compression
                else:
                    sim_group.create_dataset(attr_name, data=data, compression="gzip", compression_opts=9)


def run_abundance_light(input_dir=DEFAULT_INPUT_DIR, save_path=DEFAULT_SAVE_PATH, label="sym_MW_light",
                        target_mass=DEFAULT_TARGET_MASS, parentdir=DEFAULT_PARENTDIR,
                        mass_cut=DEFAULT_MASS_CUT, ncores=DEFAULT_NCORES, progress_every=200):

    tag = f"[{label}] " if label else ""
    print(f"{tag}mass_cut = {mass_cut:.3e}, ncores = {ncores}")
    print(f"{tag}saving to {save_path}")
    print("---------")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    run = _run_directory(input_dir, label=label, target_mass=target_mass, parentdir=parentdir,
                         mass_cut=mass_cut, ncores=ncores, progress_every=progress_every)
    valid_entries = run["entries"]

    dict_dict = {
        entry["tree_index"]: {k: v for k, v in entry.items() if k != "tree_index"}
        for entry in valid_entries
    }
    if len(dict_dict) != len(valid_entries):
        print(f"{tag}WARNING: {len(valid_entries) - len(dict_dict)} tree_index collisions "
              f"-- some trees were overwritten in the h5!")

    _write_h5(save_path, dict_dict)

    print("---------")
    print(f"{tag}wrote {len(dict_dict)} groups -> {save_path}")
    return {"processed": len(valid_entries), "errors": run["total"] - len(valid_entries)}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    p.add_argument("--label", default="sym_MW_light", help="prefix for progress lines")
    p.add_argument("--target-mass", default="",
                   help="mass-bin token to restrict to, e.g. '12.05'; "
                        "default is '' = every bin in the directory")
    p.add_argument("--parentdir", default=DEFAULT_PARENTDIR)
    p.add_argument("--mass-cut", type=float, default=DEFAULT_MASS_CUT,
                   help="Tree_Reader_Light mass_threshold, gates the z0_mass/"
                        "z0_position cut (default: %(default).2e)")
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES)
    p.add_argument("--progress-every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_abundance_light(input_dir=args.input_dir, save_path=args.save_path, label=args.label,
                        target_mass=(args.target_mass or None), parentdir=args.parentdir,
                        mass_cut=args.mass_cut, ncores=args.ncores, progress_every=args.progress_every)
