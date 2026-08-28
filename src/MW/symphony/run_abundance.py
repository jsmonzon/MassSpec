"""
Run the abundance-matching measurement over the symmetric-MW ("sym_MW")
evolved merger trees and save one .h5 file.

Modeled on MassSpec/src/epsilon_orbits/run_abundance.py, but stripped of
the epsilon/A-sweep machinery: there is exactly one tree set here, so this
just walks a directory of tree_{mass}_{idx}_evo.npz files, pushes each one
through jsm_stellarhalo.Tree_Reader, and aggregates the per-tree
write_out_abundance() dictionaries into a single h5 (one group per
tree_index).

Input set
---------
/netb/vdbosch/jsm99/data/sym_MW/DF_1/  -- 4500 evolved trees.

Two differences from the epsilon runs that matter:

  1. MASS BINS. The epsilon runs were a single bin (13.0) and therefore
     filtered filenames on that token. sym_MW spans ELEVEN bins
     (12.05, 12.06, ... 12.15 -- 4500 trees total), and we want all of
     them, so --target-mass defaults to '' (no filtering). Pass e.g.
     --target-mass 12.05 only if you deliberately want one bin.

  2. MASS CUT. Tree_Reader's mass_threshold sets the "massive" regime cut
     that every Nsub/Msub/fsub/shmf column downstream is built on. The
     epsilon runs used 7.75e10 because their hosts were 1e13. These hosts
     are ~1.1e12 with a resolution limit of psi_res ~ 1e-4 (smallest
     subhalo ~9e7), and the cut for this run is 1e8 -- essentially at the
     resolution limit, i.e. keep everything the trees resolve. Override
     with --mass-cut.

COSMOLOGY (important)
---------------------
Tree_Reader indexes cfg.Dvsample POSITIONALLY against each tree's own time
axis (read_arrays() -> NFW_vectorized(..., Delta=cfg.Dvsample, ...)), so
SatGen/src/config.py must be on the same cosmology the trees were built
with or nothing runs. These trees are Symphony (h=0.7, Om=0.286, OL=0.714)
-> Nz = 356; the "zhao" block currently active in config.py gives Nz = 354
and every tree dies with

    operands could not be broadcast together with shapes (356,) ... (354,)

_check_cosmology() below catches this once, up front, instead of letting
it fail 4500 times in the pool. Fix it by uncommenting the Symphony block
in SatGen/src/config.py (and re-commenting the zhao one).


Tree_Reader.read_arrays() takes tree_index from the filename's third
underscore-token (tree_{mass}_{idx}_evo.npz -> idx), and that token is
unique across all 4500 files in DF_1 (verified), so no two trees collide
on their h5 group name even though they come from different mass bins.

Each tree is run as: Tree_Reader(...) then compute_concentration(rng=...,
c_true="zhao") -- which populates self.without_sub / self.with_sub /
self.N_subhalos_FORCE, read unconditionally by write_out_abundance(), so
it must be called first -- then write_out_abundance().

Read the result back with jsm_processh5.ProcessH5:

    from jsm_processh5 import ProcessH5
    proc = ProcessH5("/home/jsm99/data/sym_MW", label="DF_1",
                     files=["DF_1.h5"])
    proc.process(which=("z0",))

Usage (on the cluster):
    python run_abundance.py
    python run_abundance.py --ncores 32
    python run_abundance.py --mass-cut 6.75e9
    python run_abundance.py --input-dir /netb/vdbosch/jsm99/data/sym_MW/DF_2/ \\
        --save-path /home/jsm99/data/sym_MW/DF_2.h5
    python run_abundance.py --target-mass 12.05     # one bin only
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import h5py

DEFAULT_INPUT_DIR = "/netb/vdbosch/jsm99/data/sym_MW/DF_1/"
DEFAULT_SAVE_PATH = "/home/jsm99/data/sym_MW/DF_1.h5"
DEFAULT_PARENTDIR = "/home/jsm99/SatGen/mcmc/src/"
DEFAULT_MASS_CUT = 1e8      # at the resolution limit; hosts ~1.1e12, Mres ~9e7
DEFAULT_TARGET_MASS = None  # sym_MW spans 12.05-12.15; take every bin
DEFAULT_NCORES = 16
DEFAULT_C_TRUE = "zhao"
DEFAULT_RNG_SEED = 42


def find_evo_files(input_dir, target_mass=None):
    """List tree_{mass}_{idx}_evo.npz files in input_dir, optionally restricted
    to a single mass-bin token (e.g. "12.05"), matching the filename convention
    tree_{mass}_{idx}[_evo].npz used throughout this pipeline."""

    def mass_token(filename):
        return filename.split("_")[1]

    return sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.startswith("tree") and filename.endswith("evo.npz")
        and (target_mass is None or mass_token(filename) == target_mass)
    )


def _check_cosmology(files, parentdir=DEFAULT_PARENTDIR):
    """Tree_Reader indexes cfg.Dvsample positionally against each tree's own
    time axis, so config.py's redshift grid has to be the one these trees were
    generated on. Compare lengths against the first tree and abort loudly here
    rather than letting every worker raise the same opaque broadcast error."""
    sys.path.insert(0, parentdir)
    # config.py lives in SatGen/src/, not mcmc/src/ -- jsm_stellarhalo is what
    # puts it on the path (it hardcodes parentdir="/home/jsm99/SatGen/src/"),
    # so go through the same module the workers will, and read the very same
    # cfg object they'll broadcast against.
    import jsm_stellarhalo
    cfg = jsm_stellarhalo.cfg

    tree_z = np.load(files[0])["redshift"]
    # read_arrays() deletes row 1 of every array but leaves the time axis
    # alone, so Nz must match the tree's redshift length exactly.
    if len(tree_z) != cfg.Nz:
        raise SystemExit(
            f"\nCOSMOLOGY MISMATCH\n"
            f"  {os.path.basename(files[0])}: Nz = {len(tree_z)}, zmax = {tree_z.max():.6f}\n"
            f"  config.py:{' ':16s}Nz = {cfg.Nz}, zmax = {cfg.zsample.max():.6f}\n"
            f"  (h={cfg.h}, Om={cfg.Om}, OL={cfg.OL})\n\n"
            f"Tree_Reader broadcasts cfg.Dvsample against the tree time axis, so these\n"
            f"must agree. The sym_MW trees are Symphony (h=0.7, Om=0.286, OL=0.714 -> Nz=356).\n"
            f"Uncomment the Symphony cosmology block in SatGen/src/config.py and re-run.\n")

    print(f"cosmology OK: Nz = {cfg.Nz} (h={cfg.h}, Om={cfg.Om}, OL={cfg.OL})")


def _process_one(job):
    file_i, mass_cut, parentdir, c_true, rng_seed = job
    sys.path.insert(0, parentdir)
    import jsm_stellarhalo
    try:
        tree_i = jsm_stellarhalo.Tree_Reader(file=file_i, mass_threshold=mass_cut, verbose=False)
        # write_out_abundance() reads self.without_sub / self.with_sub /
        # self.N_subhalos_FORCE, which only compute_concentration() populates,
        # so it has to run first -- otherwise you get
        # "'Tree_Reader' object has no attribute 'without_sub'".
        tree_i.compute_concentration(rng=np.random.default_rng(rng_seed), c_true=c_true)
        return tree_i.write_out_abundance()
    except Exception as e:
        return {"tree_index": None, "error": f"{os.path.basename(file_i)}: {e}"}


def _run_directory(input_dir, label="", target_mass=None, parentdir=DEFAULT_PARENTDIR,
                   mass_cut=DEFAULT_MASS_CUT, c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                   ncores=DEFAULT_NCORES, progress_every=200):
    """
    Runs the per-tree measurement (compute_concentration +
    write_out_abundance) over every evolved tree in input_dir, in parallel.
    Returns {"entries": [valid per-tree result dicts, each still carrying
    its "tree_index" key], "total": number of tree files found}.
    """
    files = find_evo_files(input_dir, target_mass=target_mass)
    Ntrees = len(files)
    tag = f"[{label}] " if label else ""
    print(f"{tag}found {Ntrees} evolved trees in {input_dir}"
          + (f" (mass bin {target_mass})" if target_mass else " (all mass bins)"))
    if Ntrees == 0:
        raise SystemExit(f"no tree_*_evo.npz files matched in {input_dir}")

    _check_cosmology(files, parentdir=parentdir)

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

    print(f"{tag}done: {len(valid_entries)} processed, {n_error} errors")
    return {"entries": valid_entries, "total": Ntrees}


def _write_h5(save_path, dict_dict):
    """dict_dict: {tree_index: {attr_name: data, ...}, ...} -> one HDF5
    group per tree_index, matching write_out_abundance()'s convention."""
    with h5py.File(save_path, "w") as f:
        for sim_name, attributes in dict_dict.items():
            sim_group = f.create_group(sim_name)
            for attr_name, data in attributes.items():
                if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
                    sim_group.create_dataset(attr_name, data=data)  # no compression
                else:
                    sim_group.create_dataset(attr_name, data=data, compression="gzip", compression_opts=9)


def run_abundance(input_dir=DEFAULT_INPUT_DIR, save_path=DEFAULT_SAVE_PATH, label="sym_MW",
                  target_mass=DEFAULT_TARGET_MASS, parentdir=DEFAULT_PARENTDIR,
                  mass_cut=DEFAULT_MASS_CUT, c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                  ncores=DEFAULT_NCORES, progress_every=200):

    tag = f"[{label}] " if label else ""
    print(f"{tag}mass_cut = {mass_cut:.3e}, c_true = {c_true}, ncores = {ncores}")
    print(f"{tag}saving to {save_path}")
    print("---------")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    run = _run_directory(input_dir, label=label, target_mass=target_mass, parentdir=parentdir,
                         mass_cut=mass_cut, c_true=c_true, rng_seed=rng_seed,
                         ncores=ncores, progress_every=progress_every)
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
    p.add_argument("--label", default="sym_MW", help="prefix for progress lines")
    p.add_argument("--target-mass", default="",
                   help="mass-bin token to restrict to, e.g. '12.05'; "
                        "default is '' = every bin in the directory")
    p.add_argument("--parentdir", default=DEFAULT_PARENTDIR)
    p.add_argument("--mass-cut", type=float, default=DEFAULT_MASS_CUT,
                   help="Tree_Reader mass_threshold (default: %(default).2e)")
    p.add_argument("--c-true", default=DEFAULT_C_TRUE, choices=["zhao", "ludlow"],
                   help="concentration model fed to compute_concentration() (default: %(default)s)")
    p.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES)
    p.add_argument("--progress-every", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_abundance(input_dir=args.input_dir, save_path=args.save_path, label=args.label,
                  target_mass=(args.target_mass or None), parentdir=args.parentdir,
                  mass_cut=args.mass_cut, c_true=args.c_true, rng_seed=args.rng_seed,
                  ncores=args.ncores, progress_every=args.progress_every)
