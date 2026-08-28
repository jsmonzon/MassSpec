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

A-scale sweep -> one combined h5
----------------------------------
apply_epsilon_ratio_to_directory.py's run_scale_sweep() writes each A
value's epsilon-scaled trees into its own scale_dir_for(A, scale_root)
directory (scale_root/scale_A{A:g}/, e.g. A=3 -> scale_A3/). Once those
directories have been evolved by jsm_SubEvo.py (i.e. each now holds
tree_*_evo.npz files, same convention as fiducial_dir/epsilon_dir above),
run_scale_sweep_abundance() below runs the same per-tree measurement over
every A directory and writes ALL of them into ONE h5 file -- a top-level
group per A value ("A3", "A6", "A9", ...), each holding the usual
one-group-per-tree_index layout underneath -- rather than one h5 per
directory the way run_abundance()/run_comparison() do. That combined file
is a drop-in for jsm_processh5.ProcessH5's grouped-file mode:

    from jsm_processh5 import ProcessH5
    proc = ProcessH5("/home/jsm99/data/epsilon", label="epsilon_A_sweep",
                      files=[("epsilon_A_sweep.h5", "A3"),
                             ("epsilon_A_sweep.h5", "A6"),
                             ("epsilon_A_sweep.h5", "A9")])
    proc.process(which=("z0",))

Usage (on the cluster):
    python run_abundance.py --a-sweep                       # A=3,6,9 -> one combined h5
    python run_abundance.py --a-sweep --A-values 3,6,9 \\
        --scale-root /netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits \\
        --save-path /home/jsm99/data/epsilon/epsilon_A_sweep.h5
"""

import argparse
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import h5py

from apply_epsilon_ratio_to_directory import scale_dir_for, DEFAULT_A_VALUES, DEFAULT_SCALE_ROOT

DEFAULT_FIDUCIAL_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/DF_1/"
DEFAULT_EPSILON_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits/"
DEFAULT_SAVE_DIR = "/home/jsm99/data/epsilon/"
DEFAULT_A_SWEEP_SAVE_PATH = "/home/jsm99/data/epsilon/epsilon_A_sweep.h5"
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


def _run_directory(input_dir, label="", target_mass=None, parentdir=DEFAULT_PARENTDIR,
                    mass_cut=DEFAULT_MASS_CUT, c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                    ncores=DEFAULT_NCORES, progress_every=200):
    """
    Runs the per-tree abundance-matching measurement (compute_concentration
    + write_out_abundance) over every evolved tree in input_dir, in
    parallel. Returns {"entries": [valid per-tree result dicts, each still
    carrying its "tree_index" key], "total": number of tree files found} --
    shared by run_abundance() (writes its own h5 immediately, one directory
    at a time) and run_scale_sweep_abundance() (collects several
    directories' worth of entries before writing one combined h5).
    """
    files = find_evo_files(input_dir, target_mass=target_mass)
    Ntrees = len(files)
    tag = f"[{label}] " if label else ""
    print(f"{tag}found {Ntrees} evolved trees in {input_dir}"
          + (f" (mass bin {target_mass})" if target_mass else ""))

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


def run_abundance(input_dir, save_path, label="", target_mass=None,
                   parentdir=DEFAULT_PARENTDIR, mass_cut=DEFAULT_MASS_CUT,
                   c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                   ncores=DEFAULT_NCORES, progress_every=200):

    tag = f"[{label}] " if label else ""
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
    _write_h5(save_path, dict_dict)

    print("---------")
    return {"processed": len(valid_entries), "errors": run["total"] - len(valid_entries)}


def run_comparison(which="both", fiducial_dir=DEFAULT_FIDUCIAL_DIR, epsilon_dir=DEFAULT_EPSILON_DIR,
                    save_dir=DEFAULT_SAVE_DIR, target_mass=DEFAULT_TARGET_MASS,
                    parentdir=DEFAULT_PARENTDIR, mass_cut=DEFAULT_MASS_CUT,
                    c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                    ncores=DEFAULT_NCORES, progress_every=200):

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


def run_scale_sweep_abundance(A_values=DEFAULT_A_VALUES, scale_root=DEFAULT_SCALE_ROOT,
                               save_path=DEFAULT_A_SWEEP_SAVE_PATH, target_mass=DEFAULT_TARGET_MASS,
                               parentdir=DEFAULT_PARENTDIR, mass_cut=DEFAULT_MASS_CUT,
                               c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED,
                               ncores=DEFAULT_NCORES, progress_every=200):
    """
    Runs the abundance-matching measurement once per A in A_values (each
    A's evolved trees living in scale_dir_for(A, scale_root) --
    scale_root/scale_A{A:g}/, the directory
    apply_epsilon_ratio_to_directory.py's run_scale_sweep() wrote the
    epsilon-scaled trees into, now evolved by jsm_SubEvo.py into
    tree_*_evo.npz files), and writes ALL of them into ONE h5 file at
    save_path: a top-level group per A value ("A3", "A6", "A9", ...), each
    holding the usual one-group-per-tree_index layout underneath -- unlike
    run_abundance()/run_comparison(), which write one h5 per directory.

    Read it back with jsm_processh5.ProcessH5's grouped-file mode:
        ProcessH5(save_dir, label="epsilon_A_sweep",
                  files=[(save_path, f"A{A:g}") for A in A_values])

    Returns a dict keyed by "A{A:g}", each value {"processed", "errors"}.
    """
    results = {}
    dict_dict = {}

    for A in A_values:
        label = f"A{A:g}"
        input_dir = scale_dir_for(A, scale_root)
        run = _run_directory(input_dir, label=label, target_mass=target_mass, parentdir=parentdir,
                              mass_cut=mass_cut, c_true=c_true, rng_seed=rng_seed,
                              ncores=ncores, progress_every=progress_every)
        valid_entries = run["entries"]
        dict_dict[label] = {
            entry["tree_index"]: {k: v for k, v in entry.items() if k != "tree_index"}
            for entry in valid_entries
        }
        results[label] = {"processed": len(valid_entries), "errors": run["total"] - len(valid_entries)}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        for label, tree_dict in dict_dict.items():
            model_group = f.create_group(label)
            for sim_name, attributes in tree_dict.items():
                sim_group = model_group.create_group(sim_name)
                for attr_name, data in attributes.items():
                    if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
                        sim_group.create_dataset(attr_name, data=data)  # no compression
                    else:
                        sim_group.create_dataset(attr_name, data=data, compression="gzip", compression_opts=9)

    print("=========")
    print(f"A-sweep done -> {save_path}")
    for label, r in results.items():
        print(f"  {label}: {r['processed']} processed, {r['errors']} errors")

    return results


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=["fiducial", "epsilon", "both"], default="both",
                   help="which tree set(s) to process (default: both); ignored if --a-sweep is set")
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

    p.add_argument("--a-sweep", action="store_true",
                   help="run run_scale_sweep_abundance() instead of run_comparison(): "
                        "process the A-scale sweep directories and write ONE combined h5 "
                        "(a top-level group per A value) instead of one h5 per --which set")
    p.add_argument("--A-values", default=",".join(f"{a:g}" for a in DEFAULT_A_VALUES),
                   help=f"comma-separated A values for --a-sweep (default: "
                        f"{','.join(f'{a:g}' for a in DEFAULT_A_VALUES)})")
    p.add_argument("--scale-root", default=DEFAULT_SCALE_ROOT,
                   help="parent directory holding --a-sweep's scale_A{A}/ input directories "
                        f"(default: {DEFAULT_SCALE_ROOT})")
    p.add_argument("--save-path", default=DEFAULT_A_SWEEP_SAVE_PATH,
                   help=f"output path for --a-sweep's single combined h5 (default: {DEFAULT_A_SWEEP_SAVE_PATH})")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.a_sweep:
        A_values = tuple(float(x) for x in args.A_values.split(","))
        run_scale_sweep_abundance(
            A_values=A_values, scale_root=args.scale_root, save_path=args.save_path,
            target_mass=(args.target_mass or None), parentdir=args.parentdir, mass_cut=args.mass_cut,
            c_true=args.c_true, rng_seed=args.rng_seed, ncores=args.ncores,
            progress_every=args.progress_every)
    else:
        run_comparison(which=args.which, fiducial_dir=args.fiducial_dir, epsilon_dir=args.epsilon_dir,
                       save_dir=args.save_dir, target_mass=(args.target_mass or None),
                       parentdir=args.parentdir, mass_cut=args.mass_cut,
                       c_true=args.c_true, rng_seed=args.rng_seed,
                       ncores=args.ncores, progress_every=args.progress_every)
