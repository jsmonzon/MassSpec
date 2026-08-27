"""
apply_epsilon_ratio_to_directory.py

Batch-applies EpsilonRatioOrbitInit to every raw (un-evolved) tree file in
a directory, writing the scaled trees into a new output directory under
the SAME filenames as the originals. Trees are processed in parallel
across multiple cores (same multiprocessing.Pool pattern jsm_SubEvo.py
uses), with progress printed as trees complete.

(Moved here from SatGen/notebooks/claude_based/, alongside
epsilon_ratio_orbit_init.py, since this is MassSpec analysis code, not
SatGen library code -- SatGen/etc/mean_MAH/ is still where the reference
MAH files live, so mean_mah_dir now defaults to an absolute cluster path
rather than one resolved relative to this file's location.)

Why same filenames, new directory: jsm_SubEvo.py discovers trees by
filename convention within whatever datadir it's pointed at (files
starting with "tree" and NOT ending in "evo.npz"; it then writes each
tree's evolved output as "<name>_evo.npz" alongside it in that same
directory). Preserving filenames means the output directory here is a
drop-in datadir for jsm_SubEvo.py -- point it there and it evolves the
epsilon-scaled trees exactly as it would the originals. The original tree
files in input_dir are left untouched, so you can run jsm_SubEvo.py on
input_dir (baseline) and on this script's output_dir (epsilon-scaled) and
compare the two "_evo.npz" sets 1:1, tree by tree.

Each tree's own mass bin (and therefore which mean_MAH reference backs
its epsilon(z)) is inferred from its own z=0 host mass -- see
EpsilonRatioOrbitInit for details. Trees were originally sampled from a
wider mass range (1e11-1e14) than mean_MAH actually covers (12.6-14.0),
so a tree whose host mass falls outside mean_mah_dir's coverage is
skipped here -- not silently matched to the nearest available bin, and
not a fatal error for the whole batch. Skipped trees are tallied by mass
bin and reported in the final summary, so a genuine mean_mah_dir/input_dir
mismatch (e.g. wrong directory entirely) is still obvious rather than
quietly swallowed. Likewise, an unexpected per-tree failure (corrupted
file, etc.) is caught, reported, and counted rather than killing the
whole pool -- mirroring how jsm_SubEvo.py's own loop() catches
AttributeError per-tree instead of dying mid-batch.

target_mass restricts processing to one mass-bin token (e.g. "13.0"),
matching the tree_{mass}_{idx}.npz filename convention used throughout
this pipeline (same check as MassSpec/src/epsilon_orbits/run_abundance.py's
find_evo_files). Useful since the raw tree pool spans multiple mass bins
in one directory but a given epsilon experiment (e.g. the A-scale sweep
below) is usually run on just one.

A-scale sweep
--------------
run_scale_sweep() runs apply_epsilon_ratio_to_directory() once per A value
in A_values, writing each set of epsilon-scaled trees into its own
"scale_A{A:g}" subdirectory of scale_root (see scale_dir_for) -- e.g.
A=3 -> scale_root/scale_A3/. This is the entry point for "run several A
values and keep each in its own directory" rather than one-off single-A
runs. Because eps_min=0.5/eps_max=1.5 (this module's single-run defaults)
saturate almost immediately once A gets much above 1 (see
epsilon_ratio_orbit_init.py's module docstring for why), run_scale_sweep
uses its own wider defaults (eps_min=0.01, eps_max=30.0) instead -- override
via its eps_min/eps_max arguments (or --eps-min/--eps-max on the CLI with
--sweep) if a given A needs something different.

Cluster deployment
-------------------
The raw N1000 zhao trees live on the cluster, outside this git repo, at
DEFAULT_INPUT_DIR below. DEFAULT_SCALE_ROOT is the "epsilon_orbits"
sibling directory already created there to receive the scaled copies, one
subdirectory per A value. mean_MAH/*.npz lives in the SatGen repo, at
DEFAULT_MEAN_MAH_DIR below -- a plain absolute cluster path, since this
script (unlike its previous home under SatGen/notebooks/) no longer sits
inside the SatGen repo and can't resolve that location relative to itself.

Usage (CLI):

    # single A, single output directory
    python apply_epsilon_ratio_to_directory.py INPUT_DIR OUTPUT_DIR \\
        MEAN_MAH_DIR --A 1.0 --target-mass 13.0 --eps-min 0.5 --eps-max 1.5 --ncores 16

    # the A=3,6,9 scale sweep (uses the cluster defaults below -- no
    # positional args needed once pushed)
    python apply_epsilon_ratio_to_directory.py --sweep

    # or override the sweep's A values / output root / clip range
    python apply_epsilon_ratio_to_directory.py --sweep --A-values 3,6,9 \\
        --scale-root /netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits \\
        --eps-min 0.01 --eps-max 30.0

    # then point jsm_SubEvo.py's datadir at each scale_A*/ directory and run it
"""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from epsilon_ratio_orbit_init import EpsilonRatioOrbitInit, infer_logM0

DEFAULT_INPUT_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/"
DEFAULT_OUTPUT_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits/"
DEFAULT_MEAN_MAH_DIR = "/home/jsm99/SatGen/etc/mean_MAH/"
DEFAULT_NCORES = 16  # matches jsm_SubGen_masspec.py / jsm_SubEvo.py's ncores

DEFAULT_SCALE_ROOT = "/netb/vdbosch/jsm99/data/mass_spec_zhao/epsilon_orbits"
DEFAULT_A_VALUES = (3.0, 6.0, 9.0)
DEFAULT_TARGET_MASS = "13.0"  # the sweep's mass bin -- 1000 trees, matching the earlier epsilon test
DEFAULT_SWEEP_EPS_MIN = 0.01
DEFAULT_SWEEP_EPS_MAX = 30.0


def scale_dir_for(A, scale_root=DEFAULT_SCALE_ROOT):
    """
    Directory-naming convention for a scale-sweep run: scale_root/scale_A{A:g}
    (e.g. A=3.0 -> ".../scale_A3", A=1.5 -> ".../scale_A1.5"). Matches the
    scale_A3 / scale_A6 / scale_A9 directories already created for the
    default sweep.
    """
    return str(Path(scale_root) / f"scale_A{A:g}")


def find_raw_tree_files(input_dir, target_mass=None):
    """
    Returns the raw (un-evolved) tree files in input_dir, using the same
    file-discovery convention as jsm_SubEvo.py: filenames starting with
    "tree" and NOT ending in "evo.npz". If target_mass is given (e.g.
    "13.0"), restricts to that mass-bin token, parsed the same way as
    run_abundance.py's find_evo_files (filename.split("_")[1]).
    """
    def mass_token(filename):
        return filename.split("_")[1]

    input_dir = Path(input_dir)
    return sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.name.startswith("tree") and not f.name.endswith("evo.npz")
        and (target_mass is None or mass_token(f.name) == target_mass)
    )


def _process_one(job):
    """
    Worker function run in each Pool process for a single tree file.
    Must stay a plain module-level function (not a closure) so it can be
    pickled out to worker processes. Returns a small, cheap-to-pickle
    status tuple -- (status, logM0, filename, detail) -- rather than the
    EpsilonRatioOrbitInit object itself, since that carries the full
    coordinates array and shipping that back across process boundaries
    for every tree would be wasteful.

    detail is only populated for "error" (always -- rare, and you want to
    know why) and for "done" when verbose=True. Skipping the per-subhalo
    summary string (and the print it would otherwise cause in the main
    process) is the point of verbose=False: at a few thousand trees, one
    print per tree is real overhead, not just log noise.
    """
    tree_file, output_dir, mean_mah_dir, A, order_filter, eps_min, eps_max, verbose = job
    tree_file = Path(tree_file)

    with np.load(tree_file) as d:
        logM0 = infer_logM0(d["mass"])
    if not (Path(mean_mah_dir) / f"{logM0:.1f}_files_mean_MAH.npz").exists():
        return ("skipped", logM0, tree_file.name, None)

    try:
        erv = EpsilonRatioOrbitInit(
            tree_file=tree_file,
            output_dir=output_dir,
            mean_mah_dir=mean_mah_dir,
            A=A,
            order_filter=order_filter,
            eps_min=eps_min,
            eps_max=eps_max,
            suffix="",  # preserve the original filename for jsm_SubEvo.py
        )
        summary = erv.summary() if verbose else None
        erv.run()
        return ("done", logM0, tree_file.name, summary)
    except Exception as e:
        return ("error", logM0, tree_file.name, str(e))


def apply_epsilon_ratio_to_directory(input_dir=DEFAULT_INPUT_DIR,
                                      output_dir=DEFAULT_OUTPUT_DIR,
                                      mean_mah_dir=DEFAULT_MEAN_MAH_DIR,
                                      A=1.0, target_mass=None,
                                      order_filter=1, eps_min=0.5, eps_max=1.5,
                                      ncores=DEFAULT_NCORES,
                                      verbose=False, progress_every=200):
    """
    Applies EpsilonRatioOrbitInit to every raw tree file in input_dir
    (optionally restricted to one mass bin via target_mass), writing each
    modified tree into output_dir under its ORIGINAL filename (no suffix),
    so the resulting directory is a drop-in datadir for jsm_SubEvo.py.
    Trees are processed across `ncores` worker processes; progress is
    printed as each tree completes.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw (un-evolved) tree_*.npz files.
        Default: the cluster path holding the N1000 zhao trees.
    output_dir : str or Path
        Directory the modified trees are written into (created if needed).
        Default: the "epsilon_orbits" sibling directory on the cluster.
    mean_mah_dir : str or Path
        Directory holding "{logM0:.1f}_files_mean_MAH.npz" reference files.
        Default: the SatGen repo's etc/mean_MAH/, as an absolute cluster path.
    A : float
        Perturbation-strength parameter passed straight through to every
        EpsilonRatioOrbitInit -- see that class for the full explanation.
        Default 1.0 reproduces the original ratio-only model; A=0 disables
        the perturbation for the whole batch.
    target_mass : str or None
        Mass-bin token to restrict input_dir's tree files to (e.g. "13.0"),
        matching the tree_{mass}_{idx}.npz naming convention. Default None
        processes every raw tree file found (subject to mean_mah coverage).
    order_filter : int
        Instantaneous order at accretion to target (default: 1).
    eps_min, eps_max : float
        epsilon(z) clip range (default: 0.5, 1.5). eps_min must be > 0.
    ncores : int
        Number of worker processes (default: 16).
    verbose : bool
        If True, print a per-tree summary line (epsilon min/max/mean,
        subhalo counts, etc.) as each tree finishes. Default False --
        with a few thousand trees this print is real per-tree overhead,
        not just noise, so it's off unless you're debugging a small run.
        The periodic progress line (every `progress_every` trees) prints
        regardless, and errors always print.
    progress_every : int
        Print a one-line progress rollup after every this-many trees
        finish (and once more at the end). Default 200.

    Returns
    -------
    dict with counts: {"processed": int, "skipped": int, "errors": int,
    "skipped_by_bin": {logM0: count}}.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    mean_mah_dir = Path(mean_mah_dir)
    tree_files = find_raw_tree_files(input_dir, target_mass=target_mass)
    total = len(tree_files)

    if not tree_files:
        print(f"no raw tree files found in {input_dir}"
              + (f" (mass bin {target_mass})" if target_mass else ""))
        return {"processed": 0, "skipped": 0, "errors": 0, "skipped_by_bin": {}}

    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "input_dir and output_dir are the same directory -- refusing to "
            "write modified trees on top of the raw ones. Pick a different "
            "output_dir (e.g. a subdirectory of input_dir)."
        )

    # create it up front so worker processes never race each other to do so
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"found {total} raw tree file(s) in {input_dir}"
          + (f" (mass bin {target_mass})" if target_mass else "")
          + f"; A={A}, eps_min={eps_min}, eps_max={eps_max}; "
          f"processing with {ncores} core(s)")

    jobs = [(f, output_dir, mean_mah_dir, A, order_filter, eps_min, eps_max, verbose) for f in tree_files]

    n_done = n_skipped = n_error = 0
    skipped_by_bin = {}
    time_start = time.time()

    with Pool(ncores) as pool:
        for i, (status, logM0, name, detail) in enumerate(
                pool.imap_unordered(_process_one, jobs), start=1):

            if status == "done":
                n_done += 1
                if verbose:
                    print(f"[{i}/{total}] {detail}")
            elif status == "skipped":
                n_skipped += 1
                skipped_by_bin[logM0] = skipped_by_bin.get(logM0, 0) + 1
            elif status == "error":
                n_error += 1
                # always printed, regardless of verbose -- rare and actionable
                print(f"[{i}/{total}] ERROR on {name}: {detail}")

            if i % progress_every == 0 or i == total:
                elapsed_min = (time.time() - time_start) / 60.0
                print(f"--- progress: {i}/{total} tree(s) handled "
                      f"({n_done} scaled, {n_skipped} skipped, {n_error} errored), "
                      f"{elapsed_min:.1f} min elapsed ---")

    print(f"\nwrote {n_done} modified tree(s) to {output_dir}")
    if skipped_by_bin:
        by_bin = ", ".join(f"{k:.1f} ({v})" for k, v in sorted(skipped_by_bin.items()))
        print(f"skipped {n_skipped} tree(s) outside mean_MAH coverage, by bin: {by_bin}")
    if n_error:
        print(f"{n_error} tree(s) errored -- see ERROR lines above")

    return {"processed": n_done, "skipped": n_skipped, "errors": n_error,
            "skipped_by_bin": skipped_by_bin}


def run_scale_sweep(A_values=DEFAULT_A_VALUES, scale_root=DEFAULT_SCALE_ROOT,
                     input_dir=DEFAULT_INPUT_DIR, mean_mah_dir=DEFAULT_MEAN_MAH_DIR,
                     target_mass=DEFAULT_TARGET_MASS, order_filter=1,
                     eps_min=DEFAULT_SWEEP_EPS_MIN, eps_max=DEFAULT_SWEEP_EPS_MAX,
                     ncores=DEFAULT_NCORES, verbose=False, progress_every=200):
    """
    Runs apply_epsilon_ratio_to_directory() once per A in A_values, writing
    each set of epsilon-scaled trees into scale_dir_for(A, scale_root)
    (scale_root/scale_A{A:g}/). Defaults to A=3,6,9 on the 13.0 mass bin
    (1000 trees), matching the three directories already created for this
    sweep, with eps_min/eps_max widened well beyond the single-run default
    of [0.5, 1.5] -- necessary at these A values, since the ratio-deviation
    term routinely exceeds that range even before A scales it up further
    (see epsilon_ratio_orbit_init.py's module docstring).

    Returns
    -------
    dict keyed by A, each value the {"processed", "skipped", "errors",
    "skipped_by_bin"} dict from that A's apply_epsilon_ratio_to_directory() call.
    """
    results = {}
    for A in A_values:
        output_dir = scale_dir_for(A, scale_root)
        print(f"\n=== scale sweep: A={A:g} -> {output_dir} ===")
        results[A] = apply_epsilon_ratio_to_directory(
            input_dir=input_dir, output_dir=output_dir, mean_mah_dir=mean_mah_dir,
            A=A, target_mass=target_mass, order_filter=order_filter,
            eps_min=eps_min, eps_max=eps_max, ncores=ncores,
            verbose=verbose, progress_every=progress_every)
    return results


def _parse_args():
    p = argparse.ArgumentParser(
        description="Apply EpsilonRatioOrbitInit to every raw tree file in a "
                     "directory, writing epsilon(z)-scaled copies (same "
                     "filenames) into a new directory usable directly by "
                     "jsm_SubEvo.py. With no arguments, uses the cluster "
                     "N1000 zhao paths below. Pass --sweep to instead run "
                     "the A=3,6,9 scale sweep into scale_root/scale_A{A}/."
    )
    p.add_argument("input_dir", type=str, nargs="?", default=DEFAULT_INPUT_DIR,
                    help=f"directory of raw (un-evolved) tree_*.npz files "
                         f"(default: {DEFAULT_INPUT_DIR})")
    p.add_argument("output_dir", type=str, nargs="?", default=DEFAULT_OUTPUT_DIR,
                    help=f"directory to write the modified trees into; ignored "
                         f"when --sweep is set (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("mean_mah_dir", type=str, nargs="?", default=DEFAULT_MEAN_MAH_DIR,
                    help=f"directory holding '{{logM0:.1f}}_files_mean_MAH.npz' "
                         f"reference files (default: {DEFAULT_MEAN_MAH_DIR})")
    p.add_argument("--A", type=float, default=1.0,
                    help="perturbation-strength parameter for a single (non-sweep) run "
                         "(default: 1.0, reproduces the original ratio-only model; "
                         "A=0 disables the perturbation entirely)")
    p.add_argument("--target-mass", default=None,
                    help="mass-bin token to restrict to, e.g. '13.0' (default: process "
                         "every raw tree found); --sweep always uses its own "
                         "--A-values/--target-mass default of '13.0' unless overridden here")
    p.add_argument("--order-filter", type=int, default=1,
                    help="instantaneous order at accretion to target (default: 1)")
    p.add_argument("--eps-min", type=float, default=None,
                    help="lower epsilon clip; must be > 0. Default 0.5 for a single run, "
                         "0.01 for --sweep")
    p.add_argument("--eps-max", type=float, default=None,
                    help="upper epsilon clip. Default 1.5 for a single run, 30.0 for --sweep")
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                    help=f"number of worker processes (default: {DEFAULT_NCORES})")
    p.add_argument("--verbose", action="store_true",
                    help="print a per-tree summary line as each tree finishes "
                         "(default: off -- at scale this print is real overhead, "
                         "not just noise; the periodic progress line and any "
                         "errors print regardless)")
    p.add_argument("--progress-every", type=int, default=200,
                    help="print a progress rollup every this-many trees (default: 200)")
    p.add_argument("--sweep", action="store_true",
                    help="run the A-scale sweep (see run_scale_sweep()) instead of a "
                         "single A run")
    p.add_argument("--A-values", default=",".join(f"{a:g}" for a in DEFAULT_A_VALUES),
                    help=f"comma-separated A values for --sweep (default: "
                         f"{','.join(f'{a:g}' for a in DEFAULT_A_VALUES)})")
    p.add_argument("--scale-root", default=DEFAULT_SCALE_ROOT,
                    help=f"parent directory for --sweep's scale_A{{A}}/ output directories "
                         f"(default: {DEFAULT_SCALE_ROOT})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.sweep:
        A_values = tuple(float(x) for x in args.A_values.split(","))
        run_scale_sweep(
            A_values=A_values,
            scale_root=args.scale_root,
            input_dir=args.input_dir,
            mean_mah_dir=args.mean_mah_dir,
            target_mass=(args.target_mass or DEFAULT_TARGET_MASS),
            order_filter=args.order_filter,
            eps_min=(args.eps_min if args.eps_min is not None else DEFAULT_SWEEP_EPS_MIN),
            eps_max=(args.eps_max if args.eps_max is not None else DEFAULT_SWEEP_EPS_MAX),
            ncores=args.ncores,
            verbose=args.verbose,
            progress_every=args.progress_every,
        )
    else:
        apply_epsilon_ratio_to_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mean_mah_dir=args.mean_mah_dir,
            A=args.A,
            target_mass=args.target_mass,
            order_filter=args.order_filter,
            eps_min=(args.eps_min if args.eps_min is not None else 0.5),
            eps_max=(args.eps_max if args.eps_max is not None else 1.5),
            ncores=args.ncores,
            verbose=args.verbose,
            progress_every=args.progress_every,
        )
