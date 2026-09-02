"""
apply_z50_ratio_to_directory.py

Batch version of the z50-ratio vr-scaling test (build_environment_test.py),
scaled up from the 3 hand-picked early/middle/late cases to the FULL
13.0-mass-bin sample (1000 trees, "the zhao data directory"). Mirrors
src/epsilon_orbits/apply_epsilon_ratio_to_directory.py's directory-batch +
multiprocessing pattern, adapted to this test's own scaling law:

    rat = (1 + z50) / (1 + <z50>)
    vr' = [1 - A*(1 - rat)] * vr   -- first-order (order==1 at accretion)
                                       subhalos only, IN PLACE on the
                                       original tree's own coordinates.

No orbit.resample_orbit(), no SatGen dependency at all -- same "write over
the original vr, don't redraw the orbit" approach as build_environment_test.py
and the epsilon test before it. Every A-scaled copy of a given tree shares
that tree's exact orbit realization; the only thing that differs is the
deterministic vr multiplier.

Why a new script rather than looping build_environment_test.py: that
script's case_rat()/apply_scaling() are written for 3 NAMED cases at a
time and aren't parallelized -- fine for 3 trees, not for 1000. This
script generalizes the same physics to every raw tree in a directory
(optionally restricted to one mass-bin token), and uses
multiprocessing.Pool the same way apply_epsilon_ratio_to_directory.py
does: a picklable module-level worker function, imap_unordered for
progress-as-you-go, per-tree exception handling so one bad file doesn't
kill the batch.

Directory layout
-----------------
For each A in A_VALUES, writes ONE subdirectory named "A{A:.2f}" under
output_root (same per-A naming this experiment has used all along, e.g.
environment_test/early/tree_A0.25.npz used "A0.25"), containing every
scaled tree under its ORIGINAL, UNCHANGED filename:

    output_root/
      A0.20/
        tree_13.0_0.npz
        tree_13.0_1.npz
        ...  (1000 files)
      A0.40/
        ...
      A0.60/
      A0.80/

Preserving filenames (rather than folding A into the filename, the way
the 3-case pilot's tree_A0.25.npz did) means each A{A:.2f}/ directory is a
drop-in datadir for a jsm_SubEvo.py-style integrator -- point it at one
A-subdirectory and it discovers/evolves those trees like any other
datadir. Same rationale apply_epsilon_ratio_to_directory.py gives for
preserving filenames.

Two phases, both parallelized
-------------------------------
Phase 1 (cheap): compute host_z50 for every raw tree found -- only the
small "mass" and "redshift" arrays are touched, not the full
coordinates/order arrays -- using the same nearest-index method
(host_z50()) used everywhere else in this project. This ALWAYS runs,
even if --mean-z50 overrides the population mean (see below), because
each tree's own z50 is still needed to compute that tree's own rat.

<z50> is then either:
  - the mean of this run's own phase-1 z50 values (default) -- the TRUE
    population mean of whatever tree set is actually pointed at, with no
    dependency on the earlier N1000 h5 summary file staying in sync with
    wherever these raw trees actually live once the server path is
    confirmed; or
  - a value passed via --mean-z50, e.g. 0.9304333832432538 -- the exact
    <z50> already established for the 3-case pilot (data/zhao/N1000/
    13.0_files.h5's precomputed host_z50, averaged) -- pass this if you
    want this full-sample run to use the IDENTICAL normalization anchor
    as early/middle/late, for direct comparability, rather than
    recomputing it from (what should be, but isn't guaranteed to be) the
    same 1000 trees.

Phase 2: for every tree, scale to every A in A_VALUES and write all 4
outputs in one job -- one job per TREE, not per (tree, A) pair, so each
raw file is only read from disk once. Scaling logic is copied from (not
imported from) build_environment_test.py's apply_scaling(), so this
script has no import-path dependency on that one and can run standalone
wherever the raw trees live.

Usage
-----
    # once the real server path is confirmed:
    python apply_z50_ratio_to_directory.py INPUT_DIR OUTPUT_ROOT \\
        --target-mass 13.0 --A-values 0.2,0.4,0.6,0.8 --ncores 16

    # reuse the pilot's own <z50> instead of recomputing it here
    python apply_z50_ratio_to_directory.py INPUT_DIR OUTPUT_ROOT \\
        --mean-z50 0.9304333832432538

    # with no arguments, uses the placeholder cluster defaults below --
    # DO NOT rely on these without checking; see "Cluster deployment"
"""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

VR_INDEX = 3  # [R, phi, z, VR, Vphi, Vz]

# --- Cluster deployment -----------------------------------------------
# PLACEHOLDER defaults. INPUT_DIR reuses the exact confirmed cluster path
# apply_epsilon_ratio_to_directory.py already uses for the same "zhao
# N1000, 13.0 mass bin" tree population -- a reasonable default since
# it's the same underlying sample, not a guess -- but the user said
# they'll confirm the real path later, so treat this as tentative until
# then. OUTPUT_ROOT is a new sibling directory alongside epsilon_orbits/,
# not yet created on the cluster -- pick/confirm a real location before
# a real run.
DEFAULT_INPUT_DIR = "/netb/vdbosch/jsm99/data/mass_spec_zhao/"
DEFAULT_OUTPUT_ROOT = "/netb/vdbosch/jsm99/data/mass_spec_zhao/environment_test_full/"
DEFAULT_TARGET_MASS = "13.0"
DEFAULT_A_VALUES = (0.2, 0.4, 0.6, 0.8)
DEFAULT_NCORES = 16  # matches apply_epsilon_ratio_to_directory.py / jsm_SubEvo.py's cluster ncores


def a_dir_for(A, output_root):
    """Per-A output directory: output_root/A{A:.2f} -- e.g. A=0.2 -> .../A0.20,
    matching this experiment's existing tree_A0.25.npz-style naming."""
    return Path(output_root) / f"A{A:.2f}"


def find_raw_tree_files(input_dir, target_mass=None):
    """Same file-discovery convention as jsm_SubEvo.py and
    apply_epsilon_ratio_to_directory.py: filenames starting with "tree",
    NOT ending in "evo.npz". target_mass (e.g. "13.0") restricts to that
    mass-bin token via filename.split("_")[1] -- same check as
    run_abundance.py's find_evo_files."""
    def mass_token(filename):
        return filename.split("_")[1]

    input_dir = Path(input_dir)
    return sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.name.startswith("tree") and not f.name.endswith("evo.npz")
        and (target_mass is None or mass_token(f.name) == target_mass)
    )


def find_nearest_idx(array, value):
    # matches jsm_ancillary.find_nearest1 exactly (nearest-neighbor, not interpolation)
    return int(np.argmin(np.abs(array - value)))


def host_z50(mass_host, redshift, target_mass=None):
    if target_mass is None:
        target_mass = mass_host[0]
    idx = find_nearest_idx(mass_host, target_mass * 0.5)
    return float(redshift[idx])


def _z50_only(tree_file):
    """Phase-1 worker: read just mass/redshift, return (filename, z50) --
    or (filename, None) on failure. Deliberately cheap: this runs once per
    tree just to get each tree's own z50 (and, by extension, the
    population mean) before the real scaling pass touches the much larger
    coordinates/order arrays."""
    tree_file = Path(tree_file)
    try:
        with np.load(tree_file) as d:
            z50 = host_z50(d["mass"][0, :], d["redshift"])
        return (tree_file.name, z50)
    except Exception:
        return (tree_file.name, None)


def apply_scaling_all_A(tree_file, A_values, rat, output_root):
    """order==1-at-accretion subhalos' VR *= m(A) = 1 - A*(1-rat), for
    every A in A_values, written to output_root/A{A:.2f}/<original
    filename>. Identical logic to apply_scaling() in
    build_environment_test.py -- copied rather than imported so this
    script has no import dependency on that one. The tree is loaded from
    disk exactly once regardless of how many A_values there are."""
    tree_file = Path(tree_file)
    with np.load(tree_file) as d:
        base_coords = d["coordinates"]
        order = d["order"]
        Nhalo = base_coords.shape[0]

        acc_index = np.full(Nhalo, -1, dtype=int)
        for i in range(1, Nhalo):
            nz = np.nonzero(base_coords[i])[0]
            if len(nz):
                acc_index[i] = nz[0]
        valid = acc_index >= 0
        acc_order = np.full(Nhalo, -1, dtype=int)
        acc_order[valid] = order[np.arange(Nhalo)[valid], acc_index[valid]]
        target_ids = np.nonzero(valid & (acc_order == 1))[0]

        other_arrays = {k: d[k] for k in
                         ("redshift", "CosmicTime", "mass", "order", "ParentID",
                          "VirialRadius", "concentration")}

    for A in A_values:
        coords = np.copy(base_coords)
        m = 1 - A * (1 - rat)
        coords[target_ids, acc_index[target_ids], VR_INDEX] *= m

        out_file = a_dir_for(A, output_root) / tree_file.name
        np.savez(out_file, coordinates=coords, **other_arrays)

    return len(target_ids)


def _scale_one(job):
    """Phase-2 worker function run in each Pool process for a single tree
    (all A_values at once -- see apply_scaling_all_A). Must stay a plain
    module-level function so it can be pickled out to worker processes."""
    tree_file, A_values, rat, output_root = job
    tree_file = Path(tree_file)
    try:
        n_targets = apply_scaling_all_A(tree_file, A_values, rat, output_root)
        return ("done", tree_file.name, n_targets, None)
    except Exception as e:
        return ("error", tree_file.name, 0, str(e))


def run(input_dir=DEFAULT_INPUT_DIR, output_root=DEFAULT_OUTPUT_ROOT,
        target_mass=DEFAULT_TARGET_MASS, A_values=DEFAULT_A_VALUES,
        mean_z50=None, ncores=DEFAULT_NCORES, progress_every=200):
    input_dir = Path(input_dir)
    output_root = Path(output_root)

    if input_dir.resolve() == output_root.resolve():
        raise ValueError(
            "input_dir and output_root are the same directory -- refusing to "
            "write scaled trees on top of the raw ones. Pick a different "
            "output_root."
        )

    tree_files = find_raw_tree_files(input_dir, target_mass=target_mass)
    total = len(tree_files)

    if not tree_files:
        print(f"no raw tree files found in {input_dir}"
              + (f" (mass bin {target_mass})" if target_mass else ""))
        return {"processed": 0, "errors": 0}

    for A in A_values:
        a_dir_for(A, output_root).mkdir(parents=True, exist_ok=True)

    print(f"found {total} raw tree file(s) in {input_dir}"
          + (f" (mass bin {target_mass})" if target_mass else "")
          + f"; A_values={list(A_values)}; {ncores} core(s)")

    # --- phase 1: host_z50 for every tree (always -- each tree needs its
    # own z50 for rat, regardless of whether <z50> itself is overridden) ---
    print(f"phase 1/2: computing host_z50 for {total} tree(s)...")
    t0 = time.time()
    with Pool(ncores) as pool:
        z50_results = list(pool.imap_unordered(_z50_only, tree_files))

    z50_by_name = {name: z for name, z in z50_results if z is not None}
    n_z50_failed = total - len(z50_by_name)
    if n_z50_failed:
        print(f"  WARNING: {n_z50_failed} tree(s) failed host_z50 computation "
              f"(corrupted or unreadable) -- excluded from <z50> and skipped in phase 2")

    if mean_z50 is not None:
        MEAN_Z50 = mean_z50
        print(f"  using provided <z50> = {MEAN_Z50:.4f} (not recomputed from this run's trees)")
    else:
        MEAN_Z50 = float(np.mean(list(z50_by_name.values())))
        print(f"  <z50> = {MEAN_Z50:.4f} (mean of {len(z50_by_name)} trees, "
              f"{(time.time()-t0)/60:.1f} min)")

    rats = np.array([(1 + z) / (1 + MEAN_Z50) for z in z50_by_name.values()])
    print(f"  rat: min={rats.min():.4f}  median={np.median(rats):.4f}  max={rats.max():.4f}")

    # --- phase 2: scale + write, one job per tree ---
    print(f"phase 2/2: scaling {len(z50_by_name)} tree(s) to A={list(A_values)} "
          f"-> {output_root}")
    jobs = [
        (input_dir / name, A_values, (1 + z50_by_name[name]) / (1 + MEAN_Z50), output_root)
        for name in z50_by_name
    ]

    n_done = n_error = 0
    t0 = time.time()
    with Pool(ncores) as pool:
        for i, (status, name, n_targets, detail) in enumerate(
                pool.imap_unordered(_scale_one, jobs), start=1):
            if status == "done":
                n_done += 1
            else:
                n_error += 1
                print(f"[{i}/{len(jobs)}] ERROR on {name}: {detail}")

            if i % progress_every == 0 or i == len(jobs):
                elapsed_min = (time.time() - t0) / 60.0
                print(f"--- progress: {i}/{len(jobs)} tree(s) handled "
                      f"({n_done} scaled, {n_error} errored), "
                      f"{elapsed_min:.1f} min elapsed ---")

    print(f"\nwrote {n_done} tree(s) x {len(A_values)} A value(s) = "
          f"{n_done * len(A_values)} file(s) under {output_root}")
    if n_error:
        print(f"{n_error} tree(s) errored -- see ERROR lines above")

    return {"processed": n_done, "errors": n_error, "mean_z50": MEAN_Z50}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Apply the z50-ratio vr-scaling test to every raw tree file "
                     "in a directory, writing one A{A:.2f}/ subdirectory per A "
                     "value (same original filenames inside). With no arguments, "
                     "uses PLACEHOLDER cluster paths -- confirm the real ones first."
    )
    p.add_argument("input_dir", type=str, nargs="?", default=DEFAULT_INPUT_DIR,
                    help=f"directory of raw (un-evolved) tree_*.npz files "
                         f"(default: {DEFAULT_INPUT_DIR})")
    p.add_argument("output_root", type=str, nargs="?", default=DEFAULT_OUTPUT_ROOT,
                    help=f"parent directory for the A{{A:.2f}}/ output subdirectories "
                         f"(default: {DEFAULT_OUTPUT_ROOT})")
    p.add_argument("--target-mass", default=DEFAULT_TARGET_MASS,
                    help=f"mass-bin token to restrict to, e.g. '13.0' (default: "
                         f"{DEFAULT_TARGET_MASS}); pass '' / None to process every "
                         f"raw tree found regardless of mass bin")
    p.add_argument("--A-values", default=",".join(f"{a:g}" for a in DEFAULT_A_VALUES),
                    help=f"comma-separated A values (default: "
                         f"{','.join(f'{a:g}' for a in DEFAULT_A_VALUES)})")
    p.add_argument("--mean-z50", type=float, default=None,
                    help="override <z50> instead of computing it from this run's own "
                         "trees (e.g. 0.9304333832432538 to match the 3-case pilot exactly)")
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                    help=f"number of worker processes for both phases (default: {DEFAULT_NCORES})")
    p.add_argument("--progress-every", type=int, default=200,
                    help="print a progress rollup every this-many trees in phase 2 (default: 200)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        input_dir=args.input_dir,
        output_root=args.output_root,
        target_mass=(args.target_mass or None),
        A_values=tuple(float(x) for x in args.A_values.split(",")),
        mean_z50=args.mean_z50,
        ncores=args.ncores,
        progress_every=args.progress_every,
    )
