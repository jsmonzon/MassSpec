"""
run_abundance.py (environment_test)

Abundance-matching counterpart to integrate_z50_ratio_to_directory.py:
runs jsm_stellarhalo.Tree_Reader's per-tree measurement (compute_concentration
+ write_out_abundance) over the now-evolved z50-ratio A-sweep trees, and
writes ALL of them into ONE combined HDF5 file -- a drop-in for
jsm_processh5.ProcessH5's grouped-file mode.

Modeled directly on src/epsilon_orbits/run_abundance.py's
run_scale_sweep_abundance() -- same physics call (compute_concentration()
must run first, since write_out_abundance() reads self.without_sub /
self.with_sub / etc, which only compute_concentration() populates -- see
that script's docstring for the exact "'Tree_Reader' object has no
attribute 'without_sub'" failure mode this avoids), same per-tree
exception handling, same final HDF5 layout (one group per tree_index
under a per-A top-level group, "A0.20"/"A0.40"/... matching this
project's own directory naming, NOT epsilon's "A3"/"A6"/"A9" integer-like
convention -- these two A conventions must never be conflated).

Revision 2026-09-03: two design changes, both raised in discussion about
streamlining the full pipeline (generate -> integrate -> abundance-match
-> process) before committing to a ~4000-tree cluster run:

  1. ONE FLAT POOL ACROSS EVERY A, not four sequential ones. The original
     version looped over A_values, spinning up a fresh multiprocessing.Pool
     and fully draining it before moving to the next A -- meaning FOUR
     separate straggler-tails instead of one, and no load-balancing across
     A directories even though tree-processing time varies tree to tree.
     This version collects every (A, tree) job that needs doing across ALL
     A directories into one flat list FIRST, then runs it through a single
     shared Pool with imap_unordered -- exactly the same flattening
     integrate_z50_ratio_to_directory.py's os.walk-based file discovery
     already does across its own A{A:.2f}/ directories, for the same
     reason.

  2. RESUMABLE, via small per-tree staging files instead of one big
     in-memory dict_dict written out only at the very end. The original
     version held every tree's full result in memory across all four A's
     and only touched disk once, when it was completely done -- so a crash
     on tree 3999 of 4000 lost everything. Now each worker writes its own
     tree's result to staging_dir/A{A:.2f}/{tree_index}.npz IMMEDIATELY
     after computing it (mirrors integrate_z50_ratio_to_directory.py's
     "<name>_evo.npz" written immediately, tree by tree). Before dispatching
     jobs, any tree whose staging file already exists is skipped entirely
     (no Tree_Reader call at all, just a path check) -- so an interrupted
     or re-run job only redoes what's actually missing. A separate, fast,
     low-risk MERGE step (run automatically at the end, or on its own via
     --merge-only) then assembles whatever's currently in staging_dir into
     the single combined HDF5 jsm_processh5.ProcessH5 expects -- so the
     expensive part (the Tree_Reader measurement) and the cheap part
     (HDF5 assembly) are now separate, independently-resumable stages, the
     same separation the streamlining discussion identified as missing.
     Pass --force to reprocess and overwrite trees that already have a
     staged result (e.g. after changing --mass-cut/--c-true/--rng-seed).

Also new: an A directory that doesn't exist yet (e.g. integration for that
A hasn't started/finished) is skipped with a warning instead of crashing
find_evo_files' os.listdir -- makes this safe to run repeatedly as the
rest of the pipeline is still catching up, which matters once this script
is chained after integration in one queued job (see the pipeline
chaining instructions being drafted alongside this).

Note: apply_z50_ratio_to_directory.py's default A_values (0.2, 0.4, 0.6,
0.8) do NOT include an A=0.00/unscaled baseline directory -- unlike the
3-case pilot, which always evolved a tree_fid.npz alongside its A-scaled
copies. If a true no-scaling comparison point is wanted for the full
sample, it would have to come from separately evolving+processing the
ORIGINAL raw trees in apply_z50_ratio_to_directory.py's own input_dir
(m(A=0)=1 regardless of rat, so the unscaled tree already IS every tree's
A=0 case) -- this script only processes what integrate_z50_ratio_to_directory.py
actually evolved (the A{A:.2f}/ directories), it does not evolve or
process anything in input_dir itself.

mass_cut=7.75e10 (the "N500", 500-particle resolution-completeness cut)
and parentdir="/home/jsm99/SatGen/mcmc/src/" are reused verbatim from
epsilon's run_abundance.py, not new guesses -- this test's raw trees are
the exact same "zhao N1000, 13.0 mass bin" population epsilon's tests use
(see apply_z50_ratio_to_directory.py's own DEFAULT_INPUT_DIR, which reuses
epsilon's confirmed cluster path for this same reason), so the same
resolution cut and the same jsm_stellarhalo.py location apply unchanged.
DEFAULT_SAVE_PATH/DEFAULT_STAGING_DIR below ARE new placeholders (an
environment_test sibling of epsilon's own "/home/jsm99/data/epsilon/"
convention) -- confirm/adjust before relying on them, same caveat as
every other placeholder path in this project's cluster scripts.

Read the final combined h5 back with jsm_processh5.ProcessH5's grouped-file mode:
    from jsm_processh5 import ProcessH5
    proc = ProcessH5("/home/jsm99/data/environment_test", label="z50_ratio_A_sweep",
                      files=[("z50_ratio_A_sweep.h5", f"A{A:.2f}")
                             for A in (0.2, 0.4, 0.6, 0.8)])
    proc.process(which=("z0",))

Usage (on the cluster, after integrate_z50_ratio_to_directory.py has
evolved at least one A directory):
    python run_abundance.py
        # measures every not-yet-staged tree across all 4 A directories,
        # then merges everything currently staged into one combined h5

    python run_abundance.py --A-values 0.2,0.4 --ncores 32
        # just a subset of A's, e.g. while the rest are still integrating

    python run_abundance.py
        # re-running later, after more A directories have finished
        # integrating, only measures the NEW trees -- already-staged ones
        # are skipped automatically

    python run_abundance.py --merge-only
        # skip the (possibly slow) measurement pass entirely and just
        # rebuild the combined h5 from whatever is currently staged --
        # useful to check progress or after changing nothing but wanting
        # a fresh h5

    python run_abundance.py --force
        # reprocess and overwrite every tree's staged result, e.g. after
        # changing --mass-cut/--c-true/--rng-seed

    python run_abundance.py --datadir /path/to/environment_test_full \\
        --staging-dir /home/jsm99/data/environment_test/_abundance_parts \\
        --save-path /home/jsm99/data/environment_test/z50_ratio_A_sweep.h5
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_z50_ratio_to_directory import a_dir_for, DEFAULT_A_VALUES, DEFAULT_OUTPUT_ROOT, DEFAULT_TARGET_MASS  # noqa: E402

DEFAULT_DATADIR = DEFAULT_OUTPUT_ROOT
DEFAULT_SAVE_DIR = "/home/jsm99/data/environment_test/"
DEFAULT_SAVE_PATH = "/home/jsm99/data/environment_test/z50_ratio_A_sweep.h5"
DEFAULT_STAGING_DIR = "/home/jsm99/data/environment_test/_abundance_parts/"
DEFAULT_PARENTDIR = "/home/jsm99/SatGen/mcmc/src/"  # reused from epsilon_orbits/run_abundance.py -- same jsm_stellarhalo.py location, confirmed working path, not a new guess
DEFAULT_MASS_CUT = 7.75e10  # "N500" resolution-completeness cut, project-wide convention (fiducial/N_particle/epsilon_orbits) -- same raw tree population as epsilon, so reused verbatim
DEFAULT_NCORES = 16
DEFAULT_C_TRUE = "zhao"     # matches this project's cosmology + epsilon_orbits/run_abundance.py's convention
DEFAULT_RNG_SEED = 42       # matches epsilon_orbits/run_abundance.py's fixed-seed convention


def find_evo_files(input_dir, target_mass=None):
    """List tree_{mass}_{idx}_evo.npz files in input_dir, optionally restricted
    to a single mass-bin token (e.g. "13.0"). Copied verbatim from
    epsilon_orbits/run_abundance.py -- same filename convention, since
    integrate_z50_ratio_to_directory.py's loop() writes evolved trees the
    exact same way jsm_SubEvo.py / integrate_environment_test.py do."""

    def mass_token(filename):
        return filename.split("_")[1]

    return sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.startswith("tree") and filename.endswith("evo.npz")
        and (target_mass is None or mass_token(filename) == target_mass)
    )


def tree_index_from_filename(path):
    """Matches jsm_stellarhalo.Tree_Reader.read_arrays()'s own tree_index
    parsing exactly (self.file.split("/")[-1].split("_")[2]) -- computed
    here from the filename alone, with no file I/O, so staging paths can
    be checked/named before (and without ever needing) a Tree_Reader call."""
    return os.path.basename(path).split("_")[2]


def staging_path_for(staging_dir, A_label, tree_index):
    return Path(staging_dir) / A_label / f"{tree_index}.npz"


def _collect_jobs(A_values, datadir, staging_dir, target_mass=None,
                   mass_cut=DEFAULT_MASS_CUT, parentdir=DEFAULT_PARENTDIR,
                   c_true=DEFAULT_C_TRUE, rng_seed=DEFAULT_RNG_SEED, force=False):
    """Flattens every (A, tree) pair that still needs abundance-matching
    into ONE job list across ALL A directories -- see module docstring,
    point 1. Skips a tree whose staging output already exists (unless
    force=True) and skips an A directory that doesn't exist yet (e.g.
    integration for that A hasn't started), printing why either way so a
    re-run's console output makes clear what's actually left to do.
    Returns (jobs, counts) where counts is {A_label: {"total", "todo"}}.
    """
    jobs = []
    counts = {}
    for A in A_values:
        label = f"A{A:.2f}"
        input_dir = a_dir_for(A, datadir)
        if not Path(input_dir).is_dir():
            print(f"[{label}] {input_dir} does not exist yet -- skipping")
            counts[label] = {"total": 0, "todo": 0}
            continue

        files = find_evo_files(str(input_dir), target_mass=target_mass)
        todo = 0
        for file_i in files:
            tree_index = tree_index_from_filename(file_i)
            out_path = staging_path_for(staging_dir, label, tree_index)
            if out_path.exists() and not force:
                continue
            jobs.append((file_i, label, tree_index, str(out_path), mass_cut, parentdir, c_true, rng_seed))
            todo += 1

        counts[label] = {"total": len(files), "todo": todo}
        skipped = len(files) - todo
        print(f"[{label}] {len(files)} evolved tree(s) found in {input_dir}, "
              f"{todo} need processing" + (f" ({skipped} already staged)" if skipped else ""))

    return jobs, counts


def _process_one(job):
    """Per-tree worker: Tree_Reader -> compute_concentration (must run
    first -- see module docstring) -> write_out_abundance -> stage the
    result to its own small npz file immediately (see module docstring,
    point 2). jsm_stellarhalo is imported inside the worker (not at module
    level) so it's picked up fresh in each Pool process via parentdir,
    matching the epsilon_orbits precedent's pattern."""
    file_i, label, tree_index, out_path, mass_cut, parentdir, c_true, rng_seed = job
    sys.path.insert(0, parentdir)
    import jsm_stellarhalo
    try:
        tree_i = jsm_stellarhalo.Tree_Reader(file=file_i, mass_threshold=mass_cut, verbose=False)
        tree_i.compute_concentration(rng=np.random.default_rng(rng_seed), c_true=c_true)
        entry = tree_i.write_out_abundance()
        attrs = {k: v for k, v in entry.items() if k != "tree_index"}
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, **attrs)
        return ("done", label, tree_index, None)
    except Exception as e:
        return ("error", label, os.path.basename(file_i), str(e))


def _write_tree_group(h5_group, tree_index, attrs):
    """One HDF5 group named tree_index under h5_group, one dataset per
    attribute -- matches write_out_abundance()'s convention (scalars
    uncompressed, arrays gzip level 9, same as epsilon_orbits/run_abundance.py)."""
    sim_group = h5_group.create_group(tree_index)
    for attr_name, data in attrs.items():
        if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
            sim_group.create_dataset(attr_name, data=data)  # no compression
        else:
            sim_group.create_dataset(attr_name, data=data, compression="gzip", compression_opts=9)


def run_measurements(jobs, ncores=DEFAULT_NCORES, progress_every=200):
    """Runs every job in `jobs` (built by _collect_jobs) through ONE shared
    Pool, regardless of which A directory each came from -- see module
    docstring, point 1. Each worker stages its own result to disk as it
    finishes (point 2); this function just tallies done/error counts per A
    for the printed summary. Returns {A_label: {"done", "errors"}}."""
    tallies = {}
    Ntotal = len(jobs)
    if Ntotal == 0:
        print("nothing to measure -- every tree already staged (or no A directories exist yet)")
        return tallies

    t0 = time.time()
    with mp.Pool(processes=ncores) as pool:
        for i, (status, label, ident, err) in enumerate(pool.imap_unordered(_process_one, jobs), 1):
            tally = tallies.setdefault(label, {"done": 0, "errors": 0})
            if status == "done":
                tally["done"] += 1
            else:
                tally["errors"] += 1
                print(f"  [{label}] ERROR: {ident}: {err}")

            if i % progress_every == 0 or i == Ntotal:
                elapsed = time.time() - t0
                print(f"--- progress: {i}/{Ntotal} ({elapsed:.0f}s elapsed) ---")

    print(f"measurement pass done in {(time.time() - t0):.0f}s")
    for label, t in tallies.items():
        print(f"  [{label}] {t['done']} done, {t['errors']} errors")
    return tallies


def merge_staging_to_h5(A_values, staging_dir, save_path):
    """Assembles whatever currently exists in staging_dir into ONE combined
    HDF5 at save_path: a top-level group per A ("A0.20", ...), each holding
    the usual one-group-per-tree_index layout underneath -- a drop-in for
    jsm_processh5.ProcessH5's grouped-file mode (see module docstring).
    Deliberately separate from run_measurements(): this is the cheap, safe,
    idempotent part -- rerunning it just rebuilds save_path from the
    current staging contents, no Tree_Reader calls involved.

    Discovers A-labels from staging_dir's own subdirectories rather than
    from the A_values argument (kept only so a caller with no staging_dir
    subfolders yet -- e.g. before any measurement pass has run -- still
    gets an informative "nothing staged" message per requested A). This
    matters because save_path is rebuilt from scratch (mode "w") on every
    call: if merge only wrote the A's passed to *this* invocation, a
    `--force --A-values 0.2` re-run (reprocessing just one A after e.g. a
    mid-run error) would silently drop every other A's already-merged
    groups from save_path. Merging everything actually on disk in
    staging_dir, regardless of what this call's --A-values happened to
    be, keeps merge a true "assemble current state" operation -- verified
    by this exact scenario during testing (see project memory)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    staged_labels = sorted(p.name for p in Path(staging_dir).iterdir() if p.is_dir()) \
        if Path(staging_dir).is_dir() else []
    requested_labels = [f"A{A:.2f}" for A in A_values]
    labels = sorted(set(staged_labels) | set(requested_labels))
    with h5py.File(save_path, "w") as f:
        for label in labels:
            model_group = f.create_group(label)
            a_dir = Path(staging_dir) / label
            if not a_dir.is_dir():
                print(f"[{label}] no staged results yet ({a_dir} does not exist) -- empty group written")
                continue
            n = 0
            for part_file in sorted(a_dir.glob("*.npz")):
                tree_index = part_file.stem
                with np.load(part_file) as d:
                    attrs = {k: d[k] for k in d.files}
                _write_tree_group(model_group, tree_index, attrs)
                n += 1
            print(f"[{label}] merged {n} tree(s) from {a_dir} into {save_path}")

    print(f"merge done -> {save_path}")


def run_z50_sweep_abundance(A_values=DEFAULT_A_VALUES, datadir=DEFAULT_DATADIR,
                             staging_dir=DEFAULT_STAGING_DIR, save_path=DEFAULT_SAVE_PATH,
                             target_mass=DEFAULT_TARGET_MASS, parentdir=DEFAULT_PARENTDIR,
                             mass_cut=DEFAULT_MASS_CUT, c_true=DEFAULT_C_TRUE,
                             rng_seed=DEFAULT_RNG_SEED, ncores=DEFAULT_NCORES,
                             progress_every=200, force=False, merge_only=False):
    """
    Main entry point. Two stages:
      1. (skipped entirely if merge_only=True) collect every (A, tree) job
         still needing abundance-matching across every A in A_values, run
         them all through one shared Pool, each result staged to disk as
         soon as it's computed.
      2. merge whatever is currently in staging_dir into one combined h5
         at save_path.

    Safe to call repeatedly as more A directories finish integrating --
    already-staged trees are skipped (unless force=True), and the merge
    step just rebuilds save_path from the current staging contents each
    time. Returns the per-A measurement tallies (empty dict if merge_only).
    """
    tallies = {}
    if not merge_only:
        jobs, counts = _collect_jobs(A_values, datadir, staging_dir, target_mass=target_mass,
                                      mass_cut=mass_cut, parentdir=parentdir, c_true=c_true,
                                      rng_seed=rng_seed, force=force)
        tallies = run_measurements(jobs, ncores=ncores, progress_every=progress_every)

    merge_staging_to_h5(A_values, staging_dir, save_path)
    return tallies


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datadir", type=str, default=DEFAULT_DATADIR,
                   help="parent directory holding the A{A:.2f}/ evolved-tree directories "
                        f"(default: apply_z50_ratio_to_directory.py's own DEFAULT_OUTPUT_ROOT, "
                        f"currently {DEFAULT_DATADIR})")
    p.add_argument("--A-values", default=",".join(f"{a:g}" for a in DEFAULT_A_VALUES),
                   help=f"comma-separated A values to process (default: "
                        f"{','.join(f'{a:g}' for a in DEFAULT_A_VALUES)})")
    p.add_argument("--staging-dir", default=DEFAULT_STAGING_DIR,
                   help=f"per-tree staged-result directory, {{staging_dir}}/A{{A:.2f}}/{{tree_index}}.npz "
                        f"(default: {DEFAULT_STAGING_DIR})")
    p.add_argument("--save-path", default=DEFAULT_SAVE_PATH,
                   help=f"output path for the single combined h5, rebuilt from staging_dir every "
                        f"run (default: {DEFAULT_SAVE_PATH})")
    p.add_argument("--target-mass", default=DEFAULT_TARGET_MASS,
                   help="mass-bin token to restrict to, e.g. '13.0' (default: %(default)s); "
                        "pass '' to disable filtering")
    p.add_argument("--parentdir", default=DEFAULT_PARENTDIR,
                   help=f"directory holding jsm_stellarhalo.py (default: {DEFAULT_PARENTDIR})")
    p.add_argument("--mass-cut", type=float, default=DEFAULT_MASS_CUT)
    p.add_argument("--c-true", default=DEFAULT_C_TRUE, choices=["zhao", "ludlow"],
                   help="concentration model fed to compute_concentration() (default: %(default)s)")
    p.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--force", action="store_true",
                   help="reprocess every tree even if it already has a staged result "
                        "(e.g. after changing --mass-cut/--c-true/--rng-seed)")
    p.add_argument("--merge-only", action="store_true",
                   help="skip the measurement pass entirely and just rebuild --save-path "
                        "from whatever is currently in --staging-dir")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    A_values = tuple(float(x) for x in args.A_values.split(","))
    run_z50_sweep_abundance(
        A_values=A_values, datadir=args.datadir, staging_dir=args.staging_dir, save_path=args.save_path,
        target_mass=(args.target_mass or None), parentdir=args.parentdir, mass_cut=args.mass_cut,
        c_true=args.c_true, rng_seed=args.rng_seed, ncores=args.ncores,
        progress_every=args.progress_every, force=args.force, merge_only=args.merge_only)
