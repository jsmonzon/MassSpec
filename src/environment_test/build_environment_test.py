"""
build_environment_test.py

Builds the "environment_test" tree set for the z50-ratio vr-scaling test
(sibling to src/epsilon_orbits/epsilon_ratio_orbit_init.py -- see project
notes for the full background).

CORRECTED 2026-09-01: the first version of this script called
orbit.resample_orbit() to redraw every subhalo's orbit from scratch before
scaling -- that made the A=0/0.25/0.5 trees three independent Monte-Carlo
realizations, not a controlled comparison. Per explicit user correction,
this version does exactly what EpsilonRatioOrbitInit does for the epsilon
test: it writes over the ORIGINAL tree's own VR values in place. No
resampling, no orbit.py/SatGen orbit dependency at all -- so all A-variants
of a case share the exact same underlying orbit realization as tree_fid,
and differ from each other and from fid ONLY by the deterministic vr
scaling below.

EXTENDED 2026-09-01: four more A values added (0.12, 0.37, 0.62, 0.75),
alongside the original 0.0/0.25/0.5 -- so A_VALUES now has 7 entries and
each case directory holds 8 files (fid + 7 A-scaled trees). The 4 new
values slot into the same in-place-scaling / no-resampling scheme as the
original 3; nothing about the method changed, only how many A's are swept.

For each of three cases -- early/middle/late forming, all drawn from the
13.0 mass bin of data/local_trees/unevolved_massspec/ -- this writes EIGHT
raw (un-evolved) trees into data/local_trees/environment_test/<case>/:

    tree_fid.npz    -- the original tree, byte-for-byte copied, untouched.
    tree_A0.00.npz  -- original tree with first-order-subhalo vr *= m(0)
                        (m(0)=1, so numerically identical to fid -- kept as
                        its own file for a uniform per-case layout).
    tree_A0.12.npz, tree_A0.25.npz, tree_A0.37.npz, tree_A0.50.npz,
    tree_A0.62.npz, tree_A0.75.npz -- original tree with first-order-subhalo
                        vr *= m(A) for each A in A_VALUES.

Scaling
-------
vr' = [1 - A*(1-rat)] * vr = [1 + A*(rat-1)] * vr, applied ONLY to
first-order subhalos (order==1 at their own accretion snapshot), exactly
mirroring EpsilonRatioOrbitInit._find_accretion/_select_targets but with a
single scalar rat per tree rather than a per-timestep epsilon(z).

rat = (1 + z50_tree) / (1 + MEAN_Z50), where z50_tree is this tree's own
host formation redshift (mass[0,:] crossing 0.5*mass[0,0], nearest-index,
matching SatGen's own host_z50 convention) and MEAN_Z50 is the population
mean of host_z50 across the full 1000-tree 13.0-mass-bin ensemble in
data/zhao/N1000/13.0_files.h5 (each tree group there already has a
precomputed host_z50 scalar, written by the full Tree_Reader when that
ensemble was built) -- confirmed 2026-09-01: mean=0.9304, median=0.9191,
std=0.403 across the 1000 trees, i.e. ~0.9 as expected for the S13 sample.

Usage
-----
    python build_environment_test.py --all
        # builds all 3 cases x 8 files (fid + 7 A values) in one go --
        # fast, pure numpy/h5py, no SatGen dependency needed at all.
        # Already-existing files are simply overwritten with identical
        # content (the scaling is deterministic), so re-running --all
        # after adding new A values is safe and won't disturb any
        # already-evolved _evo.npz trees (those live in the same
        # directory but are a separate integration step -- see
        # integrate_environment_test.py, which only touches trees that
        # don't already have a corresponding _evo.npz).

    # or one case/job at a time, if useful for debugging:
    python build_environment_test.py --case late --A 0.25
    python build_environment_test.py --setup-fid
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import h5py

# Real absolute paths on jsmonzon's machine (not the "~/mnt/<folder>"
# mount convention used inside Claude's remote-device sandbox) -- this
# script is meant to run directly in a local terminal too.
MASSSPEC_ROOT = Path("/Users/jsmonzon/Research/MassSpec")
TREE_DIR = MASSSPEC_ROOT / "data/local_trees/unevolved_massspec"
OUT_ROOT = MASSSPEC_ROOT / "data/local_trees/environment_test"
N1000_H5 = MASSSPEC_ROOT / "data/zhao/N1000/13.0_files.h5"

VR_INDEX = 3  # [R, phi, z, VR, Vphi, Vz]
A_VALUES = (0.0, 0.12, 0.25, 0.37, 0.5, 0.62, 0.75)  # extended 2026-09-01 (was 0.0/0.25/0.5)

CASES = {
    "late": "tree_13.0_30.npz",     # rat < 1 -- formed late, below-average z50
    "middle": "tree_13.0_31.npz",   # rat ~ 1 -- close to the ensemble mean
    "early": "tree_13.0_32.npz",    # rat > 1 -- formed early, above-average z50
}


def find_nearest_idx(array, value):
    # matches jsm_ancillary.find_nearest1 exactly (nearest-neighbor, not interpolation)
    return int(np.argmin(np.abs(array - value)))


def host_z50(mass_host, redshift, target_mass=None):
    if target_mass is None:
        target_mass = mass_host[0]
    idx = find_nearest_idx(mass_host, target_mass * 0.5)
    return float(redshift[idx])


def ensemble_mean_z50(h5_file=N1000_H5):
    """Population mean of the precomputed host_z50 scalar across all trees
    in the raw N1000 ensemble h5 for this mass bin -- the real ensemble
    average, as opposed to the mean-MAH-curve proxy used in the earlier
    walkthrough."""
    with h5py.File(h5_file, "r") as f:
        z50s = np.array([f[k]["host_z50"][()] for k in f.keys()])
    return float(z50s.mean())


def case_rat(case_name, mean_z50):
    tree_file = TREE_DIR / CASES[case_name]
    d = np.load(tree_file)
    z50 = host_z50(d["mass"][0, :], d["redshift"])
    rat = (1 + z50) / (1 + mean_z50)
    return tree_file, z50, rat


def apply_scaling(tree_file, A, rat, out_file):
    """order==1-at-accretion subhalos' VR *= m(A) = 1 - A*(1-rat).
    Operates directly on the ORIGINAL tree_file -- no resampling."""
    d = np.load(tree_file)
    coords = np.copy(d["coordinates"])
    order = d["order"]
    Nhalo = coords.shape[0]

    acc_index = np.full(Nhalo, -1, dtype=int)
    for i in range(1, Nhalo):
        nz = np.nonzero(coords[i])[0]
        if len(nz):
            acc_index[i] = nz[0]
    valid = acc_index >= 0
    acc_order = np.full(Nhalo, -1, dtype=int)
    acc_order[valid] = order[np.arange(Nhalo)[valid], acc_index[valid]]
    target_ids = np.nonzero(valid & (acc_order == 1))[0]

    m = 1 - A * (1 - rat)
    vr_before = coords[target_ids, acc_index[target_ids], VR_INDEX].copy()
    coords[target_ids, acc_index[target_ids], VR_INDEX] *= m
    vr_after = coords[target_ids, acc_index[target_ids], VR_INDEX]

    np.savez(
        out_file,
        redshift=d["redshift"],
        CosmicTime=d["CosmicTime"],
        mass=d["mass"],
        order=d["order"],
        ParentID=d["ParentID"],
        VirialRadius=d["VirialRadius"],
        concentration=d["concentration"],
        coordinates=coords,
    )
    return dict(
        m=m, n_targets=len(target_ids),
        vr_before_mean=float(vr_before.mean()) if len(vr_before) else float("nan"),
        vr_after_mean=float(vr_after.mean()) if len(vr_after) else float("nan"),
    )


def setup_fid():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for case, fname in CASES.items():
        case_dir = OUT_ROOT / case
        case_dir.mkdir(parents=True, exist_ok=True)
        src = TREE_DIR / fname
        dst = case_dir / "tree_fid.npz"
        shutil.copy(src, dst)
        print(f"{case}: copied {src.name} -> {dst}")


def run_one(case_name, A, mean_z50):
    tree_file, z50, rat = case_rat(case_name, mean_z50)
    case_dir = OUT_ROOT / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    out_file = case_dir / f"tree_A{A:.2f}.npz"

    stats = apply_scaling(tree_file, A, rat, out_file)

    print(
        f"{case_name} A={A:.2f}: z50={z50:.4f} rat={rat:.4f} m={stats['m']:.4f} "
        f"N_first_order={stats['n_targets']} "
        f"vr_mean {stats['vr_before_mean']:.3f} -> {stats['vr_after_mean']:.3f} "
        f"-> wrote {out_file}"
    )


def run_all():
    mean_z50 = ensemble_mean_z50()
    print(f"<z50> (N1000 13.0-bin ensemble mean) = {mean_z50:.4f}")
    setup_fid()
    for case_name in CASES:
        for A in A_VALUES:
            run_one(case_name, A, mean_z50)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="build everything: fid copies + all 21 A-scaled trees (3 cases x 7 A values)")
    p.add_argument("--setup-fid", action="store_true")
    p.add_argument("--case", type=str, choices=list(CASES.keys()))
    p.add_argument("--A", type=float)
    p.add_argument("--mean-z50", type=float, default=None,
                    help="override the ensemble mean z50 (default: recompute from N1000 h5)")
    args = p.parse_args()

    if args.all:
        run_all()
    elif args.setup_fid:
        setup_fid()
    elif args.case is not None and args.A is not None:
        mean_z50 = args.mean_z50 if args.mean_z50 is not None else ensemble_mean_z50()
        run_one(args.case, args.A, mean_z50)
    else:
        p.error("pass --all, or --setup-fid, or both --case and --A")
