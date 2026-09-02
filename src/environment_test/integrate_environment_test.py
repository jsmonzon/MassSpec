"""
integrate_environment_test.py

Integrates the 12 raw (un-evolved) trees in
data/local_trees/environment_test/{late,middle,early}/ -- the z50-ratio
vr-scaling test (see project notes / src/environment_test/build_environment_test.py
for how those trees were built).

This is deliberately NOT a rewrite of the physics: the per-tree evolution
loop below is SatGen's own jsm_SubEvo.py loop(), copied essentially
verbatim (same Green/NFW profiles, same orbit.orbit.integrate with
dynamical friction, same evolve.msub King62 tidal stripping, same
order-release/ejection mechanic) -- only the outer file-discovery and
multiprocessing driver are new, adapted to walk three case subdirectories
instead of jsm_SubEvo.py's single hardcoded datadir. Do not "clean up" or
re-derive the physics inside loop() without checking first -- see
[[feedback-use-satgen-tools]]: it's SatGen's code, reused untouched on
purpose.

What it does
------------
For every file in data/local_trees/environment_test/*/ whose name starts
with "tree" and does NOT end in "_evo.npz" (i.e. all 12 raw trees:
tree_fid.npz, tree_A0.00.npz, tree_A0.25.npz, tree_A0.50.npz in each of
late/, middle/, early/), integrates the orbits + tidal stripping from each
subhalo's own accretion snapshot down to z=0, and writes the result
alongside the input as "<name>_evo.npz" (e.g. late/tree_A0.25.npz ->
late/tree_A0.25_evo.npz) -- same naming convention jsm_SubEvo.py uses, so
any code that already knows how to read a SatGen "_evo.npz" (jsm_processh5,
jsm_stellarhalo's Tree_Reader, etc.) will work on these unchanged. Trees
that already have a corresponding "_evo.npz" are skipped, so a partial or
interrupted run can just be re-launched and it will only redo what's
missing.

Config (matching jsm_SubEvo.py's own fiducial defaults -- see paper-3
memory notes; NOT the DF-strength-sweep variants used elsewhere in
bolshoi_rep, which deliberately change lnL_pref):
    cfg.lnL_pref = 0.75      # dynamical friction strength
    cfg.evo_mode = 'arbres'  # resolution limit as a fraction of m_acc
    cfg.phi_res  = 1e-4
    alpha_type   = 'conc'    # tidal-stripping efficiency from c_sub/c_host

Cosmology: this needs SatGen/src/config.py to be on the "zhao" block (not
"Symphony") -- same requirement as orbit.py, and it was switched to zhao
on 2026-09-01 for this test. If it's been switched back since, this script
will silently integrate under the wrong cosmology (VirialOverdensity,
Dvsample, etc. all come from cfg) -- check before running if in doubt.

Runtime: unknown per-tree cost has NOT been benchmarked yet (this script
was written but not run/timed). Ordinary SatGen SubEvo runs on this same
project have used 16 cores for O(1000)-tree batches; here there are only
12 trees, sized for 8 cores below to match this machine.

Usage
-----
    python integrate_environment_test.py
        # default: processes all 12 trees under data/local_trees/environment_test/,
        # 8 worker processes

    python integrate_environment_test.py --ncores 4
    python integrate_environment_test.py --datadir /path/to/environment_test
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)

# NOTE: these are real absolute paths on jsmonzon's machine -- NOT the
# "~/mnt/<folder>" mount convention used inside Claude's remote-device
# sandbox. This script is meant to be run directly in a local terminal
# (e.g. the "subhalos" conda env), where that mount point doesn't exist.
# Edit these two constants if the repos ever move.
SATGEN_SRC = "/Users/jsmonzon/Research/SatGen/src"
DEFAULT_DATADIR = "/Users/jsmonzon/Research/MassSpec/data/local_trees/environment_test"

sys.path.insert(0, SATGEN_SRC)

import config as cfg          # noqa: E402
import cosmo as co            # noqa: E402
import evolve as ev           # noqa: E402
from profiles import NFW, Green   # noqa: E402
from orbit import orbit       # noqa: E402
import aux                    # noqa: E402

DEFAULT_NCORES = 4

Rres_factor = 10 ** -4  # (defunct, kept for parity with jsm_SubEvo.py)
alpha_type = "conc"     # 'fixed' or 'conc'

cfg.lnL_pref = 0.75
cfg.evo_mode = "arbres"
cfg.phi_res = 10 ** -4


def find_raw_tree_files(root):
    """
    Walks every immediate subdirectory of `root` (late/middle/early) and
    collects raw tree files -- same file-naming convention as
    jsm_SubEvo.py: starts with "tree", doesn't end in "_evo.npz" -- minus
    any that already have a corresponding "_evo.npz" written (so re-runs
    only do the missing work).
    """
    files_unevo = []
    files_evo = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if filename.startswith("tree") and not filename.endswith("_evo.npz"):
                files_unevo.append(full)
            if filename.endswith("_evo.npz"):
                files_evo.append(full[:-8] + ".npz")
    remaining = [f for f in files_unevo if f not in set(files_evo)]
    return sorted(remaining)


def loop(file):
    """
    SatGen's jsm_SubEvo.py loop(), unmodified in substance -- integrates
    one raw tree's orbits + tidal stripping from accretion to z=0 and
    writes "<file stem>_evo.npz" alongside it.
    """
    time_start = time.time()

    try:
        name = file[0:-4] + "_evo"

        # ---load tree
        f = np.load(file)
        redshift = f["redshift"]
        CosmicTime = f["CosmicTime"]
        mass = f["mass"]
        order = f["order"]
        ParentID = f["ParentID"]
        VirialRadius = f["VirialRadius"]
        concentration = f["concentration"]
        coordinates = f["coordinates"]

        # compute the virial overdensities for all redshifts
        VirialOverdensity = co.DeltaBN(redshift, cfg.Om, cfg.OL)  # same as Dvsample
        GreenRte = np.zeros(VirialRadius.shape) - 99.
        alphas = np.zeros(VirialRadius.shape) - 99.
        tdyns = np.zeros(VirialRadius.shape) - 99.

        # ---identify the roots of the branches
        izroot = mass.argmax(axis=1)
        idx = np.arange(mass.shape[0])
        levels = np.unique(order[order >= 0])
        izmax = mass.shape[1] - 1

        # ---get smallest host rvir from tree (defunct, kept for parity)
        min_rvir = VirialRadius[0, np.argwhere(VirialRadius[0, :] > 0)[-1][0]]
        cfg.Rres = min(0.1, min_rvir * Rres_factor)

        potentials = [0] * mass.shape[0]
        orbits = [0] * mass.shape[0]
        trelease = np.zeros(mass.shape[0])
        ejected_mass = np.zeros(mass.shape[0])

        M0 = mass[0, 0]
        min_mass = np.zeros(mass.shape[0])

        # ---evolve
        for iz in np.arange(izmax, 0, -1):
            iznext = iz - 1
            z = redshift[iz]
            tcurrent = CosmicTime[iz]
            tnext = CosmicTime[iznext]
            dt = tnext - tcurrent
            Dv = VirialOverdensity[iz]

            for level in levels:
                for id in idx:
                    if order[id, iz] != level:
                        continue
                    if iz <= izroot[id]:
                        if iz == izroot[id]:
                            za = z
                            ta = tcurrent
                            Dva = Dv
                            ma = mass[id, iz]
                            c2a = concentration[id, iz]
                            xva = coordinates[id, iz, :]

                            if np.any(np.isnan(xva)):
                                print("    WARNING: NaNs detected in init xv of id %d" % id)
                                print("    Mass fraction of tree lost: %.1e" % (ma / mass[0, 0]))
                                mass[id, :] = -99.
                                coordinates[id, :, :] = 0.
                                idx = np.delete(idx, np.argwhere(idx == id)[0])
                                continue

                            potentials[id] = Green(ma, c2a, Delta=Dva, z=za)
                            orbits[id] = orbit(xva)
                            trelease[id] = ta

                            if cfg.evo_mode == "arbres":
                                min_mass[id] = cfg.phi_res * ma
                            elif cfg.evo_mode == "withering":
                                min_mass[id] = cfg.psi_res * M0

                        ip = ParentID[id, iz]
                        p = potentials[ip]
                        s = potentials[id]

                        if s.Mh > min_mass[id]:
                            if ejected_mass[id] > 0:
                                mass[id, iz] -= ejected_mass[id]
                                ejected_mass[id] = 0
                                mass[id, iz] = max(mass[id, iz], cfg.phi_res * s.Minit)

                            s.update_mass(mass[id, iz])
                            rte = s.rte()

                        o = orbits[id]
                        xv = orbits[id].xv
                        m = s.Mh
                        m_old = m
                        r = np.sqrt(xv[0] ** 2 + xv[2] ** 2)

                        t = tnext - trelease[id]
                        k = order[ip, iznext] + 1

                        if alpha_type == "fixed":
                            alpha = 0.55
                        elif alpha_type == "conc":
                            alpha = ev.alpha_from_c2(p.ch, s.ch)

                        if m > min_mass[id]:
                            m, lt = ev.msub(s, p, xv, dt, choice="King62", alpha=alpha)
                        else:
                            pass

                        if m > min_mass[id]:
                            tdyn = p.tdyn(r)
                            o.integrate(t, p, m_old)
                            xv = o.xv
                        else:
                            tdyn = p.tdyn(cfg.Rres)
                            xv = np.array([cfg.Rres, 0., 0., 0., 0., 0.])

                        r = np.sqrt(xv[0] ** 2 + xv[2] ** 2)
                        m_old = m

                        if k > 1:
                            if (r > VirialRadius[ip, iz]) & (iz <= izroot[ip]):
                                odds = np.random.rand()
                                dyntime_frac = alphas[ip, iz] * dt / tdyns[ip, iz]
                                if odds < dyntime_frac:
                                    if ParentID[ip, iz] == ParentID[ip, iznext]:
                                        xv = aux.add_cyl_vecs(xv, coordinates[ip, iznext, :])
                                    else:
                                        xv = aux.add_cyl_vecs(xv, coordinates[ip, iz, :])
                                    orbits[id] = orbit(xv)
                                    k = order[ip, iz]
                                    ejected_mass[ip] += m
                                    ip = ParentID[ip, iz]
                                    trelease[id] = tnext

                        mass[id, iznext] = m
                        order[id, iznext] = k
                        ParentID[id, iznext] = ip
                        try:
                            VirialRadius[id, iznext] = lt
                        except UnboundLocalError:
                            print("No lt for id ", id, "iz ", iz, "masses ",
                                  np.log10(mass[id, iz]), np.log10(mass[id, iznext]), file)
                            return

                        GreenRte[id, iz] = rte
                        coordinates[id, iznext, :] = xv
                        alphas[id, iz] = alpha
                        tdyns[id, iz] = tdyn

                    else:
                        if concentration[id, iz] > 0:
                            potentials[id] = NFW(mass[id, iz], concentration[id, iz],
                                                  Delta=VirialOverdensity[iz], z=redshift[iz])

        # ---output
        np.savez(
            name,
            redshift=redshift,
            CosmicTime=CosmicTime,
            mass=mass,
            order=order,
            ParentID=ParentID,
            VirialRadius=VirialRadius,
            GreenRte=GreenRte,
            concentration=concentration,
            coordinates=coordinates,
        )
        time_end = time.time()
        print(f"done: {name}  ({(time_end - time_start) / 60.:.2f} min)", flush=True)
    except AttributeError:
        print(file, "is corrupted, skipping for now!", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datadir", type=str, default=DEFAULT_DATADIR,
                    help="root containing the late/middle/early subdirectories "
                         f"(default: {DEFAULT_DATADIR})")
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                    help=f"worker processes (default: {DEFAULT_NCORES})")
    args = p.parse_args()

    print(f"cosmology check: Om={cfg.Om}, OL={cfg.OL} "
          f"({'zhao' if abs(cfg.Om - 0.3) < 1e-6 else 'NOT zhao -- check config.py!'})")
    print(f"lnL_pref={cfg.lnL_pref}, evo_mode={cfg.evo_mode}, phi_res={cfg.phi_res}, "
          f"alpha_type={alpha_type}")

    files = find_raw_tree_files(args.datadir)
    print(f"found {len(files)} tree(s) to integrate under {args.datadir}:")
    for fpath in files:
        print("  ", fpath)

    if not files:
        print("nothing to do (all trees already have an _evo.npz, or none found)")
        return

    t0 = time.time()
    with Pool(args.ncores) as pool:
        pool.map(loop, files)
    print(f"all done in {(time.time() - t0) / 60.:.2f} min total")


if __name__ == "__main__":
    main()
