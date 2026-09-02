"""
integrate_z50_ratio_to_directory.py

Full-sample counterpart to integrate_environment_test.py: integrates the
batch-scaled raw trees written by apply_z50_ratio_to_directory.py --
output_root/A{A:.2f}/ directories, each holding up to 1000 raw trees for
the 13.0 mass bin -- instead of the pilot's 12 trees across
late/middle/early.

This is deliberately NOT a rewrite of the physics: the per-tree evolution
loop below is SatGen's own jsm_SubEvo.py loop(), copied verbatim from
integrate_environment_test.py (same Green/NFW profiles, same
orbit.orbit.integrate with dynamical friction, same evolve.msub King62
tidal stripping, same order-release/ejection mechanic). Only the outer
file-discovery/driver defaults change -- and even find_raw_tree_files()
itself needed NO changes: it already walks every subdirectory of `root`
generically (os.walk), so it picks up A0.20/A0.40/A0.60/A0.80/ exactly the
same way it picked up late/middle/early/ in the pilot. Do not "clean up"
or re-derive the physics inside loop() without checking first -- see
[[feedback-use-satgen-tools]]: it's SatGen's code, reused untouched on
purpose.

What it does
------------
Walks every subdirectory of --datadir (default: apply_z50_ratio_to_directory.py's
own DEFAULT_OUTPUT_ROOT, imported directly from that script so the two
can't drift apart), finds every raw tree file (starts with "tree", doesn't
end in "_evo.npz") that doesn't already have a corresponding "_evo.npz",
integrates orbits + tidal stripping from each subhalo's own accretion
snapshot down to z=0, and writes the result ALONGSIDE the input, in the
SAME A{A:.2f}/ directory, as "<name>_evo.npz" -- e.g.
A0.20/tree_13.0_417.npz -> A0.20/tree_13.0_417_evo.npz. Same naming
convention jsm_SubEvo.py uses, so jsm_stellarhalo.Tree_Reader(_Light),
jsm_processh5, etc. all read these unchanged. Trees that already have an
"_evo.npz" are skipped, so a partial or interrupted run can just be
re-launched and will only redo what's missing -- this matters a lot more
here than in the 12-tree pilot, since a 4000-tree cluster run is far more
likely to get interrupted partway through.

Config (matching jsm_SubEvo.py's own fiducial defaults, same as the pilot
-- see [[z50-vr-scale-test]] project memory; NOT the DF-strength-sweep
variants used elsewhere in bolshoi_rep, which deliberately change
lnL_pref):
    cfg.lnL_pref = 0.75      # dynamical friction strength
    cfg.evo_mode = 'arbres'  # resolution limit as a fraction of m_acc
    cfg.phi_res  = 1e-4
    alpha_type   = 'conc'    # tidal-stripping efficiency from c_sub/c_host

Cosmology: this needs SatGen/src/config.py to be on the "zhao" block (not
"Symphony"/vdb), same requirement as the pilot -- config.py is a REPO-WIDE
shared switch other paper-3 work also flips, so double check it's still
zhao on the server's copy of the repo right before running this (the
cosmology check line below will tell you either way; it does NOT change
config.py for you). See [[z50-vr-scale-test]] for the toggle history.

Where this runs: unlike the pilot's integrate_environment_test.py (built
for the user's own 8-core Mac, 12 trees total), this is sized for the
SERVER -- up to 4 x 1000 = 4000 trees, the same population
apply_z50_ratio_to_directory.py's batch scaling targets. Paths and
--ncores default accordingly (cluster paths, 16 cores, matching
apply_z50_ratio_to_directory.py / run_abundance.py's own cluster
defaults) -- these are NOT meant to be run as-is on a laptop.

**Path status: same caveat as apply_z50_ratio_to_directory.py.** DATADIR
defaults to that script's DEFAULT_OUTPUT_ROOT (still an UNCONFIRMED
placeholder as of 2026-09-02 -- update apply_z50_ratio_to_directory.py's
own DEFAULT_OUTPUT_ROOT once the real path is confirmed and this script
picks it up automatically, since it's imported not hardcoded here).
SATGEN_SRC is a NEW placeholder, inferred (not confirmed) from the
cluster root other scripts in this project already assume
(apply_epsilon_ratio_to_directory.py's DEFAULT_MEAN_MAH_DIR =
"/home/jsm99/SatGen/etc/mean_MAH/", run_abundance.py's DEFAULT_PARENTDIR =
"/home/jsm99/SatGen/mcmc/src/") -- "/home/jsm99/SatGen/src/" mirrors that
same repo root's core src/ directory (config.py, cosmo.py, evolve.py,
profiles.py, orbit.py, aux.py all live there, same as locally). Confirm
before running for real -- like the pilot script, these are plain module
constants (not CLI flags): SatGen's own config.py/cosmo.py/evolve.py etc.
have to be imported and configured (cfg.lnL_pref etc.) at MODULE level,
before any Pool is created, so that the pilot's already-verified pattern
(fork-based multiprocessing inherits the already-configured cfg module in
each worker -- this project's own SubEvo runs have used exactly this
pattern at O(1000)-tree/16-core cluster scale before) keeps working
unchanged. A CLI override here would need the import to happen inside a
Pool initializer instead, which is a real behavior change from what's
already been run at this scale -- not worth risking on an untested path.
Edit the two constants below directly if the repos ever move.

Runtime: NOT benchmarked at this scale. The pilot (12 trees, 8 cores) was
also never timed before running, so there's no per-tree cost estimate to
extrapolate from yet -- use --limit for a small smoke-test batch (a
handful of trees from one A directory) before committing to the full
4000-tree run, the same way apply_z50_ratio_to_directory.py was smoke-
tested on 3 trees before being called "ready."

Usage (on the cluster)
-----------------------
    python integrate_z50_ratio_to_directory.py --limit 8
        # smoke test: integrate just the first 8 discovered trees

    python integrate_z50_ratio_to_directory.py
        # full run: every tree under DEFAULT_DATADIR's A{A:.2f}/ dirs,
        # 16 worker processes

    python integrate_z50_ratio_to_directory.py --datadir /path/to/environment_test_full --ncores 32
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)

# NOTE: these are cluster paths, NOT the user's local Mac paths and NOT
# Claude's device_bash "~/mnt/<folder>" sandbox convention -- this script
# is meant to run on the server, alongside apply_z50_ratio_to_directory.py's
# own output. See the module docstring's "Path status" section: both
# constants below are placeholders pending confirmation. Edit them
# directly if the repos ever move -- deliberately module-level constants,
# not CLI flags, so SatGen's config.py/cosmo.py/etc. can be imported and
# configured once at module load, before any Pool is created (see
# docstring -- this is the same pattern integrate_environment_test.py and
# jsm_SubEvo.py itself already use, verified at O(1000)-tree/16-core
# cluster scale on this project before).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_z50_ratio_to_directory import DEFAULT_OUTPUT_ROOT  # noqa: E402

SATGEN_SRC = "/home/jsm99/SatGen/src"
DEFAULT_DATADIR = DEFAULT_OUTPUT_ROOT
DEFAULT_NCORES = 16  # cluster default -- matches apply_z50_ratio_to_directory.py / run_abundance.py

sys.path.insert(0, SATGEN_SRC)

import config as cfg          # noqa: E402
import cosmo as co            # noqa: E402
import evolve as ev           # noqa: E402
from profiles import NFW, Green   # noqa: E402
from orbit import orbit       # noqa: E402
import aux                    # noqa: E402

Rres_factor = 10 ** -4  # (defunct, kept for parity with jsm_SubEvo.py)
alpha_type = "conc"     # 'fixed' or 'conc'

cfg.lnL_pref = 0.75
cfg.evo_mode = "arbres"
cfg.phi_res = 10 ** -4


def find_raw_tree_files(root):
    """
    Walks every subdirectory of `root` (any depth -- A0.20/, A0.40/, ...,
    whatever exists) and collects raw tree files -- same file-naming
    convention as jsm_SubEvo.py: starts with "tree", doesn't end in
    "_evo.npz" -- minus any that already have a corresponding "_evo.npz"
    written (so re-runs only do the missing work). Verbatim from
    integrate_environment_test.py -- this needed no changes to generalize
    from the pilot's late/middle/early layout to A{A:.2f}/.
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
    writes "<file stem>_evo.npz" alongside it (i.e. in the SAME directory
    as the input -- no separate output location). Copied verbatim from
    integrate_environment_test.py; do not edit the physics here.
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
                            return ("error", os.path.basename(file), 0.0, f"no lt for id {id} at iz {iz}")

                        GreenRte[id, iz] = rte
                        coordinates[id, iznext, :] = xv
                        alphas[id, iz] = alpha
                        tdyns[id, iz] = tdyn

                    else:
                        if concentration[id, iz] > 0:
                            potentials[id] = NFW(mass[id, iz], concentration[id, iz],
                                                  Delta=VirialOverdensity[iz], z=redshift[iz])

        # ---output (written alongside the input file -- same directory)
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
        return ("done", os.path.basename(name), (time_end - time_start) / 60., None)
    except AttributeError as e:
        return ("error", os.path.basename(file), 0.0, str(e))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datadir", type=str, default=DEFAULT_DATADIR,
                    help="root containing the A{A:.2f}/ subdirectories "
                         f"(default: apply_z50_ratio_to_directory.py's DEFAULT_OUTPUT_ROOT, "
                         f"currently {DEFAULT_DATADIR})")
    p.add_argument("--ncores", type=int, default=DEFAULT_NCORES,
                    help=f"worker processes (default: {DEFAULT_NCORES})")
    p.add_argument("--limit", type=int, default=None,
                    help="only integrate the first N discovered trees -- for a smoke test "
                         "before committing to the full run (default: no limit, process all)")
    p.add_argument("--progress-every", type=int, default=50,
                    help="print a progress line every N completed trees (default: %(default)s)")
    args = p.parse_args()

    print(f"cosmology check: Om={cfg.Om}, OL={cfg.OL} "
          f"({'zhao' if abs(cfg.Om - 0.3) < 1e-6 else 'NOT zhao -- check config.py!'})")
    print(f"lnL_pref={cfg.lnL_pref}, evo_mode={cfg.evo_mode}, phi_res={cfg.phi_res}, "
          f"alpha_type={alpha_type}")

    files = find_raw_tree_files(args.datadir)
    print(f"found {len(files)} tree(s) to integrate under {args.datadir}")
    if args.limit is not None:
        files = files[:args.limit]
        print(f"--limit {args.limit}: only integrating the first {len(files)}")

    if not files:
        print("nothing to do (all trees already have an _evo.npz, or none found)")
        return

    t0 = time.time()
    n_done = 0
    n_error = 0
    with Pool(args.ncores) as pool:
        for i, result in enumerate(pool.imap_unordered(loop, files), 1):
            status, name, minutes, err = result
            if status == "done":
                n_done += 1
                print(f"done: {name}  ({minutes:.2f} min)", flush=True)
            else:
                n_error += 1
                print(f"ERROR: {name} is corrupted, skipping ({err})", flush=True)

            if i % args.progress_every == 0 or i == len(files):
                elapsed = (time.time() - t0) / 60.
                print(f"--- progress: {i}/{len(files)} ({n_done} done, {n_error} errors, "
                      f"{elapsed:.1f} min elapsed) ---", flush=True)

    print(f"all done in {(time.time() - t0) / 60.:.2f} min total "
          f"({n_done} integrated, {n_error} errors)")


if __name__ == "__main__":
    main()
