"""
Ludlow-vs-Zhao HOST concentration test, single tree -- step 1: build inputs.

Takes one unevolved SatGen tree and writes a matched pair into
MassSpec/data/ludlow_host_test/:

  tree_<tag>_zhao.npz    exact copy of the input (host c = Zhao, as generated)
  tree_<tag>_ludlow.npz  identical, except concentration[0,:] (the host row)
                         is replaced by the Ludlow+2016 c(z), computed at
                         every snapshot from the host's full collapsed mass
                         history (SatGen ludlow.concentration_history_Ludlow2016,
                         f=0.02, A=400). All subhalo rows stay Zhao, and the
                         TreeGen orbit initial conditions are untouched.
  ludlow_host_history_<tag>.npz   diagnostics: redshift, c_zhao, c_ludlow,
                         z_m2, converged (name doesn't start with "tree", so
                         jsm_SubEvo ignores it)

Run from anywhere:  python build_ludlow_tree.py
Requires SatGen/src/config.py to have the "zhao" cosmology block active
(checked below against the tree's own redshift grid).
"""
import sys
import shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MASSSPEC = HERE.parents[1]
SATGEN_SRC = MASSSPEC.parent / "SatGen" / "src"
sys.path.insert(0, str(SATGEN_SRC))

import config as cfg
import ludlow

TAG = "13.2_33"
TREE_IN = MASSSPEC / "data" / "local_trees" / "unevolved_massspec" / f"tree_{TAG}.npz"
OUTDIR = MASSSPEC / "data" / "ludlow_host_test"

F_CMH = 0.02    # Ludlow+2016 Sec.4.2-4.3, App.B
A_L16 = 400.    # Ludlow+2016 Sec.4.3 fn.7: Parkinson+08-style Monte Carlo trees

def main():
    f = np.load(TREE_IN)
    data = {k: f[k] for k in f.files}
    z = data["redshift"]
    if not np.allclose(cfg.zsample[:len(z)], z):
        raise RuntimeError("SatGen config.py cosmology does not match this "
                           "tree's redshift grid -- switch to the 'zhao' block.")

    mass, ParentID = data["mass"], data["ParentID"]
    c_zhao = data["concentration"][0].copy()

    c_lud, z_m2, conv = ludlow.concentration_history_Ludlow2016(
        mass, ParentID, z, cfg.Dvsample, c_guess=c_zhao, f=F_CMH, A=A_L16)

    valid = c_zhao > 0
    nbad = int((valid & ~conv).sum())
    print(f"tree {TAG}: logM0 = {np.log10(mass[0,0]):.3f}")
    print(f"  host snapshots with a Zhao c: {valid.sum()}  "
          f"Ludlow converged: {conv.sum()}  fell back to Zhao: {nbad}")
    print(f"  z=0: c_zhao = {c_zhao[0]:.3f}  c_ludlow = {c_lud[0]:.3f}  "
          f"z_-2 = {z_m2[0]:.3f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TREE_IN, OUTDIR / f"tree_{TAG}_zhao.npz")

    lud = dict(data)
    conc = data["concentration"].copy()
    conc[0] = np.where(valid, c_lud, c_zhao)   # -99 dummies kept as-is
    lud["concentration"] = conc
    np.savez(OUTDIR / f"tree_{TAG}_ludlow.npz", **lud)

    np.savez(OUTDIR / f"ludlow_host_history_{TAG}.npz", redshift=z,
             c_zhao=c_zhao, c_ludlow=conc[0], z_m2=z_m2, converged=conv,
             f_CMH=F_CMH, A_L16=A_L16)
    print(f"  wrote inputs to {OUTDIR}")

if __name__ == "__main__":
    main()
