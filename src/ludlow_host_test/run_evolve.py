"""
Ludlow-vs-Zhao HOST concentration test, single tree -- step 2: evolve.

Runs SatGen's own src/jsm_SubEvo.py (fiducial settings as set in that file:
alpha_type='conc', lnL_pref=0.75, evo_mode='arbres', phi_res=1e-4) on the
two trees in MassSpec/data/ludlow_host_test/, one core each, with the SAME
random seed for both so the paired runs differ only by the host
concentration (the RNG is only used for releasing 2nd+-order subhaloes).
Writes tree_<tag>_zhao_evo.npz and tree_<tag>_ludlow_evo.npz alongside.

Run from anywhere:  python run_evolve.py
(jsm_SubEvo skips any tree that already has an _evo file -- delete those
to re-run.)
"""
import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASSSPEC = HERE.parents[1]
SATGEN_SRC = MASSSPEC.parent / "SatGen" / "src"
DATADIR = MASSSPEC / "data" / "ludlow_host_test"
SEED = 42

env = dict(os.environ,
           SATGEN_DATADIR=str(DATADIR) + "/",
           SATGEN_NCORES="2",
           SATGEN_SEED=str(SEED))

subprocess.run([sys.executable, "jsm_SubEvo.py"], cwd=SATGEN_SRC, env=env,
               check=True)
