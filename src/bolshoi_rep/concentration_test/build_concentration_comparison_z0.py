"""
build_concentration_comparison_z0.py

Builds data/bolshoi_rep/fid_concentration_comparison_z0.csv: one wide,
one-row-per-tree table holding all three ProcessH5 concentration
definitions for the fiducial Bolshoi-replica run (fid.h5), computed once.

Why: notebooks/paper3/concentration_comparison_fid.ipynb previously called
ProcessH5 three times per run (once per conctype) directly on fid.h5
(~360MB, 1517 trees), which means re-processing the raw h5 file from
scratch on every notebook run. This script does that processing once and
caches the result as a CSV; the notebook then just reads the CSV.

Only `logc` actually depends on `conctype` -- every other z0 column
(logMvir, log1pz50, Nsub, logNsub, fsub, logfsub, MMs, logMMs) is identical
across the three runs (verified when this table was first built, and
consistent with the "analytic vs per-tree" nature of the three
definitions -- see notebooks/paper3/concentration_comparison_fid.ipynb's
intro cell). So those shared columns are taken from one run and stored
once, and the three `logc` values are stored side by side as
log_c_measured, log_c_ludlow, log_c_zhao.

Usage:
    python build_concentration_comparison_z0.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASSSPEC_ROOT = HERE.parents[2]  # src/bolshoi_rep/concentration_test -> src/bolshoi_rep -> src -> MassSpec
RESEARCH_ROOT = MASSSPEC_ROOT.parent

sys.path.insert(0, str(RESEARCH_ROOT / "SatGen" / "mcmc" / "src"))
from jsm_processh5 import ProcessH5  # noqa: E402

DATADIR = MASSSPEC_ROOT / "data" / "bolshoi_rep"
FNAME = "fid.h5"
OUTPATH = DATADIR / "fid_concentration_comparison_z0.csv"

# label -> ProcessH5 conctype
CONCTYPES = {"measured": "measured", "ludlow": "ludlow", "zhao": None}

SHARED_COLS = ["tree_index", "logMvir", "log1pz50", "Nsub", "logNsub", "fsub", "logfsub", "MMs", "logMMs"]


def main():
    tables = {}
    for label, conctype in CONCTYPES.items():
        print(f"processing fid.h5 with conctype={conctype!r} ({label})...")
        proc = ProcessH5(str(DATADIR), files=[str(DATADIR / FNAME)], conctype=conctype, verbose=False)
        tables[label] = proc.build_z0_table()

    n_trees = {label: len(df) for label, df in tables.items()}
    print(f"trees per run: {n_trees}")

    # shared (conctype-independent) columns, taken from the "measured" run
    master = tables["measured"][SHARED_COLS].copy()

    for label in CONCTYPES:
        logc = tables[label].set_index("tree_index")["logc"]
        master[f"log_c_{label}"] = master["tree_index"].map(logc)

    master.to_csv(OUTPATH, index=False)
    print(f"wrote {OUTPATH} ({master.shape[0]} rows, {master.shape[1]} columns)")


if __name__ == "__main__":
    main()
