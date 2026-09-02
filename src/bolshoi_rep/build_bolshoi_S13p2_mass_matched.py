"""
build_bolshoi_S13p2_mass_matched.py

Builds data/bolshoi_rep/bolshoiP_S13p2_mass_matched.csv: a real-BolshoiP
comparison table, one row per SatGen tree in data/bolshoi_rep/fid_z0.csv
(the "S13.2" sample, 1517 trees with host logMvir in [13.1, 13.3]), for a
direct SatGen-vs-Bolshoi comparison (see notebooks/paper3/satgen_bolshoi.ipynb).

**Rewritten 2026-09-02 -- the previous version of this script was WRONG and
its output should not be trusted if you have an old copy of the CSV lying
around.** The old version matched fid_z0.csv's masses against
BolshoiP_all_500.csv by exact-logMvir grouping, breaking ties among
degenerate-mass real halos with a SEEDED RANDOM DRAW. Two problems with
that, found 2026-09-02:
  1. It matched against the wrong real-halo sample. BolshoiP's 1517 hosts
     in the (13.1, 13.3) window actually come from HaloCatalogue's
     "relaxed" sample (relaxation cut only, no isolation cut), not "all" --
     see real_trees/recover_bolshoi_host_ids.py's docstring for the full
     derivation (confirmed bit-exact against SatGen/etc/Mhost_bolshoi_131.npy,
     the actual masses jsm_SubGen_bolshoi_rep.py used to seed each tree).
  2. Even setting that aside, matching by mass value AFTER the fact -- with
     a random tie-break -- was never necessary. SatGen/src/jsm_SubGen_
     bolshoi_rep.py generated tree_index `i` at EXACTLY
     Mhost_bolshoi_131.npy[i], which is EXACTLY the i-th halo (by host_id
     order) of the "relaxed" sample windowed to (13.1, 13.3). tree_index
     <-> real halo is a direct positional correspondence baked in at
     generation time, not something to be discovered by matching masses
     back together, and it involves no randomness at all.

This version just reads real_trees/bolshoiP_S13p2_host_ids.csv (already
built with that correct positional correspondence: tree_index, id, logMvir
-- one row per tree_index 0..1516, `id` = the real host_id, no duplicates,
no random tie-break) and joins each host_id directly against the "relaxed"
HaloCatalogue host table to pull in that SAME real halo's log1pz50, logc,
logNsub, logfsub. No grouping by mass, no candidate pool, no RNG anywhere
in this script.

Usage:
    python build_bolshoi_S13p2_mass_matched.py

(Run real_trees/recover_bolshoi_host_ids.py first if
bolshoiP_S13p2_host_ids.csv doesn't exist yet or might be stale.)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MASSSPEC_ROOT = HERE.parent.parent  # src/bolshoi_rep -> src -> MassSpec

SATGEN_ROOT = MASSSPEC_ROOT.parent / "SatGen"
sys.path.insert(0, str(SATGEN_ROOT / "mcmc" / "src"))
import jsm_simload  # noqa: E402

N_AB_SUB_PATH = MASSSPEC_ROOT.parent / "misc" / "multidark" / "bolshoiP" / "N_AB_sub.dat"
FID_Z0 = MASSSPEC_ROOT / "data" / "bolshoi_rep" / "fid_z0.csv"
HOST_IDS = MASSSPEC_ROOT / "data" / "bolshoi_rep" / "bolshoiP_S13p2_host_ids.csv"
OUTPATH = MASSSPEC_ROOT / "data" / "bolshoi_rep" / "bolshoiP_S13p2_mass_matched.csv"

KEYS = ["logMvir", "log1pz50", "logc", "logNsub", "logfsub"]


def main():
    host_ids = pd.read_csv(HOST_IDS)
    fid = pd.read_csv(FID_Z0)
    assert len(host_ids) == len(fid) == 1517, "expected 1517 rows in both fid_z0.csv and bolshoiP_S13p2_host_ids.csv"
    assert host_ids["id"].nunique() == len(host_ids), "bolshoiP_S13p2_host_ids.csv has duplicate ids -- re-run recover_bolshoi_host_ids.py"

    cat = jsm_simload.HaloCatalogue(
        sim_title="bolshoi",
        filepath=str(N_AB_SUB_PATH),
        mthresh=1.55e8 * 500,
        xoff_thresh=0.07,
        spin_thresh=0.07,
        isolation_factor=3,
    )
    host_id_unique, _groups = cat._get_groups("relaxed")
    host_table = cat._build_host_table("relaxed")
    host_table["host_id"] = host_id_unique
    host_table = host_table.set_index("host_id")

    rows = []
    for _, row in host_ids.iterrows():
        real = host_table.loc[int(row["id"])]
        rows.append({
            "tree_index": row["tree_index"],
            "id": int(row["id"]),
            **{k: real[k] for k in KEYS},
        })

    matched = pd.DataFrame(rows).sort_values("tree_index").reset_index(drop=True)
    print(f"{len(matched)} rows built (direct host_id join, no matching/randomness)")

    fid_sorted = fid.sort_values("tree_index").reset_index(drop=True)
    assert (matched["tree_index"].values == fid_sorted["tree_index"].values).all()
    assert np.allclose(matched["logMvir"].values, fid_sorted["logMvir"].values), (
        "logMvir mismatch vs fid_z0.csv -- host_id join picked the wrong halo somewhere"
    )
    print("confirmed: matched logMvir == fid_z0.csv logMvir for all 1517 trees")
    assert matched["id"].nunique() == len(matched), "duplicate id in output -- should be impossible"
    print(f"confirmed: {matched['id'].nunique()} / {len(matched)} ids unique (clean 1-to-1 mapping)")

    matched.to_csv(OUTPATH, index=False)
    print(f"wrote {OUTPATH} ({matched.shape[0]} rows, {matched.shape[1]} columns)")


if __name__ == "__main__":
    main()
