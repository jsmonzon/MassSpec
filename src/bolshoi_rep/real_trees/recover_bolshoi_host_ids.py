"""
recover_bolshoi_host_ids.py

Recovers the real consistent-trees/Rockstar halo IDs for the exact 1517
real BolshoiP halos whose masses seeded the 1517 SatGen trees in fid.h5 /
fid_z0.csv -- with NO mass-matching, NO grouping, and NO random tie-break.

**Corrected 2026-09-02 -- see git history / project memory for the earlier,
WRONG version of this script.** The first version of this script matched
fid_z0.csv's masses against BolshoiP_all_500.csv via exact-logMvir grouping
with a seeded random draw among degenerate-mass candidates (mirroring
build_bolshoi_S13p2_mass_matched.py's approach). That is NOT what was
actually asked for (a clean, non-random, no-duplicate 1-to-1 halo<->tree
mapping), and it turns out it also wasn't even matching against the right
real-halo sample in the first place (see below) -- it happened to reproduce
build_bolshoi_S13p2_mass_matched.py's own output bit-for-bit (which is why
that "verification" passed), but that script's own premise was flawed.

How the TRUE mapping was found
-------------------------------
The tree generation script that built fid.h5 (SatGen/src/jsm_SubGen_bolshoi_
rep.py) loads `samples = np.load("../etc/Mhost_bolshoi_131.npy")` and sets
each tree's host mass directly from `samples[itree]` -- so SatGen tree_index
`i` was generated at EXACTLY the mass `Mhost_bolshoi_131.npy[i]`, no matching
required at generation time. Confirmed: sorting fid_z0.csv by tree_index and
comparing element-wise to Mhost_bolshoi_131.npy gives a bit-exact match
(max abs diff ~1e-15, pure float64 noise) for all 1517 entries.

So the only remaining question was: which real-halo selection, in which
order, produced Mhost_bolshoi_131.npy in the first place? Tested every
combination of HaloCatalogue's three cut levels (all / relaxed / isolated,
i.e. HaloCatalogue._get_groups's three `sample` options) against a
logMvir in (13.1, 13.3) window (matching the user's own description: "I
naively selected all systems within 13.1 and 13.3 from the catalog"), taken
in the catalog's own native host_id-sorted order (no re-sorting):
  - sample="all" (no relaxation/isolation cut): 2168 hosts in that window
    -- too many.
  - sample="isolated" (relaxation AND isolation cuts): 1399 hosts -- too
    few. (This is the sample `build_bolshoi_S13p2_mass_matched.py` and the
    first version of this script effectively ended up validated against via
    BolshoiP_all_500.csv, incorrectly.)
  - **sample="relaxed" (relaxation cut only, no isolation cut): exactly
    1517 hosts, and taken in the catalog's native order, BIT-EXACTLY equal
    to Mhost_bolshoi_131.npy (max abs diff = 0.0).** This is the real
    source: the "relaxed" HaloCatalogue sample, mass-windowed to
    logMvir in (13.1, 13.3), row order preserved (== ascending host_id
    order, since HaloCatalogue._get_groups sorts host_id_unique).

Since host_table (from HaloCatalogue._build_host_table) has exactly one row
per host_id, and this is a straightforward positional mass-window filter
with no grouping, there is no possibility of picking the same real halo
twice -- all 1517 host_id values in the result are guaranteed distinct
(verified: 1517/1517 unique). tree_index is just the row's position (0-based)
in this exact selection -- not looked up or matched, just assigned by
construction, mirroring how jsm_SubGen_bolshoi_rep.py itself assigned
`itree`.

Output: data/bolshoi_rep/bolshoiP_S13p2_host_ids.csv (tree_index, id,
logMvir) -- one row per tree_index 0..1516, id = the real consistent-trees
halo ID, logMvir bit-identical to fid_z0.csv's own value for that tree_index.
This supersedes the previous version of this file (which had 1517 rows over
only 1315 UNIQUE ids, from the flawed random-tie-break matching).

Usage:
    python recover_bolshoi_host_ids.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MASSSPEC_ROOT = HERE.parents[1].parent  # src/bolshoi_rep/real_trees -> bolshoi_rep -> src -> MassSpec

# SatGen is a sibling of MassSpec on the real machine -- see feedback_local_script_paths
SATGEN_ROOT = MASSSPEC_ROOT.parent / "SatGen"
sys.path.insert(0, str(SATGEN_ROOT / "mcmc" / "src"))
import jsm_simload  # noqa: E402

N_AB_SUB_PATH = MASSSPEC_ROOT.parent / "misc" / "multidark" / "bolshoiP" / "N_AB_sub.dat"
SAMPLES_PATH = SATGEN_ROOT / "etc" / "Mhost_bolshoi_131.npy"
FID_Z0 = MASSSPEC_ROOT / "data" / "bolshoi_rep" / "fid_z0.csv"
OUTPATH = MASSSPEC_ROOT / "data" / "bolshoi_rep" / "bolshoiP_S13p2_host_ids.csv"

LOGMVIR_MIN = 13.1
LOGMVIR_MAX = 13.3


def main():
    samples = np.load(SAMPLES_PATH)
    print(f"Loaded {len(samples)} host masses from {SAMPLES_PATH.name} "
          f"(the exact masses SatGen's jsm_SubGen_bolshoi_rep.py used to "
          f"seed each tree_index)")

    fid = pd.read_csv(FID_Z0).sort_values("tree_index").reset_index(drop=True)
    assert len(fid) == len(samples), "fid_z0.csv row count != samples length"
    assert np.allclose(fid["logMvir"].values, samples), (
        "fid_z0.csv's logMvir (sorted by tree_index) does not match "
        f"{SAMPLES_PATH.name} element-wise -- the tree_index <-> samples "
        "correspondence this script relies on does not hold; stop and "
        "investigate before trusting anything downstream of this."
    )
    print("Confirmed: fid_z0.csv's logMvir (sorted by tree_index) is "
          "bit-identical to the samples array, element-wise.")

    cat = jsm_simload.HaloCatalogue(
        sim_title="bolshoi",
        filepath=str(N_AB_SUB_PATH),
        mthresh=1.55e8 * 500,
        xoff_thresh=0.07,
        spin_thresh=0.07,
        isolation_factor=3,
    )

    # the relaxation-cut-only sample, NOT "all" and NOT "isolated" -- see
    # module docstring for how this was determined
    host_id_unique, _groups = cat._get_groups("relaxed")
    host_table = cat._build_host_table("relaxed")
    host_table["host_id"] = host_id_unique

    cut = host_table[
        host_table["logMvir"].between(LOGMVIR_MIN, LOGMVIR_MAX, inclusive="neither")
    ].reset_index(drop=True)

    assert len(cut) == len(samples), (
        f"logMvir-windowed 'relaxed' sample has {len(cut)} rows, expected "
        f"{len(samples)} -- N_AB_sub.dat or the HaloCatalogue pipeline may "
        f"have changed since Mhost_bolshoi_131.npy was built"
    )
    assert np.array_equal(cut["logMvir"].values, samples), (
        "row order / values of the 'relaxed' mass-windowed sample do not "
        "bit-exactly match samples -- do not trust host_id assignment"
    )
    assert cut["host_id"].nunique() == len(cut), "duplicate host_id found -- unexpected"

    out = pd.DataFrame({
        "tree_index": np.arange(len(cut)),
        "id": cut["host_id"].astype(np.int64),
        "logMvir": cut["logMvir"],
    })
    out.to_csv(OUTPATH, index=False)
    print(f"wrote {OUTPATH} ({len(out)} rows, {out['id'].nunique()} unique "
          f"ids -- should be equal, confirming a clean 1-to-1 mapping)")


if __name__ == "__main__":
    main()
