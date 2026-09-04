"""
Extract real BolshoiP merger trees (host main-branch MAH only) for the exact
1517 real halos already used in the bolshoi_rep S13.2 sample, without
downloading the ~1TB tree dataset -- and without downloading hlist_1.00231.list
either. The only inputs pulled from the network are: locations.dat (~464MB,
one time, to map halo ID -> tree file + byte offset) and, per target halo,
a small Range GET carrying just that one tree's bytes out of its
multi-GB tree_X_Y_Z.dat file. Nothing else.

Halo selection comes entirely from the LOCAL file misc/multidark/bolshoiP/
N_AB_sub.dat (see recover_bolshoi_host_ids.py) -- not from hlist_1.00231.list
over the network. The `verify` phase below does peek at 20KB of the hlist's
header purely to sanity-check column names against the tree files' own
header; that's a few KB, not a download of the file, and isn't needed again
now that it's already been run once (see notes below).

End result: one main-branch MAH (scale factor, Mvir) per one of the 1517
tree_index entries in bolshoiP_S13p2_host_ids.csv, at whatever snapshot
scale-factor spacing BolshoiP's own tree files use (not resampled onto any
common/uniform grid -- that's a separate step, still to be designed, needed
before comparing directly against SatGen's own time sampling). As of the
2026-09-02 correction, bolshoiP_S13p2_host_ids.csv has a clean 1-to-1
mapping -- 1517 tree_index rows, 1517 UNIQUE real halo ids, no repeats -- so
all 1517 trees are distinct and all 1517 get fetched. `mah`'s output still
carries parallel `tree_index`/`tree_index_to_id` arrays for convenient
lookup, they just no longer have any many-to-one collisions to resolve.

RUN THIS YOURSELF IN YOUR OWN TERMINAL -- Claude cannot reach
halos.as.arizona.edu from its cloud sandbox or from the device_bash shell it
uses on this Mac (blocked by Claude's own network allowlist, not the site).

Usage (run phases in order -- each one is cheap to re-run and prints what it
found, so check the output before moving to the next phase):

    python recover_bolshoi_host_ids.py        # local, no network -- already run
    python extract_bolshoi_real_trees.py verify      # already run 2026-09-02
    python extract_bolshoi_real_trees.py locations
    python extract_bolshoi_real_trees.py extract
    python extract_bolshoi_real_trees.py mah

Requires: requests, pandas, numpy  (pip install requests pandas numpy)

CONFIRMED as of 2026-09-02 (via the user's own `verify` run):
  - hlist_1.00231.list is indeed the z=0 file (Content-Length matched the
    expected ~7.7GB).
  - The server honors HTTP Range requests (206 Partial Content on both a
    hlist and a tree-file probe).
  - Column names in this release are lowercase (`mvir`, `id`, `desc_id`,
    `scale`, `mmp?`, ...) in both hlist_*.list and tree_*.dat headers --
    handled via the case-insensitive get_col() helper below.
  - tree_*.dat headers only carry explicit (index) numbers through column
    ~36 (Tidal_ID); later columns have no index at all (unlike hlist files,
    which index everything). Not an issue here since id/desc_id/scale/mvir/
    mmp? are all <= 14, but would need a positional-fallback fix if this is
    ever extended to read later columns straight from the tree files.

CONFIRMED as of 2026-09-02, second batch (via the user's own `locations`
run): locations.dat has an explicit 4-column header
'#TreeRootID FileID Offset Filename' -- not the guessed 2/3-column layouts
from the first version of this script (that guess was wrong; see
load_locations()'s docstring for the real format and how it was sanity
checked). The `locations` phase still self-checks the byte-offset math by
fetching the very first resolved tree and confirming the returned bytes
start with "#tree <expected_id>" -- if that fails, STOP and paste back the
printed diagnostics.
"""

import sys
import re
import os
import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config -- edit these for your run
# ---------------------------------------------------------------------------

TREES_BASE = "https://halos.as.arizona.edu/simulations/BolshoiP/trees/"
HLISTS_BASE = "https://halos.as.arizona.edu/simulations/BolshoiP/hlists/"

# best guess at the z=0 snapshot filename -- only used by the legacy
# select_halos_full_sim phase, CONFIRM against the real listing if you use it
Z0_HLIST = "hlist_1.00231.list"

LOGMVIR_MIN = 13.1
LOGMVIR_MAX = 13.3

MASSSPEC_ROOT = "/Users/jsmonzon/Research/MassSpec"

# written by recover_bolshoi_host_ids.py -- id, logMvir, tree_index for the
# exact 1517 real halos already used in the bolshoi_rep S13.2 sample
TARGET_HALOS_CSV = os.path.join(
    MASSSPEC_ROOT, "data", "bolshoi_rep", "bolshoiP_S13p2_host_ids.csv"
)

# real absolute paths on this machine (NOT device_bash's ~/mnt/ convention)
OUTDIR = os.path.join(MASSSPEC_ROOT, "data", "bolshoi_real_trees")
LOCATIONS_LOCAL = os.path.join(OUTDIR, "locations.dat")
# forests.list is deliberately not downloaded -- see phase_locations()
OFFSET_TABLE_CSV = os.path.join(OUTDIR, "target_offsets.csv")
RAW_TREES_DIR = os.path.join(OUTDIR, "raw_tree_blocks")
MAH_OUT_NPZ = os.path.join(OUTDIR, "bolshoi_real_MAHs.npz")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(RAW_TREES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_indexed_header(line):
    """Parse a header line like '#scale(0) id(1) desc_scale(2) ...' into
    {colname: index}. Returns None if the line doesn't look like this."""
    line = line.lstrip("#").strip()
    tokens = re.findall(r"(\S+?)\((\d+)\)", line)
    if len(tokens) < 5:
        return None
    return {name: int(idx) for name, idx in tokens}


def range_get(url, start, end=None, timeout=60):
    """HTTP GET with a Range header. end=None means open-ended (to EOF)."""
    if end is None:
        range_header = f"bytes={start}-"
    else:
        range_header = f"bytes={start}-{end}"
    r = requests.get(url, headers={"Range": range_header}, timeout=timeout)
    return r


def find_header_columns(text_block):
    """Search a chunk of header text for the first line that parses as an
    indexed column list, trying every '#'-prefixed line."""
    for line in text_block.splitlines():
        if line.startswith("#"):
            cols = parse_indexed_header(line)
            if cols is not None:
                return cols
    return None


def get_col(cols, *names):
    """Case-insensitive lookup of a column index by any of the given
    candidate names. Confirmed 2026-09-02: this BolshoiP release's headers
    use lowercase names in both hlist_*.list and tree_*.dat ('mvir' not
    'Mvir', etc) -- always look up columns through this helper rather than
    indexing `cols` directly. Also note: tree_*.dat's header line only
    carries explicit (index) numbers through column ~36 (Tidal_ID) -- later
    columns (Mvir_all, M200b, ...) are listed with no index at all, unlike
    the hlist files which index every column. Not an issue for the
    id/desc_id/scale/mvir/mmp? columns this script needs (all <= 14), but
    worth knowing if this is ever extended to use later columns."""
    lower_map = {k.lower(): v for k, v in cols.items()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ---------------------------------------------------------------------------
# Phase 1: verify -- check Range support + inspect real headers
# ---------------------------------------------------------------------------

def phase_verify():
    print("=== Checking Range-request support and header formats ===\n")

    z0_url = HLISTS_BASE + Z0_HLIST
    print(f"HEAD {z0_url}")
    h = requests.head(z0_url, timeout=30)
    print("  status:", h.status_code)
    print("  Accept-Ranges:", h.headers.get("Accept-Ranges"))
    print("  Content-Length:", h.headers.get("Content-Length"))

    print(f"\nRange GET first 20000 bytes of {Z0_HLIST}")
    r = range_get(z0_url, 0, 19999)
    print("  status:", r.status_code, "(want 206 Partial Content)")
    text = r.text
    print("  first 40 lines:\n")
    print("\n".join(text.splitlines()[:40]))
    cols = find_header_columns(text)
    print("\n  Parsed columns:", cols)
    if cols is not None:
        for want in ("id", "mvir", "Tree_root_ID", "Orig_halo_ID"):
            idx = get_col(cols, want)
            print(f"    {want}: index {idx if idx is not None else 'NOT FOUND'}")

    print("\n\n=== Same check for one tree file (small range only) ===")
    # pick an arbitrary, likely-small tree file for this diagnostic
    probe_file = "tree_2_3_4.dat"  # listed as one of the smaller ones (~4.9GB)
    probe_url = TREES_BASE + probe_file
    print(f"Range GET first 20000 bytes of {probe_file}")
    r2 = range_get(probe_url, 0, 19999)
    print("  status:", r2.status_code)
    text2 = r2.text
    print("  first 40 lines:\n")
    print("\n".join(text2.splitlines()[:40]))
    cols2 = find_header_columns(text2)
    print("\n  Parsed tree-file columns:", cols2)

    print("\n\nIf both statuses above were 206 and both column dicts look "
          "sane (id/Mvir/scale present with real indices), Range requests "
          "work and you can move on to `locations` (make sure "
          "recover_bolshoi_host_ids.py has already been run, so "
          f"{TARGET_HALOS_CSV} exists). If not, paste this whole output "
          "back so we can adjust.")


# ---------------------------------------------------------------------------
# Legacy/optional: stream the z=0 hlist and select halos across the WHOLE
# simulation in a mass window. NOT part of the default flow -- this would
# give you a different (larger, unrelated) set of halos than the 1517
# already used in bolshoi_rep. Only run this if you deliberately want that.
# It also overwrites TARGET_HALOS_CSV -- back up bolshoiP_S13p2_host_ids.csv
# first if you want to come back to the real 1517-halo run afterward.
# ---------------------------------------------------------------------------

def phase_select_halos_full_sim():
    z0_url = HLISTS_BASE + Z0_HLIST
    print(f"Fetching header from {Z0_HLIST} to find id/Mvir columns...")
    r = range_get(z0_url, 0, 19999)
    cols = find_header_columns(r.text)
    id_idx = get_col(cols, "id") if cols else None
    mvir_idx = get_col(cols, "mvir") if cols else None
    if id_idx is None or mvir_idx is None:
        print("Could not find id/mvir columns automatically. Header text was:")
        print(r.text[:3000])
        sys.exit(1)
    print(f"Using id column {id_idx}, Mvir column {mvir_idx}")
    print(f"Streaming the full file (this transfers ~7-8GB over the wire, "
          f"but writes only matching rows to disk -- this will take a "
          f"while)...")

    matches = []
    n_seen = 0
    with requests.get(z0_url, stream=True, timeout=None) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if line.startswith("#"):
                continue
            n_seen += 1
            parts = line.split()
            try:
                mvir = float(parts[mvir_idx])
            except (IndexError, ValueError):
                continue
            if mvir <= 0:
                continue
            logmvir = np.log10(mvir)
            if LOGMVIR_MIN <= logmvir <= LOGMVIR_MAX:
                halo_id = parts[id_idx]
                matches.append((halo_id, logmvir))
            if n_seen % 2_000_000 == 0:
                print(f"  ...scanned {n_seen:,} rows, "
                      f"{len(matches)} matches so far")

    df = pd.DataFrame(matches, columns=["id", "logMvir"])
    df["id"] = df["id"].astype(np.int64)
    df.to_csv(TARGET_HALOS_CSV, index=False)
    print(f"\nWrote {len(df)} target halos to {TARGET_HALOS_CSV}")


# ---------------------------------------------------------------------------
# Phase 3: download locations.dat + forests.list in full, resolve offsets
# ---------------------------------------------------------------------------

def download_full(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping download")
        return
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=None) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("  done")


def load_locations(path):
    """Parse locations.dat. CONFIRMED format 2026-09-02 (from the user's
    actual downloaded file): a header line '#TreeRootID FileID Offset
    Filename', then one row per z=0 halo:
      - TreeRootID: the real halo id (matches '#tree <ID>' in the tree file)
      - FileID: a redundant numeric file index -- not used, Filename is
        given directly (no need to reconstruct tree_X_Y_Z.dat from it)
      - Offset: byte offset of that tree's '#tree' line within Filename.
        Sanity-checked: many different files' offset-3640 rows are each
        that file's first tree (right after a same-sized header block);
        one file (tree_0_3_1.dat) also has a row at offset ~499,127,188,
        consistent with a tree much deeper into that 16GB file -- this is
        a real byte offset with the expected huge dynamic range, not a
        small index of some other kind.
      - Filename: e.g. tree_0_3_1.dat
    (The earlier speculative 2/3-column auto-detection this replaced was
    never right -- the real file is 4 columns with an explicit header.)"""
    with open(path) as f:
        header = f.readline()
    if not header.startswith("#"):
        raise ValueError(f"Expected a '#'-prefixed header line, got: {header!r}")
    df = pd.read_csv(path, sep=r"\s+", comment="#",
                      names=["id", "file_id", "offset", "filename"])
    df["id"] = df["id"].astype(np.int64)
    df["offset"] = df["offset"].astype(np.int64)
    return df


def phase_locations():
    # forests.list is NOT downloaded: it exists to resolve cross-references
    # between trees that share subhalos/flybys at z=0, which only matters
    # when the halo you're pulling out is itself a z=0 subhalo. Our targets
    # are all top-level hosts (upid == -1), and we only want each host's own
    # main branch, which is fully self-contained inside its own "#tree
    # <root_id>" block -- no cross-tree lookups needed.
    download_full(TREES_BASE + "locations.dat", LOCATIONS_LOCAL)

    loc = load_locations(LOCATIONS_LOCAL)
    targets = pd.read_csv(TARGET_HALOS_CSV)

    merged = targets.merge(loc, on="id", how="left")
    n_missing = merged["filename"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} target halos had no match in "
              f"locations.dat -- id scheme may not line up between the "
              f"hlist and locations.dat. Inspect merged[merged.filename."
              f"isna()] before trusting the rest.")

    # end offset = next offset in the same file, or EOF (None) if last
    loc_sorted = loc.sort_values(["filename", "offset"])
    loc_sorted["next_offset"] = loc_sorted.groupby("filename")["offset"].shift(-1)
    end_lookup = loc_sorted.set_index(["filename", "offset"])["next_offset"]

    merged = merged.set_index(["filename", "offset"])
    merged["end_offset"] = end_lookup
    merged = merged.reset_index()

    merged.to_csv(OFFSET_TABLE_CSV, index=False)
    print(f"Wrote offset table for {len(merged)} target halos to "
          f"{OFFSET_TABLE_CSV}")

    # self-check: fetch the first resolved tree and confirm it starts with
    # "#tree <id>"
    row = merged.dropna(subset=["filename"]).iloc[0]
    url = TREES_BASE + row["filename"]
    end = None if pd.isna(row["end_offset"]) else int(row["end_offset"]) - 1
    print(f"\nSelf-check: fetching id={row['id']} from {row['filename']} "
          f"bytes {int(row['offset'])}-{end}")
    r = range_get(url, int(row["offset"]), end)
    first_line = r.text.splitlines()[0] if r.text else ""
    print("  first line of response:", repr(first_line))
    # CORRECTED 2026-09-02: Offset points directly at the root halo's own
    # DATA row, not at a preceding '#tree <ID>' tag line (confirmed against
    # a live fetch: the first line was a normal data row whose id column
    # already equalled the expected id). So the real check is "does the
    # id-column field of the first row match", not a '#tree' string match.
    id_col = get_col(find_header_columns(range_get(url, 0, 19999).text) or {}, "id")
    fields = first_line.split()
    ok = id_col is not None and len(fields) > id_col and fields[id_col] == str(int(row["id"]))
    if ok:
        print("  MATCH -- offset resolves directly to the root halo's data "
              "row (no '#tree' tag prefix at this offset -- that's expected, "
              "see load_locations()/main_branch_mah()'s notes).")
    else:
        print(f"  MISMATCH -- expected id column (index {id_col}) to read "
              f"{int(row['id'])}. Do not trust the rest of the batch yet; "
              f"paste this output back.")


# ---------------------------------------------------------------------------
# Phase 4: extract each target tree's raw bytes
# ---------------------------------------------------------------------------

def phase_extract():
    merged = pd.read_csv(OFFSET_TABLE_CSV)
    merged = merged.dropna(subset=["filename"])

    header_cache = {}

    for i, row in merged.iterrows():
        halo_id = int(row["id"])
        fname = row["filename"]
        start = int(row["offset"])
        end = None if pd.isna(row["end_offset"]) else int(row["end_offset"]) - 1
        out_path = os.path.join(RAW_TREES_DIR, f"tree_{halo_id}.dat")
        if os.path.exists(out_path):
            continue

        url = TREES_BASE + fname
        if fname not in header_cache:
            hdr = range_get(url, 0, 19999).text
            cols = find_header_columns(hdr)
            header_cache[fname] = cols
            # save the raw header text once per file too, for reference
            with open(os.path.join(RAW_TREES_DIR, f"_header_{fname}.txt"), "w") as f:
                f.write(hdr)

        r = range_get(url, start, end)
        with open(out_path, "w") as f:
            f.write(r.text)

        if i % 100 == 0:
            print(f"  ...{i}/{len(merged)} trees fetched")

    print(f"Done. Raw tree blocks in {RAW_TREES_DIR}")


# ---------------------------------------------------------------------------
# Phase 5: parse each raw tree block -> host main-branch MAH
# ---------------------------------------------------------------------------

def main_branch_mah(tree_text, root_id, cols):
    """Walk the main branch (most-massive-progenitor chain) of a single
    tree's raw byte range and return (scale, mvir) arrays, oldest to z=0.

    CORRECTED 2026-09-02: earlier assumed the fetched block starts with a
    '#tree <id>' tag line and read root_id off of it. It doesn't -- offsets
    in locations.dat point straight at the root halo's own data row (see
    load_locations()'s docstring), so root_id must be passed in from the
    caller (who already knows it from the offset table) rather than parsed
    out of the text. The very last line of a fetched block CAN be a stray
    '#tree <next_id>' tag (the next tree's, sitting right at the boundary
    this block's end_offset was computed to stop before) -- that's handled
    by the existing "skip any line starting with #" check below, same as
    it always was; nothing special needed for it."""
    lines = tree_text.splitlines()

    id_i = get_col(cols, "id")
    desc_i = get_col(cols, "desc_id")
    scale_i = get_col(cols, "scale")
    mvir_i = get_col(cols, "mvir")
    mmp_i = get_col(cols, "mmp?")  # may be absent in some releases

    rows = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        rows.append(parts)

    by_id = {p[id_i]: p for p in rows}
    progenitors_of = {}
    for p in rows:
        progenitors_of.setdefault(p[desc_i], []).append(p)

    chain = []
    current_id = str(root_id)
    while current_id in by_id or current_id in progenitors_of:
        row = by_id.get(current_id)
        if row is not None:
            chain.append(row)
        progs = progenitors_of.get(current_id, [])
        if not progs:
            break
        if mmp_i is not None:
            mmp_progs = [p for p in progs if p[mmp_i] == "1"]
            progs = mmp_progs if mmp_progs else progs
        # most massive among candidates
        next_row = max(progs, key=lambda p: float(p[mvir_i]))
        current_id = next_row[id_i]

    scale = np.array([float(r[scale_i]) for r in chain])
    mvir = np.array([float(r[mvir_i]) for r in chain])
    order = np.argsort(scale)
    return scale[order], mvir[order]


def phase_mah():
    header_files = [f for f in os.listdir(RAW_TREES_DIR)
                     if f.startswith("_header_")]
    if not header_files:
        print("No cached headers found -- run `extract` first.")
        sys.exit(1)

    # build filename -> cols by re-parsing cached header files
    header_cols = {}
    for hf in header_files:
        fname = hf[len("_header_"):-len(".txt")]
        with open(os.path.join(RAW_TREES_DIR, hf)) as f:
            header_cols[fname] = find_header_columns(f.read())

    merged = pd.read_csv(OFFSET_TABLE_CSV).dropna(subset=["filename"])

    # one MAH per unique real halo (id). As of the 2026-09-02 correction to
    # recover_bolshoi_host_ids.py, ids in TARGET_HALOS_CSV are already all
    # unique (1517 tree_index rows, 1517 distinct ids) -- .unique() here is
    # just a defensive no-op, not doing any real deduplication anymore.
    result = {}
    for halo_id in merged["id"].unique():
        halo_id = int(halo_id)
        tree_path = os.path.join(RAW_TREES_DIR, f"tree_{halo_id}.dat")
        if not os.path.exists(tree_path):
            continue
        fname = merged.loc[merged["id"] == halo_id, "filename"].iloc[0]
        cols = header_cols[fname]
        with open(tree_path) as f:
            text = f.read()
        try:
            scale, mvir = main_branch_mah(text, halo_id, cols)
        except Exception as e:
            print(f"  failed on halo {halo_id}: {e}")
            continue
        result[f"scale_{halo_id}"] = scale
        result[f"mvir_{halo_id}"] = mvir

    # tree_index -> id lookup, so a downstream comparison to fid_z0.csv can
    # go straight from tree_index to the right scale_<id>/mvir_<id> pair
    if "tree_index" in merged.columns:
        result["tree_index"] = merged["tree_index"].values
        result["tree_index_to_id"] = merged["id"].values

    np.savez(MAH_OUT_NPZ, **result)
    n_halos = len(result) // 2
    print(f"Saved {n_halos} unique host MAHs ({len(merged)} tree_index rows "
          f"total) to {MAH_OUT_NPZ}")
    print("Look up a given tree_index's MAH via the parallel "
          "tree_index/tree_index_to_id arrays, e.g.:\n"
          "  d = np.load(MAH_OUT_NPZ)\n"
          "  i = np.where(d['tree_index'] == YOUR_TREE_INDEX)[0][0]\n"
          "  hid = d['tree_index_to_id'][i]\n"
          "  scale, mvir = d[f'scale_{hid}'], d[f'mvir_{hid}']")


# ---------------------------------------------------------------------------

PHASES = {
    "verify": phase_verify,
    "select_halos_full_sim": phase_select_halos_full_sim,  # legacy/optional, see note above
    "locations": phase_locations,
    "extract": phase_extract,
    "mah": phase_mah,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PHASES:
        print(f"Usage: python {sys.argv[0]} <phase>")
        print(f"Phases: {', '.join(PHASES)}")
        sys.exit(1)
    PHASES[sys.argv[1]]()
