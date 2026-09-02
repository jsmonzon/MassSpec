"""
to_share.py -- minimal, dependency-light reader for the h5 files this
project's Tree_Reader / Tree_Reader_Light classes write
(SatGen/mcmc/src/jsm_stellarhalo.py). No project code required -- just
h5py (and pandas, for the optional summary table):

    pip install h5py pandas

Layout: one group per tree_index (e.g. epsilon.h5, DF_1_light.h5), or one
top-level group per model/variant with tree groups nested underneath
(e.g. epsilon_A_sweep.h5 -> "A3"/"A6"/"A9"). Detected automatically.

Usage:
    python to_share.py path/to/file.h5
    python to_share.py path/to/grouped_file.h5 <group_name>

Two field sets show up, depending on which class wrote the file:

LIGHT (Tree_Reader_Light, e.g. DF_1_light.h5) -- host histories plus a
bare per-subhalo census:
    host_MAH, host_concentration, host_Rvir  full (Ntime,) host histories
    host_z10 / z50 / z90                     host formation-redshift scalars
    survives                                 bool, alive at z=0 (per subhalo)
    z0_mass, z0_position                     z=0 values -- NaN unless that
                                              subhalo survives, sits within
                                              the host's z=0 Rvir, and is
                                              above the run's mass cut
    zacc_mass, zacc_concentration, zacc_redshift
                                              values when each subhalo's
                                              branch fell into the host's
                                              main progenitor
    (z=0 subhalo concentration isn't included here -- SatGen never saves it)
"""

import sys

import h5py
import numpy as np


def list_trees(h5path, group=None):
    """Return the sorted list of tree_index keys in a file (or one of its
    top-level groups, for a grouped file)."""
    with h5py.File(h5path, "r") as f:
        root = f[group] if group else f
        return sorted(root.keys(), key=int)


def load_tree(h5path, tree_index, group=None):
    """Load one tree's fields into a plain dict: {field_name: value}, where
    value is either a python scalar or a numpy array. See this module's
    docstring for what each field means."""
    with h5py.File(h5path, "r") as f:
        root = f[group] if group else f
        node = root[str(tree_index)]
        return {name: dset[()] for name, dset in node.items()}


def summary_table(h5path, group=None, scalar_only=True):
    """Build a pandas DataFrame with one row per tree, one column per field
    that's a scalar for every tree (arrays -- e.g. MAH, host_c(t) -- are
    skipped by default; pass scalar_only=False to keep everything as
    object columns instead)."""
    import pandas as pd

    rows = {}
    for tree_index in list_trees(h5path, group=group):
        fields = load_tree(h5path, tree_index, group=group)
        if scalar_only:
            fields = {k: v for k, v in fields.items() if np.ndim(v) == 0}
        rows[tree_index] = fields
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("tree_index").reset_index()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    h5path = sys.argv[1]
    group = sys.argv[2] if len(sys.argv) > 2 else None

    with h5py.File(h5path, "r") as f:
        top_keys = list(f.keys())
        looks_grouped = bool(top_keys) and isinstance(f[top_keys[0]], h5py.Group) \
            and any(isinstance(v, h5py.Group) for v in f[top_keys[0]].values())

    if looks_grouped and group is None:
        print(f"{h5path} looks like a grouped file -- top-level groups: {top_keys}")
        print("pass one as a second argument, e.g.:")
        print(f"    python to_share.py {h5path} {top_keys[0]}")
        sys.exit(0)

    trees = list_trees(h5path, group=group)
    print(f"{h5path}" + (f" / {group}" if group else "") + f": {len(trees)} trees")
    print(f"first tree_index: {trees[0]}")

    example = load_tree(h5path, trees[0], group=group)
    print(f"\nfields in tree {trees[0]} ({len(example)} total) "
          "-- see this file's module docstring for what each one means:")
    for name, value in sorted(example.items()):
        kind = "scalar" if np.ndim(value) == 0 else f"array{np.shape(value)}"
        print(f"  {name:28s} {kind}")

    print("\nbuilding a scalar-only summary table (pandas) ...")
    df = summary_table(h5path, group=group)
    print(df.head())
