"""
Builds notebooks/paper3/env_dev.ipynb -- a working/dev notebook combining
the two earlier scratch explorations of the z50-ratio vr-scaling law
(notebooks/scratch/z50_vr_scale_explore.py and z50_vr_scale_feel.py) with
the later VR-vs-A diagnostic (individual first-order subhalos' VR traced
across the A sweep, for the late/middle/early pilot trees).

This sits alongside environment_test.ipynb (the actual evolved-tree
results notebook) as dev/scratch material: it works only with the raw
(un-evolved) local trees in data/local_trees/unevolved_massspec -- no
orbit integration, just inspecting the vr' = [1-A(1-rat)]*vr scaling law
itself before any tree gets evolved.

Two notes baked into the notebook text:
- the explore/feel sections use the OLDER "mean_MAH ensemble-curve" proxy
  for <z50> (mean_curve_z50(), reading SatGen/etc/mean_MAH/*.npz) -- this
  is DIFFERENT from the canonical true N1000-ensemble mean
  (<z50> = 0.9304333832432538, from data/zhao/N1000/13.0_files.h5) used
  everywhere else in this project (build_environment_test.py,
  apply_z50_ratio_to_directory.py, environment_test.ipynb). The rat
  values in the explore/feel sections should not be read as the
  canonical ones.
- the final VR-vs-A diagnostic cell uses the canonical mean instead, and
  is kept close to what was actually shown in-conversation.

Run with: python3 make_env_dev_notebook.py <output_path>
"""
import sys
import nbformat as nbf

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "env_dev.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# env_dev -- z50-ratio vr-scaling, dev/scratch notebook

Working notebook for the z50-ratio vr-scaling law, `vr' = [1-A(1-rat)] vr` with `rat = (1+z50)/(1+\\langle z50\\rangle)`, applied in-place to first-order-at-accretion subhalos only. This combines two earlier scratch explorations (`notebooks/scratch/z50_vr_scale_explore.py`, `z50_vr_scale_feel.py`) plus a later VR-vs-A diagnostic, all working directly with the **raw, un-evolved** local trees in `data/local_trees/unevolved_massspec` -- no orbit integration happens anywhere in this notebook, it's purely about the scaling law applied to each subhalo's VR at its own accretion snapshot.

For the actual evolved-tree results (SHMF, radial distribution, Nsub/fsub vs A, host MAH), see the sibling notebook `environment_test.ipynb`.

**Caveat on `<z50>`:** the "explore" and "feel" sections below (mirroring the original scratch scripts) use an older proxy for the ensemble-mean formation redshift -- `mean_curve_z50()`, which reads the *ensemble-mean MAH curve* per mass bin (`SatGen/etc/mean_MAH/{logM0}_files_mean_MAH.npz`) and computes z50 off of that single averaged curve. This is **not** the same as the canonical `<z50> = 0.9304333832432538`, which is the true mean of `host_z50` computed *per-tree* across the full 1000-tree 13.0-mass-bin ensemble (`data/zhao/N1000/13.0_files.h5`) and is what's used everywhere else in this project (`build_environment_test.py`, `apply_z50_ratio_to_directory.py`, `environment_test.ipynb`). The `rat` values in the explore/feel sections below are therefore only approximately comparable to the canonical ones -- the final diagnostic section uses the canonical mean instead."""
))

cells.append(nbf.v4.new_code_cell(
"""%load_ext autoreload
%autoreload 2"""
))

cells.append(nbf.v4.new_code_cell(
"""import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import warnings; warnings.simplefilter('ignore')"""
))

cells.append(nbf.v4.new_code_cell(
"""plt.style.use('../../../SatGen/notebooks/paper1/paper.mplstyle')
double_textwidth = 7.0  # inches
single_textwidth = 3.5  # inches"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Shared setup and helpers

`TREE_DIR` holds the raw (un-evolved) local trees used throughout this notebook; `MEAN_MAH_DIR` holds the per-mass-bin ensemble-mean MAH curves used by the older `mean_curve_z50` proxy (explore/feel sections only -- see caveat above).

Helper functions are defined once here and reused everywhere below (the original two scratch scripts each redefined their own copies, sometimes with different signatures -- unified here):
- `find_nearest_idx` -- nearest-neighbor index lookup (matches `jsm_ancillary.find_nearest1`)
- `host_z50(mass_host, redshift, target_mass=None)` -- formation redshift where mass first crosses half of `target_mass` (default: `mass_host[0]`, i.e. z=0 mass)
- `load_tree_z50(tree_file)` -- loads a raw tree `.npz` and returns a dict with its host z50 plus the raw arrays needed downstream
- `mean_curve_z50(logM0)` -- z50 of the ensemble-mean MAH curve for a given mass bin (older proxy, see caveat)
- `first_order_vr(tree)` -- VR (and accretion redshift) of every first-order-at-accretion subhalo; accepts either a `load_tree_z50` dict or a raw tree-file path"""
))

cells.append(nbf.v4.new_code_cell(
"""TREE_DIR = Path("../../data/local_trees/unevolved_massspec")
MEAN_MAH_DIR = Path("../../../SatGen/etc/mean_MAH")
VR_INDEX = 3  # [R, phi, z, VR, Vphi, Vz]


def find_nearest_idx(array, value):
    # matches jsm_ancillary.find_nearest1 exactly (nearest-neighbor, not interpolation)
    return int(np.argmin(np.abs(array - value)))


def host_z50(mass_host, redshift, target_mass=None):
    if target_mass is None:
        target_mass = mass_host[0]
    idx = find_nearest_idx(mass_host, target_mass * 0.5)
    return redshift[idx], idx


def load_tree_z50(tree_file):
    tree_file = Path(tree_file)
    d = np.load(tree_file)
    mass = d["mass"]
    redshift = d["redshift"]
    order = d["order"]
    coords = d["coordinates"]
    logM0 = round(float(np.log10(mass[0, 0])), 1)
    z50, idx50 = host_z50(mass[0, :], redshift)
    return dict(file=tree_file.name, logM0=logM0, z50=z50, idx50=idx50,
                mass=mass, redshift=redshift, order=order, coords=coords)


def mean_curve_z50(logM0):
    f = MEAN_MAH_DIR / f"{logM0:.1f}_files_mean_MAH.npz"
    d = np.load(f)
    z50, idx50 = host_z50(d["M"], d["z"])
    return z50, f.name


def first_order_vr(tree):
    \"\"\"Mirrors EpsilonRatioOrbitInit._find_accretion / _select_targets, order_filter=1.
    `tree` may be a load_tree_z50() dict, or a raw tree-file path (str/Path).\"\"\"
    if not isinstance(tree, dict):
        tree = load_tree_z50(tree)
    coords = tree["coords"]
    order = tree["order"]
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
    vr = coords[target_ids, acc_index[target_ids], VR_INDEX]
    zacc = tree["redshift"][acc_index[target_ids]]
    return target_ids, vr, zacc"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Explore: z50/rat and m(A) for the 13.0 mass-bin local trees

From `z50_vr_scale_explore.py`. Four steps: (1) each 13.0-bin local tree's own z50 vs. the mean-MAH-curve proxy z50; (2) the resulting multiplicative factor $m(A) = 1-A(1-{\\rm rat})$ across a grid of A; (3) applying that factor to one tree's actual first-order-subhalo VR population; (4) the same z50/rat computation swept across every local tree (11.0-14.0) that has mean-MAH coverage, as a broader sanity check."""
))

cells.append(nbf.v4.new_code_cell(
"""print("=" * 78)
print("Step 1: host z50 for each 13.0-bin local tree, vs. the ensemble-mean curve")
print("=" * 78)

A_values = [0.1, 0.2, 0.3, 0.4, 0.5]
trees_130 = sorted(TREE_DIR.glob("tree_13.0_*.npz"))

results = {}
for tf in trees_130:
    t = load_tree_z50(tf)
    zmean, mean_file = mean_curve_z50(t["logM0"])
    rat = (1 + t["z50"]) / (1 + zmean)
    results[tf.name] = dict(t=t, zmean=zmean, mean_file=mean_file, rat=rat)
    print(f"{tf.name}: logM0={t['logM0']:.1f}  z50_tree={t['z50']:.4f}  "
          f"z50_mean({mean_file})={zmean:.4f}  rat=(1+z50)/(1+<z50>)={rat:.4f}")"""
))

cells.append(nbf.v4.new_code_cell(
"""print("=" * 78)
print("Step 2: multiplicative factor m(A) = 1 - A*(1-rat) = 1 + A*(rat-1), for A in", A_values)
print("=" * 78)
for name, r in results.items():
    rat = r["rat"]
    print(f"\\n{name}  (rat={rat:.4f}, rat-1={rat-1:+.4f}):")
    for A in A_values:
        m = 1 - A * (1 - rat)
        pct = (m - 1) * 100
        print(f"  A={A:.1f}:  m={m:.4f}   ({pct:+.2f}% change to vr)")"""
))

cells.append(nbf.v4.new_code_cell(
"""print("=" * 78)
print("Step 3: apply to the actual first-order-subhalo VR population of one tree")
print("=" * 78)
primary_name = "tree_13.0_30.npz"
tree = load_tree_z50(TREE_DIR / primary_name)
target_ids, vr, zacc = first_order_vr(tree)
rat = results[primary_name]["rat"]
print(f"tree: {primary_name}  N first-order subhalos (at accretion): {len(vr)}")
print(f"VR (raw, un-evolved orbit, at accretion) stats [SatGen internal units]:")
print(f"  min={vr.min():.4f}  max={vr.max():.4f}  mean={vr.mean():.4f}  median={np.median(vr):.4f}")
print(f"  fraction with VR<0 (infalling): {np.mean(vr<0):.1%}")
print()
print(f"rat = {rat:.4f}  ->  since rat {'>' if rat>1 else '<'} 1, m(A) will be "
      f"{'>1 (SPEEDS UP infall/outflow)' if rat>1 else '<1 (SLOWS DOWN)'} for all A>0")
print()
for A in A_values:
    m = 1 - A * (1 - rat)
    vr_new = vr * m
    print(f"A={A:.1f}  m={m:.4f}:  new VR mean={vr_new.mean():.4f} (was {vr.mean():.4f}), "
          f"median={np.median(vr_new):.4f} (was {np.median(vr):.4f}), "
          f"max|delta_vr| at extreme subhalo={np.max(np.abs(vr_new-vr)):.4f}")"""
))

cells.append(nbf.v4.new_code_cell(
"""print("=" * 78)
print("Step 4: sanity check -- same computation for ALL local trees (11.0-14.0), where mean_MAH exists")
print("=" * 78)
for tf in sorted(TREE_DIR.glob("tree_*.npz")):
    d = np.load(tf)
    mass = d["mass"]; redshift = d["redshift"]
    logM0 = round(float(np.log10(mass[0, 0])), 1)
    mean_file = MEAN_MAH_DIR / f"{logM0:.1f}_files_mean_MAH.npz"
    if not mean_file.exists():
        print(f"{tf.name}: logM0={logM0:.1f}  -- no mean_MAH reference, skipped")
        continue
    z50, _ = host_z50(mass[0, :], redshift)
    zmean, _ = mean_curve_z50(logM0)
    rat = (1 + z50) / (1 + zmean)
    print(f"{tf.name}: logM0={logM0:.1f}  z50={z50:.4f}  z50_mean={zmean:.4f}  rat={rat:.4f}")"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## Feel: a visual sense of the scaling, for the three pilot trees

From `z50_vr_scale_feel.py`. Three panels: (1) $m(A)$ vs A for the three 13.0-bin pilot trees (`tree_13.0_30/31/32.npz` -- later adopted as the late/middle/early cases); (2) rat vs $\\log_{10}M_{\\rm host}(z{=}0)$ across every local tree with mean-MAH coverage, the three pilots highlighted in red; (3) the VR distribution of `tree_13.0_30`'s first-order subhalos, before and after scaling at a few A values."""
))

cells.append(nbf.v4.new_code_cell(
"""A_grid = np.linspace(0, 0.5, 100)

# gather rat for all 13.0-bin trees + wider context (11.0-14.0)
trees = sorted(TREE_DIR.glob("tree_*.npz"))
rat_by_tree = {}
for tf in trees:
    d = np.load(tf)
    mass, redshift = d["mass"], d["redshift"]
    logM0 = round(float(np.log10(mass[0, 0])), 1)
    mf = MEAN_MAH_DIR / f"{logM0:.1f}_files_mean_MAH.npz"
    if not mf.exists():
        continue
    z50 = host_z50(mass[0, :], redshift)[0]
    zmean = mean_curve_z50(logM0)[0]
    rat_by_tree[tf.name] = (1 + z50) / (1 + zmean)

fig, axes = plt.subplots(1, 3, figsize=(double_textwidth + 1.5, single_textwidth + 1))

# Panel 1: m(A) vs A for the three 13.0-bin pilot trees
ax = axes[0]
highlight = ["tree_13.0_30.npz", "tree_13.0_31.npz", "tree_13.0_32.npz"]
colors = ["#1f77b4", "#7f7f7f", "#d62728"]  # late, middle, early
for name, c in zip(highlight, colors):
    rat = rat_by_tree[name]
    m = 1 - A_grid * (1 - rat)
    ax.plot(A_grid, m, color=c, lw=2, label=f"{name}\\nrat={rat:.2f}")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_xlabel("A")
ax.set_ylabel("m(A) = 1 - A(1-rat)   (multiplies vr)")
ax.set_title("Scaling factor vs A\\n(13.0 mass-bin pilot trees)")
ax.legend(fontsize=6, loc="best")

# Panel 2: rat distribution across all local trees with mean_MAH coverage
ax = axes[1]
names = list(rat_by_tree.keys())
rats = np.array([rat_by_tree[n] for n in names])
logM0s = np.array([round(float(np.log10(np.load(TREE_DIR / n)["mass"][0, 0])), 1) for n in names])
ax.scatter(logM0s, rats, c="#2ca02c")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
for name in highlight:
    i = names.index(name)
    ax.scatter([logM0s[i]], [rats[i]], color="#d62728", s=90, zorder=5, edgecolor="k")
ax.set_xlabel("log$_{10}$ M$_{\\\\rm host}$(z=0)")
ax.set_ylabel("rat = (1+z50) / (1+<z50>)")
ax.set_title("rat across local trees\\n(red = the 3 highlighted 13.0 trees)")

# Panel 3: VR distribution before/after scaling, for tree_13.0_30 (late, rat<1)
ax = axes[2]
_, vr, _ = first_order_vr(TREE_DIR / "tree_13.0_30.npz")
rat = rat_by_tree["tree_13.0_30.npz"]
bins = np.linspace(vr.min(), vr.max(), 30)
ax.hist(vr, bins=bins, histtype="step", lw=2, color="k", label="raw vr (A=0)")
for A, c in zip([0.1, 0.3, 0.5], ["#9ecae1", "#4292c6", "#08519c"]):
    m = 1 - A * (1 - rat)
    ax.hist(vr * m, bins=bins, histtype="step", lw=1.5, color=c, label=f"A={A:.1f} (m={m:.3f})")
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("VR at accretion [SatGen internal units]")
ax.set_ylabel("N first-order subhalos")
ax.set_title(f"tree_13.0_30 (rat={rat:.2f})\\nVR before/after scaling")
ax.legend(fontsize=6)

plt.tight_layout(w_pad=2.5)
# plt.savefig("../../figures/env_dev_feel.pdf", bbox_inches="tight")  # uncomment to save
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## VR-vs-A diagnostic: individual subhalo trajectories, all three pilot cases

Each panel is one of the three pilot cases (late/middle/early -- same trees adopted for `environment_test.ipynb`'s three cases). Each line traces one first-order subhalo's own $VR' = m(A)\\cdot VR_{\\rm raw}$ as A sweeps from 0 to 0.8 -- **every** first-order subhalo of that tree, not a sample (N=155/193/398 for late/middle/early), so the fan of lines in each panel is a direct read of how much of the population is being scaled and by how much. Lines are drawn thin and semi-transparent (`lw=0.5, alpha=0.35`) since hundreds of them overlap. Unlike the explore/feel sections above, `rat` here uses the **canonical** ensemble mean, `<z50> = 0.9304333832432538` (true per-tree mean across the full 1000-tree 13.0-mass-bin sample), matching `environment_test.ipynb` and `apply_z50_ratio_to_directory.py`.

**Lines are colored by each subhalo's own UNSCALED (A=0) radial velocity**, not arbitrarily -- a diverging (blue-white-red) colormap, normalized linearly across the full observed VR range (`mcolors.Normalize`, not centered at VR=0 -- the range isn't symmetric once every subhalo is included), on ONE shared scale across all three panels (built from every shown subhalo in all three cases combined, not per-panel), so a color means the same raw VR in every panel and the colorbar is directly comparable case to case. `CASE_DATA` (built here) and this color scale (`VR_NORM`/`VR_CMAP`) are reused by the z=0-mass-vs-A cell below, so both diagnostics color the identical subhalo the identical color. Axes share y (`sharey=True`) so the width of the fan is directly comparable panel to panel.

Late (rat<1) trees flatten every subhalo's VR toward 0 as A increases; early (rat>1) trees do the opposite, amplifying VR magnitude -- each subhalo's line has slope set by its own raw VR (the scaling is purely multiplicative, so subhalos that start further from 0 move further as A increases). With the full population shown, the fan's width at A=0.8 is a direct visual read of how much the scaling law moves VRs around for that host, and the darkest blue/red lines (most extreme raw VR) fan out the most."""
))

cells.append(nbf.v4.new_code_cell(
"""MEAN_Z50_CANON = 0.9304333832432538  # true N1000-ensemble mean, 13.0 mass bin (see project notes)

PILOT_CASES = {
    "late": "tree_13.0_30.npz",
    "middle": "tree_13.0_31.npz",
    "early": "tree_13.0_32.npz",
}
CASE_COLORS = {"early": "#d62728", "middle": "#7f7f7f", "late": "#1f77b4"}
CASE_LIST = ["late", "middle", "early"]

A_grid_full = np.linspace(0, 0.8, 50)

# Gather each case's FULL first-order-subhalo selection up front (reused by this cell
# and the z=0-mass-vs-A cell below) -- every first-order subhalo, not a sample, so the
# fan of lines actually conveys how much of the population is being scaled and by how
# much (rather than just a handful of illustrative examples).
CASE_DATA = {}
for case in CASE_LIST:
    tree = load_tree_z50(TREE_DIR / PILOT_CASES[case])
    rat = (1 + tree["z50"]) / (1 + MEAN_Z50_CANON)
    target_ids, vr, _ = first_order_vr(tree)
    CASE_DATA[case] = dict(rat=rat, show_ids=target_ids, vr_show=vr)
    print(f"{case}: N first-order subhalos = {len(target_ids)}")

# color by each subhalo's own raw (A=0) VR -- diverging, ONE shared scale across all
# three panels so colors mean the same thing in every panel
all_vr_show = np.concatenate([d["vr_show"] for d in CASE_DATA.values()])
VR_NORM = mcolors.Normalize(vmin=all_vr_show.min(), vmax=all_vr_show.max())
VR_CMAP = plt.cm.coolwarm

fig, axes = plt.subplots(1, 3, figsize=(double_textwidth, single_textwidth), sharex=True, sharey=True)

for ax, case in zip(axes, CASE_LIST):
    rat = CASE_DATA[case]["rat"]
    vr_show = CASE_DATA[case]["vr_show"]
    m_of_A = 1 - A_grid_full * (1 - rat)
    for vr_i in vr_show:
        ax.plot(A_grid_full, vr_i * m_of_A, color=VR_CMAP(VR_NORM(vr_i)), lw=0.5, alpha=0.35)

    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_title(f"{case}, N={len(vr_show)}\\nrat={rat:.3f}", color=CASE_COLORS[case], fontsize=9)
    ax.set_xlabel("A")

axes[0].set_ylabel("VR [SatGen internal units]")
fig.colorbar(plt.cm.ScalarMappable(norm=VR_NORM, cmap=VR_CMAP), ax=axes,
             label="raw VR at accretion, A=0 [SatGen units]", shrink=0.85)
# plt.savefig("../../figures/env_dev_vr_vs_A.pdf", bbox_inches="tight")  # uncomment to save
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## z=0 subhalo mass vs A: the same subhalos, after full integration

**This is the one cell in the notebook that breaks the "raw trees only" rule above** -- VR-vs-A above is a pure scaling-law diagnostic (no evolution needed, since it just applies $m(A)$ to each subhalo's own accretion-time VR), but a subhalo's z=0 mass only exists after actually integrating orbits + tidal stripping, which this notebook otherwise never does. So this cell reads the **already-evolved** pilot trees from `environment_test.ipynb`'s own dataset (`data/local_trees/environment_test/<case>/tree_A{A:.2f}_evo.npz`, the same 7-value A sweep, 0.00-0.75) instead of the raw trees in `TREE_DIR`.

Same subhalo selection AND the same raw-VR color scale as the VR-vs-A panel above (reuses `CASE_DATA`, `VR_NORM`, `VR_CMAP` directly -- run that cell first) -- now every first-order subhalo of each tree, not a sample, matching the panel above. Confirmed by direct comparison that halo indexing is identical between the raw tree in `TREE_DIR` and its A-scaled counterparts in `environment_test/<case>/`, since `build_environment_test.py` only ever mutates first-order subhalos' VR in place and never reorders or drops rows. Unlike VR-vs-A's continuous `A_grid_full`, this can only show the 7 **actual** integrated A values -- there's no physics to interpolate between them.

Plotted as $\\log_{10}[m(A)/m(A{=}0)]$ -- each subhalo's own z=0 mass relative to its own A=0 mass, rather than raw mass -- so every line starts at 0 by construction and the panels are directly comparable regardless of how many orders of magnitude the subhalos themselves span (raw z=0 mass made this hard to read: a late/middle/early subhalo's mass differs by orders of magnitude from its neighbors in the same panel, swamping the A-driven change on a shared axis). All three panels share the same y-scale, so the size of the late vs. early effect can be compared directly by eye. Lines are thin and semi-transparent (`lw=0.5, alpha=0.35`, no markers) for the same reason as the panel above -- hundreds of subhalos overlap, and the point is the shape and spread of the whole fan, not any single trajectory.

**Measured, not just expected -- and it's counter to the naive intuition.** The naive expectation is: late (rat<1, damped VR) subhalos should gain mass as A increases (less radial orbits -> less stripping), early (rat>1, amplified VR) should lose mass (more radial orbits -> more stripping), middle should barely move. Looking at every first-order subhalo's actual A=0 -> A=0.75 change tells a different story: **late loses mass on net** (31% of its subhalos show a clear net loss of >0.05 dex vs. only 3% a clear net gain; median $\\log_{10}$ ratio $\\approx -0.01$, mean $\\approx -0.08$), while **early gains mass on net** (41% clear net gain vs. only 4% clear net loss; median $\\approx +0.02$, mean $\\approx +0.14$, pulled up by a long tail of large individual gains past +3 dex). Middle behaves as expected (both fractions <3%, median $\\approx 0$). The correlation between a subhalo's own raw VR and its final mass change is weak-to-moderate in late and early ($r\\approx 0.2$ and $-0.27$) and near zero in middle -- so raw VR alone doesn't cleanly predict which subhalos swing which way.

**Caveat before reading too much into this:** SatGen's tidal-stripping/ejection step includes an *unseeded* stochastic draw at each timestep (see project notes on the k-order release mechanic), and each A value here was integrated as a fully separate run. That means part of the A=0 vs. A=0.75 difference for any one subhalo is genuine run-to-run noise from that randomness, not purely the deterministic VR shift -- tidal stripping near the disruption threshold is also known to be highly sensitive to small orbital perturbations, so a subhalo that's marginally destroyed in one run and survives comfortably in another (hence the extreme individual outliers, especially in the early panel) is plausible either way. Disentangling how much of this net late-loses/early-gains pattern is a real VR-scaling effect vs. stochastic noise would need repeat realizations per (subhalo, A) -- this cell doesn't attempt that, it's a diagnostic showing what the current single-realization pilot integration actually produced. Worth flagging: this revises the more hand-wavy answer given earlier from the N=8 pilot sample, and is worth checking against the full 1000-tree Nsub/fsub-vs-A result once that's available, since that aggregate (survival-count-based) measure need not move the same way as individual subhalos' masses."""
))

cells.append(nbf.v4.new_code_cell(
"""EVOLVED_DATA_ROOT = Path("../../data/local_trees/environment_test")  # environment_test.ipynb's dataset
PILOT_A_VALUES = (0.00, 0.12, 0.25, 0.37, 0.50, 0.62, 0.75)  # same sweep as environment_test.ipynb


def evo_tree_file(case, A):
    return EVOLVED_DATA_ROOT / case / f"tree_A{A:.2f}_evo.npz"


fig, axes = plt.subplots(1, 3, figsize=(double_textwidth, single_textwidth), sharex=True, sharey=True)

for ax, case in zip(axes, CASE_LIST):
    rat = CASE_DATA[case]["rat"]
    show_ids = CASE_DATA[case]["show_ids"]
    vr_show = CASE_DATA[case]["vr_show"]

    mass_z0 = np.full((len(PILOT_A_VALUES), len(show_ids)), np.nan)
    for k, A in enumerate(PILOT_A_VALUES):
        with np.load(evo_tree_file(case, A)) as d:
            m = d["mass"][show_ids, 0]  # z=0 is CosmicTime/redshift column index 0
        mass_z0[k] = np.where(m > 0, m, np.nan)  # disrupted/invalid entries (-99) -> gap in the line

    log_ratio = np.log10(mass_z0 / mass_z0[0])  # relative to each subhalo's own A=0 mass

    for j, vr_i in enumerate(vr_show):
        ax.plot(PILOT_A_VALUES, log_ratio[:, j], color=VR_CMAP(VR_NORM(vr_i)), lw=0.5, alpha=0.35)

    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_title(f"{case}, N={len(vr_show)}\\nrat={rat:.3f}", color=CASE_COLORS[case], fontsize=9)
    ax.set_xlabel("A")

axes[0].set_ylabel(r"$\\log_{10}[m(A)\\,/\\,m(A{=}0)]$")
fig.colorbar(plt.cm.ScalarMappable(norm=VR_NORM, cmap=VR_CMAP), ax=axes,
             label="raw VR at accretion, A=0 [SatGen units]", shrink=0.85)
# plt.savefig("../../figures/env_dev_mass_vs_A.pdf", bbox_inches="tight")  # uncomment to save
plt.show()"""
))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open(OUT_PATH, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT_PATH)
