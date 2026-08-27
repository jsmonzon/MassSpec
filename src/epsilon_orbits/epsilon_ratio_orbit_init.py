"""
epsilon_ratio_orbit_init.py

Scales (rather than additively perturbs) the initial radial velocity of
first-order subhalos in a raw (un-evolved) SatGen merger tree, using a
per-timestep epsilon(z) built from the ratio of the tree's own host mass
history to the ensemble mean MAH for its mass bin.

Background
----------
This is a sibling to epsilon_orbit_init.py. That script adds a constant
(or callable-of-z) perturbation directly to VR. Here, epsilon is derived
from the data itself:

    ratio(z) = M_host_unique(z) / M_host_mean(z)
    epsilon(z) = clip( 1 + A * (ratio(z) - 1), eps_min, eps_max )

evaluated on SatGen's native cfg.zsample grid. Both a raw tree's own main
branch (mass[0, :]) and the SatGen/etc/mean_MAH/*.npz arrays already
live on that same 354-point grid, so no interpolation is needed -- the
two arrays line up index-for-index. Any timestep where the tree's own
main-branch mass is unresolved (NaN -- its branch has dropped below the
tree's resolution limit at that redshift) is masked to epsilon = 1 (no
perturbation), rather than left as NaN or extrapolated.

A is a free parameter controlling how strongly a tree's own accretion
history (relative to the 1000-tree ensemble mean for its mass bin) is
allowed to shift its orbit:

    A = 0  ->  epsilon = 1 for every tree, regardless of ratio (no
               perturbation -- a clean "off" switch).
    A = 1  ->  epsilon = ratio exactly, reproducing this script's
               original (pre-A) behavior: 1 + 1*(ratio - 1) = ratio.
    A > 1  ->  amplifies each tree's deviation from the ensemble mean.
               A fast-growing tree (ratio > 1) gets pushed to a bigger
               speed-up, a slow-growing tree (ratio < 1) to a bigger
               slow-down, before clipping.
    0<A<1  ->  damps the deviation (a gentler version of the A=1 case).

Because ratio is centered near 1 (not 0), this deviation form -- not a
plain epsilon = A*ratio -- is what makes A a genuine dial: A*ratio would
never go below 1 for A>0 (ratio is always positive), so it could only
ever speed subhalos up, never slow them down, and would blow past any
fixed clip range almost immediately for A much bigger than 1. Across a
real sample of first-order accretion events (13.0 mass bin), ratio
ranged from about 0.14 to 2.6 (median ~1.1), so a plain A=10 * ratio
would run from ~1.4 to ~26 -- saturating a [0.5, 1.5] clip for
essentially every tree. The deviation form only pushes eps below 1 for
below-average trees, and reproduces the original model exactly at A=1,
so a caller sweeping A actually gets a graded family of models rather
than an instant flip to "everyone gets clipped to eps_max."

eps_min must stay strictly positive (enforced in __init__ -- see
"Sign safety" below): epsilon multiplies VR, so an epsilon that reaches
0 or goes negative would zero out or reverse a subhalo's initial radial
velocity, not just slow it down. eps_min/eps_max are NOT automatically
rescaled with A -- they're plain, explicit clip bounds a caller sets
(same as always); at large A they'll typically need to be widened by
hand (e.g. via --eps-min/--eps-max, or apply_epsilon_ratio_to_directory.py's
run_scale_sweep()) to avoid saturating almost every tree to one bound.

For each first-order subhalo (order == 1 at its own accretion snapshot),
VR at that snapshot is *multiplied* by epsilon(z_acc) -- not added to,
unlike epsilon_orbit_init.py. Since epsilon > 0 is enforced, VR's sign
is always preserved: a subhalo only ever falls in faster/slower (or
moves outward faster/slower), never reverses direction. Nothing else in
the tree is touched: mass, order, ParentID, VirialRadius, concentration,
and every other coordinate are passed through unchanged, exactly as in
epsilon_orbit_init.py.

Which mean_MAH file backs a given tree is chosen by the tree's own z=0
host mass (mass[0, 0]), rounded to the nearest 0.1 dex, matching the
"{logM0:.1f}_files_mean_MAH.npz" naming convention used in
SatGen/etc/mean_MAH/. A tree whose host mass falls outside the
available bins raises rather than silently reusing the nearest one.
"""

import argparse
from pathlib import Path

import numpy as np



def infer_logM0(mass, decimals=1):
    """
    Rounds a tree's own z=0 host mass, mass[0, 0], to `decimals` dex --
    this is the key used to look up its mean_MAH reference file
    ("{logM0:.1f}_files_mean_MAH.npz"). Exposed as a standalone function
    (rather than kept private to the class) so batch drivers can pre-check
    mass-bin coverage before constructing an EpsilonRatioOrbitInit -- see
    apply_epsilon_ratio_to_directory.py, which uses this to skip trees
    outside the sampled range instead of raising.
    """
    return round(float(np.log10(mass[0, 0])), decimals)


class EpsilonRatioOrbitInit:
    """
    Reads a raw (un-evolved) SatGen merger tree .npz file and writes out a
    copy in which the radial velocity component (VR) of the initial orbit
    has been scaled by

        epsilon(z_acc) = clip(1 + A * (ratio(z_acc) - 1), eps_min, eps_max)
        ratio(z) = M_unique(z) / M_mean(z)

    for subhalos of a given instantaneous order at accretion (default:
    first-order, order == 1).

    Syntax:

        erv = EpsilonRatioOrbitInit(tree_file, output_dir, mean_mah_dir, A=1.0)
        erv.run()

    where

        tree_file: path to a raw tree .npz file, as produced by TreeGen
            (str or pathlib.Path)
        output_dir: directory the modified tree is written into; created
            if it doesn't already exist (str or pathlib.Path)
        mean_mah_dir: directory holding the "{logM0:.1f}_files_mean_MAH.npz"
            reference files (e.g. SatGen/etc/mean_MAH/)
        A: free parameter scaling how strongly a tree's own MAH deviation
            from the ensemble mean shifts epsilon away from 1 (float,
            default 1.0 -- reproduces the original ratio-only model;
            A=0 disables the perturbation entirely; see the module
            docstring for the full explanation and worked numbers)
        order_filter: only subhalos with this instantaneous order at their
            accretion snapshot are scaled (int, default 1, i.e.
            first-order subhalos)
        eps_min, eps_max: epsilon is clipped to this range (floats,
            default 0.5, 1.5). eps_min must be > 0 -- see "Sign safety"
            below -- and is NOT automatically widened as A grows; pick
            eps_min/eps_max to suit whatever A you're using.
        suffix: appended to the input file's stem to build the output
            filename (str, default "_epsilon")

    Sign safety
    -----------
    epsilon multiplies VR, so epsilon must stay strictly positive or a
    subhalo's initial radial velocity could be zeroed out (epsilon = 0)
    or have its direction reversed (epsilon < 0) rather than simply
    scaled. __init__ raises ValueError if eps_min <= 0, regardless of A --
    this is a hard floor, not something that relaxes at high A.

    Attributes set in __init__ (before any perturbation is applied):

        self.data: the raw NpzFile object for the input tree (unmodified)
        self.coordinates: a modified COPY of data["coordinates"]; identical
            to the input until self.apply_epsilon() is called
        self.mass, self.order, self.redshift: convenience references into
            self.data
        self.logM0: the tree's own z=0 host mass, log10(Msun), rounded to
            1 decimal -- used to pick the matching mean_MAH reference file
        self.mean_mah_file: path to the mean_MAH reference used
        self.A: the perturbation-strength parameter used
        self.ratio_z: (Ntime,) array, the raw (unclipped) M_unique(z)/M_mean(z)
            ratio at every output redshift step -- kept around for
            diagnostics (see summary()), independent of A/eps_min/eps_max
        self.epsilon_z: (Ntime,) array, epsilon(z) = clip(1 + A*(ratio_z - 1),
            eps_min, eps_max), evaluated at every output redshift step,
            already clipped and NaN-masked
        self.Nhalo, self.Ntime: tree dimensions
        self.acc_index: (Nhalo,) accretion-snapshot index per subhalo, or
            -1 where no valid initial orbit was found (host, or the rare
            NaN-coordinate edge case noted in orbit.resample_orbit)
        self.acc_order: (Nhalo,) instantaneous order at accretion, or -1
            where self.acc_index is -1
        self.target_ids: subhalo indices selected for scaling
        self.output_file: path the modified tree will be/was written to

    Methods:

        erv.apply_epsilon(): scales self.coordinates in place. Calling it
            more than once will double (or triple, ...) apply epsilon, so
            call it exactly once before erv.write().
        erv.write(): saves the modified tree to self.output_file, creating
            output_dir if needed. Returns self.output_file.
        erv.run(): apply_epsilon() + write(), returns self.output_file.
        erv.summary(): one-line diagnostic string -- see its docstring for
            what it reports (including how much of the population is
            saturating the eps_min/eps_max clip, which is the first thing
            to check when sweeping A).
    """

    VR_INDEX = 3  # [R, phi, z, VR, Vphi, Vz]

    def __init__(self, tree_file, output_dir, mean_mah_dir, A=1.0, order_filter=1,
                 eps_min=0.5, eps_max=1.5, suffix="_epsilon"):
        if eps_min <= 0:
            raise ValueError(
                f"eps_min must be > 0 (got {eps_min}) -- epsilon multiplies VR, "
                f"so a non-positive eps_min could zero out or reverse a "
                f"subhalo's initial radial velocity instead of just scaling it. "
                f"This floor applies regardless of A."
            )

        self.tree_file = Path(tree_file)
        self.output_dir = Path(output_dir)
        self.mean_mah_dir = Path(mean_mah_dir)
        self.A = A
        self.order_filter = order_filter
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.suffix = suffix

        self.output_file = (
            self.output_dir / f"{self.tree_file.stem}{self.suffix}{self.tree_file.suffix}"
        )

        self.data = np.load(self.tree_file)
        self.coordinates = np.copy(self.data["coordinates"])
        self.mass = self.data["mass"]
        self.order = self.data["order"]
        self.redshift = self.data["redshift"]
        self.Nhalo, self.Ntime = self.mass.shape

        self.logM0 = infer_logM0(self.mass)
        self.mean_mah_file = self.mean_mah_dir / f"{self.logM0:.1f}_files_mean_MAH.npz"
        if not self.mean_mah_file.exists():
            raise FileNotFoundError(
                f"{self.tree_file.name}: no mean MAH reference for host mass "
                f"bin logM0={self.logM0:.1f} (looked for {self.mean_mah_file})"
            )
        mean_data = np.load(self.mean_mah_file)
        M_mean = mean_data["M"]
        z_mean = mean_data["z"]
        if M_mean.shape != self.redshift.shape or not np.allclose(z_mean, self.redshift):
            raise ValueError(
                f"{self.tree_file.name}: tree's redshift grid doesn't match "
                f"{self.mean_mah_file.name}'s -- can't index epsilon(z) directly "
                f"without interpolation."
            )

        self.ratio_z, self.epsilon_z = self._build_epsilon(self.mass[0, :], M_mean)

        self._find_accretion()
        self._select_targets()
        self._applied = False

    def _build_epsilon(self, M_unique, M_mean):
        """
        Returns (ratio_z, epsilon_z): the raw ratio(z) = M_unique/M_mean
        (unclipped, kept for diagnostics) and epsilon(z) =
        clip(1 + A*(ratio(z) - 1), eps_min, eps_max), with any non-finite
        ratio (unresolved main branch at that step) masked to epsilon = 1.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = M_unique / M_mean
        eps = 1.0 + self.A * (ratio - 1.0)
        eps = np.clip(eps, self.eps_min, self.eps_max)
        eps = np.where(np.isfinite(ratio), eps, 1.0)
        return ratio, eps

    def _find_accretion(self):
        """
        Locates each subhalo's accretion snapshot: the first (highest-z)
        time index where its coordinates entry is nonzero. Mirrors the
        approach used in orbit.resample_orbit and epsilon_orbit_init.py.
        sub_ii = 0 (the host) is always left at -1, since the host has no
        orbit of its own.
        """
        acc_index = np.full(self.Nhalo, -1, dtype=int)
        for sub_ii in range(1, self.Nhalo):
            nz = np.nonzero(self.coordinates[sub_ii])[0]
            if len(nz) == 0:
                continue
            acc_index[sub_ii] = nz[0]
        self.acc_index = acc_index
        self.valid = acc_index >= 0

    def _select_targets(self):
        """
        Determines each valid subhalo's instantaneous order at its own
        accretion snapshot, then selects the subset matching order_filter.
        """
        acc_order = np.full(self.Nhalo, -1, dtype=int)
        acc_order[self.valid] = self.order[
            np.arange(self.Nhalo)[self.valid], self.acc_index[self.valid]
        ]
        self.acc_order = acc_order
        self.target_ids = np.nonzero(self.valid & (acc_order == self.order_filter))[0]

    def _epsilon_at(self, sub_ii):
        iz = self.acc_index[sub_ii]
        return self.epsilon_z[iz]

    def apply_epsilon(self):
        """Multiplies VR by epsilon(z_acc) at the accretion snapshot of each targeted subhalo."""
        for sub_ii in self.target_ids:
            iz = self.acc_index[sub_ii]
            self.coordinates[sub_ii, iz, self.VR_INDEX] *= self._epsilon_at(sub_ii)
        self._applied = True
        return self.coordinates

    def write(self):
        """Writes the (possibly scaled) tree to self.output_file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.output_file,
            redshift=self.data["redshift"],
            CosmicTime=self.data["CosmicTime"],
            mass=self.data["mass"],
            order=self.data["order"],
            ParentID=self.data["ParentID"],
            VirialRadius=self.data["VirialRadius"],
            concentration=self.data["concentration"],
            coordinates=self.coordinates,
        )
        return self.output_file

    def run(self):
        """apply_epsilon() + write(), returns the output path."""
        if not self._applied:
            self.apply_epsilon()
        return self.write()

    def summary(self):
        """
        One-line diagnostic covering both the targeted population and how
        much of it is saturating the eps_min/eps_max clip -- the latter is
        the first thing worth checking whenever A, eps_min, or eps_max
        changes, since a saturated population means epsilon has stopped
        tracking each tree's own MAH history and just become a near-flat
        multiplier.
        """
        acc_iz = self.acc_index[self.target_ids]
        raw_ratio = self.ratio_z[acc_iz]
        raw_eps = 1.0 + self.A * (raw_ratio - 1.0)  # pre-clip, for diagnostics only
        eps_targeted = self.epsilon_z[acc_iz]

        finite = np.isfinite(raw_eps)
        n_finite = int(finite.sum())
        frac_saturated = (
            float(np.mean((raw_eps[finite] < self.eps_min) | (raw_eps[finite] > self.eps_max)))
            if n_finite else float("nan")
        )

        return (
            f"{self.tree_file.name}: logM0={self.logM0:.1f} (-> {self.mean_mah_file.name}), "
            f"A={self.A}, eps_min={self.eps_min}, eps_max={self.eps_max}, "
            f"Nhalo={self.Nhalo}, valid subhalos={int(self.valid.sum())}, "
            f"order=={self.order_filter} at accretion={len(self.target_ids)}, "
            f"epsilon (clipped) at targeted accretion snapshots: "
            f"min={eps_targeted.min():.3f}, max={eps_targeted.max():.3f}, mean={eps_targeted.mean():.3f}; "
            f"pre-clip epsilon range: min={np.nanmin(raw_eps):.3f}, max={np.nanmax(raw_eps):.3f}; "
            f"fraction saturating [eps_min, eps_max]: {frac_saturated:.1%}"
        )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Scale first-order subhalos' initial radial velocity by "
                     "epsilon(z_acc) = clip(1 + A*(ratio(z_acc)-1), eps_min, eps_max), "
                     "ratio(z) = M_unique(z)/M_mean(z), in a raw SatGen tree file."
    )
    p.add_argument("tree_file", type=str, help="path to a raw (un-evolved) tree .npz file")
    p.add_argument("output_dir", type=str, help="directory to write the scaled tree into")
    p.add_argument("mean_mah_dir", type=str,
                    help="directory holding '{logM0:.1f}_files_mean_MAH.npz' reference files")
    p.add_argument("--A", type=float, default=1.0,
                    help="perturbation-strength parameter (default: 1.0, reproduces the "
                         "original ratio-only model; A=0 disables the perturbation entirely; "
                         "see the module docstring for how A interacts with eps_min/eps_max)")
    p.add_argument("--order-filter", type=int, default=1,
                    help="instantaneous order at accretion to target (default: 1)")
    p.add_argument("--eps-min", type=float, default=0.5,
                    help="lower epsilon clip (default: 0.5); must be > 0")
    p.add_argument("--eps-max", type=float, default=1.5, help="upper epsilon clip (default: 1.5)")
    p.add_argument("--suffix", type=str, default="_epsilon",
                    help="suffix appended to the output filename stem (default: _epsilon)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    erv = EpsilonRatioOrbitInit(
        tree_file=args.tree_file,
        output_dir=args.output_dir,
        mean_mah_dir=args.mean_mah_dir,
        A=args.A,
        order_filter=args.order_filter,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        suffix=args.suffix,
    )
    print(erv.summary())
    out = erv.run()
    print(f"wrote: {out}")
