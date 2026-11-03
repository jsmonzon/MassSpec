"""
Step 1 of the Ludlow host test (see master.py): for every unevolved
Bolshoi-Rep S13.2 tree, replace the host's concentration history with the
Ludlow+2016 c(z) IN MEMORY and evolve it with SatGen's jsm_SubEvo.evolve_tree.
Only the evolved tree is written (to out_dir, same file name as the fid run's
_evo file, so tree_index lines up 1-to-1 with fid.h5).
"""
import os
import sys
import json
import time
import numpy as np
import multiprocessing as mp

with open("config.json", "r") as f:
    config = json.load(f)

sys.path.insert(0, os.path.join(config["satgen_dir"], "src"))
import config as cfg
import ludlow
import jsm_SubEvo  # sets the fiducial SubEvo options (alpha_type, lnL_pref, evo_mode, phi_res) on import

UNEVO_DIR = config["unevo_dir"]
FID_DIR = config["fid_dir"]
OUT_DIR = config["out_dir"]
HIST_DIR = os.path.join(OUT_DIR, "ludlow_c_history")


def tree_files():
    return sorted(os.path.join(UNEVO_DIR, fn) for fn in os.listdir(UNEVO_DIR)
                  if fn.startswith("tree") and fn.endswith(".npz") and not fn.endswith("evo.npz"))


def out_path(file_i):
    return os.path.join(OUT_DIR, os.path.basename(file_i)[:-4] + "_evo.npz")


def tree_index(file_i):
    return int(os.path.basename(file_i)[:-4].split("_")[2])  # tree_<logM>_<i>.npz


def preflight(files):
    """Stop before launching ~1500 jobs if anything is inconsistent."""
    print(f"SubEvo settings: alpha_type={jsm_SubEvo.alpha_type}, lnL_pref={cfg.lnL_pref}, "
          f"evo_mode={cfg.evo_mode}, phi_res={cfg.phi_res}")
    print(f"cosmology: h={cfg.h}, Om={cfg.Om}, OL={cfg.OL}  ({len(cfg.zsample)} snapshots)")
    print(f"Ludlow constants: f={config['f_CMH']}, A={config['A_L16']}")
    print(f"found {len(files)} unevolved trees in {UNEVO_DIR}")
    if len(files) != 1517:
        print("  WARNING: expected 1517 trees for the S13.2 sample")
    if not files:
        sys.exit("no trees found -- check unevo_dir")

    # cosmology must match the trees' snapshot grid (first, middle, last tree)
    for fi in (files[0], files[len(files) // 2], files[-1]):
        z = np.load(fi)["redshift"]
        if len(z) != len(cfg.zsample) or not np.allclose(cfg.zsample, z):
            sys.exit(f"SatGen config.py cosmology does not match {os.path.basename(fi)} "
                     f"({len(z)} snapshots vs {len(cfg.zsample)}) -- activate the BolshoiP block.")

    # every tree should have a fid (zhao-arm) counterpart, evolved on the same grid
    missing = [fi for fi in files if not os.path.exists(os.path.join(FID_DIR, os.path.basename(out_path(fi))))]
    if missing:
        print(f"  WARNING: {len(missing)} trees have no fid _evo counterpart, e.g. {os.path.basename(missing[0])}")
    else:
        zf = np.load(os.path.join(FID_DIR, os.path.basename(out_path(files[0]))))["redshift"]
        if not np.allclose(zf, cfg.zsample):
            sys.exit("the fid (zhao-arm) evolved trees are on a different snapshot grid than config.py")
        print("  all trees have a fid (zhao-arm) counterpart on the same snapshot grid")


def process_file(file_i):
    """Ludlow host c(z) -> evolve -> save. Returns a small status dict."""
    t0 = time.time()
    name = os.path.basename(file_i)
    try:
        f = np.load(file_i)
        data = {k: np.array(f[k]) for k in f.files}

        c_zhao = data["concentration"][0].copy()
        valid = c_zhao > 0
        c_lud, z_m2, conv = ludlow.concentration_history_Ludlow2016(
            data["mass"], data["ParentID"], data["redshift"], cfg.Dvsample,
            c_guess=c_zhao, f=config["f_CMH"], A=config["A_L16"])
        data["concentration"][0] = np.where(valid, c_lud, c_zhao)  # -99 dummies kept

        np.random.seed(config["evo_seed"] + tree_index(file_i))
        out = jsm_SubEvo.evolve_tree(data, label=file_i)
        if out is None:
            return {"name": name, "ok": False, "error": "evolve_tree returned None (no lt)", "dt": time.time() - t0}

        # write atomically so an interrupted job never leaves a truncated _evo.npz
        final = out_path(file_i)
        tmp = final[:-4] + ".partial.npz"
        np.savez(tmp, **out)
        os.replace(tmp, final)

        np.savez(os.path.join(HIST_DIR, name[:-4] + "_ludlow_c.npz"),
                 redshift=data["redshift"], c_zhao=c_zhao, c_ludlow=data["concentration"][0],
                 z_m2=z_m2, converged=conv, f_CMH=config["f_CMH"], A_L16=config["A_L16"])

        return {"name": name, "ok": True, "dt": time.time() - t0,
                "c0_zhao": c_zhao[0], "c0_ludlow": data["concentration"][0][0],
                "n_fallback": int((valid & ~conv).sum())}
    except Exception as e:
        return {"name": name, "ok": False, "error": repr(e), "dt": time.time() - t0}


if __name__ == "__main__":
    os.makedirs(HIST_DIR, exist_ok=True)
    files = tree_files()
    preflight(files)

    todo = [fi for fi in files if not os.path.exists(out_path(fi))]
    print(f"{len(files) - len(todo)} already evolved in {OUT_DIR} (skipped); {len(todo)} to go "
          f"on {config['N_cpus']} cores", flush=True)

    t_start = time.time()
    n_err = 0
    n_fallback = 0
    PRINT_EVERY = 100
    with mp.Pool(processes=config["N_cpus"]) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, todo), 1):
            if r["ok"]:
                n_fallback += r["n_fallback"] > 0
            else:
                n_err += 1
                print(f"  ERROR {r['name']}: {r['error']}", flush=True)  # errors are always reported right away
            if i % PRINT_EVERY == 0 or i == len(todo):
                elapsed = time.time() - t_start
                eta = elapsed / i * (len(todo) - i)
                print(f"[{i:4d}/{len(todo)}] errors: {n_err} | trees with Zhao fallback snapshots: {n_fallback} | "
                      f"elapsed {elapsed / 3600:5.2f} h, ETA {eta / 3600:5.2f} h", flush=True)

    print(f"done: {len(todo) - n_err} evolved, {n_err} errors, {(time.time() - t_start) / 3600:.2f} h total")
    if n_err:
        sys.exit(1)
