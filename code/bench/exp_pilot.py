"""E17: pilot validation -- does a cheap coverage pilot predict the larger-budget gain?

For every task large enough to support a pilot (k_pilot=10) and a larger budget
(k_full in {20,30,50}), we draw independent support sets for the pilot and the full
run, and record the paired gains (kmeans - random). A decision policy adopts coverage
only when the pilot gain is positive; we then score the policy's realised utility
against always-random (0), always-coverage, and an oracle.

What makes the pilot-then-decide rule informative is precisely whether the pilot
predicts the full-budget outcome, which is what this script measures.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import GRID_CONFIGS, OUT, build, prep, task_seed  # noqa: E402
from bench.run import model_factory, select  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

MODELS = ("knn", "ridge")
PILOT_K = 10
PILOT_REPS = 5          # cheap pilot: five random supports
FULL_REPS = 10          # evaluation: ten supports


def rho(y, pred):
    v = spearmanr(y, pred).correlation
    return float(v) if np.isfinite(v) else 0.0


def run_task(Xi, yi, kp, kf, seed):
    n = len(yi)
    Z = prep(Xi)
    rows = []
    rng_p = np.random.default_rng(seed)
    rng_f = np.random.default_rng(seed + 7919)
    for rep in range(FULL_REPS):
        # the pilot budget is only run for the first PILOT_REPS repetitions;
        # each repetition otherwise contributes one support pair at each budget
        budgets = ((kp, rng_p), (kf, rng_f)) if rep < PILOT_REPS else ((kf, rng_f),)
        for k, rng in budgets:
            sel_r = select("random", Z, k, rng)
            sel_k = select("kmeans", Z, k, rng)
            q_r = np.setdiff1d(np.arange(n), sel_r)
            q_k = np.setdiff1d(np.arange(n), sel_k)
            for mname in MODELS:
                m = model_factory(mname, rep)
                m.fit(Xi[sel_r], yi[sel_r]); rr = rho(yi[q_r], m.predict(Xi[q_r]))
                m = model_factory(mname, rep)
                m.fit(Xi[sel_k], yi[sel_k]); rk = rho(yi[q_k], m.predict(Xi[q_k]))
                rows.append({"rep": rep, "k": k, "model": mname,
                             "pilot": (k == kp and rep < PILOT_REPS), "gain": rk - rr})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "pilot_pairs.csv"))
    args = ap.parse_args()

    rows, t0 = [], time.time()
    for ds, target, ms, inc in GRID_CONFIGS:
        if ds in ("D4", "D8"):
            continue
        print(f"== {ds} {target or ''} ==", flush=True)
        r, tasks = build(ds, target, ms, inc)
        X, y = r["X"], r["y"]
        for ti, task in enumerate(tasks, 1):
            idx = task["idx"]; n = len(idx)
            kp = PILOT_K
            valid = [k for k in (20, 30, 50) if k <= n - 10 and k >= kp + 10]
            if not valid:
                continue
            kf = max(valid)
            Xi, yi = X[idx], y[idx]
            df = run_task(Xi, yi, kp, kf, task_seed(ds, target, task["group"]))
            p = df[(df.k == kp) & (df.rep < PILOT_REPS)].groupby("rep").gain.mean()
            f = df[df.k == kf].groupby("rep").gain.mean()
            rows.append({"dataset": ds, "target": target or "main",
                         "group": task["group"], "n": n, "kp": kp, "kf": kf,
                         "pilot_gain": p.mean(), "pilot_sd": p.std(ddof=0),
                         "full_gain": f.mean(), "full_sd": f.std(ddof=0)})
            if ti % 10 == 0:
                print(f"  {ti}/{len(tasks)} tasks {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)

    d = pd.DataFrame(rows)
    d["family"] = [f"{a} {b}" for a, b in zip(d.dataset, d.target)]
    d.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(d)} tasks)")

    # ---------------- policy evaluation ----------------
    def evaluate(sub, thr=0.0):
        pilot, full = sub.pilot_gain.values, sub.full_gain.values
        adopt = pilot > thr
        util = np.where(adopt, full, 0.0)
        oracle = np.maximum(full, 0.0)
        return {"n": len(sub), "pilot_thr": thr,
                "always_random": 0.0, "always_coverage": full.mean(),
                "policy": util.mean(), "oracle": oracle.mean(),
                "regret": (oracle - util).mean(),
                "sensitivity": ((adopt) & (full > 0)).sum() / max(1, (full > 0).sum()),
                "specificity": ((~adopt) & (full <= 0)).sum() / max(1, (full <= 0).sum())}

    summ = [evaluate(d, t) for t in (-0.05, 0.0, 0.02, 0.05)]
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "pilot_summary.csv", index=False)
    print("\n=== policy (all tasks) ===")
    print(summ.round(3).to_string(index=False))

    fam = []
    for fam_name, sub in d.groupby("family"):
        row = evaluate(sub, 0.0); row["family"] = fam_name
        fam.append(row)
    fam = pd.DataFrame(fam).sort_values("always_coverage", ascending=False)
    fam.to_csv(OUT / "pilot_by_family.csv", index=False)
    print("\n=== policy by family ===")
    print(fam.round(3).to_string(index=False))

    q = d[["pilot_gain", "full_gain"]].dropna()
    slope, icpt = np.polyfit(q.pilot_gain, q.full_gain, 1)
    pred = slope * q.pilot_gain + icpt
    ss = 1 - ((q.full_gain - pred) ** 2).sum() / ((q.full_gain - q.full_gain.mean()) ** 2).sum()
    calib = pd.DataFrame([{"spearman": spearmanr(q.pilot_gain, q.full_gain).correlation,
                           "ols_slope": slope, "ols_intercept": icpt, "r2": ss,
                           "n": len(q)}])
    calib.to_csv(OUT / "pilot_calibration.csv", index=False)
    print("\n=== calibration ===")
    print(calib.round(3).to_string(index=False))


if __name__ == "__main__":
    main()