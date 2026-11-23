"""E1 prototype: task-level risk-controlled adoption of coverage selection.

Decision unit is a task (dataset x target x group), not a molecule. The score is
the k=10 coverage pilot gain s = rho(kmeans) - rho(random). The rule adopts
coverage iff s > lambda. The risk we control is the conditional probability that
adoption was a mistake,

    R(lambda) = P(full_gain < 0 | s > lambda),

i.e. "we switched to coverage and the large-budget run came out worse than
random". Coverage of the decision is the adoption rate c(lambda) = P(s > lambda).

Estimators
----------
plug-in   : lambda-hat = inf{lambda : S/m <= alpha}, S = #adopted negatives,
            m = #adopted, with m >= MIN_ADOPT.
CP-UCB    : lambda-hat = inf{lambda : Clopper-Pearson UCB_{1-delta}(S, m) <= alpha}
            (distribution-free under exchangeable calibration tasks).
weights   : optional dataset-equal weighting, so D2 (31 tasks) does not dominate.

Protocols
---------
split     : repeated random task-level calibration/test splits (split conformal).
cv+       : K-fold cross-conformal, folds stratified within dataset.
cluster   : leave-one-dataset-out; calibration and test datasets are disjoint, so
            the exchangeability assumption is only over datasets (8 of them).

Outputs land in results/bench_v2/e1_*.csv and results/figures/v2/figE1_*.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "bench_v2"
FIGDIR = ROOT / "results" / "figures" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

LAM_GRID = np.round(np.arange(-0.12, 0.241, 0.005), 3)
MIN_ADOPT = 10          # do not trust a conditional risk estimated from <10 tasks
DELTA = 0.10            # conformal confidence level (1 - delta)
N_EVAL = 400            # split-conformal repetitions
Z_LCB = 1.645           # for the conservative pilot-score variant


# ----------------------------------------------------------------- statistics
def cp_upper(s, m, conf=1.0 - DELTA):
    """Clopper-Pearson upper bound on a binomial proportion."""
    if m <= 0:
        return 0.0
    if s >= m:
        return 1.0
    return float(beta.ppf(conf, s + 1, m - s))


def risk_point(sub, lam, weights=None):
    """Conditional negative rate among adopted tasks, optionally dataset-weighted."""
    a = sub.pilot_gain.values > lam
    m = int(a.sum())
    if m == 0:
        return np.nan, 0
    neg = (sub.full_gain.values < 0) & a
    if weights is None:
        return float(neg.sum() / m), m
    w = weights.reindex(sub.dataset.values).fillna(1.0).values
    num = float((w[neg]).sum())
    den = float(w[a].sum())
    return (num / den if den > 0 else np.nan), m


def evaluate(sub, lam):
    """Policy metrics for one threshold on a task set."""
    full = sub.full_gain.values
    adopt = sub.pilot_gain.values > lam
    n = len(sub)
    m = int(adopt.sum())
    out = {"n": n, "n_adopt": m, "adopt_rate": m / n,
           "oracle": float(np.maximum(full, 0).mean()),
           "always_coverage": float(full.mean())}
    if m == 0:
        out.update({"risk": np.nan, "harm_incidence": 0.0, "mean_full_adopt": np.nan,
                    "utility": 0.0, "regret": out["oracle"]})
    else:
        out.update({
            "risk": float((full[adopt] < 0).mean()),
            "harm_incidence": float(((full < 0) & adopt).mean()),
            "mean_full_adopt": float(full[adopt].mean()),
            "utility": float(full[adopt].sum() / n),
            "regret": float(out["oracle"] - full[adopt].sum() / n)})
    return out


# ---------------------------------------------------------------- selection
def select_lambda(cal, alpha, mode="ucb", min_adopt=MIN_ADOPT, weights=None,
                  conf=1.0 - DELTA):
    """Smallest lambda whose risk satisfies the target; +inf means never adopt."""
    if cal.empty:
        return np.inf
    for lam in LAM_GRID:
        risk, m = risk_point(cal, lam, weights)
        if m < min_adopt:
            continue
        if mode == "plugin":
            if risk <= alpha:
                return float(lam)
        elif mode == "ucb":
            a = cal.pilot_gain.values > lam
            s = int(((cal.full_gain.values < 0) & a).sum())
            if cp_upper(s, m, conf) <= alpha:
                return float(lam)
        else:
            raise ValueError(mode)
    return np.inf


# ------------------------------------------------------------------- curves
def curve_table(df, grid=LAM_GRID, n_boot=2000, seed=0):
    """Point estimates plus cluster-bootstrap CIs resampling datasets."""
    rng = np.random.default_rng(seed)
    datasets = sorted(df.dataset.unique())
    blocks = {ds: (df[df.dataset == ds].pilot_gain.values,
                   df[df.dataset == ds].full_gain.values) for ds in datasets}

    def stats(p, f):
        """All thresholds at once for one (bootstrapped) task set."""
        adopt = p[None, :] > grid[:, None]                 # L x N
        n = len(f)
        m = adopt.sum(1)
        neg = f < 0
        s = (adopt & neg[None, :]).sum(1)
        risk = np.where(m > 0, s / np.maximum(m, 1), np.nan)
        util = (adopt * f[None, :]).sum(1) / n
        oracle = np.maximum(f, 0).mean()
        return {"m": m, "adopt": m / n, "risk": risk, "util": util,
                "harm": (adopt & neg[None, :]).sum(1) / n, "oracle": oracle}

    base = stats(np.concatenate([blocks[d][0] for d in datasets]),
                 np.concatenate([blocks[d][1] for d in datasets]))
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(datasets, size=len(datasets), replace=True)
        p = np.concatenate([blocks[d][0] for d in pick])
        f = np.concatenate([blocks[d][1] for d in pick])
        boots.append(stats(p, f))
    allfull = np.concatenate([blocks[d][1] for d in datasets])
    out = pd.DataFrame({
        "lambda": grid, "n_adopt": base["m"], "adopt_rate": base["adopt"],
        "risk": base["risk"], "harm_incidence": base["harm"],
        "utility": base["util"], "regret": base["oracle"] - base["util"],
        "always_coverage": float(allfull.mean())})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key, col in (("risk", "risk"), ("adopt", "adopt_rate"), ("util", "utility")):
            arr = np.array([b[key] for b in boots])       # n_boot x L
            out[f"{col}_lo"] = np.nanpercentile(arr, 5, axis=0)
            out[f"{col}_hi"] = np.nanpercentile(arr, 95, axis=0)
    return out


# ---------------------------------------------------------------- protocols
def split_conformal(df, alpha, mode, reps=N_EVAL, test_frac=0.5, seed=0,
                    weights=None):
    rng = np.random.default_rng(seed)
    rows = []
    n_test = int(round(test_frac * len(df)))
    for r in range(reps):
        idx = rng.permutation(len(df))
        te = df.iloc[idx[:n_test]]
        ca = df.iloc[idx[n_test:]]
        lam = select_lambda(ca, alpha, mode, weights=weights)
        e = evaluate(te, lam)
        if np.isfinite(lam):
            a = ca.pilot_gain.values > lam
            s = int(((ca.full_gain.values < 0) & a).sum())
            e["cal_risk"] = float(s / a.sum())
            e["cal_ucb"] = cp_upper(s, int(a.sum()))
        else:
            e["cal_risk"] = np.nan
            e["cal_ucb"] = np.nan
        base = evaluate(te, 0.0)          # fixed delta > 0 baseline, same test half
        e.update({"base_utility": base["utility"], "base_regret": base["regret"],
                  "base_risk": base["risk"], "base_adopt_rate": base["adopt_rate"]})
        e.update({"rep": r, "alpha": alpha, "mode": mode, "lambda": lam,
                  "violation": bool(np.isfinite(e["risk"]) and e["risk"] > alpha)})
        rows.append(e)
    return pd.DataFrame(rows)


def cv_plus(df, alpha, mode, K=5, seed=0, weights=None):
    """Cross-conformal: fold k is scored by a lambda fitted on the other folds.

    Folds are taken within each dataset so every fold keeps every dataset.
    """
    rng = np.random.default_rng(seed)
    fold = np.empty(len(df), dtype=int)
    for _, sub in df.groupby("dataset"):
        order = rng.permutation(len(sub))
        fold[sub.index.values[order]] = np.arange(len(sub)) % K
    rows, per_task = [], []
    for k in range(K):
        ca = df[fold != k]
        te = df[fold == k]
        lam = select_lambda(ca, alpha, mode, weights=weights)
        e = evaluate(te, lam)
        e.update({"fold": k, "alpha": alpha, "mode": mode, "lambda": lam})
        rows.append(e)
        t = te[["dataset", "family", "pilot_gain", "full_gain"]].copy()
        t["adopt"] = t.pilot_gain > lam
        t["fold"] = k
        per_task.append(t)
    pool = pd.concat(per_task, ignore_index=True)
    full = pool.full_gain.values
    adopt = pool.adopt.values
    pooled = {
        "n": len(pool), "n_adopt": int(adopt.sum()), "adopt_rate": float(adopt.mean()),
        "risk": float((full[adopt] < 0).mean()) if adopt.any() else np.nan,
        "harm_incidence": float(((full < 0) & adopt).mean()),
        "mean_full_adopt": float(full[adopt].mean()) if adopt.any() else np.nan,
        "utility": float(full[adopt].sum() / len(pool)),
        "oracle": float(np.maximum(full, 0).mean()),
        "always_coverage": float(full.mean())}
    pooled["regret"] = pooled["oracle"] - pooled["utility"]
    return pd.DataFrame(rows), pooled


def cluster_lodo(df, alpha, mode, weights=None):
    """Leave-one-dataset-out: calibration and test datasets are disjoint.

    Returns per-dataset rows plus a task-weighted pooled summary; the per-dataset
    mean (every dataset weighted equally) is a different estimand and is kept
    only for the violation check.
    """
    rows, per_task = [], []
    for ds in sorted(df.dataset.unique()):
        ca = df[df.dataset != ds]
        te = df[df.dataset == ds]
        lam = select_lambda(ca, alpha, mode, weights=weights)
        e = evaluate(te, lam)
        e.update({"holdout": ds, "alpha": alpha, "mode": mode, "lambda": lam,
                  "violation": bool(np.isfinite(e["risk"]) and e["risk"] > alpha)})
        rows.append(e)
        t = te[["dataset", "family", "pilot_gain", "full_gain"]].copy()
        t["adopt"] = t.pilot_gain > lam
        per_task.append(t)
    pool = pd.concat(per_task, ignore_index=True)
    full, adopt = pool.full_gain.values, pool.adopt.values
    pooled = {
        "n": len(pool), "n_adopt": int(adopt.sum()), "adopt_rate": float(adopt.mean()),
        "risk": float((full[adopt] < 0).mean()) if adopt.any() else np.nan,
        "harm_incidence": float(((full < 0) & adopt).mean()),
        "mean_full_adopt": float(full[adopt].mean()) if adopt.any() else np.nan,
        "utility": float(full[adopt].sum() / len(pool)),
        "oracle": float(np.maximum(full, 0).mean()),
        "always_coverage": float(full.mean())}
    pooled["regret"] = pooled["oracle"] - pooled["utility"]
    return pd.DataFrame(rows), pooled


def cluster_boot_ucb(df, lam, alpha, n_boot=2000, seed=0, conf=1.0 - DELTA):
    """Dataset-resampling upper bound on the conditional risk at a fixed lambda."""
    rng = np.random.default_rng(seed)
    datasets = df.dataset.unique()
    boot = {ds: df[df.dataset == ds] for ds in datasets}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(datasets, size=len(datasets), replace=True)
        b = pd.concat([boot[d] for d in pick], ignore_index=True)
        e = evaluate(b, lam)
        if e["n_adopt"] >= 5:
            vals.append(e["risk"])
    return float(np.nanpercentile(vals, 100 * conf)) if vals else np.nan


def feasibility(df, alphas, mode="ucb", weights=None):
    """Largest adoption rate each nominal alpha can be certified for."""
    rows = []
    for a in alphas:
        lam = select_lambda(df, a, mode, weights=weights)
        e = evaluate(df, lam) if np.isfinite(lam) else {
            "n_adopt": 0, "adopt_rate": 0.0, "risk": np.nan, "utility": 0.0,
            "regret": float(np.maximum(df.full_gain.values, 0).mean()),
            "mean_full_adopt": np.nan}
        rows.append({"alpha": a, "lambda": lam, "adopt_rate": e["adopt_rate"],
                     "n_adopt": e["n_adopt"], "risk": e["risk"],
                     "utility": e["utility"], "regret": e["regret"]})
    return pd.DataFrame(rows)


def family_table(df, alpha_list=(0.15, 0.2)):
    rows = []
    for fam, sub in df.groupby("family"):
        row = {"family": fam, "dataset": sub.dataset.iloc[0],
               "target": sub.target.iloc[0], "n": len(sub),
               "mean_pilot": sub.pilot_gain.mean(),
               "mean_full": sub.full_gain.mean(),
               "sd_full": sub.full_gain.std(ddof=0),
               "neg_rate": float((sub.full_gain < 0).mean()),
               "pilot_pos_rate": float((sub.pilot_gain > 0).mean()),
               "oracle_adopt": bool(sub.full_gain.mean() > 0)}
        for a in alpha_list:
            lam = select_lambda(df, a, "ucb")
            adopt = float((sub.pilot_gain > lam).mean()) if np.isfinite(lam) else 0.0
            sel = sub[sub.pilot_gain > lam]
            row[f"adopt_frac_alpha{a}"] = adopt
            row[f"risk_alpha{a}"] = (float((sel.full_gain < 0).mean())
                                     if len(sel) else np.nan)
            row[f"mean_full_adopt_alpha{a}"] = (float(sel.full_gain.mean())
                                                if len(sel) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_full", ascending=False)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(OUT / "pilot_pairs.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    df = df.dropna(subset=["pilot_gain", "full_gain"]).reset_index(drop=True)
    print(f"tasks={len(df)} datasets={df.dataset.nunique()} families={df.family.nunique()}")
    print(f"always-coverage={df.full_gain.mean():.4f} "
          f"neg-rate={float((df.full_gain < 0).mean()):.3f} "
          f"spearman={df[['pilot_gain', 'full_gain']].corr(method='spearman').iloc[0, 1]:.3f}")

    # dataset-equal weights for the robustness variant
    w = 1.0 / df.groupby("dataset").size()
    weights = (w / w.sum()).reindex(df.dataset.values)
    weights.index = df.dataset.values

    # 1. risk-coverage curve (cluster bootstrap CI)
    curve = curve_table(df)
    curve.to_csv(OUT / "e1_curve.csv", index=False)
    print("wrote e1_curve.csv")

    # 2. feasibility frontier: which alpha can be certified
    feas = feasibility(df, np.round(np.arange(0.05, 0.51, 0.01), 2))
    feas.to_csv(OUT / "e1_feasibility.csv", index=False)
    ok = feas[np.isfinite(feas["lambda"])]
    print("certifiable alphas:", ok.alpha.min() if len(ok) else None)

    # 3. split conformal
    splits = []
    for a in (0.10, 0.15, 0.20, 0.30):
        for mode in ("ucb", "plugin"):
            splits.append(split_conformal(df, a, mode))
    splits = pd.concat(splits, ignore_index=True)
    splits.to_csv(OUT / "e1_split.csv", index=False)
    ssum = (splits.groupby(["alpha", "mode"])
            .agg(n_reps=("rep", "size"),
                 adopt_rate=("adopt_rate", "mean"),
                 risk_mean=("risk", "mean"),
                 risk_median=("risk", "median"),
                 utility=("utility", "mean"),
                 regret=("regret", "mean"),
                 base_utility=("base_utility", "mean"),
                 base_regret=("base_regret", "mean"),
                 cal_ucb=("cal_ucb", "mean"),
                 violation_rate=("violation", "mean"),
                 never_adopt=("lambda", lambda s: float(np.isinf(s).mean())))
            .reset_index())
    ssum.to_csv(OUT / "e1_split_summary.csv", index=False)
    print("\n=== split conformal ===")
    print(ssum.round(3).to_string(index=False))

    # 4. cross-conformal
    cvr, cvpool = [], []
    for a in (0.15, 0.20):
        for mode in ("ucb", "plugin"):
            f, p = cv_plus(df, a, mode)
            f["alpha"], f["mode"] = a, mode
            cvr.append(f)
            p.update({"alpha": a, "mode": mode})
            cvpool.append(p)
    pd.concat(cvr, ignore_index=True).to_csv(OUT / "e1_cvplus_folds.csv", index=False)
    cvp = pd.DataFrame(cvpool)
    cvp.to_csv(OUT / "e1_cvplus.csv", index=False)
    print("\n=== cross-conformal (pooled out-of-fold) ===")
    print(cvp.round(3).to_string(index=False))

    # 5. cluster conformal (leave-dataset-out)
    cl, clpool = [], []
    for a in (0.15, 0.20):
        for mode in ("ucb", "plugin"):
            r, p = cluster_lodo(df, a, mode)
            cl.append(r)
            p.update({"alpha": a, "mode": mode})
            clpool.append(p)
    cl = pd.concat(cl, ignore_index=True)
    cl.to_csv(OUT / "e1_cluster_lodo.csv", index=False)
    clp = pd.DataFrame(clpool)
    clp.to_csv(OUT / "e1_cluster.csv", index=False)
    clsum = (cl.groupby(["alpha", "mode"])
             .agg(n_datasets=("holdout", "size"),
                  adopt_rate=("adopt_rate", "mean"),
                  risk_mean=("risk", "mean"),
                  risk_max=("risk", "max"),
                  utility_datasetavg=("utility", "mean"),
                  regret_datasetavg=("regret", "mean"),
                  violation_rate=("violation", "mean"))
             .reset_index())
    clsum.to_csv(OUT / "e1_cluster_summary.csv", index=False)
    print("\n=== cluster conformal (leave-one-dataset-out, task-weighted pooled) ===")
    print(clp.round(3).to_string(index=False))
    print("\nper-dataset violation summary:")
    print(clsum.round(3).to_string(index=False))

    # 6. dataset-weighted variant at alpha=0.15/0.20
    wt = feasibility(df, (0.15, 0.20), mode="plugin", weights=weights)
    wt["variant"] = "dataset_weighted"
    wt.to_csv(OUT / "e1_weighted.csv", index=False)

    # 7. per-family decision table under the two certified rules
    fam = family_table(df, alpha_list=(0.15, 0.20))
    fam["decided_adopt_a15"] = fam["adopt_frac_alpha0.15"] > 0
    fam["decided_adopt_a20"] = fam["adopt_frac_alpha0.2"] > 0
    fam["decision_agrees"] = fam["decided_adopt_a20"] == fam["oracle_adopt"]
    fam.to_csv(OUT / "e1_family_table.csv", index=False)
    print("\n=== per-family (certified rule, alpha=0.20) ===")
    cols = ["family", "n", "mean_pilot", "mean_full", "neg_rate",
            "adopt_frac_alpha0.2", "risk_alpha0.2", "oracle_adopt"]
    print(fam[cols].round(3).to_string(index=False))

    summary = pd.DataFrame([{
        "n_tasks": len(df), "n_datasets": df.dataset.nunique(),
        "n_families": df.family.nunique(),
        "always_coverage": float(df.full_gain.mean()),
        "neg_rate": float((df.full_gain < 0).mean()),
        "oracle": float(np.maximum(df.full_gain, 0).mean()),
        "pilot_full_spearman": float(df[["pilot_gain", "full_gain"]]
                                     .corr(method="spearman").iloc[0, 1]),
        "min_cp_ucb": float(min(cp_upper(int(((df.full_gain < 0) &
                                              (df.pilot_gain > l)).sum()),
                                         int((df.pilot_gain > l).sum()))
                                for l in LAM_GRID
                                if (df.pilot_gain > l).sum() >= MIN_ADOPT))}])
    summary.to_csv(OUT / "e1_summary.csv", index=False)
    print("\nwrote e1_summary.csv")


if __name__ == "__main__":
    main()