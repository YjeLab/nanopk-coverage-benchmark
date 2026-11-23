"""E18: can cheap task meta-features predict where coverage selection helps?

Two ad-hoc diagnostics account for little variance (R^2 = 0.07). This script replaces
them with a battery of structure-aware meta-features computed from (X, y) alone -- label
smoothness in feature space, cluster-label alignment, intrinsic dimension, label
entropy -- and test predictability with leave-one-dataset-out validation.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew, spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import GRID_CONFIGS, OUT, build, prep  # noqa: E402
from bench.exp_common import task_gain_table  # noqa: E402

rng = np.random.default_rng(0)


def meta_features(Z, y):
    n, d = Z.shape
    out = {"n": n, "d": d, "log_n": np.log(n)}
    # intrinsic dimension: PCs for 90% variance
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    v = s ** 2
    v = v / v.sum()
    out["pca90_frac"] = float(np.searchsorted(np.cumsum(v), 0.90) + 1) / d
    p = v / v.sum()
    out["eff_rank_frac"] = float(np.exp(-(p * np.log(p + 1e-12)).sum())) / d
    # label smoothness: Spearman(y, mean of 5 nearest-neighbour labels)
    k = min(6, n - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, ind = nn.kneighbors(Z)
    neigh = y[ind[:, 1:]].mean(1) if k > 1 else y
    out["knn_consistency"] = spearmanr(y, neigh).correlation if n > 5 else 0.0
    # cluster-label alignment eta^2
    c = int(max(2, min(20, round(np.sqrt(n)))))
    c = min(c, n)
    lab = KMeans(n_clusters=c, n_init=5, random_state=0).fit_predict(Z)
    tot = ((y - y.mean()) ** 2).sum()
    within = sum(((y[lab == j] - y[lab == j].mean()) ** 2).sum()
                 for j in range(c) if (lab == j).any())
    out["eta2"] = float(1 - within / tot) if tot > 0 else 0.0
    # feature-space separability
    if 2 <= c < n:
        sub = np.arange(n) if n <= 600 else rng.choice(n, 600, replace=False)
        if len(np.unique(lab[sub])) > 1:
            out["silhouette"] = float(silhouette_score(Z[sub], lab[sub]))
        else:
            out["silhouette"] = 0.0
    else:
        out["silhouette"] = 0.0
    # label distribution
    out["std_y"] = float(np.std(y))
    out["skew_y"] = float(skew(y)) if n > 5 else 0.0
    out["kurt_y"] = float(kurtosis(y)) if n > 5 else 0.0
    cnt, _ = np.histogram(y, bins=min(10, max(2, n // 5)))
    out["label_entropy"] = float(entropy(cnt + 1e-9) / np.log(len(cnt)))
    out["unique_frac"] = len(np.unique(y)) / n
    # feature-label coupling
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    ys = (y - y.mean()) / (y.std() + 1e-9)
    cor = np.abs((Zs * ys[:, None]).mean(0))
    out["max_abs_corr"] = float(cor.max())
    out["mean_abs_corr"] = float(cor.mean())
    # cheap learnability: 3-fold ridge CV R^2
    if n >= 12:
        pred = cross_val_predict(Ridge(alpha=10.0), Z, y, cv=3)
        out["ridge_cv_r2"] = float(r2_score(y, pred))
    else:
        out["ridge_cv_r2"] = 0.0
    return out


def main():
    rows, t0 = [], time.time()
    for ds, target, ms, inc in GRID_CONFIGS:
        print(f"== {ds} {target or ''} ==", flush=True)
        r, tasks = build(ds, target, ms, inc)
        X, y = r["X"], r["y"]
        for task in tasks:
            idx = task["idx"]
            if len(idx) < 12:
                continue
            Xi, yi = X[idx], y[idx]
            mask = ~np.all(np.isnan(Xi), axis=0)
            Xi = Xi[:, mask]
            Z = prep(Xi)
            f = meta_features(Z, yi)
            f.update({"dataset": ds, "task": r["task"], "group": task["group"],
                      "family": f"{ds} {target or 'main'}"})
            rows.append(f)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "task_meta.csv", index=False)
    print(f"meta features: {len(d)} tasks, {time.time()-t0:.0f}s")

    g = task_gain_table()
    m = d.merge(g[["dataset", "task", "group", "gain", "random", "n"]],
                on=["dataset", "task", "group"], how="inner", suffixes=("", "_gain"))
    m.to_csv(OUT / "task_meta_merged.csv", index=False)
    print(f"merged tasks: {len(m)}")

    feats = [c for c in d.columns
             if c not in ("dataset", "task", "group", "family")]
    print("\n=== univariate Spearman with gain ===")
    uni = []
    for c in feats:
        q = m[[c, "gain"]].dropna()
        if len(q) > 5:
            rho = spearmanr(q[c], q.gain).correlation
            uni.append({"feature": c, "spearman": rho, "n": len(q)})
    uni = pd.DataFrame(uni).sort_values("spearman")
    uni.to_csv(OUT / "meta_univariate.csv", index=False)
    print(uni.round(3).to_string(index=False))

    # leave-one-dataset-out
    Xf = m[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    yv = m.gain.values
    pred = np.full(len(m), np.nan)
    for ds in m.dataset.unique():
        te = (m.dataset == ds).values
        if te.sum() == 0 or (~te).sum() < 10:
            continue
        model = HistGradientBoostingRegressor(max_iter=200, max_depth=2,
                                              learning_rate=0.05, random_state=0)
        model.fit(Xf[~te], yv[~te])
        pred[te] = model.predict(Xf[te])
    ok = np.isfinite(pred)
    loo = pd.DataFrame({"dataset": m.dataset[ok], "family": m.family[ok],
                        "gain": yv[ok], "pred": pred[ok],
                        "group": m.group[ok]})
    loo.to_csv(OUT / "meta_loo.csv", index=False)
    adopt = loo.pred.values > 0
    util = np.where(adopt, loo.gain.values, 0.0)
    summ = pd.DataFrame([{
        "n": int(ok.sum()), "r2": r2_score(yv[ok], pred[ok]),
        "spearman": spearmanr(yv[ok], pred[ok]).correlation,
        "decision_acc": float((adopt == (loo.gain.values > 0)).mean()),
        "policy_utility": util.mean(), "always_coverage": loo.gain.mean(),
        "oracle": np.maximum(loo.gain.values, 0).mean(),
    }])
    summ.to_csv(OUT / "meta_summary.csv", index=False)
    print("\n=== leave-one-dataset-out ===")
    print(summ.round(3).to_string(index=False))

    # permutation importance on the full data
    model = HistGradientBoostingRegressor(max_iter=200, max_depth=2,
                                          learning_rate=0.05, random_state=0).fit(Xf, yv)
    base = model.score(Xf, yv)
    imp = []
    for j, c in enumerate(feats):
        Xp = Xf.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append({"feature": c, "drop": base - model.score(Xp, yv)})
    pd.DataFrame(imp).sort_values("drop", ascending=False).to_csv(
        OUT / "meta_importance.csv", index=False)


if __name__ == "__main__":
    main()