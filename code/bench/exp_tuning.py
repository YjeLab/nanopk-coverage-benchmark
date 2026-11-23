"""E15: does hyper-parameter tuning change the 'no consistent winner' conclusion?

Predictors are first compared at fixed hyper-parameters (kNN k=5, ridge alpha=1,
fixed GBM). Here every predictor is tuned by inner 3-fold CV on the support set, and we
compare tuned vs fixed rankings and coverage gains.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import OUT, build, prep  # noqa: E402
from bench.run import model_factory, select  # noqa: E402

K = 20
REPS = 3
CONFIGS = [
    ("D1", None, 20, False),
    ("D5", "particle_size", 10, True),
    ("D5", "EE", 10, True),
    ("D5", "LC", 10, True),
    ("D7", "pdi_std", 10, True),
    ("D9", None, 10, True),
    ("D10", None, 20, True),
]
GRIDS = {
    "knn": [{"n_neighbors": v} for v in (3, 5, 10)],
    "ridge": [{"alpha": v} for v in (0.1, 1.0, 10.0)],
    "gbm": [{"max_depth": a, "learning_rate": b}
            for a, b in ((2, 0.03), (2, 0.06), (3, 0.06), (3, 0.1))],
}


def build_model(name, params):
    pre = [SimpleImputer(strategy="median")]
    if name != "gbm":
        pre.append(StandardScaler())
    if name == "knn":
        est = KNeighborsRegressor(weights="distance", **params)
    elif name == "ridge":
        est = Ridge(**params)
    else:
        est = HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=3,
                                            random_state=0, **params)
    return make_pipeline(*pre, est)


def tune(Xs, ys, name, rng):
    if len(ys) < 6:
        return GRIDS[name][0]
    kf = KFold(3, shuffle=True, random_state=int(rng.integers(1e6)))
    best, best_score = GRIDS[name][0], -np.inf
    for p in GRIDS[name]:
        scores = []
        for tr, va in kf.split(Xs):
            pp = dict(p)
            if name == "knn":
                pp["n_neighbors"] = max(1, min(pp["n_neighbors"], len(tr) - 1))
            try:
                m = build_model(name, pp)
                m.fit(Xs[tr], ys[tr])
                v = spearmanr(ys[va], m.predict(Xs[va])).correlation
                scores.append(v if np.isfinite(v) else 0.0)
            except Exception:
                scores.append(-1.0)
        s = float(np.mean(scores))
        if s > best_score:
            best, best_score = dict(p), s
    return best


def main():
    rows = []
    for ds, target, ms, inc in CONFIGS:
        print(f"== {ds} {target or ''} ==", flush=True)
        r, tasks = build(ds, target, ms, inc)
        X, y = r["X"], r["y"]
        for task in tasks:
            idx = task["idx"]; n = len(idx)
            if n < K + 10:
                continue
            Xi, yi = X[idx], y[idx]
            Z = prep(Xi)
            rng = np.random.default_rng(abs(hash((ds, target or "", task["group"]))) % (2 ** 31))
            for rep in range(REPS):
                for strat in ("random", "kmeans"):
                    sel = select(strat, Z, K, rng)
                    if sel is None or len(sel) < 6:
                        continue
                    q = np.setdiff1d(np.arange(n), sel)
                    for name in ("knn", "ridge", "gbm"):
                        fp = dict(GRIDS[name][0])
                        if name == "knn":
                            fp["n_neighbors"] = max(1, min(fp["n_neighbors"], len(sel) - 1))
                        fixed = build_model(name, fp)
                        fixed.fit(Xi[sel], yi[sel])
                        rf = spearmanr(yi[q], fixed.predict(Xi[q])).correlation
                        params = tune(Xi[sel], yi[sel], name, rng)
                        if name == "knn":
                            params = dict(params)
                            params["n_neighbors"] = max(1, min(params["n_neighbors"], len(sel) - 1))
                        tuned = build_model(name, params)
                        tuned.fit(Xi[sel], yi[sel])
                        rt = spearmanr(yi[q], tuned.predict(Xi[q])).correlation
                        rows.append({"dataset": ds, "target": target or "main",
                                     "group": task["group"], "rep": rep,
                                     "strategy": strat, "model": name,
                                     "fixed_rho": float(rf) if np.isfinite(rf) else 0.0,
                                     "tuned_rho": float(rt) if np.isfinite(rt) else 0.0,
                                     "params": str(params)})
            print("  task done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tuning.csv", index=False)
    s = d.groupby(["dataset", "target", "model"])[["fixed_rho", "tuned_rho"]].mean()
    s["delta"] = s.tuned_rho - s.fixed_rho
    print(s.round(3).to_string())
    s.to_csv(OUT / "tuning_summary.csv")


if __name__ == "__main__":
    main()