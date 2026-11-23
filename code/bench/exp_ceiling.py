"""E1: within-task 5-fold ceiling (kNN / Ridge)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.data import load  # noqa: E402

CONFIGS = [("D1", None, 20), ("D2", None, 60), ("D3", None, 60),
           ("D5", "particle_size", 10), ("D5", "EE", 10),
           ("D6", "encapsulation_efficiency_pct", 10),
           ("D7", "pdi_std", 10), ("D9", None, 10), ("D10", None, 20)]


def main():
    rows = []
    for ds, target, ms in CONFIGS:
        r = load(ds, target=target) if target else load(ds)
        g = np.asarray(r["groups"])
        tasks = [np.where(g == u)[0] for u in pd.unique(g) if np.sum(g == u) >= ms]
        tasks = [t for t in tasks if len(t) >= 30]
        if not tasks:
            tasks = [np.arange(len(r["y"]))]
        for idx in tasks:
            Xi, yi = r["X"][idx], r["y"][idx]
            if len(idx) < 30:
                continue
            imp = SimpleImputer(strategy="median").fit(Xi)
            X = StandardScaler().fit_transform(imp.transform(Xi))
            for name in ["knn", "ridge", "gbm"]:
                pred = np.zeros(len(yi))
                for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                    if name == "knn":
                        m = KNeighborsRegressor(n_neighbors=5, weights="distance")
                    elif name == "ridge":
                        m = Ridge(alpha=1.0)
                    else:
                        m = HistGradientBoostingRegressor(
                            max_iter=200, max_depth=3, learning_rate=0.06,
                            min_samples_leaf=3, random_state=0)
                    m.fit(X[tr], yi[tr])
                    pred[te] = m.predict(X[te])
                rho = spearmanr(yi, pred).correlation
                rows.append({"dataset": ds, "target": target or "main",
                             "unit": int(idx[0]), "n": len(idx), "model": name,
                             "ceiling": float(rho) if np.isfinite(rho) else np.nan})
        print(f"{ds} {target} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/bench/ceiling.csv", index=False)
    print(out.groupby(["dataset", "model"])["ceiling"].mean().round(3).to_string())


if __name__ == "__main__":
    main()