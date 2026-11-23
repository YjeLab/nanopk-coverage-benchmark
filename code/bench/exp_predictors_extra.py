"""E11: additional predictors (RF, GP, MLP) vs kNN/Ridge/GBM at fixed supports."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.data import load  # noqa: E402
from bench.run import model_factory, select  # noqa: E402

K = 20
REPS = 3
CONFIGS = [
    ("D1", None, 20), ("D5", "particle_size", 10), ("D5", "EE", 10),
    ("D7", "pdi_std", 10), ("D9", None, 10), ("D10", None, 20),
]


def models(seed):
    return {
        "knn": model_factory("knn", seed),
        "ridge": model_factory("ridge", seed),
        "gbm": model_factory("gbm", seed),
        "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                    random_state=seed, n_jobs=-1),
        "gp": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            GaussianProcessRegressor(
                                kernel=RBF(1.0) + WhiteKernel(0.1),
                                normalize_y=True, alpha=1e-3, random_state=seed)),
        "mlp": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64,), max_iter=300,
                                          random_state=seed)),
    }


def main():
    rows = []
    for ds, target, ms in CONFIGS:
        r = load(ds, target=target) if target else load(ds)
        g = np.asarray(r["groups"])
        tasks = [np.where(g == u)[0] for u in pd.unique(g) if np.sum(g == u) >= ms]
        for idx in tasks:
            Xi, yi = r["X"][idx], r["y"][idx]
            n = len(idx)
            if n < K + 5:
                continue
            imp = SimpleImputer(strategy="median").fit(Xi)
            Z = StandardScaler().fit_transform(imp.transform(Xi))
            rng = np.random.default_rng(zlib.crc32(f"{ds}|{target}|{idx[0]}".encode()) % (2**31))
            for rep in range(REPS):
                for strat in ["random", "kmeans"]:
                    sel = select(strat, Z, K, rng)
                    q = np.setdiff1d(np.arange(n), sel)
                    for name, m in models(rep).items():
                        try:
                            m.fit(Xi[sel], yi[sel])
                            rho = spearmanr(yi[q], m.predict(Xi[q])).correlation
                        except Exception:
                            rho = np.nan
                        rows.append({"dataset": ds, "target": target or "main",
                                     "unit": int(idx[0]), "rep": rep, "strategy": strat,
                                     "model": name,
                                     "spearman": float(rho) if np.isfinite(rho) else np.nan})
        print(f"{ds} {target} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/bench/predictors_extra.csv", index=False)
    print(out.groupby(["dataset", "model"])["spearman"].mean().round(3).to_string())


if __name__ == "__main__":
    main()