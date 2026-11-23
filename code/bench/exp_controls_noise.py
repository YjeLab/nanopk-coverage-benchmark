"""E12: label-shuffle and label-noise controls for the coverage gain."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.data import load  # noqa: E402
from bench.run import select  # noqa: E402

K = 20
REPS = 5
CONFIGS = [("D1", None, 20), ("D5", "particle_size", 10)]


def tasks_from(r, ms):
    g = np.asarray(r["groups"])
    return [np.where(g == u)[0] for u in pd.unique(g) if np.sum(g == u) >= ms]


def eval_task(Xi, yi, rng):
    n = len(yi)
    imp = SimpleImputer(strategy="median").fit(Xi)
    sc = StandardScaler().fit(imp.transform(Xi))
    Z = sc.transform(imp.transform(Xi))
    out = {}
    for rep in range(REPS):
        sel = {
            "random": select("random", Z, K, rng),
            "kmeans": select("kmeans", Z, K, rng),
        }
        for strat, a in sel.items():
            q = np.setdiff1d(np.arange(n), a)
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              KNeighborsRegressor(n_neighbors=min(5, len(a)),
                                                  weights="distance"))
            m.fit(Xi[a], yi[a])
            rho = spearmanr(yi[q], m.predict(Xi[q])).correlation
            out.setdefault(strat, []).append(rho if np.isfinite(rho) else 0.0)
    return {k: np.mean(v) for k, v in out.items()}


def main():
    rows = []
    for ds, target, ms in CONFIGS:
        r = load(ds, target=target) if target else load(ds)
        tasks = tasks_from(r, ms)
        for idx in tasks:
            Xi, yi = r["X"][idx], r["y"][idx]
            n = len(idx)
            if n < K + 5:
                continue
            rng = np.random.default_rng(zlib.crc32(f"{ds}|{target}|{idx[0]}".encode()) % (2**31))
            settings = {"original": yi}
            settings["shuffle"] = rng.permutation(yi)
            sd = yi.std() if yi.std() > 1e-9 else 1.0
            settings["noise_0.5"] = yi + 0.5 * sd * rng.normal(size=n)
            settings["noise_1.0"] = yi + 1.0 * sd * rng.normal(size=n)
            for name, yv in settings.items():
                res = eval_task(Xi, yv, rng)
                rows.append({"dataset": ds, "target": target or "main",
                             "setting": name, "random": res["random"],
                             "kmeans": res["kmeans"],
                             "gain": res["kmeans"] - res["random"]})
        print(f"{ds} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/bench/controls_noise.csv", index=False)
    print(out.groupby(["dataset", "setting"])[["random", "kmeans", "gain"]]
             .mean().round(3).to_string())


if __name__ == "__main__":
    main()