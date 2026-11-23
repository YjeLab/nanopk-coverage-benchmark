"""E11 supplement: TabICL vs kNN at fixed supports (GPU)."""
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
REPS = 3
CONFIGS = [("D1", None, 20), ("D5", "particle_size", 10), ("D7", "pdi_std", 10)]


def main():
    from tabicl import TabICLRegressor
    rows = []
    for ds, target, ms in CONFIGS:
        r = load(ds, target=target) if target else load(ds)
        g = np.asarray(r["groups"])
        tasks = [np.where(g == u)[0] for u in pd.unique(g) if np.sum(g == u) >= ms]
        tasks = [t for t in tasks if len(t) >= K + 10]
        if not tasks:
            tasks = [np.arange(len(r["y"]))]
        for idx in tasks:
            Xi, yi = r["X"][idx], r["y"][idx]
            n = len(idx)
            imp = SimpleImputer(strategy="median").fit(Xi)
            Z = StandardScaler().fit_transform(imp.transform(Xi))
            rng = np.random.default_rng(zlib.crc32(f"{ds}|{target}|{idx[0]}".encode()) % (2**31))
            for rep in range(REPS):
                for strat in ["random", "kmeans"]:
                    sel = select(strat, Z, K, rng)
                    q = np.setdiff1d(np.arange(n), sel)
                    Xa, Xq = Xi[sel], Xi[q]
                    for name in ["tabicl", "knn"]:
                        try:
                            if name == "tabicl":
                                m = make_pipeline(SimpleImputer(strategy="median"),
                                                  TabICLRegressor())
                            else:
                                m = make_pipeline(SimpleImputer(strategy="median"),
                                                  StandardScaler(),
                                                  KNeighborsRegressor(n_neighbors=5,
                                                                      weights="distance"))
                            m.fit(Xa, yi[sel])
                            rho = spearmanr(yi[q], m.predict(Xq)).correlation
                        except Exception as e:
                            print("fail", name, e, flush=True)
                            rho = np.nan
                        rows.append({"dataset": ds, "target": target or "main",
                                     "unit": int(idx[0]), "rep": rep,
                                     "strategy": strat, "model": name,
                                     "spearman": float(rho) if np.isfinite(rho) else np.nan})
        print(f"{ds} {target} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/bench/tabicl_bench.csv", index=False)
    print(out.groupby(["dataset", "model"])["spearman"].mean().round(3).to_string())


if __name__ == "__main__":
    main()