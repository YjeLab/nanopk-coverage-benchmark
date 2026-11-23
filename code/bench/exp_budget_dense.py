"""Dense budget grid for Fig 2b: more K thresholds so the budget panel matches the
endpoint-family panel in information content.

k in {5,10,15,20,25,30,40,50}; random vs k-means; kNN + Ridge; 5 supports per fold.
Writes results/bench_v2/budget_dense.csv incrementally.
"""
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import GRID_CONFIGS, OUT, build, prep  # noqa: E402
from bench.run import model_factory, select  # noqa: E402

KS = [5, 10, 15, 20, 25, 30, 40, 50]
REPS = 5
MODELS = ("knn", "ridge")
STRATS = ("random", "kmeans")
MIN_Q = 10


def rho(y, p):
    v = spearmanr(y, p).correlation
    return float(v) if np.isfinite(v) else 0.0


def main():
    out = OUT / "budget_dense.csv"
    rows, t0 = [], time.time()
    for ds, target, ms, inc in GRID_CONFIGS:
        if ds in ("D4",):
            continue
        print(f"== {ds} {target or ''} ==", flush=True)
        r, tasks = build(ds, target, ms, inc)
        X, y = r["X"], r["y"]
        for ti, task in enumerate(tasks, 1):
            idx = task["idx"]; n = len(idx)
            Xi, yi = X[idx], y[idx]
            Z = prep(Xi)
            rng = np.random.default_rng(
                zlib.crc32(f"{ds}|{target}|{task['group']}".encode()) % (2 ** 31))
            for k in KS:
                if k > n - MIN_Q:
                    continue
                for rep in range(REPS):
                    for strat in STRATS:
                        sel = select(strat, Z, k, rng)
                        if sel is None or len(sel) < 5 or len(sel) >= n - 4:
                            continue
                        q = np.setdiff1d(np.arange(n), sel)
                        for mname in MODELS:
                            m = model_factory(mname, rep)
                            m.fit(Xi[sel], yi[sel])
                            rows.append({"dataset": ds, "target": target or "main",
                                         "group": task["group"], "n": n, "k": k,
                                         "rep": rep, "strategy": strat,
                                         "model": mname, "spearman": rho(yi[q], m.predict(Xi[q]))})
            if ti % 10 == 0:
                print(f"  {ti}/{len(tasks)} tasks {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()