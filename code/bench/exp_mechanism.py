"""E21: mechanism experiment on synthetic tasks.

The discussion claimed coverage selection helps when the label varies along directions
encoded by the features. Here we manipulate that alignment directly and measure the
coverage gain, so the mechanism is tested rather than narrated.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import OUT, prep  # noqa: E402
from bench.run import model_factory, select  # noqa: E402

REPS = 20
SEEDS = 4
K = 10


def generate(n, d, c, sep, align, noise, rng):
    centers = rng.normal(0, sep, size=(c, d))
    lab = rng.integers(0, c, n)
    X = centers[lab] + rng.normal(0, 1, size=(n, d))
    w = rng.normal(0, 1, d); w /= np.linalg.norm(w)
    f = X @ w
    f = (f - f.mean()) / (f.std() + 1e-9)
    g = rng.normal(0, 1, n); g = (g - g.mean()) / (g.std() + 1e-9)
    y = align * f + np.sqrt(max(0.0, 1 - align ** 2)) * g + rng.normal(0, noise, n)
    return X, y


def gain_for(params, seed):
    X, y = generate(**params, rng=np.random.default_rng(seed))
    Z = prep(X)
    n = len(y)
    rng = np.random.default_rng(seed + 1)
    gains, rr, rk = [], [], []
    for _ in range(REPS):
        sr = select("random", Z, K, rng)
        sk = select("kmeans", Z, K, rng)
        qr = np.setdiff1d(np.arange(n), sr)
        qk = np.setdiff1d(np.arange(n), sk)
        for mname in ("knn", "ridge"):
            m = model_factory(mname, 0)
            m.fit(X[sr], y[sr]); a = spearmanr(y[qr], m.predict(X[qr])).correlation
            m = model_factory(mname, 0)
            m.fit(X[sk], y[sk]); b = spearmanr(y[qk], m.predict(X[qk])).correlation
            a = 0.0 if not np.isfinite(a) else a
            b = 0.0 if not np.isfinite(b) else b
            gains.append(b - a); rr.append(a); rk.append(b)
    return float(np.mean(gains)), float(np.mean(rr)), float(np.mean(rk))


BASE = dict(n=300, d=20, c=8, sep=3.0, align=0.8, noise=0.3)
SWEEPS = {
    "alignment": ("align", [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]),
    "noise": ("noise", [0.0, 0.2, 0.4, 0.8, 1.2]),
    "n": ("n", [100, 200, 400, 800]),
    "clusters": ("c", [1, 2, 4, 8, 16]),
    "separation": ("sep", [0.5, 1.0, 2.0, 4.0]),
}


def main():
    rows, seed_rows = [], []
    for i, (name, (key, vals)) in enumerate(SWEEPS.items()):
        for j, v in enumerate(vals):
            p = dict(BASE); p[key] = v
            gs = [gain_for(p, seed=1000 * i + j + s * 10007) for s in range(SEEDS)]
            for s, (gg, rr_, rk_) in enumerate(gs):
                seed_rows.append({"sweep": name, "value": v, "seed": s,
                                  "gain": gg, "random_rho": rr_, "kmeans_rho": rk_})
            g = float(np.mean([x[0] for x in gs]))
            rr = float(np.mean([x[1] for x in gs]))
            rk = float(np.mean([x[2] for x in gs]))
            rows.append({**{f"param_{k}": p[k] for k in BASE},
                         "sweep": name, "value": v,
                         "gain": g, "random_rho": rr, "kmeans_rho": rk,
                         "gain_sd": float(np.std([x[0] for x in gs], ddof=0))})
            print(f"{name}={v}: gain={g:+.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "mechanism_sweeps.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "mechanism_seeds.csv", index=False)

    # 2-D alignment x noise grid for a heatmap
    grid = []
    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for nz in [0.0, 0.3, 0.6, 1.0]:
            p = dict(BASE); p.update(align=a, noise=nz)
            g, rr, rk = gain_for(p, seed=5000 + int(a * 10) * 7 + int(nz * 10))
            grid.append({"align": a, "noise": nz, "gain": g})
            print(f"grid align={a} noise={nz}: gain={g:+.3f}", flush=True)
    pd.DataFrame(grid).to_csv(OUT / "mechanism_grid.csv", index=False)


if __name__ == "__main__":
    main()