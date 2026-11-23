"""E10: extra selection strategies (facility location, uncertainty) vs random/kmeans."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.data import load  # noqa: E402
from bench.run import select  # noqa: E402

K = 20
REPS = 5
CONFIGS = [
    ("D1", None, 20, "T1"),
    ("D5", "particle_size", 10, "PLGA_size"),
    ("D6", "encapsulation_efficiency_pct", 10, "LipoDB_EE"),
    ("D7", "pdi_std", 10, "LNPAtlas_PDI"),
    ("D9", None, 10, "MHT_SAR"),
    ("D10", None, 20, "LNPCD"),
]


def tasks_from(r, min_samples):
    g = np.asarray(r["groups"])
    return [np.where(g == u)[0] for u in pd.unique(g)
            if np.sum(g == u) >= min_samples]


def facility_location(Z, k):
    n = len(Z)
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    scale = np.median(D[D > 0]) + 1e-9
    S = np.exp(-D / scale)
    best = np.zeros(n)
    chosen = []
    for _ in range(k):
        gain = np.maximum(S - best[:, None], 0).sum(0)
        gain[chosen] = -1
        j = int(np.argmax(gain))
        chosen.append(j)
        best = np.maximum(best, S[:, j])
    return np.array(chosen)


def uncertainty(Z, y, k, rng):
    n = len(Z)
    pilot = rng.choice(n, size=min(5, n - 2), replace=False)
    preds = []
    for b in range(5):
        idx = rng.choice(pilot, size=len(pilot), replace=True)
        m = Ridge(alpha=1.0).fit(Z[idx], y[idx])
        preds.append(m.predict(Z))
    var = np.var(np.vstack(preds), axis=0)
    return np.argsort(-var)[:k]


def eval_config(ds, target, min_samples):
    r = load(ds, target=target) if target else load(ds)
    tasks = tasks_from(r, min_samples)
    rows = []
    for idx in tasks:
        Xi, yi = r["X"][idx], r["y"][idx]
        n = len(idx)
        if n < K + 5:
            continue
        imp = SimpleImputer(strategy="median").fit(Xi)
        sc = StandardScaler().fit(imp.transform(Xi))
        Z = sc.transform(imp.transform(Xi))
        rng = np.random.default_rng(zlib.crc32(f"{ds}|{target}|{idx[0]}".encode()) % (2**31))
        for rep in range(REPS):
            for strat in ["random", "kmeans", "facility", "uncertainty"]:
                if strat in ("facility",):
                    sel = facility_location(Z, K)
                elif strat == "uncertainty":
                    sel = uncertainty(Z, yi, K, rng)
                else:
                    sel = select(strat, Z, K, rng)
                if sel is None or len(sel) < 5:
                    sel = rng.choice(n, size=K, replace=False)
                q = np.setdiff1d(np.arange(n), sel)
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  KNeighborsRegressor(n_neighbors=min(5, len(sel)),
                                                      weights="distance"))
                m.fit(Xi[sel], yi[sel])
                rho = spearmanr(yi[q], m.predict(Xi[q])).correlation
                rows.append({"dataset": ds, "target": target or "main", "k": K,
                             "rep": rep, "strategy": strat,
                             "spearman": float(rho) if np.isfinite(rho) else 0.0})
        print(f"  {ds} task done", flush=True)
    return rows


def main():
    rows = []
    for ds, target, ms, name in CONFIGS:
        print(f"== {name} ==", flush=True)
        rows.extend(eval_config(ds, target, ms))
    r = pd.DataFrame(rows)
    r.to_csv("results/bench/selection_extra.csv", index=False)
    print(r.groupby(["dataset", "strategy"])["spearman"].mean().round(3).to_string())


if __name__ == "__main__":
    main()