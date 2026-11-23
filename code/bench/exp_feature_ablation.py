"""E8 feature ablation: does the coverage gain depend on the feature space?"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.data import load_d1, load_d2, load_d3, load_d6, load_d7  # noqa: E402
from bench.run import model_factory, select  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

K = 20
REPS = 5


def tasks_from(r, min_samples):
    groups = np.asarray(r["groups"])
    out = []
    for u in pd.unique(groups):
        idx = np.where(groups == u)[0]
        if len(idx) >= min_samples:
            out.append(idx)
    return out


def eval_config(name, setting, r, min_samples):
    tasks = [t for t in tasks_from(r, min_samples) if len(t) >= K + 5]
    if not tasks:
        tasks = [np.arange(len(r["y"]))]
    gains = []
    for idx in tasks:
        Xi, yi = r["X"][idx], r["y"][idx]
        n = len(idx)
        if n < K + 5:
            continue
        imp = SimpleImputer(strategy="median").fit(Xi)
        Z = StandardScaler().fit_transform(imp.transform(Xi))
        rng = np.random.default_rng(zlib.crc32(f"{name}|{setting}|{idx[0]}".encode()) % (2**31))
        for rep in range(REPS):
            rhos = {}
            for strat in ["random", "kmeans"]:
                sel = select(strat, Z, K, rng)
                if sel is None or len(sel) < 5:
                    sel = rng.choice(n, size=K, replace=False)
                q = np.setdiff1d(np.arange(n), sel)
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  KNeighborsRegressor(n_neighbors=min(5, len(sel)),
                                                      weights="distance"))
                m.fit(Xi[sel], yi[sel])
                rho = spearmanr(yi[q], m.predict(Xi[q])).correlation
                rhos[strat] = rho if np.isfinite(rho) else 0.0
            gains.append(rhos["kmeans"] - rhos["random"])
    return {"dataset": name, "feature_setting": setting, "tasks": len(tasks),
            "gain_mean": float(np.mean(gains)) if gains else np.nan,
            "positive": float(np.mean(np.array(gains) > 0)) if gains else np.nan}


def main():
    rows = []
    # D2/D3: fingerprint bits
    for name, loader, ms in [("D2", load_d2, 60), ("D3", load_d3, 60)]:
        for bits in [128, 256, 512, 1024]:
            r = loader(n_bits=bits)
            rows.append(eval_config(name, f"fp{bits}", r, ms))
            print(rows[-1], flush=True)
    # D7: molar ratios
    for t in ["pdi_std", "particle_size_nm_std"]:
        for inc in [True, False]:
            r = load_d7(target=t, include_ratios=inc)
            rows.append(eval_config("D7", f"{t}_ratios={inc}", r, 10))
            print(rows[-1], flush=True)
    # D6: categorical cap
    for t in ["encapsulation_efficiency_pct", "particle_size_nm"]:
        for cap in [5, 10, 20]:
            r = load_d6(target=t, max_cat=cap)
            rows.append(eval_config("D6", f"{t}_catcap={cap}", r, 10))
            print(rows[-1], flush=True)
    # D1: feature groups
    for g in ["drug", "form", "phys", "drug+form", "all"]:
        r = load_d1(group=g)
        rows.append(eval_config("D1", f"group={g}", r, 20))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/bench/ablation_features.csv", index=False)
    print("\n" + out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()