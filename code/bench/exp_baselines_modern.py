"""E16: strong, non-strawman selection baselines at a fixed budget.

Adds quality-diversity (greedy DPP) and stratified samplers to the original
random / k-means / k-center set, so the conclusion is not an artefact of weak baselines.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import OUT, build, prep  # noqa: E402
from bench.run import model_factory, select  # noqa: E402

K = 20
REPS = 10
CONFIGS = [
    ("D1", None, 20, False),
    ("D5", "particle_size", 10, True),
    ("D5", "EE", 10, True),
    ("D5", "LC", 10, True),
    ("D6", "particle_size_nm", 10, True),
    ("D7", "pdi_std", 10, True),
    ("D9", None, 10, True),
    ("D10", None, 20, True),
]


def dpp_select(Z, k):
    n = len(Z)
    L = (Z @ Z.T) / Z.shape[1]
    chosen = []
    for _ in range(min(k, n)):
        if not chosen:
            gain = np.diag(L).copy()
        else:
            Ls = L[:, chosen]
            sol = np.linalg.solve(L[np.ix_(chosen, chosen)] + 1e-9 * np.eye(len(chosen)),
                                  Ls.T)
            gain = np.diag(L) - np.einsum("kn,kn->n", Ls.T, sol)
        gain[chosen] = -np.inf
        chosen.append(int(np.argmax(gain)))
    return np.array(chosen)


def stratified_random(Z, k, rng):
    n = len(Z)
    lab = KMeans(n_clusters=min(k, n), n_init=10,
                 random_state=int(rng.integers(1e6))).fit_predict(Z)
    out = [int(rng.choice(np.where(lab == c)[0])) for c in np.unique(lab)]
    if len(out) < k:
        rest = np.setdiff1d(np.arange(n), out)
        out += list(rng.choice(rest, size=k - len(out), replace=False))
    return np.array(out[:k])


def pc_stratified(Z, k, rng):
    n = len(Z)
    pc = Z[:, 0]
    order = np.argsort(pc)
    bins = np.array_split(order, k)
    return np.array([int(rng.choice(b)) for b in bins if len(b)])


STRATS = ["random", "kmeans", "kcenter", "dpp", "strat_cluster", "strat_pc"]


def choose(strat, Z, rng):
    if strat == "dpp":
        return dpp_select(Z, K)
    if strat == "strat_cluster":
        return stratified_random(Z, K, rng)
    if strat == "strat_pc":
        return pc_stratified(Z, K, rng)
    return select(strat, Z, K, rng)


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
                for strat in STRATS:
                    sel = choose(strat, Z, rng)
                    if len(sel) < 5:
                        continue
                    q = np.setdiff1d(np.arange(n), sel)
                    for mname in ("knn", "ridge"):
                        m = model_factory(mname, rep)
                        m.fit(Xi[sel], yi[sel])
                        v = spearmanr(yi[q], m.predict(Xi[q])).correlation
                        rows.append({"dataset": ds, "target": target or "main",
                                     "group": task["group"], "n": n, "rep": rep,
                                     "strategy": strat, "model": mname,
                                     "spearman": float(v) if np.isfinite(v) else 0.0})
            print(f"  task done n={n}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "baselines_modern.csv", index=False)
    piv = d.pivot_table(index=["dataset", "target"],
                        columns="strategy", values="spearman", aggfunc="mean")
    print(piv.round(3).to_string())
    piv.to_csv(OUT / "baselines_modern_summary.csv")


if __name__ == "__main__":
    main()