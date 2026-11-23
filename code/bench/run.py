"""Grid runner: selection x predictor x budget over benchmark tasks."""
import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.tasks import build_tasks  # noqa: E402


def metrics(y_true, pred):
    rho = spearmanr(y_true, pred).correlation
    n = len(y_true)
    nq = max(1, int(round(0.1 * n)))
    true_top = set(np.argsort(-y_true)[:nq].tolist())
    pred_q = np.argsort(-pred)[:nq].tolist()
    recall = len(true_top & set(pred_q)) / nq
    mm = min(10, n)
    pred_m = np.argsort(-pred)[:mm]
    prec = float(np.mean([1.0 if i in true_top else 0.0 for i in pred_m]))
    rank_of = np.argsort(np.argsort(-y_true))
    regret = float(np.mean(rank_of[pred_m]) / max(1, n - 1))
    return (float(rho) if np.isfinite(rho) else 0.0), recall, prec, regret


def select(strategy, Z, k, rng):
    n = len(Z)
    if strategy == "random":
        return rng.choice(n, size=k, replace=False)
    if strategy == "kmeans":
        km = KMeans(n_clusters=k, n_init=20, random_state=int(rng.integers(1e6)))
        lab = km.fit_predict(Z)
        out = []
        for c in range(k):
            idx = np.where(lab == c)[0]
            if len(idx):
                out.append(idx[np.argmin(np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1))])
        return np.array(out)
    if strategy == "kcenter":
        chosen = [int(rng.integers(n))]
        d = np.linalg.norm(Z - Z[chosen[0]], axis=1)
        for _ in range(k - 1):
            nxt = int(np.argmax(d)); chosen.append(nxt)
            d = np.minimum(d, np.linalg.norm(Z - Z[nxt], axis=1))
        return np.array(chosen)
    if strategy == "leverage":
        G = Z.T @ Z + 1e-6 * np.eye(Z.shape[1])
        h = np.einsum("ij,jk,ik->i", Z, np.linalg.pinv(G), Z)
        return np.argsort(-h)[:k]
    raise ValueError(strategy)


def model_factory(name, seed=0):
    if name == "knn":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             KNeighborsRegressor(n_neighbors=5, weights="distance"))
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=1.0))
    if name == "gbm":
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingRegressor(
                                 max_iter=200, max_depth=3, learning_rate=0.06,
                                 min_samples_leaf=3, random_state=seed))
    if name == "tabicl":
        from tabicl import TabICLRegressor
        return make_pipeline(SimpleImputer(strategy="median"), TabICLRegressor())
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--min-queries", type=int, default=5)
    ap.add_argument("--include-global", action="store_true")
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "kmeans", "kcenter", "leverage", "cross_context"])
    ap.add_argument("--models", nargs="+", default=["knn", "ridge", "gbm"])
    ap.add_argument("--out", default="results/bench")
    args = ap.parse_args()

    r, tasks = build_tasks(args.dataset, args.target, args.min_samples, args.include_global)
    X, y = r["X"], r["y"]
    print(f"[{args.dataset}] rows={len(y)} tasks={len(tasks)} X={X.shape[1]} "
          f"target={r['target']}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{args.dataset}_{args.target or 'main'}.csv"

    rows = []
    t0 = time.time()
    for ti, task in enumerate(tasks, 1):
        idx = task["idx"]
        Xi, yi = X[idx], y[idx]
        n = len(idx)
        imp = SimpleImputer(strategy="median").fit(Xi)
        Z = StandardScaler().fit_transform(imp.transform(Xi))
        rng = np.random.default_rng(zlib.crc32(f"{args.dataset}|{task['group']}".encode()) % (2**31))
        donors = [t for t in tasks if t["group"] != task["group"]]
        for k in args.ks:
            if k > n - args.min_queries:
                continue
            for rep in range(args.reps):
                for strategy in args.strategies:
                    if strategy == "cross_context":
                        cand = [t for t in donors if len(t["idx"]) >= k]
                        if not cand:
                            continue
                        d = cand[int(rng.integers(len(cand)))]
                        a_idx = rng.choice(len(d["idx"]), size=k, replace=False)
                        aX, aY = X[d["idx"]][a_idx], y[d["idx"]][a_idx]
                        q = np.arange(n)
                    else:
                        sel = select(strategy, Z, k, rng)
                        if sel is None or len(sel) >= n - 4:
                            continue
                        q = np.setdiff1d(np.arange(n), sel)
                        aX, aY = Xi[sel], yi[sel]
                    for model in args.models:
                        try:
                            m = model_factory(model, rep)
                            m.fit(aX, aY)
                            pred = m.predict(Xi[q])
                            rho, rec, prec, reg = metrics(yi[q], pred)
                        except Exception as e:
                            if model == "tabicl":
                                print(f"tabicl fail {task['group']} k={k}: {e}", flush=True)
                            continue
                        rows.append({"dataset": args.dataset, "task": r["task"],
                                     "group": task["group"], "k": k, "rep": rep,
                                     "strategy": strategy, "model": model,
                                     "n": n, "n_q": len(q), "spearman": rho,
                                     "recall10": rec, "prec10": prec, "regret10": reg})
        if ti % 5 == 0:
            print(f"  {ti}/{len(tasks)} tasks, {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(outfile, index=False)

    rdf = pd.DataFrame(rows)
    print("\n=== mean by strategy/model/k ===")
    print(rdf.groupby(["strategy", "model", "k"])["spearman"].mean()
             .unstack().round(3).to_string())
    rdf.to_csv(outfile, index=False)
    print(f"\nwrote {outfile}")


if __name__ == "__main__":
    main()