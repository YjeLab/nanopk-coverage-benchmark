"""E14: meta-regression of task-level coverage gain and within-task budget slope.

Notes
-----
- Task-level rows use each task's largest budget and kNN/ridge (main comparison).
- The aggregate gain-by-k curve is unbalanced: only the large D2/D3 studies support k=50,
  so budget effects are also tested within task (paired deltas and a slope on log k).
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)

# ---------- within-task budget analysis (all budgets, kNN/ridge) ----------
raw = pd.concat([pd.read_csv(f) for f in glob.glob("results/bench/D*_*.csv")],
                ignore_index=True)
raw = raw[raw.model.isin(["knn", "ridge"])]
p = raw.pivot_table(index=["dataset", "task", "group", "k"], columns="strategy",
                    values="spearman", aggfunc="mean").reset_index()
p = p.dropna(subset=["random", "kmeans"])
p["gain"] = p.kmeans - p.random

print("=== gain by budget (task-level, unbalanced) ===")
for k, g in p.groupby("k"):
    v = g.gain.values
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    print(f"  k={int(k):3d} n={len(v):3d} mean={v.mean():+.3f} "
          f"CI=[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}] "
          f"pos={100*(v > 0).mean():.0f}%")

w = p.pivot_table(index=["dataset", "task", "group"], columns="k", values="gain")
for a, b in [(10, 20), (10, 30), (10, 50)]:
    if a in w.columns and b in w.columns:
        q = w[[a, b]].dropna()
        dv = (q[b] - q[a]).values
        bs = [rng.choice(dv, len(dv)).mean() for _ in range(2000)]
        print(f"  paired k{b}-k{a}: n={len(dv)} delta={dv.mean():+.3f} "
              f"CI=[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")

slopes = []
for _, row in w.iterrows():
    q = row.dropna()
    if q.index.nunique() >= 2:
        slopes.append(np.polyfit(np.log(q.index.values), q.values, 1)[0])
slopes = np.array(slopes)
bs = [rng.choice(slopes, len(slopes)).mean() for _ in range(2000)]
print(f"=== within-task slope of gain on log k ===\n  n={len(slopes)} "
      f"mean={slopes.mean():+.3f} CI=[{np.percentile(bs, 2.5):+.3f},"
      f"{np.percentile(bs, 97.5):+.3f}] frac<0={100*(slopes < 0).mean():.0f}%")
pd.DataFrame({"task": range(len(slopes)), "slope": slopes}).to_csv(
    "results/bench/budget_slopes.csv", index=False)

# ---------- task-level meta-regression at each task's largest budget ----------
cd = pd.read_csv("results/bench/cross_dataset.csv")
d = (cd.groupby(["dataset", "task", "group"])
       .agg(random=("random", "mean"), kmeans=("kmeans", "mean"),
            cross=("cross", "mean"), k=("k", "max"))
       .reset_index())
d["gain"] = d.kmeans - d.random
d["log_tasks"] = np.log(d.groupby(["dataset", "task"])["group"].transform("nunique"))

print("\n=== single-feature Spearman with gain ===")
for x in ["random", "cross", "log_tasks"]:
    q = d[[x, "gain"]].dropna()
    print(f"  {x:10s} rho={spearmanr(q[x], q.gain).correlation:+.3f} (n={len(q)})")

feats = ["random", "cross", "log_tasks"]
q = d[feats + ["gain"]].dropna()
X, y = q[feats].values, q.gain.values
m = LinearRegression().fit(X, y)
print(f"\n=== OLS: gain ~ random + cross + log_tasks ===\n  R2={r2_score(y, m.predict(X)):.3f}, n={len(q)}")
coefs = {}
for name, coef in zip(feats, m.coef_):
    boots = [LinearRegression().fit(X[i], y[i]).coef_[feats.index(name)]
             for i in (rng.integers(0, len(q), len(q)) for _ in range(1000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    coefs[name] = coef
    print(f"  {name:10s} coef={coef:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

pd.DataFrame({"feature": list(coefs), "coef": list(coefs.values())}).to_csv(
    "results/bench/meta_regression.csv", index=False)
d.to_csv("results/bench/task_level.csv", index=False)