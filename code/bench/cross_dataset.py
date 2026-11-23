"""Cross-dataset analysis: coverage gain vs cross-context transfer and learnability."""
import glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

frames = [pd.read_csv(f) for f in glob.glob("results/bench/D*.csv")]
r = pd.concat(frames, ignore_index=True)

rows = []
for (ds, task, group), sub in r.groupby(["dataset", "task", "group"]):
    # use the largest common k with random/kmeans/cross for fixed models
    for model in ["knn", "ridge"]:
        s = sub[sub.model == model]
        ks = [k for k in [10, 20, 30, 50] if k in s.k.values]
        if not ks:
            continue
        k = max(ks)
        def m(strat):
            v = s[(s.k == k) & (s.strategy == strat)]["spearman"]
            return v.mean() if len(v) else np.nan
        rnd, km, cc = m("random"), m("kmeans"), m("cross_context")
        rows.append({"dataset": ds, "task": task, "group": group, "model": model,
                     "k": k, "random": rnd, "kmeans": km, "cross": cc,
                     "gain": km - rnd})

t = pd.DataFrame(rows)
t.to_csv("results/bench/cross_dataset.csv", index=False)

print("=== per dataset/task (mean over tasks, models) ===")
g = (t.groupby(["dataset", "task"])
       .agg(random=("random", "mean"), kmeans=("kmeans", "mean"),
            cross=("cross", "mean"), gain=("gain", "mean"),
            tasks=("group", "nunique"))
       .reset_index().sort_values("gain", ascending=False))
print(g.round(3).to_string(index=False))

print("\n=== correlations across dataset×task ===")
for x in ["cross", "random"]:
    q = g[[x, "gain"]].dropna()
    print(f"  gain vs {x}: Spearman {spearmanr(q[x], q['gain']).correlation:+.3f} (n={len(q)})")

print("\n=== correlations across tasks (all) ===")
for x in ["cross", "random"]:
    q = t[[x, "gain"]].dropna()
    print(f"  gain vs {x}: Spearman {spearmanr(q[x], q['gain']).correlation:+.3f} (n={len(q)})")