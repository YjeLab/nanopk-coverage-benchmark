"""E7: task-level bootstrap CIs for coverage gain; E13: decision metrics summary."""
import glob
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
frames = [pd.read_csv(f) for f in glob.glob("results/bench/D*.csv")]
r = pd.concat(frames, ignore_index=True)

# --- E7: per-task gain (kmeans - random), fixed models, best common k
rows = []
for (ds, task, group), sub in r.groupby(["dataset", "task", "group"]):
    for model in ["knn", "ridge"]:
        s = sub[sub.model == model]
        ks = sorted(s.k.unique())
        if not ks:
            continue
        k = ks[-1]
        p = s[s.k == k].pivot_table(index="strategy", values="spearman", aggfunc="mean")["spearman"]
        if "kmeans" in p and "random" in p:
            rows.append({"dataset": ds, "task": task, "group": group, "model": model,
                         "k": k, "gain": p["kmeans"] - p["random"],
                         "random": p["random"], "kmeans": p["kmeans"]})
g = pd.DataFrame(rows)

out = []
for (ds, task), sub in g.groupby(["dataset", "task"]):
    v = sub.groupby("group")["gain"].mean().values
    if len(v) == 0:
        continue
    boots = [rng.choice(v, len(v), replace=True).mean() for _ in range(1000)]
    out.append({"dataset": ds, "task": task, "tasks": len(v),
                "gain_mean": v.mean(), "ci_lo": np.percentile(boots, 2.5),
                "ci_hi": np.percentile(boots, 97.5),
                "positive": (v > 0).mean()})
ci = pd.DataFrame(out).sort_values("gain_mean", ascending=False)
ci.to_csv("results/bench/ci_table.csv", index=False)
print("=== E7: per-task gain with 95% bootstrap CI ===")
print(ci.round(3).to_string(index=False))

all_v = g.groupby(["dataset", "task", "group"])["gain"].mean().values
boots = [rng.choice(all_v, len(all_v), replace=True).mean() for _ in range(1000)]
print(f"\noverall tasks={len(all_v)} mean={all_v.mean():+.3f} "
      f"CI=[{np.percentile(boots,2.5):+.3f},{np.percentile(boots,97.5):+.3f}] "
      f"positive={(all_v>0).mean():.1%}")

# --- E13: decision metrics at k=20 (or largest common), random vs kmeans
k_use = 20
rows = []
for (ds, task), sub in r.groupby(["dataset", "task"]):
    s = sub[(sub.k == k_use) & (sub.model.isin(["knn", "ridge"]))]
    if s.empty:
        continue
    for strat in ["random", "kmeans"]:
        q = s[s.strategy == strat]
        if q.empty:
            continue
        per_task = q.groupby("group")[["recall10", "prec10", "regret10"]].mean()
        rows.append({"dataset": ds, "task": task, "strategy": strat,
                     "recall10": per_task["recall10"].mean(),
                     "prec10": per_task["prec10"].mean(),
                     "regret10": per_task["regret10"].mean(),
                     "tasks": len(per_task)})
dm = pd.DataFrame(rows)
dm.to_csv("results/bench/decision_metrics.csv", index=False)
print(f"\n=== E13: decision metrics at k={k_use} ===")
print(dm.round(3).to_string(index=False))