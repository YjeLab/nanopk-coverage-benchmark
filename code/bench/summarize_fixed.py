"""Fixed-model summary (kNN / Ridge), no best-per-repetition selection."""
import glob
import pandas as pd
import numpy as np

frames = [pd.read_csv(f) for f in glob.glob("results/bench/D*.csv")]
r = pd.concat(frames, ignore_index=True)

rows = []
for (ds, task, k, strat, model), sub in r.groupby(["dataset", "task", "k", "strategy", "model"]):
    if model not in ("knn", "ridge"):
        continue
    per_task = sub.groupby("group")["spearman"].mean()
    rows.append({"dataset": ds, "task": task, "k": k, "strategy": strat, "model": model,
                 "mean": per_task.mean(), "min": per_task.min(), "max": per_task.max(),
                 "tasks": len(per_task)})

t = pd.DataFrame(rows).sort_values(["dataset", "task", "model", "strategy", "k"])
t.to_csv("results/bench/summary_fixed.csv", index=False)

for model in ["knn", "ridge"]:
    print(f"\n=== {model}: mean Spearman ===")
    q = t[t.model == model].pivot_table(index=["dataset", "task", "strategy"],
                                        columns="k", values="mean")
    print(q.round(3).to_string())

print("\n=== coverage gain (kmeans - random), knn ===")
for model in ["knn", "ridge"]:
    print(f"[{model}]")
    q = (t[t.model == model]
         .pivot_table(index=["dataset", "task", "k"], columns="strategy", values="mean"))
    if "kmeans" in q and "random" in q:
        g = (q["kmeans"] - q["random"]).reset_index()
        print(g.pivot(index=["dataset", "task"], columns="k", values=0).round(3).to_string())