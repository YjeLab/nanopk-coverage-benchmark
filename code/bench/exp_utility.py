"""E20: cost-benefit / decision-curve analysis of coverage selection.

Turns the Spearman and recall deltas into a decision-relevant quantity: net benefit as a
function of the value of one true top-hit relative to the cost of one measurement.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import OUT, family_name  # noqa: E402

K = 20
rng = np.random.default_rng(0)

frames = [pd.read_csv(f) for f in sorted(glob.glob("results/bench/D*.csv"))]
r = pd.concat(frames, ignore_index=True)
r = r[r.model.isin(["knn", "ridge"]) & r.strategy.isin(["random", "kmeans"])]

rows = []
for (ds, task, group), sub in r.groupby(["dataset", "task", "group"]):
    k = sub.k.max()
    s = sub[sub.k == k]
    piv = s.pivot_table(index="strategy", values=["recall10", "prec10", "regret10"],
                        aggfunc="mean")
    if not {"random", "kmeans"}.issubset(piv.index):
        continue
    n = s.n.iloc[0]
    n_top = max(1, int(round(0.1 * n)))
    rows.append({
        "dataset": ds, "task": task, "group": group, "n": n, "n_top": n_top,
        "recall_random": piv.loc["random", "recall10"],
        "recall_kmeans": piv.loc["kmeans", "recall10"],
        "prec_random": piv.loc["random", "prec10"],
        "prec_kmeans": piv.loc["kmeans", "prec10"],
        "regret_random": piv.loc["random", "regret10"],
        "regret_kmeans": piv.loc["kmeans", "regret10"],
    })
d = pd.DataFrame(rows)
d["family"] = [family_name(a, b) for a, b in zip(d.dataset, d.task)]
d["extra_hits"] = (d.recall_kmeans - d.recall_random) * d.n_top
d["extra_top10"] = d.prec_kmeans - d.prec_random
d.to_csv(OUT / "utility_tasks.csv", index=False)

# net benefit / c_meas = R * extra_hits - K, where R = value_per_hit / cost_per_measurement
Rs = np.logspace(-1, 3, 60)
curves = []
for R in Rs:
    for fam, sub in [("ALL", d)] + list(d.groupby("family")):
        net = R * sub.extra_hits.values - K
        curves.append({"R": R, "family": fam, "mean_net": net.mean(),
                       "share_positive": (net > 0).mean()})
curves = pd.DataFrame(curves)
curves.to_csv(OUT / "utility_curves.csv", index=False)

# break-even R per family (median over tasks that gain)
be = []
for fam, sub in [("ALL", d)] + list(d.groupby("family")):
    pos = sub[sub.extra_hits > 0]
    be.append({"family": fam, "tasks": len(sub),
               "gain_tasks": len(pos),
               "median_extra_hits": sub.extra_hits.median(),
               "breakeven_R": (K / pos.extra_hits).median() if len(pos) else np.nan})
be = pd.DataFrame(be).sort_values("breakeven_R")
be.to_csv(OUT / "utility_breakeven.csv", index=False)

print("=== break-even R (value per hit / cost per measurement) ===")
print(be.round(3).to_string(index=False))

# overall decision metrics at k=20
mm = d[["recall_random", "recall_kmeans", "prec_random", "prec_kmeans",
        "regret_random", "regret_kmeans"]].mean()
print("\n=== mean decision metrics at k=20 ===")
print(mm.round(3).to_string())