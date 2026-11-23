"""Cost-benefit of the pilot itself.

The existing cost-benefit analysis (exp_utility.py) asks whether adopting coverage
is worth K measurements: net = R * extra_hits - K, where R is the value of one
true top hit divided by the cost of one measurement.

Here we ask the same question for the pilot. A pilot that measures 5 random and
5 coverage support sets of size 10 touches N_p distinct formulations. It is worth
running only if the hits it recovers outweigh that cost:

    net_pilot = R * extra_hits_pilot - N_p   ->   break-even R = N_p / extra_hits_pilot

where extra_hits_pilot is the top-10% hit gain of tasks the pilot sends to
coverage (pilot gain > 0) and zero otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from bench.exp_common import BENCH, OUT  # noqa: E402

K_FULL = 20          # full-budget measurements used in the utility section
SUPPORTS = 10        # 5 random + 5 coverage support sets
SUPPORT_K = 10       # formulations per support set


def expected_union(n, draws=SUPPORTS * SUPPORT_K):
    """Expected number of distinct formulations touched by `draws` draws of size 1."""
    return n * (1.0 - (1.0 - 1.0 / n) ** draws)


def main():
    pilot = pd.read_csv(OUT / "pilot_pairs.csv")
    grid = pd.concat([pd.read_csv(f) for f in sorted(BENCH.glob("D*.csv"))],
                     ignore_index=True)
    grid = grid[grid.model.isin(["knn", "ridge"])
                & grid.strategy.isin(["random", "kmeans"])]

    util = pd.read_csv(OUT / "utility_tasks.csv")
    codes = util[["dataset", "task"]].drop_duplicates()
    def task_code(ds, target):
        sub = codes[codes.dataset == ds]
        hit = sub[sub.task.str.contains(target, regex=False)]
        if len(hit) == 1:
            return hit.task.iloc[0]
        if target == "main" and len(sub) == 1:
            return sub.task.iloc[0]
        raise KeyError(f"{ds}/{target}: candidates {sub.task.tolist()}")

    pilot["task"] = [task_code(d_, t) for d_, t in zip(pilot.dataset, pilot.target)]
    d = pilot.merge(util[["dataset", "task", "group", "extra_hits"]],
                    on=["dataset", "task", "group"], how="left")
    d = d[d.extra_hits.notna()].copy()
    d["pilot_cost"] = expected_union(d.n.values)
    d["adopt"] = d.pilot_gain > 0
    d["extra_hits_pilot"] = np.where(d.adopt, d.extra_hits, 0.0)
    d.to_csv(OUT / "pilot_cost_tasks.csv", index=False)

    mean_all = d.extra_hits.mean()
    mean_pilot = d.extra_hits_pilot.mean()
    mean_pos = d.loc[d.adopt, "extra_hits"].mean()
    npos = int(d.adopt.sum())
    cost_med = d.pilot_cost.median()
    cost_mean = d.pilot_cost.mean()

    be_cov = K_FULL / mean_all if mean_all > 0 else np.nan
    be_pil = cost_mean / mean_pilot if mean_pilot > 0 else np.nan
    be_pil_med = cost_med / mean_pilot if mean_pilot > 0 else np.nan

    # median-based break-even, matching the convention of the utility section
    pos_all = d.loc[d.extra_hits > 0, "extra_hits"]
    pos_pil = d.loc[d.adopt & (d.extra_hits > 0), "extra_hits"]
    be_cov_med = K_FULL / pos_all.median() if len(pos_all) else np.nan
    be_pil_med = cost_med / pos_pil.median() if len(pos_pil) else np.nan

    out = pd.DataFrame([{
        "n_tasks": len(d), "n_adopt": npos, "adopt_rate": npos / len(d),
        "mean_extra_hits_all_coverage": mean_all,
        "mean_extra_hits_pilot": mean_pilot,
        "mean_extra_hits_when_adopted": mean_pos,
        "pilot_cost_mean": cost_mean, "pilot_cost_median": cost_med,
        "breakeven_R_coverage": be_cov,
        "breakeven_R_pilot_mean_cost": be_pil,
        "breakeven_R_pilot_median_cost": be_pil_med,
        "breakeven_R_coverage_median": be_cov_med,
        "breakeven_R_pilot_median_pos": be_pil_med,
        "n_positive_hits_all": int(len(pos_all)),
        "n_positive_hits_pilot": int(len(pos_pil)),
        "pilot_recovers_share_of_value": mean_pilot / mean_all if mean_all else np.nan,
    }])
    out.to_csv(OUT / "pilot_cost_summary.csv", index=False)
    print(out.round(3).T.to_string())
    print()
    print("pilot cost distribution (distinct formulations):",
          np.percentile(d.pilot_cost, [10, 50, 90]).round(0))
    print("tasks where pilot says adopt but extra_hits < 0:",
          int(((d.adopt) & (d.extra_hits < 0)).sum()))


if __name__ == "__main__":
    main()