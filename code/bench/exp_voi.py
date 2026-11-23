"""Empirical value of information of the k=10 pilot.

Compares the pilot policy ("adopt coverage if pilot gain > 0") with always-coverage
in two units: extra hits and Spearman utility. Bootstrap over tasks.
"""
import numpy as np
import pandas as pd

OUT = "results/bench_v2"
RNG = np.random.default_rng(20261122)
N_BOOT = 20000


def boot_ci(x, rng, n_boot=N_BOOT):
    n = len(x)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = x[draws].mean(axis=1)
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float((means > 0).mean())


def main():
    d = pd.read_csv(f"{OUT}/pilot_cost_tasks.csv")
    adopt = d["adopt"].astype(str).str.lower() == "true"

    always_hits = d["extra_hits"].to_numpy()
    policy_hits = np.where(adopt, d["extra_hits_pilot"].to_numpy(), 0.0)
    dhits = policy_hits - always_hits

    u = d["full_gain"].to_numpy()
    policy_u = np.where(adopt, u, 0.0)
    du = policy_u - u

    h = boot_ci(dhits, RNG)
    v = boot_ci(du, RNG)

    rows = [
        dict(metric="delta_extra_hits", mean=h[0], ci_lo=h[1], ci_hi=h[2], p_positive=h[3]),
        dict(metric="delta_spearman_utility", mean=v[0], ci_lo=v[1], ci_hi=v[2], p_positive=v[3]),
        dict(metric="pilot_cost_median", mean=float(d["pilot_cost"].median()), ci_lo=np.nan, ci_hi=np.nan, p_positive=np.nan),
        dict(metric="always_coverage_hits", mean=float(always_hits.mean()), ci_lo=np.nan, ci_hi=np.nan, p_positive=np.nan),
        dict(metric="pilot_policy_hits", mean=float(policy_hits.mean()), ci_lo=np.nan, ci_hi=np.nan, p_positive=np.nan),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/voi_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()