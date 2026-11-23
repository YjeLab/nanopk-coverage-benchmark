"""Consistency checks for the cost-benefit analysis.

Break-even is easy to define in two incompatible ways: dividing K by the per-task
extra hits *only on tasks that gained* is neither the zero crossing of the mean
net-benefit curve nor a single well-defined counterfactual.  Two are reported here,
separately:

  * incremental comparison (coverage vs random, same measurement budget):
    mean extra top-10% hits and its bootstrap CI.  Incremental cost is zero, so
    coverage is preferred iff this mean is positive.
  * program-level comparison (screening with coverage vs no screening):
    break-even value/cost ratio = K / mean(coverage top-10% hits).

Also computes: overall gain excluding synthetic D8; split-half/Spearman-Brown
reliability and attenuation correction for the pilot-full correlation; Fisher-z
power.  Writes utility_cost_benefit.csv and revision_checks.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parents[2] / "results" / "bench_v2"
RNG = np.random.default_rng(20261122)
NBOOT = 5000


def echo(s=""):
    print(s)
    lines.append(s)


lines = []


def boot_ci(values, stat, nboot=NBOOT):
    values = np.asarray(values, float)
    vals = [stat(RNG.choice(values, size=len(values), replace=True)) for _ in range(nboot)]
    vals = np.asarray(vals)
    fin = vals[np.isfinite(vals)]
    return (np.percentile(fin, 2.5), np.percentile(fin, 97.5)) if len(fin) else (np.nan, np.nan)


def bootstrap_mean(values_by_family, nboot=NBOOT):
    fams = list(values_by_family)
    means = []
    for _ in range(nboot):
        pick = RNG.choice(len(fams), size=len(fams), replace=True)
        pool = np.concatenate([values_by_family[f] for f in (fams[i] for i in pick)])
        means.append(pool.mean())
    return np.percentile(means, [2.5, 97.5])


def bootstrap_task_mean(values, nboot=NBOOT):
    values = np.asarray(values)
    means = [RNG.choice(values, size=len(values), replace=True).mean() for _ in range(nboot)]
    return np.percentile(means, [2.5, 97.5])


# ---------------------------------------------------------------- 1. exclude D8
echo("=" * 74)
echo("1. OVERALL GAIN, EXCLUDING SYNTHETIC D8")
echo("=" * 74)
tg = pd.read_csv(OUT / "task_gain.csv").dropna(subset=["gain"])
tg_real = tg[~tg.dataset.eq("D8")]
for label, d in [("all tasks", tg), ("excluding D8", tg_real)]:
    lo, hi = bootstrap_task_mean(d.gain.values)
    flo, fhi = bootstrap_mean({f: g.gain.values for f, g in d.groupby("family")})
    echo(f"{label:16s} n={len(d):3d} mean={d.gain.mean():+.4f} pos={100*(d.gain>0).mean():.1f}% "
         f"task95%=[{lo:+.4f},{hi:+.4f}] family95%=[{flo:+.4f},{fhi:+.4f}]")
echo(f"D8 only          n={len(tg)-len(tg_real):3d} mean={tg[tg.dataset.eq('D8')].gain.mean():+.4f}")

# ---------------------------------------------------------------- 2. cost-benefit
echo()
echo("=" * 74)
echo("2. COST-BENEFIT")
echo("=" * 74)
ut = pd.read_csv(OUT / "utility_tasks.csv")
tg = pd.read_csv(OUT / "task_gain.csv")[["dataset", "task", "group", "k"]]
ut = ut.merge(tg, on=["dataset", "task", "group"], how="left")
assert ut.k.notna().all()
ut["cov_hits"] = ut.recall_kmeans * ut.n_top
ut["rand_hits"] = ut.recall_random * ut.n_top
rows = []


def add(sub, name, synthetic):
    x = sub.extra_hits.values
    ch = sub.cov_hits.values
    kk = sub.k.values
    n = len(x)
    m = x.mean()
    dlo, dhi = boot_ci(x, np.mean)
    p_nonpos = np.mean([RNG.choice(x, size=n, replace=True).mean() <= 0
                        for _ in range(NBOOT)])
    if ch.mean() > 0:
        R = kk.mean() / ch.mean()
        rlo, rhi = boot_ci(np.column_stack([kk, ch]),
                           lambda v: v[:, 0].mean() / v[:, 1].mean())
    else:
        R, rlo, rhi = np.inf, np.nan, np.nan
    rows.append({"family": name, "n": n, "synthetic": synthetic,
                 "mean_delta": m, "delta_ci_lo": dlo, "delta_ci_hi": dhi,
                 "share_pos": (x > 0).mean(), "p_mean_nonpos": p_nonpos,
                 "mean_k": kk.mean(), "mean_cov_hits": ch.mean(), "program_R": R,
                 "program_ci_lo": rlo, "program_ci_hi": rhi})
    echo(f"{name:14s} n={n:3d} meanΔ={m:+.3f} CI=[{dlo:+.3f},{dhi:+.3f}] "
         f"pos={100*(x>0).mean():.0f}% P(mean<=0)={p_nonpos:.3f} meanK={kk.mean():.1f} "
         f"programR={R:6.2f} [{rlo:.1f},{rhi:.1f}]")


add(ut, "ALL", False)
add(ut[~ut.dataset.eq("D8")], "ALL_noD8", False)
for fam, sub in ut.groupby("family"):
    add(sub, fam, bool(sub.dataset.eq("D8").all()))
cb = pd.DataFrame(rows)
cb.to_csv(OUT / "utility_cost_benefit.csv", index=False)

# ---------------------------------------------------------------- 3. attenuation
echo()
echo("=" * 74)
echo("3. ATTENUATION (SPEARMAN-BROWN) CORRECTION")
echo("=" * 74)
pp = pd.read_csv(OUT / "pilot_pairs.csv").dropna(subset=["pilot_gain", "full_gain"])
P_REPS, F_REPS = 5, 10
rel_p = 1 - ((pp.pilot_sd ** 2).mean() / P_REPS) / pp.pilot_gain.var(ddof=1)
rel_f = 1 - ((pp.full_sd ** 2).mean() / F_REPS) / pp.full_gain.var(ddof=1)
r_obs = spearmanr(pp.pilot_gain, pp.full_gain).correlation
r_dis = r_obs / np.sqrt(rel_p * rel_f)
echo(f"n={len(pp)} observed r={r_obs:.4f} reliabilities={rel_p:.3f}/{rel_f:.3f} "
     f"disattenuated={r_dis:.4f} R2={r_dis**2:.3f}")

# ---------------------------------------------------------------- 4. power
echo()
echo("=" * 74)
echo("4. DETECTABLE CORRELATION (FISHER Z, ALPHA=.05, POWER=.80)")
echo("=" * 74)
z_alpha, z_power = 1.959964, 0.841621
echo(f"n=103 smallest detectable |rho| = {np.tanh((z_alpha+z_power)/np.sqrt(100)):.3f}")
for r in (0.2, 0.25, 0.3):
    n = (z_alpha + z_power) ** 2 / np.arctanh(r) ** 2 + 3
    echo(f"rho={r:.2f} required n = {n:.0f}")

(OUT / "revision_checks.txt").write_text("\n".join(lines), encoding="utf-8")
echo()
echo("wrote utility_cost_benefit.csv and revision_checks.txt")