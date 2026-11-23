"""E19: hierarchical (mixed-effects) analysis of task-level coverage gain.

Two limitations of a flat task-level analysis are addressed:
  * a task-level bootstrap treats heterogeneous tasks as exchangeable;
  * a per-family CI with n=1 has zero width.
Here tasks are nested in endpoint-task families; we report a mixed-effects intercept
(random intercept per family) and a family-cluster bootstrap, plus per-family tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.exp_common import OUT, task_gain_table  # noqa: E402

RNG = np.random.default_rng(0)
d = task_gain_table()
d.to_csv(OUT / "task_gain.csv", index=False)

# ---------------- overall mixed-effects intercept ----------------
m = smf.mixedlm("gain ~ 1", d, groups=d["family"]).fit(reml=True)
fe = float(m.fe_params["Intercept"])
se = float(m.bse_fe["Intercept"])
var_family = float(m.cov_re.iloc[0, 0])
var_resid = float(m.scale)
icc = var_family / (var_family + var_resid)

# family-cluster bootstrap: resample families, keep all their tasks
fams = d["family"].unique()
boots = []
for _ in range(4000):
    pick = RNG.choice(fams, len(fams), replace=True)
    v = pd.concat([d[d.family == f] for f in pick])["gain"].values
    boots.append(v.mean())
lo, hi = np.percentile(boots, [2.5, 97.5])

# naive task bootstrap, for contrast
task_boots = [RNG.choice(d.gain.values, len(d)).mean() for _ in range(4000)]
tlo, thi = np.percentile(task_boots, [2.5, 97.5])

overall = pd.DataFrame([{
    "n_tasks": len(d), "n_families": d.family.nunique(),
    "mean": d.gain.mean(), "mixedlm_intercept": fe, "mixedlm_se": se,
    "cluster_ci_lo": lo, "cluster_ci_hi": hi,
    "naive_ci_lo": tlo, "naive_ci_hi": thi,
    "icc_family": icc, "positive_share": (d.gain > 0).mean(),
}])
overall.to_csv(OUT / "hier_overall.csv", index=False)
print(overall.round(3).to_string(index=False))

# ---------------- per-family estimates + tests ----------------
rows = []
for fam, sub in d.groupby("family"):
    v = sub.gain.values
    bs = [RNG.choice(v, len(v)).mean() for _ in range(2000)] if len(v) > 1 else [v[0]]
    if len(v) >= 5 and np.any(v != 0):
        p = wilcoxon(v, alternative="greater").pvalue
    else:
        p = binomtest(int((v > 0).sum()), len(v), 0.5, alternative="greater").pvalue
    rows.append({"family": fam, "tasks": len(v), "mean_gain": v.mean(),
                 "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                 "positive": (v > 0).mean(), "p_greater": p,
                 "dataset": sub.dataset.iloc[0]})
fam = pd.DataFrame(rows).sort_values("mean_gain", ascending=False)
fam.to_csv(OUT / "hier_families.csv", index=False)
print("\n" + fam.round(3).to_string(index=False))

# ---------------- within-task budget slope, mixed model ----------------
import glob
raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/bench/D*.csv"))],
                ignore_index=True)
raw = raw[raw.model.isin(["knn", "ridge"])]
p = raw.pivot_table(index=["dataset", "task", "group", "k"], columns="strategy",
                    values="spearman", aggfunc="mean").reset_index()
p = p.dropna(subset=["random", "kmeans"])
p["gain"] = p.kmeans - p.random
p["logk"] = np.log(p.k)
p["tid"] = p.dataset + "|" + p.task + "|" + p.group
p["family"] = [f"{a} {b}" for a, b in zip(p.dataset, p.task)]
ms = smf.mixedlm("gain ~ logk", p, groups=p["tid"]).fit(reml=True)
slope = float(ms.fe_params["logk"])
slo, shi = slope - 1.96 * float(ms.bse_fe["logk"]), slope + 1.96 * float(ms.bse_fe["logk"])
# random-slope model: lets each task have its own log-k slope (more conservative)
try:
    mr = smf.mixedlm("gain ~ logk", p, groups=p["tid"], re_formula="~logk").fit(reml=True)
    slope_r = float(mr.fe_params["logk"])
    rlo, rhi = (slope_r - 1.96 * float(mr.bse_fe["logk"]),
                slope_r + 1.96 * float(mr.bse_fe["logk"]))
    ok_r = True
except Exception:
    slope_r, rlo, rhi, ok_r = np.nan, np.nan, np.nan, False
pd.DataFrame([{"slope_logk": slope, "se": float(ms.bse_fe["logk"]),
               "ci_lo": slo, "ci_hi": shi,
               "slope_logk_randomslope": slope_r,
               "ci_lo_rs": rlo, "ci_hi_rs": rhi, "randomslope_ok": ok_r,
               "n_obs": len(p), "n_tasks": p.tid.nunique()}]).to_csv(
    OUT / "hier_budget.csv", index=False)
print(f"\nwithin-task mixed slope on log k (random intercept): {slope:+.4f} "
      f"CI [{slo:+.4f}, {shi:+.4f}] n_obs={len(p)}")
print(f"within-task mixed slope on log k (random slope):    {slope_r:+.4f} "
      f"CI [{rlo:+.4f}, {rhi:+.4f}]" if ok_r else "random-slope model failed")