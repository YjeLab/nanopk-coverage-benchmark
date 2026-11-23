"""Additional robustness checks for the benchmark.

Six checks:
  A. cross-dataset molecule / study overlap audit (independence of the 175 tasks)
  B. primary-estimand reconciliation table (0.041 / 0.047 / 0.040 / 0.050 / 0.067)
  C. hit-count dictionary (recall -> coverage hits -> extra hits -> break-even R)
  D. fixed-task-subset budget curve (is the aggregate decline task composition?)
  E. Holm sensitivity on top of Benjamini-Hochberg
  F. overlap-excluded sensitivity for the overall gain

Writes results/bench_v2/editor_*.{csv,txt}. Reads only; never overwrites existing results.
"""
import sys
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _holm(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * (n - np.arange(n))
    adj = np.maximum.accumulate(adj)
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from bench.data import load  # noqa: E402

OUT = ROOT / "results" / "bench_v2"
RNG = np.random.default_rng(20261123)
NBOOT = 4000
lines = []


def echo(s=""):
    print(s)
    lines.append(str(s))


# ------------------------------------------------------------------ utils
def canon(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s)) if s is not None and str(s) != "nan" else None
        out.append(Chem.MolToSmiles(m) if m is not None else None)
    return out


def cluster_boot_ci(d, col="gain", nboot=NBOOT):
    fams = d.family.unique()
    vals = []
    for _ in range(nboot):
        pick = RNG.choice(fams, len(fams), replace=True)
        v = pd.concat([d[d.family == f] for f in pick])[col].values
        vals.append(v.mean())
    return np.percentile(vals, [2.5, 97.5])


# ------------------------------------------------------------------ A. overlap
echo("=" * 78)
echo("A. CROSS-DATASET OVERLAP AUDIT")
echo("=" * 78)

MOLCOL = {"D2": "IL_SMILES", "D3": "smiles", "D7": "ionizable_lipid_smiles", "D10": "smiles"}
mol_by_ds, task_mols = {}, {}
for ds, col in MOLCOL.items():
    r = load(ds)
    d = r["df"].copy()
    if len(d) != len(r["groups"]):
        raise RuntimeError(f"{ds}: df rows {len(d)} != groups {len(r['groups'])}")
    d["_group"] = np.asarray(r["groups"]).astype(str)
    d = d.assign(_canon=canon(d[col].tolist()))
    d = d[d._canon.notna()]
    mol_by_ds[ds] = set(d._canon)
    task_mols[ds] = d.groupby("_group")["_canon"].apply(set).to_dict()
    echo(f"{ds}: rows={len(d)} unique molecules={len(mol_by_ds[ds])} tasks={len(task_mols[ds])}")

ov = []
for a, b in combinations(MOLCOL, 2):
    inter = mol_by_ds[a] & mol_by_ds[b]
    ov.append({"pair": f"{a}-{b}", "n_shared": len(inter),
               "jaccard": len(inter) / len(mol_by_ds[a] | mol_by_ds[b]),
               "share_of_%s" % a: len(inter) / len(mol_by_ds[a]),
               "share_of_%s" % b: len(inter) / len(mol_by_ds[b])})
ov = pd.DataFrame(ov)
ov.to_csv(OUT / "editor_overlap_pairs.csv", index=False)
echo("\npairwise shared molecules:")
echo(ov.round(4).to_string(index=False))

# task-level: does a task contain molecules that also occur in another dataset?
rows = []
for ds in MOLCOL:
    others = set().union(*[v for k, v in mol_by_ds.items() if k != ds])
    for g, s in task_mols[ds].items():
        sh = s & others
        rows.append({"dataset": ds, "group": g, "n_mol": len(s), "n_shared": len(sh),
                     "frac_shared": len(sh) / max(1, len(s))})
tl = pd.DataFrame(rows)
# global tasks contain the whole dataset: give them the dataset-level share
extra = []
for ds in MOLCOL:
    others = set().union(*[v for k, v in mol_by_ds.items() if k != ds])
    frac = len(mol_by_ds[ds] & others) / max(1, len(mol_by_ds[ds]))
    extra.append({"dataset": ds, "group": "global", "n_mol": len(mol_by_ds[ds]),
                  "n_shared": len(mol_by_ds[ds] & others), "frac_shared": frac})
tl = pd.concat([tl, pd.DataFrame(extra)], ignore_index=True)
tl.to_csv(OUT / "editor_overlap_tasks.csv", index=False)
echo("\nshare of task molecules found in another dataset (task level):")
echo(tl.groupby("dataset").agg(tasks=("group", "size"), mean_frac=("frac_shared", "mean"),
                               max_frac=("frac_shared", "max"),
                               tasks_any=("n_shared", lambda x: int((x > 0).sum())),
                               tasks_half=("frac_shared", lambda x: int((x >= .5).sum()))
                               ).round(3).to_string())

# study-identifier overlap (are the same experiments reused?)
ids = {}
r2 = load("D2")["df"]; r3 = load("D3")["df"]; r7 = load("D7")["df"]; r10 = load("D10")["df"]
ids["D2 Experiment_ID"] = set(r2["Experiment_ID"].astype(str))
ids["D3 experiment unit"] = set(r3["unit"].astype(str))
ids["D7 paper_doi"] = set(r7["paper_doi"].astype(str))
ids["D10 Experiment_ID"] = set(r10["Experiment_ID"].astype(str))
echo("\nstudy identifiers:")
for k, v in ids.items():
    echo(f"  {k}: {len(v)} unique")
for a, b in combinations(ids, 2):
    echo(f"  shared {a} / {b}: {len(ids[a] & ids[b])}")

# ------------------------------------------------------------------ B/C. estimand + hits
echo()
echo("=" * 78)
echo("B. PRIMARY-ESTIMAND RECONCILIATION")
echo("=" * 78)

tg = pd.read_csv(OUT / "task_gain.csv").dropna(subset=["gain"])
ho = pd.read_csv(OUT / "hier_overall.csv").iloc[0]
dense = pd.read_csv(OUT / "budget_dense.csv")
dense = dense[dense.model.isin(["knn", "ridge"]) & dense.strategy.isin(["random", "kmeans"])]
dp = dense.pivot_table(index=["dataset", "target", "group", "k"], columns="strategy",
                       values="spearman", aggfunc="mean").reset_index()
dp["gain"] = dp.kmeans - dp.random
main = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results/bench/D*.csv")))],
                 ignore_index=True)
main = main[main.model.isin(["knn", "ridge"]) & main.strategy.isin(["random", "kmeans"])]
mp = main.pivot_table(index=["dataset", "task", "group", "k"], columns="strategy",
                      values="spearman", aggfunc="mean").reset_index()
mp["gain"] = mp.kmeans - mp.random

ds_eq = tg.groupby("dataset").gain.mean().mean()
est = pd.DataFrame([
    {"estimand": "task-equal mean, each task at its own max k", "value": tg.gain.mean(),
     "lo": ho.cluster_ci_lo, "hi": ho.cluster_ci_hi, "n": len(tg), "ci": "family-cluster bootstrap",
     "role": "PRIMARY"},
    {"estimand": "same, naive task bootstrap", "value": tg.gain.mean(),
     "lo": ho.naive_ci_lo, "hi": ho.naive_ci_hi, "n": len(tg), "ci": "task bootstrap",
     "role": "sensitivity"},
    {"estimand": "mixed-effects intercept, random intercept per family",
     "value": ho.mixedlm_intercept, "lo": ho.mixedlm_intercept - 1.96 * ho.mixedlm_se,
     "hi": ho.mixedlm_intercept + 1.96 * ho.mixedlm_se, "n": len(tg), "ci": "Wald",
     "role": "sensitivity (ICC=%.3f)" % ho.icc_family},
    {"estimand": "dense grid, k=10 fold means", "value": dp[dp.k == 10].gain.mean(),
     "lo": np.nan, "hi": np.nan, "n": dp[dp.k == 10].groupby(["dataset", "target", "group"]).ngroups,
     "ci": "-", "role": "sensitivity (budget-explicit)"},
    {"estimand": "main grid, k=10 fold means", "value": mp[mp.k == 10].gain.mean(),
     "lo": np.nan, "hi": np.nan, "n": mp[mp.k == 10].groupby(["dataset", "task", "group"]).ngroups,
     "ci": "-", "role": "sensitivity (budget-explicit)"},
    {"estimand": "dataset-equal mean of task means", "value": ds_eq,
     "lo": np.nan, "hi": np.nan, "n": tg.dataset.nunique(), "ci": "-", "role": "sensitivity (weighting)"},
])
est.to_csv(OUT / "editor_estimands.csv", index=False)
echo(est.round(4).to_string(index=False))
echo(f"family ICC = {ho.icc_family:.3f}; mixed-model SE = {ho.mixedlm_se:.4f}")

echo()
echo("=" * 78)
echo("C. HIT-COUNT DICTIONARY")
echo("=" * 78)
ut = pd.read_csv(OUT / "utility_tasks.csv")
ut["cov_hits"] = ut.recall_kmeans * ut.n_top
ut["rand_hits"] = ut.recall_random * ut.n_top
cb = pd.read_csv(OUT / "utility_cost_benefit.csv")
allrow = cb[cb.family == "ALL"].iloc[0]
hits = pd.DataFrame([{
    "quantity": "mean |top-10% set| (n_top)", "value": ut.n_top.mean(), "unit": "candidates"},
    {"quantity": "mean recall@10%, random", "value": ut.recall_random.mean(), "unit": "fraction"},
    {"quantity": "mean recall@10%, coverage", "value": ut.recall_kmeans.mean(), "unit": "fraction"},
    {"quantity": "mean coverage hits (absolute)", "value": ut.cov_hits.mean(), "unit": "candidates"},
    {"quantity": "mean extra hits (coverage - random)", "value": ut.extra_hits.mean(), "unit": "candidates"},
    {"quantity": "mean measurement budget K", "value": ut.k.mean() if "k" in ut else allrow.mean_k,
     "unit": "formulations"},
    {"quantity": "program break-even R = mean K / mean coverage hits", "value": allrow.program_R,
     "unit": "value per hit / cost per measurement"},
])
hits.to_csv(OUT / "editor_hit_dictionary.csv", index=False)
echo(hits.round(4).to_string(index=False))
pcs = pd.read_csv(OUT / "pilot_cost_summary.csv").iloc[0]
echo(f"\npilot subset (n={int(pcs.n_tasks)}): mean EXTRA hits, always-coverage "
     f"{pcs.mean_extra_hits_all_coverage:.3f}; pilot rule {pcs.mean_extra_hits_pilot:.3f}; "
     f"difference {pcs.mean_extra_hits_all_coverage - pcs.mean_extra_hits_pilot:+.3f}")

# ------------------------------------------------------------------ D. budget on fixed subset
echo()
echo("=" * 78)
echo("D. BUDGET CURVE ON A FIXED TASK SUBSET")
echo("=" * 78)
tid = ["dataset", "target", "group"]
ks = sorted(dp.k.unique())
sets = {k: set(map(tuple, dp[dp.k == k][tid].values)) for k in ks}
common = set.intersection(*sets.values())
echo(f"k levels: {ks}; tasks present at every k: {len(common)}")
cur = []
for k in ks:
    sub = dp[dp.k == k]
    c = sub[sub[tid].apply(lambda r: tuple(r) in common, axis=1)]
    cur.append({"k": k, "n_all": len(sub), "mean_all": sub.gain.mean(),
                "n_fixed": len(c), "mean_fixed": c.gain.mean()})
cur = pd.DataFrame(cur)
cur.to_csv(OUT / "editor_budget_fixed_subset.csv", index=False)
echo(cur.round(4).to_string(index=False))
x = np.log(cur.k.values); y = cur.mean_fixed.values
sl = np.polyfit(x, y, 1)[0]
echo(f"slope of fixed-subset curve on log k = {sl:+.4f} over k={ks[0]}..{ks[-1]}")
sl_all = np.polyfit(np.log(cur.k.values), cur.mean_all.values, 1)[0]
echo(f"slope of all-task curve on log k       = {sl_all:+.4f}")

# ------------------------------------------------------------------ E. Holm
echo()
echo("=" * 78)
echo("E. MULTIPLE-COMPARISON SENSITIVITY (Holm vs BH)")
echo("=" * 78)
fam = pd.read_csv(OUT / "hier_families.csv")
fam = fam[fam.tasks >= 2].copy()
p = fam.p_greater.values
fam["q_bh"] = _bh(p)
fam["p_holm"] = _holm(p)
fam = fam.sort_values("p_greater")
fam[["family", "tasks", "mean_gain", "p_greater", "q_bh", "p_holm"]].round(4).to_csv(
    OUT / "editor_multiple_comparison.csv", index=False)
echo(fam[["family", "tasks", "mean_gain", "p_greater", "q_bh", "p_holm"]].round(4).to_string(index=False))
echo(f"\nsignificant at 0.05: BH {int((fam.q_bh < .05).sum())} / {len(fam)}; "
     f"Holm {int((fam.p_holm < .05).sum())} / {len(fam)}")

# ------------------------------------------------------------------ F. overlap-excluded sensitivity
echo()
echo("=" * 78)
echo("F. OVERALL GAIN EXCLUDING TASKS WITH CROSS-DATASET MOLECULE SHARING")
echo("=" * 78)
tgm = tg.copy()
tgm["group"] = tgm.group.astype(str)
tl["group"] = tl.group.astype(str)
tgm = tgm.merge(tl[["dataset", "group", "frac_shared"]], on=["dataset", "group"], how="left")
echo(f"tasks with a matched overlap record: {tgm.frac_shared.notna().sum()} / {len(tgm)}")
for thr in (0.5, 0.2, 0.0):
    keep = tgm[(tgm.frac_shared.isna()) | (tgm.frac_shared <= thr)]
    if len(keep) < 20:
        echo(f"threshold {thr}: skipped (n={len(keep)})")
        continue
    lo, hi = cluster_boot_ci(keep)
    echo(f"exclude tasks with >= {thr:.0%} shared molecules: n={len(keep):3d} "
         f"mean={keep.gain.mean():+.4f} CI=[{lo:+.4f},{hi:+.4f}]")


# ------------------------------------------------------------------ G. leave-one-dataset-out
echo()
echo("=" * 78)
echo("G. LEAVE-ONE-DATASET-OUT SENSITIVITY (D2/D3 overlap)")
echo("=" * 78)
loo = []
for ds in sorted(tg.dataset.unique()):
    keep = tg[tg.dataset != ds]
    lo, hi = cluster_boot_ci(keep)
    loo.append({"dropped": ds, "n": len(keep), "mean": keep.gain.mean(),
                "ci_lo": lo, "ci_hi": hi})
loo = pd.DataFrame(loo)
loo.to_csv(OUT / "editor_leave_one_dataset_out.csv", index=False)
echo(loo.round(4).to_string(index=False))
for pair in (["D2"], ["D3"], ["D2", "D3"]):
    keep = tg[~tg.dataset.isin(pair)]
    lo, hi = cluster_boot_ci(keep)
    echo(f"drop {'+'.join(pair)}: n={len(keep):3d} mean={keep.gain.mean():+.4f} "
         f"CI=[{lo:+.4f},{hi:+.4f}]")

(OUT / "editor_revision_report.txt").write_text("\n".join(lines), encoding="utf-8")
echo()
echo("wrote editor_*.csv and editor_revision_report.txt")
