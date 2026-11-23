"""Shared configs/helpers for the v2 closed-loop experiments (E15-E22)."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.data import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "bench_v2"
BENCH = ROOT / "results" / "bench"
OUT.mkdir(parents=True, exist_ok=True)

# (dataset, target, min_samples, include_global) -- matches run_main_grid.sh
GRID_CONFIGS = [
    ("D1", None, 20, False),
    ("D2", None, 60, False),
    ("D3", None, 60, False),
    ("D4", None, 5, True),
    ("D5", "particle_size", 10, True),
    ("D5", "EE", 10, True),
    ("D5", "LC", 10, True),
    ("D6", "particle_size_nm", 10, True),
    ("D6", "encapsulation_efficiency_pct", 10, True),
    ("D7", "particle_size_nm_std", 10, True),
    ("D7", "pdi_std", 10, True),
    ("D7", "zeta_potential_mv_std", 10, True),
    ("D7", "encapsulation_efficiency_percent_std", 10, True),
    ("D8", "BBB_transport_score_0_100", 100, True),
    ("D8", "overall_performance_score_0_100", 100, True),
    ("D9", None, 10, True),
    ("D10", None, 20, True),
]

FAMILY_RENAME = {
    "T1_cellline_ic50": "D1 IC50",
    "T2_study_delivery": "D2 delivery",
    "T3_experiment_delivery": "D3 delivery",
    "T6_study_tumor_delivery": "D4 tumour",
    "T4_plga_particle_size": "D5 size",
    "T4_plga_EE": "D5 EE",
    "T4_plga_LC": "D5 LC",
    "T5_lipodb_particle_size_nm": "D6 size",
    "T5_lipodb_encapsulation_efficiency_pct": "D6 EE",
    "T7_lnpatlas_particle_size_nm_std": "D7 size",
    "T7_lnpatlas_pdi_std": "D7 PDI",
    "T7_lnpatlas_zeta_potential_mv_std": "D7 zeta",
    "T7_lnpatlas_encapsulation_efficiency_percent_std": "D7 EE",
    "T8_synthetic_BBB_transport_score_0_100": "D8 BBB",
    "T8_synthetic_overall_performance_score_0_100": "D8 overall",
    "T9_mht_sar": "D9 SAR",
    "T10_lnpcd_viability": "D10 viability",
}


def family_name(dataset, task):
    return FAMILY_RENAME.get(task, f"{dataset} {task}")


def build(ds, target, min_samples, include_global):
    r = load(ds, target=target) if target else load(ds)
    g = np.asarray(r["groups"])
    tasks = []
    for u in pd.unique(g):
        idx = np.where(g == u)[0]
        if len(idx) >= min_samples:
            tasks.append({"group": str(u), "idx": idx})
    if include_global and len(g) >= min_samples:
        tasks.append({"group": "global", "idx": np.arange(len(g))})
    return r, tasks


def prep(Xi):
    imp = SimpleImputer(strategy="median").fit(Xi)
    return StandardScaler().fit_transform(imp.transform(Xi))


def task_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


def task_gain_table(models=("knn", "ridge")):
    """Canonical task-level coverage gain (kmeans - random) at each task's max k."""
    frames = [pd.read_csv(f) for f in sorted(BENCH.glob("D*.csv"))]
    if not frames:
        raise FileNotFoundError("no results/bench/D*.csv; run the main grid first")
    r = pd.concat(frames, ignore_index=True)
    r = r[r.model.isin(models) & r.strategy.isin(["random", "kmeans"])]
    rows = []
    for (ds, task, group), sub in r.groupby(["dataset", "task", "group"]):
        k = sub.k.max()
        p = sub[sub.k == k].pivot_table(index="strategy", values="spearman",
                                        aggfunc="mean")["spearman"]
        if "random" in p and "kmeans" in p:
            rows.append({"dataset": ds, "task": task, "group": group, "k": k,
                         "random": p["random"], "kmeans": p["kmeans"],
                         "gain": p["kmeans"] - p["random"], "n": sub.n.iloc[0]})
    d = pd.DataFrame(rows)
    d["family"] = [family_name(a, b) for a, b in zip(d.dataset, d.task)]
    return d