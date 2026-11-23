"""Print dataset/task statistics for the benchmark."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.data import load  # noqa: E402


def show(name, **kw):
    r = load(name, **kw)
    n = len(r["y"])
    groups = pd.Series(r["groups"])
    sizes = groups.value_counts()
    X = r["X"]
    print(f"{name} {r['task']:28s} rows={n:4d} groups={groups.nunique():3d} "
          f"median/group={sizes.median():5.1f} >=10:{(sizes>=10).sum():3d} "
          f"X={X.shape[1]:4d} y=[{np.nanmin(r['y']):.2f},{np.nanmax(r['y']):.2f}] "
          f"feat_na={np.isnan(X).mean():.2f}")


print("=== D1-D4 ===")
for name in ["D1", "D2", "D3", "D4"]:
    show(name)
print("=== D5 (PLGA) ===")
for t in ["particle_size", "EE", "LC"]:
    show("D5", target=t)
print("=== D6 (LipoDB) ===")
for t in ["particle_size_nm", "pdi", "zeta_potential_mV", "encapsulation_efficiency_pct"]:
    show("D6", target=t)
print("=== D7 (LNP Atlas) ===")
for t in ["particle_size_nm_std", "pdi_std", "zeta_potential_mv_std",
          "encapsulation_efficiency_percent_std"]:
    show("D7", target=t)