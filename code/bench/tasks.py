"""Task construction: a task is a group (cell line / study / experiment / paper / global)."""
import numpy as np
import pandas as pd

from bench.data import load


def build_tasks(name, target=None, min_samples=10, include_global=False):
    r = load(name, target=target) if target else load(name)
    groups = np.asarray(r["groups"])
    tasks = []
    for u in pd.unique(groups):
        idx = np.where(groups == u)[0]
        if len(idx) >= min_samples:
            tasks.append({"group": str(u), "idx": idx})
    if include_global and len(groups) >= min_samples:
        tasks.append({"group": "global", "idx": np.arange(len(groups))})
    return r, tasks