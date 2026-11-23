"""Certification frontier: score quality (Spearman rho) x calibration size n.

Simulates (pilot gain, full gain) pairs from a Gaussian copula calibrated to the
empirical marginals of the 103 pilot tasks, then applies the same
Clopper-Pearson rule used in the paper. For each cell we report the median
minimum certifiable alpha and the share of draws that certify alpha=0.20/0.30.
"""
import math

import numpy as np
import pandas as pd

OUT = "results/bench_v2"
RHO = [0.2, 0.4, 0.6, 0.8]
NS = [100, 250, 500, 1000, 2000, 5000]
REPS = 100
LAMS = np.arange(-0.12, 0.2401, 0.01)


def betacf(a, b, x):
    maxit, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = fpmin if abs(d) < fpmin else d
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1 - x) / b


def cp_upper(s, m, delta=0.10):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if betai(s + 1, m - s, mid) < 1 - delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def phi(z):
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def min_alpha(s, y):
    best = 1.0
    for lam in LAMS:
        sel = s > lam
        m = int(sel.sum())
        if m < 10:
            continue
        k = int((y[sel] < 0).sum())
        best = min(best, cp_upper(k, m))
    return best


def main():
    d = pd.read_csv(f"{OUT}/pilot_pairs.csv")
    s0, y0 = d["pilot_gain"].to_numpy(), d["full_gain"].to_numpy()
    rng = np.random.default_rng(20261122)

    rows = []
    for rho in RHO:
        for n in NS:
            vals = []
            for _ in range(REPS):
                a = rng.standard_normal(n)
                b = rng.standard_normal(n)
                z1 = a
                z2 = rho * a + math.sqrt(max(0.0, 1 - rho * rho)) * b
                s = np.quantile(s0, phi(z1))
                y = np.quantile(y0, phi(z2))
                vals.append(min_alpha(s, y))
            vals = np.asarray(vals)
            rows.append(dict(
                rho=rho, n=n, reps=REPS,
                min_alpha_median=float(np.median(vals)),
                q10=float(np.quantile(vals, 0.10)),
                q90=float(np.quantile(vals, 0.90)),
                cert_rate_020=float((vals <= 0.20).mean()),
                cert_rate_030=float((vals <= 0.30).mean()),
            ))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/frontier.csv", index=False)


if __name__ == "__main__":
    main()