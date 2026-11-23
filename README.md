# Gains and predictability of coverage sampling in nanomedicine formulation screening: a multi-dataset benchmark

**覆盖式采样在纳米药物配方筛选中的收益与可预测性：一个多数据集基准**

## About this repository

Formulation screening in nanomedicine can measure only a small number of candidates per round, so the choice of the initial measurement set largely determines how informative the subsequent ranking is. Coverage-based (space-filling) selection — picking candidates that are spread out in feature space, for example by clustering followed by per-cluster representatives, or by greedy farthest-point sampling — is widely recommended in image classification and materials informatics, yet its value for formulation screening had not been examined systematically.

This repository accompanies a benchmark study that compares coverage-based selection against random sampling over multiple public nanomedicine formulation datasets, a range of predictors, and a range of measurement budgets. The study asks three questions: whether coverage selection is on average better than random under an identical measurement budget; whether the benefit for a given endpoint can be judged in advance from cheap proxies or from a small pilot experiment; and whether a risk guarantee on the adoption decision can be certified at the available scale. The accompanying paper reports a small but consistent average gain that is endpoint-dependent, shows that neither cheap proxies nor a pilot provide a usable ex-ante criterion, and quantifies the calibration cost of a certified guarantee.

## What is in here

Only the source code needed to reproduce the analyses reported in the paper is included:

- `code/bench/` — data loaders and task construction, the sampling-strategy × predictor × budget grid runner, and the statistical analyses (hierarchical mixed-effects models, family-cluster bootstrap, decision-metric summaries, calibration and risk control, cost–benefit).
- `code/fp_utils.py` — molecular fingerprint and descriptor utilities with an on-disk cache.

## What is not in here

This repository is a partial source release, not a runnable distribution.

- **No data.** None of the underlying datasets are redistributed. They are public resources, cited in the paper, and remain subject to their own licences.
- **No intermediate artefacts.** Feature caches, task definitions and result tables are archived separately, with a DOI, and are not tracked here.
- **No plotting code.** The figure scripts used for the manuscript are not part of this release; the figures are drawn from the archived result tables.
- **No support.** The code is provided as-is, at the version used for the reported results. It depends on the layout of the accompanying data archive; paths, environment and dependencies outside the published analysis are not maintained, and issues are not monitored.

## Licence

Source code is released under the MIT Licence (see `LICENSE`). The datasets referenced by this work are not covered by that licence; consult each dataset's own terms before use.
