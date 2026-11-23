#!/usr/bin/env bash
# Canonical main grid: leave at least 10 queries per fold (min-queries 10).
set -e
PY="${PYTHON:-python3}"
RUN="code/bench/run.py"
OUT="results/bench"
LOG="results/logs"
mkdir -p "$OUT" "$LOG"

$PY $RUN --dataset D1 --min-samples 20 --min-queries 10 --ks 10 20 \
    --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D1.log 2>&1

$PY $RUN --dataset D2 --min-samples 60 --min-queries 10 --ks 10 20 30 50 \
    --reps 5 --models knn ridge gbm --out $OUT > $LOG/minq10_D2.log 2>&1

$PY $RUN --dataset D3 --min-samples 60 --min-queries 10 --ks 10 20 30 50 \
    --reps 5 --models knn ridge gbm --out $OUT > $LOG/minq10_D3.log 2>&1

$PY $RUN --dataset D4 --min-samples 5 --include-global --min-queries 10 \
    --ks 5 10 --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D4.log 2>&1

for t in particle_size EE LC; do
  $PY $RUN --dataset D5 --target $t --min-samples 10 --include-global --min-queries 10 \
      --ks 10 20 30 --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D5_$t.log 2>&1
done

for t in particle_size_nm encapsulation_efficiency_pct; do
  $PY $RUN --dataset D6 --target $t --min-samples 10 --include-global --min-queries 10 \
      --ks 10 20 30 --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D6_$t.log 2>&1
done

for t in particle_size_nm_std pdi_std zeta_potential_mv_std encapsulation_efficiency_percent_std; do
  $PY $RUN --dataset D7 --target $t --min-samples 10 --include-global --min-queries 10 \
      --ks 10 20 30 --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D7_$t.log 2>&1
done

for t in BBB_transport_score_0_100 overall_performance_score_0_100; do
  $PY $RUN --dataset D8 --target $t --min-samples 100 --include-global --min-queries 10 \
      --ks 10 20 30 --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D8_$t.log 2>&1
done

$PY $RUN --dataset D9 --min-samples 10 --include-global --min-queries 10 \
    --ks 10 20 30 --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D9.log 2>&1

$PY $RUN --dataset D10 --min-samples 20 --include-global --min-queries 10 \
    --ks 10 20 30 --reps 10 --models knn ridge gbm --out $OUT > $LOG/minq10_D10.log 2>&1

echo "main grid done"