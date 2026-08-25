#!/usr/bin/env bash
set -o pipefail

JAX_PLATFORMS=cpu \
MPLCONFIGDIR=/tmp/brainx-mpl-cache \
/home/yixinliu/anaconda3/envs/braincell-released/bin/python sparse_ei_network.py \
  --config runs/20260824T151959+0800-production-brunel-seeds5/config.json \
  --output-dir runs/20260824T151959+0800-production-brunel-seeds5/results \
  2>&1 | tee -a runs/20260824T151959+0800-production-brunel-seeds5/run.log

run_exit=${PIPESTATUS[0]}
printf '%s\n' "${run_exit}" > runs/20260824T151959+0800-production-brunel-seeds5/exit_code
exit "${run_exit}"
