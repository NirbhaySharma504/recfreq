#!/bin/bash
# eval-only run 3; most diverse architectures first
cd ~/recfreq
for e in E1 E2a E2c E2b; do
  .venv/bin/python -m recfreq.run3 results/runs --exp $e --out figs_run3 2>&1 | tee -a results/run3.log
done
echo "run3 finished $(date)" | tee -a results/run3.log
