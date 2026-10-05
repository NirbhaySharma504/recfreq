#!/bin/bash
# runs once the E5 training queue has finished: law-scope fits and verdicts (plan §18), then figures
cd ~/recfreq
while ! grep -q "queue finished" results/queue_E5.log; do sleep 60; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m recfreq.verify.law_scope --out verify_out --procs 12 > verify_out/law_scope.txt 2>&1
.venv/bin/python -m recfreq.verify.plot_scope --out figs_scope > verify_out/plot_scope.log 2>&1
echo "after_e5 finished $(date)" > verify_out/after_e5_done.txt
