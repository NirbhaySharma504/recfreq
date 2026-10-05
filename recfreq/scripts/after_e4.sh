#!/bin/bash
# runs once the E4 training queue has finished: fits, E4 summary, then the GPU audit checks
cd ~/recfreq
while ! grep -q "queue finished" results/queue_E4.log; do sleep 60; done
while [ ! -f verify_out/cpu_done.txt ]; do sleep 30; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m recfreq.fit_rules results/runs --exp E4r,E4h --out figs_fit_E4 --procs 10 > verify_out/fit_E4.log 2>&1
.venv/bin/python -m recfreq.verify.e4 > verify_out/e4.txt 2>&1
.venv/bin/python -m recfreq.verify.gpu_checks ood > verify_out/ood.txt 2>&1
.venv/bin/python -m recfreq.verify.gpu_checks precision > verify_out/precision.txt 2>&1
.venv/bin/python -m recfreq.verify.gpu_checks slope > verify_out/slope.txt 2>&1
echo "after_e4 finished $(date)" > verify_out/after_e4_done.txt
