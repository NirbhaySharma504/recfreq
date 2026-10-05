#!/bin/bash
# Overnight batch (Oct 5): positional-encoding control, conflict-enriched training,
# uniform-typo corner, (h, eps) boundary grid. Then one analysis per experiment.
cd ~/recfreq
LOG=results/queue_E2.log
echo "start $(date)" | tee -a $LOG
.venv/bin/python scripts/queue.py configs/E2a.json configs/E2c.json configs/E2u.json configs/E2b.json \
    --max-parallel 4 2>&1 | tee -a $LOG
echo "queue end $(date)" | tee -a $LOG
A=".venv/bin/python -m recfreq.analyze results/runs"
$A --exp E1,E2a --out figs_E2a            > results/analysis_E2a.txt 2>&1
$A --exp E1,E2c --out figs_E2c            > results/analysis_E2c.txt 2>&1
$A --exp E2u    --out figs_E2u            > results/analysis_E2u.txt 2>&1
$A --exp E1,E2b --out figs_E2b --curves 0 > results/analysis_E2b.txt 2>&1
echo "analysis end $(date)" | tee -a $LOG
