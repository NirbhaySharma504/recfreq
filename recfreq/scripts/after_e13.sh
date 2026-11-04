#!/bin/bash
# Plan §20: babysit the E13/E14 queues on the shared server, then run survival + wide probes and grade §20.1.
# If a queue dies (killed, OOM because the GPU is shared) or gives up on runs, restart both queues, at most 3 times.
# bash scripts/after_e13.sh   (from ~/recfreq on the server)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=logs/after_e13.log
TOTAL=148
Q0="configs/E13.json configs/E14_base.json configs/E14_hp.json configs/E14_arch.json"
Q1="configs/E14_arch.json configs/E14_hp.json configs/E14_base.json configs/E13.json"
queues_alive() { pgrep -u "$USER" -f "^[^ ]*python[0-9.]* scripts/queue.py" > /dev/null; }
n_done() { ls results/runs/E1[34]_*/done.json 2>/dev/null | wc -l; }
restarts=0
while true; do
  while queues_alive; do sleep 60; done
  d=$(n_done)
  echo "$(date +%T) no queue running; $d/$TOTAL runs done" >> $LOG
  [ "$d" -ge "$TOTAL" ] && break
  if [ $restarts -ge 3 ]; then echo "$(date +%T) giving up after 3 restarts" >> $LOG; break; fi
  restarts=$((restarts+1))
  echo "$(date +%T) restart $restarts" >> $LOG
  CUDA_VISIBLE_DEVICES=0 setsid nohup $PY scripts/queue.py $Q0 --max-parallel 6 >> logs/queue_gpu0.log 2>&1 < /dev/null &
  sleep 20
  CUDA_VISIBLE_DEVICES=1 setsid nohup $PY scripts/queue.py $Q1 --max-parallel 4 >> logs/queue_gpu1.log 2>&1 < /dev/null &
  sleep 60
done
CUDA_VISIBLE_DEVICES=0 $PY -m recfreq.survival_probes results/runs --match E13_,E14_ --seeds 0,1,2 --device cuda >> $LOG 2>&1
echo "$(date +%T) probes finished" >> $LOG
$PY -m recfreq.verify.validity score --out verify_out --procs 12 > logs/score_s20.txt 2>&1
echo "$(date +%T) scoring finished" >> $LOG
