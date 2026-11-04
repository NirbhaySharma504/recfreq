#!/bin/bash
# Quick status of the recfreq runs and GPU usage on the shared server.
#   ssh gpu 'bash ~/recfreq/scripts/status.sh'
cd ~/recfreq || exit 1
echo "=== $(date '+%a %d %b %H:%M:%S')  on $(hostname)"

echo; echo "=== queues"
if pgrep -u "$USER" -f "^[^ ]*python[0-9.]* scripts/queue.py" > /dev/null; then
  pgrep -u "$USER" -af "^[^ ]*python[0-9.]* scripts/queue.py" | sed -E 's/.*queue.py /  running: /; s/ --max-parallel/  (parallel/; s/$/)/'
else
  echo "  no queue running"
fi
pgrep -u "$USER" -f "^bash scripts/after_e13.sh" > /dev/null && echo "  post-processing watcher: waiting/running" || echo "  post-processing watcher: not running"

echo; echo "=== progress (E13 = validation sweep, E14 = robustness)"
for e in E13 E14; do
  d=$(ls results/runs/${e}_*/done.json 2>/dev/null | wc -l)
  t=$( [ $e = E13 ] && echo 72 || echo 76 )
  echo "  $e: $d / $t done"
done
fails=$(grep -h -E "FAIL|GIVE UP" logs/queue_gpu*.log 2>/dev/null | wc -l)
echo "  failed attempts so far: $fails"
[ "$fails" -gt 0 ] && grep -h -E "FAIL|GIVE UP" logs/queue_gpu*.log | tail -5 | sed 's/^/    /'

echo; echo "=== my training jobs"
n=0
for pid in $(pgrep -u "$USER" -f recfreq.train); do
  out=$(tr '\0' ' ' < /proc/$pid/cmdline | sed -E 's/.*--out ([^ ]+).*/\1/')
  gpu=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep CUDA_VISIBLE_DEVICES | cut -d= -f2)
  step=$(tail -1 "$out/log.jsonl" 2>/dev/null | python3 -c 'import json,sys
try: d=json.loads(sys.stdin.read()); print(d.get("step","?"))
except Exception: print("starting")')
  printf "  GPU %s  step %-6s %s\n" "${gpu:-?}" "$step" "$(basename "$out")"
  n=$((n+1))
done
[ $n -eq 0 ] && echo "  none"

echo; echo "=== GPUs (memory used by me vs everyone)"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > /tmp/.st_gpu_$USER
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits > /tmp/.st_app_$USER
while IFS=', ' read -r idx uuid rest; do
  name=$(echo "$rest" | cut -d, -f1); util=$(echo "$rest" | awk -F', ' '{print $2}')
  used=$(echo "$rest" | awk -F', ' '{print $3}'); tot=$(echo "$rest" | awk -F', ' '{print $4}')
  mine=0; others=""
  while IFS=', ' read -r u pid mem; do
    [ "$u" = "$uuid" ] || continue
    who=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ "$who" = "$USER" ]; then mine=$((mine+mem)); else others="$others ${who:-?}"; fi
  done < /tmp/.st_app_$USER
  others=$(echo $others | tr ' ' '\n' | sort | uniq -c | awk '{printf "%s(%s) ", $2, $1}')
  printf "  GPU %s %-40s busy %3s%%  mem %6s/%s MiB  mine %6s MiB  others: %s\n" "$idx" "$name" "$util" "$used" "$tot" "$mine" "${others:-none}"
done < /tmp/.st_gpu_$USER
rm -f /tmp/.st_gpu_$USER /tmp/.st_app_$USER

echo; echo "=== my CPU / RAM"
ps -u "$USER" -o rss=,pcpu= | awk '{r+=$1; c+=$2} END {printf "  RAM %.1f GB, CPU %.0f%% (of %d cores x 100%%)\n", r/1048576, c, '"$(nproc)"'}'
