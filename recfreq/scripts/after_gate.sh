#!/bin/bash
# wait for the E0 queue, require both gate runs to pass, then run E1
cd ~/recfreq
while ! grep -q "queue finished" results/queue_E0.log; do sleep 30; done
ok=$(.venv/bin/python -c "import json,glob; r=[json.load(open(f))[\"gate_pass\"] for f in glob.glob(\"results/runs/E0_*/done.json\")]; print(int(len(r)==2 and all(r)))")
if [ "$ok" != "1" ]; then echo "GATE FAILED, not starting E1" | tee -a results/queue_E1.log; exit 1; fi
echo "gate passed, starting E1" | tee -a results/queue_E1.log
.venv/bin/python scripts/queue.py configs/E1.json --max-parallel 4 2>&1 | tee -a results/queue_E1.log
.venv/bin/python -m recfreq.analyze results/runs --exp E1 --out figs > results/analysis_E1.txt 2>&1
