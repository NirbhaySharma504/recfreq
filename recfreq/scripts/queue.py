"""Run a grid of training jobs on one GPU with N in parallel. Resumable, retries failures.

python scripts/queue.py configs/E1.json [configs/E2a.json ...] --max-parallel 4
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expand(cfg):
    grid = cfg["grid"]
    names = list(grid)
    for combo in itertools.product(*[grid[n] for n in names]):
        args = {**cfg.get("base", {})}
        for n, v in zip(names, combo):
            if isinstance(v, dict):  # a grouped setting, e.g. {"h": 0.01, "eps": 0.1}
                args.update(v)
            else:
                args[n] = v
        rid = "{exp}_h{h}_e{eps}_L{layers}_{pe}_{m}".format(
            exp=cfg["exp"], h=args.get("h", 0), eps=args.get("eps", 0), layers=args.get("layers", 2),
            pe=args.get("pe", "alibi_learn"), m="mlp" if args.get("mlp", 0) else "attn")
        if args.get("typo", "structured") != "structured":
            rid += "_" + args["typo"]
        if args.get("conflict_frac", 0):
            rid += "_cf{}".format(args["conflict_frac"])
        if args.get("d_model", 128) != 128:
            rid += "_d{}".format(args["d_model"])
        if args.get("steps", 20000) != 20000:
            rid += "_st{}".format(args["steps"])
        if args.get("lr", 1e-3) != 1e-3:
            rid += "_lr{}".format(args["lr"])
        if args.get("batch", 128) != 128:
            rid += "_b{}".format(args["batch"])
        if args.get("V", 64) != 64:
            rid += "_V{}".format(args["V"])
        if args.get("K", 16) != 16:
            rid += "_K{}".format(args["K"])
        if args.get("n_pairs", 256) != 256:
            rid += "_N{}".format(args["n_pairs"])
        if args.get("key_dist", "zipf") != "zipf":
            rid += "_keys{}".format(args["key_dist"])
        rid += "_s{}".format(args.get("seed", 0))
        yield rid, args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--results", default=os.path.join(ROOT, "results", "runs"))
    a = p.parse_args()
    todo = []
    for path in a.configs:
        for rid, args in expand(json.load(open(path))):
            out = os.path.join(a.results, rid)
            if os.path.exists(os.path.join(out, "done.json")):
                continue
            todo.append((rid, args, out))
    print(f"{len(todo)} runs to do", flush=True)
    attempts = {rid: 0 for rid, _, _ in todo}
    running = {}
    while todo or running:
        while todo and len(running) < a.max_parallel:
            rid, args, out = todo.pop(0)
            os.makedirs(out, exist_ok=True)
            cmd = [sys.executable, "-m", "recfreq.train", "--out", out] + sum(
                [[f"--{k}", str(v)] for k, v in args.items()], [])
            logf = open(os.path.join(out, "stdout.log"), "a")
            running[rid] = (subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT), args, out, logf)
            attempts[rid] += 1
            print(time.strftime("%H:%M:%S"), "start", rid, f"(attempt {attempts[rid]})", flush=True)
        time.sleep(5)
        for rid in list(running):
            proc, args, out, logf = running[rid]
            if proc.poll() is None:
                continue
            logf.close()
            del running[rid]
            if proc.returncode == 0:
                print(time.strftime("%H:%M:%S"), "done ", rid, flush=True)
            elif attempts[rid] < a.retries:
                print(time.strftime("%H:%M:%S"), "FAIL ", rid, "rc", proc.returncode, "-> retry", flush=True)
                todo.append((rid, args, out))
            else:
                print(time.strftime("%H:%M:%S"), "GIVE UP", rid, flush=True)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
