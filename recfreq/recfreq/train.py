"""Train one model on switch-typo KV data. Resumable; logs JSONL; runs probes at the end.

python -m recfreq.train --h 0.01 --eps 0.1 --layers 2 --pe alibi_learn --seed 0 --out results/runs/<id>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import TaskConfig, first_occurrence, sample_batch
from .model import ModelConfig, Transformer
from .oracle import bayes_metrics, bayes_predictive
from .probes import conflict_batch

EVAL_SEED = 1234


def value_logits(cfg: TaskConfig, logits: torch.Tensor) -> torch.Tensor:
    """Logits at key positions restricted to value tokens: (B, N, V)."""
    return logits[:, 1::2, 1 + cfg.K : 1 + cfg.K + cfg.V]


def lr_at(step, total, warmup, lr, min_frac=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def make_eval_set(task: TaskConfig, B: int, device) -> dict:
    g = torch.Generator(device=device).manual_seed(EVAL_SEED)
    batch = sample_batch(task, B, g, device)
    batch["bayes"] = bayes_predictive(task, batch["keys"], batch["vals"])
    batch["repeat"] = ~first_occurrence(batch["keys"], task.K)
    batch["bayes_all"] = bayes_metrics(batch["bayes"], batch["vals"])
    batch["bayes_rep"] = bayes_metrics(batch["bayes"], batch["vals"], batch["repeat"])
    return batch


@torch.no_grad()
def evaluate(model, task: TaskConfig, ev: dict, chunk: int = 128) -> dict:
    model.eval()
    ces, accs = [], []
    for i in range(0, ev["tokens"].shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ev["tokens"].is_cuda):
            lg = value_logits(task, model(ev["tokens"][i : i + chunk])).float()
        v = ev["vals"][i : i + chunk]
        ces.append(F.cross_entropy(lg.transpose(1, 2), v, reduction="none"))
        accs.append((lg.argmax(-1) == v).float())
    model.train()
    ce, acc, rep = torch.cat(ces), torch.cat(accs), ev["repeat"].float()
    out = {
        "ce": ce.mean().item(),
        "acc": acc.mean().item(),
        "ce_rep": (ce * rep).sum().item() / rep.sum().item(),
        "acc_rep": (acc * rep).sum().item() / rep.sum().item(),
        "bayes_ce": ev["bayes_all"]["ce"],
        "bayes_acc": ev["bayes_all"]["acc"],
        "bayes_acc_rep": ev["bayes_rep"]["acc"],
    }
    out["excess"] = out["ce"] - out["bayes_ce"]
    return out


@torch.no_grad()
def induction_heads(model, task: TaskConfig, ev: dict, n: int = 64) -> list[dict]:
    """Per head: attention mass from each key position to same-key value positions
    (positions whose previous token is the current key), averaged over repeat queries."""
    model.eval()
    tok = ev["tokens"][:n]
    _, attns = model(tok, return_attn=True)
    model.train()
    keys = ev["keys"][:n]
    N = keys.shape[1]
    qpos = 1 + 2 * torch.arange(N, device=tok.device)
    vpos = 2 + 2 * torch.arange(N, device=tok.device)
    same = (keys[:, :, None] == keys[:, None, :]) & (
        torch.arange(N, device=tok.device)[None, :, None] > torch.arange(N, device=tok.device)[None, None, :]
    )  # (B, query pair, earlier pair)
    rep = ev["repeat"][:n].float()
    slopes = model.slopes()
    rows = []
    for l, a in enumerate(attns):
        a = a.float()[:, :, qpos][:, :, :, vpos]  # (B, H, Nq, Nv)
        mass = (a * same[:, None].float()).sum(-1)  # (B, H, Nq)
        score = (mass * rep[:, None]).sum((0, 2)) / rep.sum()
        for hd in range(a.shape[1]):
            rows.append({"layer": l, "head": hd, "ind_score": score[hd].item(),
                         "slope": None if slopes is None else slopes[l][hd]})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--typo", default="structured")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--V", type=int, default=64)
    p.add_argument("--n_pairs", type=int, default=256)
    p.add_argument("--key_dist", default="zipf")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mlp", type=int, default=0)
    p.add_argument("--pe", default="alibi_learn")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batch", type=int, default=512)
    p.add_argument("--probes", type=int, default=1)
    p.add_argument("--probe_fills", type=int, default=256)
    p.add_argument("--gate", type=int, default=0, help="enforce the h=eps=0 gate thresholds at the end")
    p.add_argument("--conflict_frac", type=float, default=0.0,
                   help="loss weight on extra probe-shaped conflict sequences (query sampled from Bayes)")
    p.add_argument("--conflict_batch", type=int, default=32)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    done_path = os.path.join(args.out, "done.json")
    if os.path.exists(done_path):
        print("already done:", args.out)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    task = TaskConfig(K=args.K, V=args.V, n_pairs=args.n_pairs, h=args.h, eps=args.eps,
                      typo=args.typo, key_dist=args.key_dist)
    mcfg = ModelConfig(vocab=task.vocab, max_len=task.seq_len, d_model=args.d_model, n_heads=args.heads,
                       n_layers=args.layers, mlp=bool(args.mlp), pe=args.pe)
    model = Transformer(mcfg).to(device)
    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    no_decay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    g = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ev = make_eval_set(task, args.eval_batch, device)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log_path = os.path.join(args.out, "log.jsonl")
    step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        g.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        step = ck["step"]
        print(f"resumed at step {step}")
    else:
        m0 = evaluate(model, task, ev)
        print("init", m0)
        if abs(m0["ce"] - math.log(task.V)) > 0.3:
            raise SystemExit(f"initial loss {m0['ce']:.3f} not within 0.3 of ln V = {math.log(task.V):.3f}")
        with open(log_path, "w") as f:
            f.write(json.dumps({"step": 0, **m0, "slopes": model.slopes()}) + "\n")

    t0 = time.time()
    while step < args.steps:
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, args.steps, args.warmup, args.lr)
        batch = sample_batch(task, args.batch, g, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = value_logits(task, model(batch["tokens"]))
        loss = F.cross_entropy(lg.float().transpose(1, 2), batch["vals"])
        if args.conflict_frac > 0:
            cb = conflict_batch(task, args.conflict_batch, g, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lq = value_logits(task, model(cb["tokens"]))[:, -1].float()
            loss_c = F.cross_entropy(lq, cb["vals"][:, -1])
            loss = (1 - args.conflict_frac) * loss + args.conflict_frac * loss_c
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, task, ev)
            rec = {"step": step, "train_loss": loss.item(), **m, "slopes": model.slopes(),
                   "sec": round(time.time() - t0, 1)}
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "slopes"}),
                  flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(), "step": step,
                        "mcfg": mcfg.__dict__, "task": task.__dict__}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    final = evaluate(model, task, ev)
    heads = induction_heads(model, task, ev)
    summary = {"final": final, "heads": heads, "args": vars(args)}
    if args.gate:
        summary["gate_pass"] = bool(final["acc_rep"] >= 0.99 and final["excess"] <= 0.02)
        print("GATE", "PASS" if summary["gate_pass"] else "FAIL", final)
    if args.probes:
        from .probes import run_probes
        df = run_probes(model, task, device, fills=args.probe_fills)
        df.to_parquet(os.path.join(args.out, "probes.parquet"))
    with open(done_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("done", args.out)


if __name__ == "__main__":
    main()
