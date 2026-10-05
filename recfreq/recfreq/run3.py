"""Run-3 evaluation on trained checkpoints (no training): causal ablations, slope surgery,
new probe layouts, and per-copy attention for the content-vs-distance test.

Per run it writes <out>/<run>.surf.parquet (probe log-odds per condition, layout and cell),
<out>/<run>.attn.parquet (copy heads' attention to each copy) and <out>/<run>.json (loss per condition).

python -m recfreq.run3 results/runs --exp E1,E2a,E2b,E2c --mech figs_mech --out figs_run3
"""
from __future__ import annotations

import argparse
import copy
import glob
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import first_occurrence, sample_batch
from .mech import load_model
from .probes import GAPS, NC, NW, key_predictive, probe_batch
from .train import value_logits

LAYOUTS = {"sp4": (4, 1), "sp2": (2, 1), "sp8": (8, 1), "swap4": (4, -1)}  # name -> (spacing, w_shift)


@torch.no_grad()
def eval_ce(model, task, tokens, vals, repeat, head_masks=None, chunk=64):
    ces = []
    for i in range(0, tokens.shape[0], chunk):
        lg = value_logits(task, model(tokens[i:i + chunk], head_masks=head_masks)).float()
        ces.append(F.cross_entropy(lg.transpose(1, 2), vals[i:i + chunk], reduction="none"))
    ce = torch.cat(ces)
    rep = repeat.float()
    return {"ce": ce.mean().item(), "ce_rep": ((ce * rep).sum() / rep.sum()).item()}


@torch.no_grad()
def surface(model, task, layout: str, fills: int, seed: int, head_masks=None, attn_heads=(), bayes=False):
    """Mean model log p(c)/p(w) per probe cell for a layout; optionally exact Bayes and copy-head attention."""
    sp, ws = LAYOUTS[layout]
    g = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    dev = next(model.parameters()).device
    qp = task.n_pairs - 1
    qi, lo_k = 1 + 2 * qp, 1 + task.K
    ar = torch.arange(fills, device=dev)
    rows, att = [], []
    for n_c, n_w, G in itertools.product(NC, NW, GAPS):
        tok, keys, vals, c, w = probe_batch(task, n_c, n_w, G, fills, g, dev, sp=sp, w_shift=ws)
        need_attn = len(attn_heads) > 0
        out = model(tok, return_attn=need_attn, head_masks=head_masks)
        logits, attns = (out if need_attn else (out, None))
        lp = logits[:, qi, lo_k:lo_k + task.V].float().log_softmax(-1)
        lo = lp[ar, c] - lp[ar, w]
        rec = {"layout": layout, "n_c": n_c, "n_w": n_w, "G": G, "lo": lo.mean().item(), "lo_sd": lo.std().item()}
        w_pos = [qp - sp * j for j in range(1, n_w + 1)]
        c_pos = [qp - sp * n_w - G - sp * i for i in range(n_c)]
        if bayes:
            pos = sorted(c_pos + w_pos)
            pred = key_predictive(task, pos, vals[:, pos], qp)
            rec["lo_bayes"] = (pred[ar, c].log() - pred[ar, w].log()).mean().item()
        rows.append(rec)
        for (l, h) in attn_heads:
            a = attns[l][:16, h, qi]
            for p_list, recent in ((c_pos, 0), (w_pos, 1)):
                variant = int(recent == 1) if ws == 1 else int(recent == 0)
                for p in p_list:
                    for b, x in enumerate(a[:, 2 + 2 * p].tolist()):
                        att.append({"layout": layout, "head": f"L{l}H{h}", "n_c": n_c, "n_w": n_w, "G": G, "fill": b,
                                    "d": qp - p, "recent": recent, "variant": variant, "a": x})
    return pd.DataFrame(rows), pd.DataFrame(att)


def masks_for(model, heads):
    H, Ls = model.cfg.n_heads, model.cfg.n_layers
    m = {l: torch.ones(H, device=next(model.parameters()).device) for l in range(Ls)}
    for l, h in heads:
        m[l][h] = 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--exp", default="E1,E2a,E2b,E2c")
    ap.add_argument("--mech", default="figs_mech")
    ap.add_argument("--out", default="figs_run3")
    ap.add_argument("--fills", type=int, default=32, help="fills per cell for the intervention surfaces")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    heads_tab = pd.read_csv(os.path.join(a.mech, "mech_heads.csv"))
    dirs = sorted(d for e in a.exp.split(",") for d in glob.glob(os.path.join(a.runs_dir, f"{e}*")))
    for d in dirs:
        name = os.path.basename(d)
        cfgj = json.load(open(os.path.join(d, "config.json")))
        if cfgj["h"] == 0 or cfgj["eps"] == 0 or os.path.exists(os.path.join(a.out, f"{name}.json")):
            continue
        hr = heads_tab[heads_tab.run == name].sort_values("share", ascending=False)
        if hr.empty:
            continue
        copy_h = hr[hr.share >= 0.1]
        if copy_h.empty:  # MLP models: use the two heads with the largest share
            copy_h = hr.head(2)
        copy_heads = [(int(t[1]), int(t[3])) for t in copy_h["head"]]
        rest = hr[~hr["head"].isin(copy_h["head"])].assign(absshare=lambda x: x.share.abs()).sort_values("absshare")
        control = [(int(t[1]), int(t[3])) for t in rest["head"].head(len(copy_heads))]
        model, task = load_model(d, dev)
        g = torch.Generator(device=dev).manual_seed(4321)
        ev = sample_batch(task, 256, g, dev)
        rep = ~first_occurrence(ev["keys"], task.K)
        late = [(l, h) for l in range(1, model.cfg.n_layers) for h in range(model.cfg.n_heads)]
        # pre-registered: copy heads vs control heads; post-hoc: every late head alone, and all late heads
        light = False  # every model gets the full set of interventions
        conds = {"base": None, "ablate_copy": copy_heads, "ablate_all_late": late}
        if not light:
            conds["ablate_control"] = control
            for l, h in late:
                conds[f"ablate_L{l}H{h}"] = [(l, h)]
        surfs, atts, losses = [], [], {}
        for cname, heads in conds.items():
            hm = None if heads is None else masks_for(model, heads)
            s, _ = surface(model, task, "sp4", a.fills, 7, head_masks=hm)
            surfs.append(s.assign(cond=cname))
            losses[cname] = eval_ce(model, task, ev["tokens"], ev["vals"], rep, head_masks=hm)
        if model.cfg.pe == "alibi_learn":
            for tag, hs in ((("copy", copy_heads),) if light else (("copy", copy_heads), ("late", late))):
                for scale in (0.5, 2.0):
                    m2 = copy.deepcopy(model)
                    for l, h in hs:
                        m2.blocks[l].attn.log_slope.data[h] += math.log(scale)
                    s, _ = surface(m2, task, "sp4", a.fills, 7)
                    cname = f"slope_{tag}_x{scale}"
                    surfs.append(s.assign(cond=cname))
                    losses[cname] = eval_ce(m2, task, ev["tokens"], ev["vals"], rep)
                    del m2
        for lay in ("sp2", "sp8", "swap4"):
            s, att = surface(model, task, lay, 64, 11, bayes=True, attn_heads=late if lay == "swap4" else ())
            surfs.append(s.assign(cond="base"))
            if len(att):
                atts.append(att)
        _, att = surface(model, task, "sp4", 16, 11, attn_heads=late)
        atts.append(att)
        pd.concat(surfs).to_parquet(os.path.join(a.out, f"{name}.surf.parquet"))
        pd.concat(atts).to_parquet(os.path.join(a.out, f"{name}.attn.parquet"))
        json.dump({"run": name, "copy_heads": copy_heads, "control_heads": control, "losses": losses,
                   **{k: cfgj[k] for k in ("h", "eps", "layers", "mlp", "pe", "seed")},
                   "conflict_frac": cfgj.get("conflict_frac", 0)}, open(os.path.join(a.out, f"{name}.json"), "w"))
        b = losses["base"]["ce_rep"]
        print(f"{name:<50} copy {copy_heads} ctrl {control}  repeat-CE base {b:.3f} -> ablate copy "
              f"{losses['ablate_copy']['ce_rep']:.3f}, all late heads {losses['ablate_all_late']['ce_rep']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
