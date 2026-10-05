"""
generate_induction_datasets.py

Synthetic dataset generator for testing the "multi-offset shift head" hypothesis
about induction heads in attention-only transformers (extending Elhage et al.,
2021, "A Mathematical Framework for Transformer Circuits").

Hypothesis under test
----------------------
Induction heads deeper than 2 layers are not built from a single previous-token
head at offset -1 alone, but from a family of "shift heads" at offsets
-1, -2, -3, ... that get K-composed together. This would let one induction
head match n-grams of varying length and stay robust to local noise.

What this script produces
--------------------------
Every generated sequence is a stream of random "filler" tokens with one or
more embedded *induction triggers*. A trigger of order n consists of:

    - a first occurrence of an n-token unit  [u_1, u_2, ..., u_{n-1}, u_n]
    - later, a second occurrence of the (n-1)-token context [u_1, ..., u_{n-1}]
      (optionally corrupted with noise), immediately followed by u_n again.

A model that has learned order-n induction should predict u_n as soon as it
sees the second occurrence of the context, i.e. at the token position right
before u_n is placed. That position (`query_pos`) and the position holding
the correct answer (`answer_pos`) are recorded in the metadata for every
trigger, so you can compute accuracy/loss exactly at the positions that
matter instead of averaging over the whole (mostly random-filler) sequence.

    order=2 ("bigram")   -> context length 1  -> needs only a -1 shift head
    order=3 ("trigram")  -> context length 2  -> needs -1 and -2 composed
    order=4 ("quadgram") -> context length 3  -> needs -1, -2 and -3 composed

Noise types (used for the skip / robustness experiments):
    None            -> clean context, no corruption
    "insertion"     -> one random token is inserted into the context, which
                        shifts the relative offsets of every token after it
                        by one position (tests whether the model can still
                        find the match using heads other than -1)
    "substitution"  -> one context token is replaced by a random different
                        token (tests robustness to a corrupted position
                        rather than a shifted position)

Three experiment phases (see README.md written alongside the data):
    1. isolated/    - pure bigram / trigram / quadgram datasets (single order,
                       no noise) to prove longer n-grams need more offsets.
    2. skip/        - quadgram clean vs. quadgram+insertion vs.
                       quadgram+substitution, for ablation robustness tests.
    3. mixed/       - sequences containing a random mix of orders and noise
                       types, for testing whether a single induction head
                       dynamically shifts its attended offset.

Usage
-----
    python generate_induction_datasets.py --outdir ./induction_data

Everything is pure numpy + stdlib. Token ids are plain integers in
[0, vocab_size); plug them directly into an nn.Embedding / TransformerLens
model. No tokenizer is involved on purpose, so the n-gram structure is exact.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Core event / sequence construction
# --------------------------------------------------------------------------- #

@dataclass
class InductionEvent:
    order: int                     # n-gram order (2, 3, 4, ...)
    first_pos: int                 # start index of the first occurrence of the unit
    second_pos: int                # start index of the (possibly noisy) second occurrence
    query_pos: int                 # position whose next-token logit should predict the answer
    answer_pos: int                # position holding the correct token (== query_pos + 1)
    is_noisy: bool
    noise_type: Optional[str]      # None | "insertion" | "substitution"
    noise_pos: Optional[int]       # absolute sequence position of the injected/substituted token
    unit: List[int]                # the original clean n-gram tokens [u_1, ..., u_n]


def _filler(rng: np.random.Generator, vocab_size: int, n: int) -> List[int]:
    return [int(x) for x in rng.integers(0, vocab_size, size=n)]


def build_sequence(
    rng: np.random.Generator,
    seq_len: int,
    vocab_size: int,
    trigger_specs: Sequence[dict],
    min_gap: int = 5,
    max_extra_gap: int = 10,
    initial_filler_range: Tuple[int, int] = (3, 10),
    inter_filler_range: Tuple[int, int] = (3, 15),
) -> Tuple[List[int], List[InductionEvent]]:
    """Build one sequence of exactly `seq_len` tokens containing the requested
    triggers (in order). If there isn't enough room left for a trigger, it is
    silently dropped (so short seq_len / many triggers just yields fewer
    realized triggers -- check the returned events list, don't assume len ==
    len(trigger_specs))."""

    tokens: List[int] = []
    events: List[InductionEvent] = []

    remaining = seq_len
    f0 = min(int(rng.integers(*initial_filler_range)), remaining)
    tokens.extend(_filler(rng, vocab_size, f0))
    remaining = seq_len - len(tokens)

    for spec in trigger_specs:
        n = spec["order"]
        noise_type = spec.get("noise_type")
        assert noise_type in (None, "insertion", "substitution")
        assert n >= 2

        second_len = n if noise_type in (None, "substitution") else n + 1
        gap_len = int(rng.integers(min_gap, min_gap + max_extra_gap))
        block_len = n + gap_len + second_len
        # leave a little slack so later fillers/triggers aren't totally starved
        if block_len + 5 > remaining:
            break

        # n distinct tokens so the "context" is unambiguous
        unit = [int(x) for x in rng.choice(vocab_size, size=n, replace=False)]

        first_pos = len(tokens)
        tokens.extend(unit)

        gap_tokens = _filler(rng, vocab_size, gap_len)
        tokens.extend(gap_tokens)

        second_pos = len(tokens)
        context = unit[:-1].copy()
        noise_pos_abs = None

        if noise_type == "insertion":
            ins_idx = int(rng.integers(0, len(context) + 1))
            noise_tok = int(rng.integers(0, vocab_size))
            context = context[:ins_idx] + [noise_tok] + context[ins_idx:]
            noise_pos_abs = second_pos + ins_idx
        elif noise_type == "substitution":
            sub_idx = int(rng.integers(0, len(context)))
            orig = context[sub_idx]
            noise_tok = orig
            while noise_tok == orig:
                noise_tok = int(rng.integers(0, vocab_size))
            context[sub_idx] = noise_tok
            noise_pos_abs = second_pos + sub_idx

        tokens.extend(context)
        answer_pos = len(tokens)
        answer_tok = unit[-1]
        tokens.append(answer_tok)
        query_pos = answer_pos - 1

        events.append(
            InductionEvent(
                order=n,
                first_pos=first_pos,
                second_pos=second_pos,
                query_pos=query_pos,
                answer_pos=answer_pos,
                is_noisy=noise_type is not None,
                noise_type=noise_type,
                noise_pos=noise_pos_abs,
                unit=unit,
            )
        )

        remaining = seq_len - len(tokens)
        fi = min(int(rng.integers(*inter_filler_range)), remaining)
        tokens.extend(_filler(rng, vocab_size, fi))
        remaining = seq_len - len(tokens)

    # pad the tail with filler to hit exactly seq_len
    remaining = seq_len - len(tokens)
    if remaining > 0:
        tokens.extend(_filler(rng, vocab_size, remaining))
    tokens = tokens[:seq_len]
    return tokens, events


# --------------------------------------------------------------------------- #
# Dataset-level generation
# --------------------------------------------------------------------------- #

def generate_dataset(
    n_sequences: int,
    seq_len: int,
    vocab_size: int,
    order_choices: Sequence[int],
    noise_type_choices: Sequence[Optional[str]],
    noise_type_probs: Sequence[float],
    triggers_per_seq: Tuple[int, int] = (3, 6),
    seed: int = 0,
) -> Tuple[np.ndarray, List[List[InductionEvent]]]:
    """Generate a full dataset. Each sequence independently samples its own
    number of triggers and, per trigger, an order from `order_choices` and a
    noise type from `noise_type_choices` (weighted by `noise_type_probs`)."""

    rng = np.random.default_rng(seed)
    all_tokens = np.zeros((n_sequences, seq_len), dtype=np.int32)
    all_events: List[List[InductionEvent]] = []

    for i in range(n_sequences):
        n_triggers = int(rng.integers(triggers_per_seq[0], triggers_per_seq[1] + 1))
        trigger_specs = []
        for _ in range(n_triggers):
            order = int(rng.choice(order_choices))
            noise_type = rng.choice(noise_type_choices, p=noise_type_probs)
            noise_type = None if noise_type == "None" or noise_type is None else str(noise_type)
            trigger_specs.append({"order": order, "noise_type": noise_type})

        tokens, events = build_sequence(rng, seq_len, vocab_size, trigger_specs)
        all_tokens[i] = tokens
        all_events.append(events)

    return all_tokens, all_events


# --------------------------------------------------------------------------- #
# Saving / loading
# --------------------------------------------------------------------------- #

def save_dataset(path_prefix: str, tokens: np.ndarray, events: List[List[InductionEvent]], config: dict) -> None:
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    np.save(path_prefix + "_tokens.npy", tokens)
    with open(path_prefix + "_events.jsonl", "w") as f:
        for seq_id, seq_events in enumerate(events):
            f.write(json.dumps({"seq_id": seq_id, "events": [asdict(e) for e in seq_events]}) + "\n")
    with open(path_prefix + "_config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_dataset(path_prefix: str) -> Tuple[np.ndarray, List[List[dict]], dict]:
    tokens = np.load(path_prefix + "_tokens.npy")
    events: List[List[dict]] = []
    with open(path_prefix + "_events.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            events.append(rec["events"])
    with open(path_prefix + "_config.json") as f:
        config = json.load(f)
    return tokens, events, config


# --------------------------------------------------------------------------- #
# Analysis helpers (for the ablation / accuracy experiments described in the setup)
# --------------------------------------------------------------------------- #

def induction_accuracy(
    predicted_tokens: np.ndarray,
    tokens: np.ndarray,
    events: List[List[dict]],
    order: Optional[int] = None,
    noise_type: Optional[str] = "__any__",
) -> float:
    """Compute induction accuracy at the recorded query positions only.

    predicted_tokens: (n_sequences, seq_len) array of argmax model
        predictions (predicted_tokens[i, t] = model's guess for the token at
        position t+1, given tokens[i, :t+1]).
    order: filter to a specific n-gram order, or None for all orders.
    noise_type: filter to "__any__" (default, no filtering), None (clean
        triggers only), or "insertion"/"substitution".
    """
    correct = 0
    total = 0
    for i, seq_events in enumerate(events):
        for ev in seq_events:
            if order is not None and ev["order"] != order:
                continue
            if noise_type != "__any__" and ev["noise_type"] != noise_type:
                continue
            q = ev["query_pos"]
            a_pos = ev["answer_pos"]
            total += 1
            if int(predicted_tokens[i, q]) == int(tokens[i, a_pos]):
                correct += 1
    return correct / total if total > 0 else float("nan")


def decode_and_print_examples(
    tokens: np.ndarray,
    events: List[List[InductionEvent]],
    n_examples: int = 3,
) -> None:
    for i in range(min(n_examples, len(tokens))):
        seq = tokens[i]
        seq_events = events[i]
        marks = {}
        for ev in seq_events:
            n = ev.order
            for k in range(n):
                marks.setdefault(ev.first_pos + k, []).append(f"U{n}")
            ctx_len = n - 1 + (1 if ev.noise_type == "insertion" else 0)
            for k in range(ctx_len):
                marks.setdefault(ev.second_pos + k, []).append(f"C{n}")
            marks.setdefault(ev.answer_pos, []).append(f"A{n}!")
            if ev.noise_pos is not None:
                marks.setdefault(ev.noise_pos, []).append("N")
        parts = []
        for t, tok in enumerate(seq):
            tag = "".join(marks.get(t, []))
            parts.append(f"{tok}{('[' + tag + ']') if tag else ''}")
        print(f"--- sequence {i} ({len(seq_events)} triggers) ---")
        print(" ".join(parts))
        for ev in seq_events:
            print(
                f"    order={ev.order} noise={ev.noise_type} "
                f"first={ev.first_pos} second={ev.second_pos} "
                f"query={ev.query_pos} answer={ev.answer_pos} unit={ev.unit}"
            )
        print()


# --------------------------------------------------------------------------- #
# Main: build the three experiment phases
# --------------------------------------------------------------------------- #

def _make_split(name: str, outdir: str, gen_kwargs: dict, config_extra: dict, seed: int, n_total: int, train_frac: float, seq_len: int, vocab_size: int):
    n_train = int(n_total * train_frac)
    n_val = n_total - n_train

    tokens_tr, events_tr = generate_dataset(n_sequences=n_train, seed=seed, **gen_kwargs)
    tokens_va, events_va = generate_dataset(n_sequences=n_val, seed=seed + 1, **gen_kwargs)

    cfg = {
        "name": name,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "n_train": n_train,
        "n_val": n_val,
        **{k: v for k, v in gen_kwargs.items() if k != "n_sequences"},
        **config_extra,
    }
    save_dataset(os.path.join(outdir, name, name + "_train"), tokens_tr, events_tr, cfg)
    save_dataset(os.path.join(outdir, name, name + "_val"), tokens_va, events_va, cfg)

    n_events_tr = sum(len(e) for e in events_tr)
    print(f"[{name}] train: {n_train} seqs / {n_events_tr} triggers   "
          f"val: {n_val} seqs / {sum(len(e) for e in events_va)} triggers")
    return tokens_tr, events_tr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=str, default="./induction_data")
    p.add_argument("--vocab_size", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--n_isolated", type=int, default=4000, help="sequences per isolated-length dataset (bigram/trigram/quadgram)")
    p.add_argument("--n_skip", type=int, default=4000, help="sequences per skip/noise dataset")
    p.add_argument("--n_mixed", type=int, default=8000, help="sequences in the mixed/multiplexed dataset")
    p.add_argument("--triggers_min", type=int, default=3)
    p.add_argument("--triggers_max", type=int, default=6)
    p.add_argument("--print_examples", type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    trig_range = (args.triggers_min, args.triggers_max)

    print("=== Phase 1: isolated length tests (bigram / trigram / quadgram) ===")
    last_example = None
    for order, name in [(2, "isolated_bigram"), (3, "isolated_trigram"), (4, "isolated_quadgram")]:
        tokens, events = _make_split(
            name, args.outdir,
            gen_kwargs=dict(
                seq_len=args.seq_len, vocab_size=args.vocab_size,
                order_choices=[order], noise_type_choices=[None], noise_type_probs=[1.0],
                triggers_per_seq=trig_range,
            ),
            config_extra={"phase": 1, "order": order, "noise_type": None},
            seed=args.seed + order, n_total=args.n_isolated, train_frac=args.train_frac,
            seq_len=args.seq_len, vocab_size=args.vocab_size,
        )
        last_example = (tokens, events)

    print("\n=== Phase 2: isolated skip variants (quadgram clean / insertion / substitution) ===")
    for noise_type, name in [(None, "skip_quadgram_clean"),
                              ("insertion", "skip_quadgram_insertion"),
                              ("substitution", "skip_quadgram_substitution")]:
        _make_split(
            name, args.outdir,
            gen_kwargs=dict(
                seq_len=args.seq_len, vocab_size=args.vocab_size,
                order_choices=[4], noise_type_choices=[noise_type], noise_type_probs=[1.0],
                triggers_per_seq=trig_range,
            ),
            config_extra={"phase": 2, "order": 4, "noise_type": noise_type},
            seed=args.seed + 100 + hash(str(noise_type)) % 1000,
            n_total=args.n_skip, train_frac=args.train_frac,
            seq_len=args.seq_len, vocab_size=args.vocab_size,
        )

    print("\n=== Phase 3: mixed / multiplexed dataset (random order + random noise per trigger) ===")
    tokens_mix, events_mix = _make_split(
        "mixed", args.outdir,
        gen_kwargs=dict(
            seq_len=args.seq_len, vocab_size=args.vocab_size,
            order_choices=[2, 3, 4],
            noise_type_choices=[None, "insertion", "substitution"],
            noise_type_probs=[0.5, 0.25, 0.25],
            triggers_per_seq=(max(4, trig_range[0]), max(8, trig_range[1])),
        ),
        config_extra={"phase": 3},
        seed=args.seed + 500, n_total=args.n_mixed, train_frac=args.train_frac,
        seq_len=args.seq_len, vocab_size=args.vocab_size,
    )

    if args.print_examples > 0:
        print("\n=== Example sequences (mixed dataset) ===")
        print("Legend: U{n}=first occurrence of an order-n unit, C{n}=second-occurrence context,")
        print("        N=injected/substituted noise token, A{n}!=answer token (the induction target)\n")
        decode_and_print_examples(tokens_mix, events_mix, n_examples=args.print_examples)

    print(f"\nAll datasets written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
