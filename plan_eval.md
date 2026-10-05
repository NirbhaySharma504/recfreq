# Plan: full verification audit of the recency-vs-frequency results

## Context

Over 5 runs we trained 123 models (2 gate + 36 + 69 + 16) and ran many analyses (rule fits, decomposition, ablations, slope surgery, path ablation). Several earlier claims had to be corrected along the way (counter-sum explanation, "conflicts are rare", the P1 counter form, test 1's control). Before writing the workshop paper, we want one systematic pass that checks:

1. the code is right,
2. the experiments were done right,
3. the results are what they should be and the reported numbers are reproducible,
4. the results support the paper's claims, with statistics and alternatives ruled out,
5. there is a concrete, predictive formulation, even if only in this controlled setting,

plus verification of every citation and the novelty claim.

**Outcome:** a verification report (new artifact page), a claims registry that recomputes every headline number from raw files, corrections to the main report wherever a number or claim fails, and a final "formulation" section ready for the paper.

All work lives in a new `recfreq/verify/` module plus new tests. New GPU checks get predictions written into the plan (§17) **before** they run, as in runs 3–5.

---

## Phase 0 · Freeze and inventory (CPU, ~15 min)

- Copy all result summaries (`results/runs/*/{config,done}.json`, `probes.parquet`, `figs_*/*.csv`) into a dated snapshot folder on the server; record file hashes in a `MANIFEST.csv`.
- Confirm: 123 `done.json`, all configs V=64, K=16, n_pairs=256, lr=1e-3, batch=128 (already checked: true), no NaN in any `log.jsonl`.

## Phase 1 · Made right: code correctness (CPU, ~1.5 h)

**1.1 Re-run the 20 existing tests, then add tests** (`tests/`):
- `test_calibration.py`: the generator and the oracle agree. Sample 20k sequences per (h, ε) ∈ {(0.01, 0.1), (0.03, 0.2), (0, 0.2) uniform}; bin oracle predictive probabilities; empirical frequency must match within 2 SE (reliability check, ECE < 0.01); mean −log p* must equal the oracle CE within MC error. (The existing brute-force test only checks the oracle math against itself.)
- `test_recovery.py`: `layouts.BayesFree` recovers known (h′, ε′) from synthetic Bayes surfaces with probe-level noise (sd = median `lo_model_sd/√fills`), across the grid, including h = 0.003 where boundaries fall outside the probe range; report recovery error and the h′–ε′ trade-off ridge. Same for `layouts.fit_counter` / `fit_counter2` on known counters.
- `test_probes_layouts.py`: sp2/sp8/swap4 layouts: tau absent from background, copy positions and values as specified, query value never visible, Lay distances equal `probe_batch` positions.
- `test_model_pe.py`: ALiBi bias sign and per-token distance; RoPE score depends only on i−j (shift invariance); learned-abs and RoPE paths equal SDPA.
- `test_mech_paths.py`: promote the inline checks (DLA reconstruction < 1e-3, `paths.forward_paths` == model, `run5.decompose_scores` reproduces attention) to unit tests on 2 checkpoints.

**1.2 Line-by-line review checklist** of `data.py`, `oracle.py`, `train.py` (value-position indexing `logits[:, 1::2]`, conflict-enrichment weighting), `probes.py`, `fit_rules.py`/`layouts.py` (CV masks, refinement acceptance, bounds), `analyze.py` (`boundary` interpolation), `analyze3.py`, `mech.py`, `paths.py`, `run5.py`. Record each finding.

**1.3 Precision check:** stored probes were computed under bf16 autocast, run-3 surfaces in fp32. Recompute the stored surfaces in fp32 for 12 models (2 per architecture); quantify the change in fitted h′, ε′ and R².

## Phase 2 · Done right: procedure (CPU + GPU)

**2.1 Run integrity (CPU):** convergence: excess loss at 15k vs 20k steps for every run (flag runs still improving > 10%); gate pass; seeds distinct; identical eval set per (h, ε).

**2.2 Probe out-of-distribution check (GPU eval, ~15 min):** for 20 models, compare loss on the *background* positions of probe sequences with ordinary-data loss, and the copy heads' total attention mass on probes vs ordinary data. Probes must not be OOD.

**2.3 Reproducibility (GPU, ~45 min):** retrain 4 models with their original seeds (E1 L2 ALiBi C s0, D s0; E2a L2 RoPE C s0; E1 L3 C s0) and 2 configs with 2 new seeds each (L2 ALiBi C, L2 RoPE C, seeds 3–4). Compare excess loss, probe surface correlation, h′, ε′, Bayes-vs-counter gap.

**2.4 Hyperparameter sweep (GPU, ~2.5–3 h, 6 jobs in parallel now that the GPU memory is free):** points C and D × {L2 ALiBi, L2 RoPE} × {lr 3e-4, lr 3e-3, batch 256 at lr 1e-3} × 2 seeds = 24 runs. First extend `scripts/queue.py` run IDs with `_lr…` / `_b…` so they don't collide. Check h′/h, Bayes-vs-counter gap, cliff ratio.

**2.5 Pre-registration audit (CPU):** one table of every pre-registered prediction (runs 3–5) vs verdict, confirming each threshold was applied exactly as written; list every post-hoc analysis and every deviation (NoPE→RoPE swap before launch, light mode reverted, test 1 control flaw, P1 counter form).

**2.6 Train/test separation audit:** confirm fit vs test data never overlap (stored probes seed 777 vs run-3 surfaces seed 11; note the B2 swap4 partial overlap).

## Phase 3 · Results are what we expect (CPU, ~1.5 h)

**3.1 Claims registry** (`verify/registry.py`): one row per headline number in the report: id, reported value, source files, an independent recompute function written fresh (not reusing the analysis script that produced it). Output a diff table; every mismatch beyond rounding is fixed in the report.

**3.2 Theory checks:** Prop. 2 boundary vs exact oracle over a wide (h, ε, n_w) grid, mapping where the small-hG formula is accurate; n_c-independence; per-copy weight = log((1−ε)/ε).

**3.3 Internal consistency:** the 0.31 per-copy law vs log((1−ε′)/ε′) from the fitted ε′; cliff ratio vs h′/h; slope-surgery h′ changes vs the cross-model relation between learned slope and h′.

**3.4 Fit robustness:** refit h′, ε′ with a second optimizer and a finer grid; bootstrap CIs from cell noise (`lo_model_sd`) and across seeds; check every fitted scale b > 0; profile-likelihood plots for 4 representative models.

## Phase 4 · Do the results support the claims? (CPU, ~1.5 h)

**4.1 Final claim list C1–C7** with exact wording, e.g. C1 Prop. 2 is exact; C2 models = Bayes with wrong (h′, ε′); C3 not generic (null models); C4 generalizes to unseen layouts; C5 slope surgery causal; C6 mechanism Σγ_h(A_c−A_w); C7 miscalibration cheap and persistent with training. Each with: pre-registered or post-hoc, effect size, CI, n, failure cases.

**4.2 Statistics:** seed-blocked bootstrap CIs; sign or binomial tests for monotonicity and win rates; paired held-out R² differences; AIC/BIC for families with different parameter counts; a multiple-comparison note.

**4.3 Null models and alternatives (pre-registered in §17):**
- Shuffled-cell surfaces.
- Generic smooth monotone surfaces: random logistic-in-log G with linear n_w and log n_c terms plus probe noise. Bayes(h′, ε′) must fit these clearly worse than it fits the models.
- Random counter sums: done; re-verify.
- Probe OOD: 2.2. Under-training: E3 + 2.4. Positional encoding: E2a. Hyperparameters: 2.4.

**4.4 Final hypothesis scorecard:** original (a)(b)(c), RQ1–RQ3, all run 3–5 predictions.

## Phase 5 · Concrete formulation (CPU + GPU eval)

**5.1 Predictive law, leave-one-setting-out (CPU):** fit h′(h) = α·h^β and logit ε′ = γ + δ·logit ε (plus median scale/offset) on 11 of the 12 grid settings; predict the held-out setting's full probe surface from its (h, ε) alone; score R² against the trained models (3 seeds), with baselines: true Bayes, the mean surface, a counter with average parameters. This tests "given the data statistics, predict the trained model's behaviour".

**5.2 Causal elasticity (GPU eval, ~25 min):** finer slope surgery (×0.25, 0.5, 0.71, 1, 1.41, 2, 4) on all 57 ALiBi models; fit log h′ = k·log(scale) + c per model; report k with CI. Also record the loss vs scale curve.

**5.3 Loss flatness:** curvature of ordinary-data loss in log-slope from 5.2: how cheap the miscalibration is.

**5.4 Formulation write-up:** Prop. 2 (exact, with validity domain); Law 1 (behaviour = Bayes with h′(h), ε′(ε), CIs, predictive R²); Law 2 (h′ ∝ slope^k, causal); mechanism equation; loss flatness; stated scope limits (2–3 layers, width ≤ 256, one task family).

## Phase 6 · Literature (web, ~1 h)

Verify every citation in plan §2 (title, authors, venue, year, arXiv ID) against arXiv or the venue page; read the abstracts of the 6 closest papers (D'Angelo 2026, Selective Induction Heads, Edelman 2024, Qin 2026, Dudley 2026, Bajaj 2025); write a novelty statement and flag any overlap with our claims; fix the plan's related-work table.

---

## Pre-registration (§17, written before Phase 2–5 GPU/null runs)

- 2.3: same-seed retrain within 15% on h′ and ε′, surface correlation > 0.95.
- 2.4: for ALiBi, h′/h > 2 at every lr/batch; Bayes beats the counter (held-out) in ≥ 90% of runs.
- 4.3: Bayes(h′, ε′) median R² on generic smooth surfaces < 0.80 (models: 0.94).
- 5.1: leave-one-setting-out median R² ≥ 0.75 and above the true-Bayes baseline.
- 5.2: log h′ linear in log scale (median R² > 0.9), k in [0.7, 1.3].

## Files

- **New:** `recfreq/verify/{registry,calibration,recovery,nulls,predictive_law,stats,ood,slope_sweep,report}.py`; `tests/test_{calibration,recovery,probes_layouts,model_pe,mech_paths}.py`; `configs/E4_repro.json`, `configs/E4_hparams.json`.
- **Edit:** `scripts/queue.py` (run IDs for lr/batch); report HTML (corrections plus a link to the audit); `research_plan_recency_vs_frequency.md` (§17 predictions and verdicts, related-work fixes).
- **Reuse:** `layouts.{Lay,BayesFree,fit_counter,fit_counter2,r2}`, `fit_rules.bayes_lo_grid`, `analyze.{load,boundary}`, `oracle.bayes_predictive`, `probes.{probe_batch,key_predictive}`, `run3.surface`, `paths.forward_paths`, `mech.decompose`, `train.main`, `scripts/queue.py`.

## Verification of this audit

- `pytest -q` passes (old + new tests).
- `python -m recfreq.verify.registry` prints a diff table with zero unexplained mismatches.
- One command (`scripts/verify_all.sh`) reruns every CPU check from the snapshot.
- The verification report lists, per criterion (1–5 + literature): checks run, pass/fail, numbers, and every correction made to the main report.

**Estimated time:** about 6–8 h wall clock; about 4 h of GPU (2.2, 2.3, 2.4, 5.2), with CPU phases running in parallel.
