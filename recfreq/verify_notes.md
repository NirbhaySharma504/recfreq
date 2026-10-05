# Verification audit — working notes (5 Oct 2026)

## Phase 0 · Freeze
- Snapshot `~/recfreq/snapshots/2026-10-05/` + `MANIFEST.txt` (sha256, 647 files). 123 runs complete; no non-finite values in any log.

## Phase 1 · Code correctness
- Tests: 36 pass (20 old + 16 new: calibration ×4, recovery ×2, probe layouts ×5, PE ×2, mechanism exactness ×2, + 1).
- Generator ↔ oracle calibration (independent of the brute-force check): ECE < 0.01, mean NLL = expected entropy (|z| < 4) at (0.01, 0.1), (0.03, 0.2), (0.003, 0.05), uniform (0, 0.2). **Pass.**
- Recovery: BayesFree recovers known (h′, ε′): median error < 25% where the boundary is in the probe range; fits reach the noise floor everywhere (R² ≥ true-parameter R²; residual ≤ 1.11 × noise). Weak identifiability of h′ on flat surfaces (h = 0.003, ε = 0.3: fit 0.0051 vs 0.003). Counter recovers λ within 5%. **Pass** — the first test version used an absolute R² threshold, which is wrong for flat surfaces (noise is a large share of the variance); replaced with a noise-floor criterion after diagnosis. *Deviation noted.*
- Independent Bayes filter (fresh code) = layouts code to 0.0.
- Code review (data, oracle, train, probes, conflict enrichment, layouts/fit_rules, analyze, analyze3, mech, paths, run5): no correctness bugs found. Design notes:
  - probe backgrounds use 15 keys (target key excluded) with Zipf over 15 ranks;
  - stored probes scored under bf16 autocast (run-3/4/5 surfaces in fp32) → precision check 1.3;
  - `fit_rules`/`layouts` hard-code V = 64, N = 256 — consistent with every config;
  - h′ fits are weakly identified when the switch-over lies outside the probed gaps.

## Phase 2 · Procedure
- Convergence: 21 of 131 runs still improving > 10% (and > 0.002 nats) in the last 25% of steps (E2c 10/12, E2a 6/24); absolute improvement median 0.0014, max 0.0054 nats. Cosine-schedule tail. Behavioural conclusions rely on E3 (3× steps: h′ ×0.76–1.17), not on loss convergence.
- Gate: 100% repeat-sighting accuracy, |excess| ≈ 1e-5 nats.

- Reproducibility (V2.3, pre-registered): same-seed retrains reproduce h′ within 4%, ε′ within 2%, surface corr 0.9998–1.0000, loss to 4 decimals (4/4); new seeds 3–4 inside the original seeds' range (4/4). **Pass.**
- Hyperparameters (V2.4, pre-registered; lr 3e-4, 3e-3, batch 256; C, D; ALiBi, RoPE; 2 seeds = 24 runs): ALiBi h′/h 3.3–6.3 (> 2 in every run ✓); Bayes(h′, ε′) beats the best counter held-out in 24/24 (≥ 90% ✓); no run with h′/h ≤ 1.5 (against-outcome absent ✓); RoPE h′/h 1.8–4.2; cliff ratio 1.2–3.6. **Pass.**

## Pre-registration audit (runs 3–5)
| Test | As written | Verdict as reported | Strict re-check |
|---|---|---|---|
| R3-1 copy vs control ablation | copy > 70% |LO| drop; control < 20% | failed | failed (bad control choice) |
| R3-2 slope surgery | monotone **in all seeds** | passed | **h′ monotone 56/57 — strictly fails "all"**; the exception (E2b h0.01 ε0.3 s1) is an h′–ε′ trade-off (ε′ 0.456 → 0.178); model-free G* monotone 39/39 |
| R3-3 layouts | Bayes beats counters; refit within ~2× | passed | passed (90% of 168 sp2/sp8 pairs; swap 94% by MAE; refit 97%). Swap metric changed from R² to MAE after seeing that Bayes is constant there — *deviation noted* |
| R3-4 content vs distance | no direction | finding | finding |
| R3-5 random counter sums | claim weakens if Bayes-shaped | claim stands | stands |
| R3-6 scale | h′ moves < 2× | ALiBi yes; RoPE nearly closes | same |
| R4-A1/A2 | | passed | passed |
| R4-A3, B1, B2 | | failed | failed |
| R5-5A-1 | | passed | passed |
| R5-5A-2, 5B-1 | | failed | failed |
| R5-5B-2 | ≥ 50% from layer-0 | borderline | borderline (median 0.58; 63% of models) |

Other deviations: NoPE replaced by RoPE before the overnight launch (NoPE did not converge); E2b "light mode" enabled then reverted before any E2b model ran (no effect); P1 post-hoc used a bounded counter form, corrected in run 5; per-copy theory check redesigned (fixed-G slope was the wrong quantity).

## Phase 3 · Results
- Claims registry: **31/31** headline numbers reproduce from raw files with fresh code.
- Corrections found:
  1. slope-surgery loss changes are +0.006 (×0.5) and +0.016 (×2) nats, not "+0.01/+0.02";
  2. **"models use about a third of the Bayes evidence per recent copy" overstates the gap**: the 0.31 law compares with log((1−ε)/ε), but that theory value is exact only for h·G* ≲ 0.3 (≤ 2% error) and overstates the exact Bayes weight by 30–55% for h·G* ≈ 1–3. Against the exact Bayes weight on the same probes, model/Bayes = 0.25, 0.24, 0.34, 0.46, 0.46, 0.47, 0.78, 0.68, 1.06 (median ≈ 0.46);
  3. registry R9 initially mixed layouts (registry bug, not a report error).
- Theory: Prop. 2 boundary formula within median 10–15% of exact (max ~50% at h·G* ≈ 3); n_c-independence: ≤ 0.11 nats change from n_c = 2 to 16; per-copy weight = log((1−ε)/ε) within 2% for h·G* ≤ 0.3.

- Fit robustness (84 models): second optimizer (61×60 grid + Nelder–Mead) gives the same h′ (median |log ratio| 0.000; within 25% in 99%) and ε′ (100%). Wild-bootstrap 90% interval width: h′ ×1.30, ε′ ×1.14 (median). h′ > true h significantly in 92% of models. Fitted scale b > 0 in all (min 0.50).
- Laws with seed-resampled 95% CIs: h′ = 1.50 [1.29, 1.78] · h^0.74 [0.71, 0.78]; logit ε′ = −0.33 [−0.47, −0.04] + 0.44 [0.35, 0.57] · logit ε.

## Phase 4 · Support
- Nulls (pre-registered V4.3): generic smooth monotone surfaces → Bayes(h′, ε′) median R² 0.73 (< 0.80 ✓; 10% of them > 0.93); shuffled → 0.07 (< 0.2 ✓). Models: 0.95.

## Phase 5 · Formulation
- Predictive law (pre-registered V5.1): leave-one-setting-out median R² **0.867** (≥ 0.75 ✓), above true Bayes with in-sample scale/offset (0.738 ✓); beats mean surface in 100%, in-sample true Bayes in 75%. Weak: (0.003, 0.3) 0.57; (0.03, 0.05) 0.72.

## Phase 6 · Literature
- 17 citations checked. Corrections: Zhou et al. 2410.05493 title is "An Information-Theoretic Approach to Understanding Transformers' In-Context Learning of Variable-Order Markov Chains" (AISTATS 2026); Makkuva et al. 2402.04161 published at ICLR 2025 as "Attention with Markov: A Curious Case of Single-Layer Transformers"; Rajaraman et al. NeurIPS 2024 venue not confirmed from arXiv. Selective Induction Heads = ICLR 2025 ✓ (proceedings). Qin et al. = arXiv 2604.10946, ICLR 2026 ✓. Hegazy et al. title "Recency Biased Causal Attention for Time-series Forecasting", AISTATS 2026 ✓. Crosbie & Shutova NAACL Findings 2025 ✓. Behrens, Biggio, Zdeborová ICML 2025 ✓.
- Novelty: no prior work found on conflicting old/recent continuations with a Bayes switch/typo trade-off for induction heads; checked 2606.12058 (copy-head phase transitions) and 2508.03934 (Markov estimation): no overlap.
