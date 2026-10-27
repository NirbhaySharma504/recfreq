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

# Run 7 — region of validity and explanation (27 Oct 2026, overnight, iterative; plan §19)

## Round 1 (02:26–02:48)
- Prop. 2 redone: the paper's boundary dropped three terms (same-value redraw, old copies as typos of c−1, observed-token odds). Corrected: within 10% of exact in 91% of 920 cases (median 0.9%); the paper's: 20% (median 47% off; 77% off at ε ≥ 0.3).
- H-lin ("ALiBi's linear distance distorts the sense of time"): refuted. The linear-survival surrogate gives h′/h of only 1–3.7, and the survival probes show calibrated time.
- H-support: per-cell deviation from Bayes falls as the situation becomes more frequent in training (within-cell Spearman −0.73 across 19 settings). Post hoc.

## Round 2 (02:48–03:10)
- Switch points: the models' G* ∝ h^−0.93, nearly calibrated, but extra recent copies move it only ~0.5× as much as in Bayes.
- Natural law (frozen 02:55, before E6): log G* = −0.739 + 0.933·log(1/h) + 0.327·log(1/r_B). Plain reading: the clock is right, evidence counts at about 1/3.
- RoPE has the same form with evidence weight 0.70.
- Wide window: ε = 0.4 models are not flat. Their floor (−1.5 nats) lies outside Bayes's range (±0.41), which is why the Bayes-form law fails there.
- GPU: E6 (50 runs, region sweep) and E7 (16 runs, conflict-dose) queued at 02:26 and 02:49.
- Next: E6 → R2-P2, R2-P3, R2-P4 and the region map; E7 → R2-P1 and R2-P5.

## Round 3 (03:18; E8 pre-registered: context length at the low-h edge)
- E8: N = 512 at h ∈ {0.0003, 0.001}; R3-P1 in plan §19.5.

## Round 4 (04:58–05:25)
- Ops: the server reached 31/31 GB RAM with 27 parallel jobs, and 4 E7 jobs were OOM-killed (they resume from checkpoints). Two fixes in `queue.py`: no new job while free RAM < 3 GB, and jobs run in their own session so stopping a queue no longer kills them. Earlier, killing the old E6/E7 queues killed their jobs through tmux hangup; they resumed from checkpoints and lost at most 500 steps each.
- Region map on 16/25 E6 seed-0 models (`verify/region_map.py`):
  - **h′ extrapolates.** The frozen h′ law matches the fitted h′ from h = 0.002 to 0.15. Below that, h′ levels off at about 0.01.
  - **ε′ fails off the fitted range** because it depends on h (0.33 at h ≤ 0.001 down to 0.08 at h = 0.15, at ε = 0.1).
  - **Cross-term law:** the run-6 cross-term law (out of sample) halves the ε′ error but misses the drop at h ≥ 0.06.
- **Evidence weight** (`verify/evidence_weight.py`). w = switch-point shift per recent copy ÷ Bayes's, in clock units h·G*. For ALiBi it rises with h:

  | h | 0.001 | 0.003 | 0.01 | 0.03 | 0.06 | 0.1 |
  |---|---|---|---|---|---|---|
  | w | 0.10 | 0.26 | 0.40 | 0.61 | 0.78 | ≥ 1 |

  Spearman with log h is 0.77 (n = 65). w also rises with ε. RoPE: 0.8–1.1.
- **The switch point at zero net evidence** (x_ref, in units of 1/h) is 0.4–0.6 across h = 0.005–0.06. The model switches about 2× earlier than Bayes, by a constant factor.
- **R3-P2 FAIL.** The single-sighting survival at h = 0.0003–0.001 is calibrated out to d = 248. The low-h failure is in combining an old copy with new ones: at h = 0.0003 an extra recent copy moves the log-odds by 0.05 nats at G ≤ 32 and by 1.4 nats at G = 128. Time and evidence interact, where Bayes adds them.
- **Excess time × evidence interaction** (additive-fit residual, model minus Bayes) does *not* line up with where the law fails (Spearman −0.40). Dropped.
- **Restricted Bayes fits** (`verify/clock_fixed.py`). RMSE relative to the free (h′, ε′, a, b) fit, ALiBi medians:

  | Fit | RMSE ratio |
  |---|---|
  | true h, free ε′ | 1.86 |
  | free h′, true ε | 1.35 |
  | true h, free V_eff and ε′ | 1.24 (V_eff at the grid edge of 4 for h ≥ 0.003) |
  | true h and ε | 2.09 |

  So within the Bayes family the probe surfaces want h′ > h even though the single-sighting clock is right. h′ stands for "switches early, by a near-constant factor", not "forgets faster". RoPE: every restriction is within ×1.13.
- E9 launched (K ∈ {8, 32, 64} at h ∈ {0.0003, 0.001}; R3-P3).

## Round 4b (05:55–06:05)
- **First E7 models** (dose 0.4 at C and D, dose 0.1 at C; read before the low doses exist; verdicts later). Medians:

  | | f = 0 | f = 0.1 | f = 0.2 (E2c) | f = 0.4 |
  |---|---|---|---|---|
  | w at C (h = 0.01, ε = 0.1) | 0.34 | 0.81 | 0.87 | 0.84 |
  | w at D (h = 0.03, ε = 0.2) | 0.66 | — | 0.59 | 0.51 |

  - At C, extra conflict training *raises* the evidence weight w; at D it doesn't.
  - Deviation from Bayes doesn't fall at either setting.
  - The early-switch factor x_ref (0.36–0.67) is untouched by conflict training.
- **V runs (E5d).** The model's G* moves little with V (34 / 40 / 42 at C for V = 32 / 64 / 128) while Bayes's moves a lot (30 / 53 / 87). That fits the value prior log V being weighted by the same w as the copy evidence: log(h·G*_model) ≈ α + w·log(h·G*_Bayes).
- **Tried and dropped:** a law with exact clock units and w = w0 + w1·log h + w2·logit ε, fitted on pre-run-7 models. It is worse in-sample than the natural law (0.22 vs about 0.15 median |log err|), because switch gaps scale as h^−0.93, not h^−1. Not pre-registered and not kept: it would be curve-fitting.

## Round 5 (06:36–07:50)
- **E6 seed 0 complete** (25 models). Verdicts in plan §19.7:
  - R2-P3: a PASS, b FAIL, c FAIL (the frozen law beats the natural law on new crossings);
  - R2-P2: PASS (per-cell deviation predicted from training-data rates on unseen settings, Spearman 0.67).
- **Region map.** The pre-registered 1.25× criterion is stricter than the law manages on its own fitting grid (4/12). Post hoc, "as good as inside the box" (ratio ≤ 1.51) covers h ≈ 0.001–0.045 at ε ≈ 0.05–0.25. Fails at ε ≥ 0.35 (×2.7–12), ε ≤ 0.02 (×1.9–2.9) and h ≥ 0.06 (×2.6–6.5). h = 0.0003 is not measurable.
- **Survival** calibrated in all 58 default-task models (median |Δs| ≤ 0.041).
- **R3-P1 FAIL** (E8, context 512): G* moves ×1.34 at h = 0.0003 and ×1.13 at h = 0.001.
- **R3-P3 interim** (E9 seed 0): K = 32 moves G* by ×1.12 / ×1.16; K = 8 by ×0.65 at h = 0.0003. Long-gap support is not the main cause of the low-h edge.
- **E10 pre-registered** (R5-P1: is the low-h evidence weight just undertrained? 80k steps).
- Ops: E7 and E9 seed 1 are paused (SIGSTOP) to give E7 s0, E9 s0 and E10 the GPU. Resume with `scripts/sigjobs.sh CONT '^E(7|9)_.*_s1$'`.

## Round 6 (08:10–08:37)
- **R3-P3 interim FAIL** (E9 seed 0): 4.6–9× more long-gap training data moves G* by only ×1.11–1.20. K = 8 (almost no long gaps) moves it ×0.65 / ×0.83.
- **R2-P1, R2-P5** (E7 seed 0 complete): conflict dose does not lower the deviation from Bayes (a FAIL at C and D). It does lower h′/h (b PASS at both). C2 steps 0.35 → about 0.5 at any f > 0 (R2-P5 FAIL: not graded).
- **Key post-hoc finding** (run-3 surfaces): rescaling layer-1 ALiBi slopes moves w (×2 → w ×1.57; ×0.5 → ×0.59) and moves x_ref the opposite way.
  - Correlational: λ rises with h (ρ 0.89), and w tracks λ (ρ 0.72–0.75). Conflict-trained models break this tie: high w at small λ.
- **H-tie** (pre-registered R5-P2, R5-P3): one slope sets both the clock and the evidence weight.
- **R5-P3** (surgery on 25 unseen E6 models): ×0.5 PASS (w ×0.64, 20/20); ×2 FAIL narrowly (×1.28 against the ≥ 1.3 bound; 14/16 in the right direction).
- New slide d_mech ("One ALiBi slope does two jobs").

## Round 7 (09:17–09:55)
- **R3-P3 final** (both E9 seeds): FAIL. K = 32 / 64 move G* by only ×1.07–1.27. Long-gap support is not the low-h cause.
- **E10** (80k steps, all GPU given to it, about 1,245 steps/min per job): **R5-P1 FAIL** (w ×0.92 at h = 0.0003, ×1.79 at h = 0.001, ×1.10 at h = 0.01) and **R5-P2 FAIL**. Training length is not the low-h cause either.
- **Low-h edge:** consistent with the slope tie at its extreme (smallest λ, smallest w), but no data-side change moves it.
- **E11 pre-registered** (R6-P1): RoPE at h = 0.0003 / 0.001 / 0.1. If w stays in [0.6, 1.4], the drift is specific to ALiBi.
- Ops lesson: the GPU is time-sliced and the CPU has 14 cores, so total throughput is about one job's worth. Pausing low-priority jobs (SIGSTOP) gives the priority job about 8× speed.

## Round 8 (27 Oct 12:55 IST): E7 seed 1, final dose verdicts

`python -m recfreq.verify.dose` with both E7 seeds. Every verdict is the same as with seed 0:
- **R2-P1a:** FAIL in C and D.
- **R2-P1b:** PASS in both.
- **R2-P1c:** PASS in C, FAIL in D.
- **R2-P5:** FAIL. C2 now rises monotonically with f (Spearman 1.00), but by only ×1.41 (needed > ×1.5).

Conflict data raises the evidence weight a little, with most of the gain at the first 1%. It does not bring the models to Bayes.

Next: E6 seed 1 (region edge stability; R2-P2, R2-P3 and R3-P4 finals).

## Round 9 (27 Oct 13:40 IST): E6 seed 1, finals, round-7 pre-registration

- **Region map with both seeds:** no setting changes side (pre-registered criterion 8/44 works; post-hoc ≤ 1.51× unchanged). The edges are stable.
- **Verdicts on all 50 E6 models:**
  - R2-P3: a PASS (0.279), b FAIL (0.76), c FAIL (frozen law better);
  - R2-P2: PASS (0.664; β −1.06);
  - R3-P4: PASS on the unread seed-1 models (Spearman 0.67).
- **Slope surgery replicated on seed 1:** ×0.56 and ×1.37.
- **Clock:** calibrated in all 124 models with survival probes.
- **New (post hoc):** at ε ≥ 0.35, probe behaviour is seed-dependent at equal loss. At h = 0.01, ε = 0.35, w is 1.26 vs 0.08. The slide's per-copy numbers were replaced by the direct probe measure.
- **Slides updated** (`d_region`, `d_why`, `d_mech`, `d_edges`), with figures regenerated by `plot_region` (which now rebuilds `slope_vs_w.csv`).
- **Next:** round 7 (E12, §19.13) tests whether conflict data lifts w at very low h for ALiBi and RoPE. It is pre-registered at 13:41. The launch is waiting on the server: the laptop moved to a network that cannot reach it.
