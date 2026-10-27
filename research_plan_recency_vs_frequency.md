# Recency vs. Frequency in Induction Heads — Research Plan

**Authors:** Nirbhay Sharma, Gathik Jindal
**Status:** planning document, Oct 2026
**Purpose:** hand this file to a Claude Code session. It contains the diagnosis of the old notebook, a literature survey, the theory with checked formulas, the experiment design, the code spec and the A6000 run plan.

---

## 0. TL;DR

- **Question (from Sir):** when a pattern's continuation appears *often but long ago* and a conflicting continuation (a typo, or a real change) appears *rarely but recently*, which one does an induction head copy?
- **The reframing that makes it a paper:** recency and frequency are each the *right* answer under different data-generating processes.
  - If continuations can **change** (switch rate `h`), the recent copy is informative.
  - If copies are only **corrupted** (typo rate `ε`), the frequent copy is informative.
- **Two formulas to test against each other:**
  1. **Mechanism (Prop. 1).** A softmax induction head with positional bias `λ·d` is an *exponentially discounted counter*. It picks the old value `c` iff `log(n_c / n_w) > λ·(d_c − d_w) − β·δ`. This depends on `n_c`, and the distance enters **linearly**.
  2. **Bayes-optimal (Prop. 2).** Under the switch + typo model, the optimal predictor picks the recent value `w` iff `(1−h)^G / (1 − (1−h)^G) < (1/V)·((1−ε)/ε)^{n_w}`. For small `hG` this becomes `log(h·G) > log V − n_w·log((1−ε)/ε)`. It is **independent of `n_c`**, and the gap `G` enters **logarithmically**. Checked against an exact HMM filter to within about 5%.
- **The paper's claim, whichever way it falls:** train small transformers on controlled `(h, ε)` data and measure which rule they implement, how the learned `λ` tracks `h` and `ε`, and how depth, MLPs and positional encoding move them between the two rules.
- **The old notebook failed for fixable reasons** (§1). The new task design (§5) has a computable Bayes ceiling, so we always know whether training worked.
- **Headline test = `n_c`-sensitivity** (§4.4, Cor. 1). Any sum of positive copying heads is strictly increasing in `n_c`; Bayes is flat in `n_c`. The "linear vs. log G" test is secondary because a mixture of heads with different slopes can look logarithmic.
- **Current scope = the reduced first run in §8.1** (gate + 4 `(h, ε)` points × 3 architectures × 3 seeds, learnable ALiBi only). Everything else in §8 waits until that run is clean.

---

## 1. Why the earlier notebooks gave no result

> **Correction (Oct 4).** The notebook in this folder (`week2 induction head.ipynb`) is the *multi-offset shift-head* notebook, not the typo-frequency one described below. The typo-frequency notebook and `generate_typo_frequency_dataset.py` are not in this folder, so the table below is kept only as a record and can't be re-checked here.
>
> **What the attached shift-head notebook actually shows:** training loss stuck at 5.299 ≈ ln 200 (uniform prediction) at every depth 0–4, and induction accuracy 0.3–0.9% (chance) in every phase. Causes: about 680 optimizer steps in total (8 epochs × ~85 steps of batch 128), loss over every position while only ~3% of positions carry induction signal, N(0,1) embeddings, no warmup or schedule. The run never left the uniform plateau.
>
> **`models.py` correction:** it *does* have pre-LayerNorm, a final LayerNorm and an untied unembedding. Its weaknesses are N(0,1) embedding init and default `nn.Linear` init. The new code does not reuse it.
>
> ⚠️ **Open issue:** the results slides in `paper2.pdf` (90–98% accuracy, a 1-layer model at ~98% on bigrams, ablation bars, K-composition heatmap) do not match any output in the attached notebook, and a 1-layer attention-only model cannot do induction. Find the run that produced them, or mark them as placeholders, before reusing anything from that deck.

Numbers from the typo-frequency notebook outputs (val set, 13.5k events; not re-checkable from this folder):

| layers | final train loss | correct | distractor (typo) | other |
|---|---|---|---|---|
| 0 | 5.295 | 0.005 | 0.005 | 0.990 |
| 1 | 4.43 | 0.115 | 0.025 | 0.861 |
| 2 | 4.00 | 0.138 | 0.034 | 0.829 |
| 3 | **4.82** | 0.111 | 0.019 | 0.870 |
| 4 | 3.78 | 0.101 | 0.017 | 0.882 |

Baselines by construction: recency = 0.000, frequency = 0.99–1.00.

**Diagnosis, most important first:**

1. **The models never learned induction.**
   - About 85% of predictions are neither the correct token nor the typo.
   - A 1-layer attention-only model *cannot* implement induction, yet it scores the same 11% as the 4-layer model.
   - So the 10–15% comes from a 1-layer shortcut: "boost tokens that occur often in the context". The correct token occurs 4–6 times, so it wins sometimes.
   - Depth added nothing. The induction circuit never formed.
2. **The optimization is broken.**
   - Initial loss was **72.6** against `ln(200) = 5.3` for uniform predictions, so the logits are huge at init. The model used there was presumably not the `models.py` in this folder (that one starts near ln 200).
   - The 3-layer model ends *worse* than the 1-layer model (4.82 vs 4.43): instability.
   - There was no warmup, no LR decay and no gradient clipping.
   - Check `models.py` for all of this. I didn't have that file or `generate_typo_frequency_dataset.py`.
3. **Most of the loss is unlearnable noise.**
   - The 0-layer loss is exactly `ln 200`, so filler tokens are uniform random.
   - Only about 10% of positions (completions after a repeated prefix) carry induction signal, so the gradient is dominated by noise.
4. **Frequency and recency are confounded.**
   - "Correct" is always far *and* frequent; "typo" is always near *and* rare.
   - The baselines are exactly 0 and 1 by construction.
   - Even a perfect model can't tell you whether it counts or ignores the nearest match.
5. **There's no notion of a right answer.**
   - Nothing in the training distribution makes the typo *wrong* or *right*. Typo completions are trained on as ordinary targets.
   - So the "correct" label is our opinion, not something the data rewards.
6. **There's no ceiling.**
   - Without a Bayes-optimal reference loss, you can't tell "the model is bad" from "the task is impossible".
7. **Spurious matches.**
   - With V = 200 and L = 256, each token appears about 1.3 times by chance in the filler.
   - So bigram prefixes collide with random filler occurrences.
8. **Practical issues.** Evaluation produced CUDA OOM warnings on the 8 GB card, and there was a single seed.

**Lesson for the new code.** Every experiment starts with a gate: on clean data (`h = 0, ε = 0`), the 2-layer model must reach at least 99% accuracy on *repeat* occurrences of a key and get within 0.02 nats of the Bayes loss (see §6 for the exact definition). Otherwise stop and fix training.

---

## 2. Literature survey

Novelty claim to defend: *nobody puts count and recency in controlled conflict for induction heads, derives the Bayes-optimal trade-off, and checks which rule trained transformers implement.* I checked the closest papers this session.

### 2.1 Closest — must cite and position against

| Paper | What it does | How we differ |
|---|---|---|
| **D'Angelo, Yüksel, Narashiman, Flammarion — *Induction Heads Interpolate N-Grams*, ICML 2026** ([arXiv 2607.02800](https://arxiv.org/abs/2607.02800)) | Order-k Markov chains (stationary). The induction circuit is a *soft context-matching* estimator: partial matches are weighted exponentially by overlap (≈ Jelinek-Mercer smoothing), and a BOS token gives Dirichlet pseudo-counts. | Their exponential-overlap weighting **is our `β·δ` typo term**, so cite it for that. They study **no recency, no nonstationarity, no conflicting continuations**. We add the positional (recency) axis and the Bayes comparison under switching. |
| **d'Angelo, Croce, Flammarion — *Selective Induction Heads*, ICLR 2025** ([arXiv 2509.08184](https://arxiv.org/abs/2509.08184)) | Interleaved Markov chains with different lags. A 3-layer construction selects the lag and copies; converges to maximum likelihood. | They choose *which offset*; we choose *which occurrence* (old/frequent vs. recent). Same "selection" flavour, different axis. |
| **Edelman, Edelman, Goel, Malach, Tsilivis — *Evolution of Statistical Induction Heads*, NeurIPS 2024** ([arXiv 2402.11004](https://arxiv.org/abs/2402.11004)) | In-context Markov chains; transformers learn bigram *counting* in stages (uniform → unigram → bigram). | Stationary data, so counting is optimal and recency never matters. This is our `h = 0` corner. |
| **Qin, Jiang, Zhu — *Learning to Adapt: In-Context Learning Beyond Stationarity*, ICLR 2026** | Drifting linear regression (AR weights). Gated linear attention acts as an adaptive filter with a learnable recency bias and beats plain linear attention. | Closest "optimal recency" paper, but it covers linear regression and linear/gated attention, with no induction heads, discrete tokens or typos. We do softmax induction on discrete sequences, with typos versus switches. |
| **Dudley, Bi, Liu, Oymak — *In-Context Learning Under Regime Change*, 2026** ([arXiv 2604.16988](https://arxiv.org/abs/2604.16988)) | Piecewise-linear regression with one change point; transformers can approximate Bayesian model averaging; positional encoding helps. | Constructive, regression only, no mechanism or induction heads. Supports our "Bayes under change" framing. |
| **Bajaj, Mistry, Maini, Aggarwal, Tiganj — *Beyond Semantics: How Temporal Biases Shape Retrieval in Transformer and State-Space Models*, Oct 2025** ([arXiv 2510.22752](https://arxiv.org/abs/2510.22752)) and ***Emergence of Episodic Memory in Transformers*** ([arXiv 2502.06902](https://arxiv.org/abs/2502.06902), same group) | Repeated tokens in context: primacy/recency/contiguity biases; ablating induction heads removes them; NoPE models still get weak recency via the causal mask. | Behavioural characterization with no frequency-versus-recency conflict and no formula. Useful evidence that recency exists in induction heads. **Check the author order on arXiv before citing.** |
| ***Temporal Dependencies in In-Context Learning: The Role of Induction Heads*, 2026** ([arXiv 2604.01094](https://arxiv.org/abs/2604.01094)) | Open LLMs show a +1-lag serial-recall bias; ablating induction heads removes it. | Same: recency/lag only. |

### 2.2 Theory of depth and Markov induction (for the depth axis)

- **Rajaraman et al., *Transformers on Markov Data: Constant Depth Suffices*, NeurIPS 2024 (venue not confirmed on the arXiv page — check before citing)** ([2407.17686](https://arxiv.org/abs/2407.17686)). Three layers suffice for any k-th-order source with LayerNorm/MLPs; attention-only constructions use O(log k) layers.
- **Sanford, Hsu & Telgarsky, *Transformers, Parallel Computation, and Logarithmic Depth*, ICML 2024.** k-hop induction needs Θ(log k) depth (the lower bound is conditional on a conjecture).
- ***What One Cannot, Two Can*, 2025** ([2508.07208](https://arxiv.org/abs/2508.07208)). Two-layer transformers provably represent induction heads on any-order Markov chains.
- **Nichani, Damian & Lee, *How Transformers Learn Causal Structure with Gradient Descent*, ICML 2024.**
- **Zhou, Tian & Diggavi, *An Information-Theoretic Approach to Understanding Transformers' In-Context Learning of Variable-Order Markov Chains*, AISTATS 2026** ([2410.05493](https://arxiv.org/abs/2410.05493)). *(Title corrected by the audit; the earlier title was wrong.)*
- **Bietti et al., *Birth of a Transformer: A Memory Viewpoint*, NeurIPS 2023.** Global bigram versus in-context induction; associative-memory view of the weights.
- **Makkuva et al., *Attention with Markov: A Curious Case of Single-Layer Transformers*, ICLR 2025** ([2402.04161](https://arxiv.org/abs/2402.04161); arXiv title "Attention with Markov: A Framework for Principled Analysis of Transformers via Markov Chains").

### 2.3 Positional encoding and where recency comes from

- **Press et al., ALiBi, ICLR 2022.** A linear bias `−m·d` makes our Prop. 1 exact; we make the slope **learnable** so λ can be read off directly.
- **Su et al., RoPE.**
- **Kazemnejad et al., NoPE, NeurIPS 2023.**
- **Wu, Wang, Jegelka & Jadbabaie, *On the Emergence of Position Bias in Transformers*, ICML 2025** ([2502.01951](https://arxiv.org/abs/2502.01951)). The causal mask creates primacy across layers; relative-PE decay creates recency within a head; the two trade off. This matters for us: depth can push towards *older* occurrences.
- **Hegazy, Mahoney & Erichson, *Recency Biased Causal Attention for Time-series Forecasting*, AISTATS 2026** ([2502.06151](https://arxiv.org/abs/2502.06151)). Power-law recency beats exponential for time series. Motivates testing linear versus log distance kernels.

### 2.4 Typos and fuzzy matching

- **Crosbie & Shutova, *Induction Heads as an Essential Mechanism for Pattern Matching in ICL*, NAACL Findings 2025.** Induction heads do fuzzy prefix matching.
- **D'Angelo et al. 2026** (above). Partial-match weighting is exponential in overlap.

### 2.5 Behavioural LLM evidence (motivation paragraph only)

- **Qiao et al., *Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates*, Feb 2026** ([2603.12271](https://arxiv.org/abs/2603.12271)). When a fact is updated several times in context, earliest-state accuracy stays high and latest-state accuracy drops: primacy beats recency.
- **Naphade, *Rational Synthesizers or Heuristic Followers?*, Jan 2026** ([2601.06189](https://arxiv.org/abs/2601.06189)). In RAG with conflicting evidence, LLMs favour evidence presented first, and "paraphrasing an argument can be more persuasive than providing distinct independent support" — repetition is mistaken for corroboration.
- ⚠️ **Verify every citation before writing.** Several arXiv IDs here (2607.xxxxx, 2604.xxxxx, 2603.xxxxx) were not checked against the actual arXiv pages, and author lists and venues can be wrong.
- ⚠️ **Correction to the earlier notes:** arXiv 2506.06485 does **not** support a "majority bias" claim (it is about context-versus-parametric-memory conflicts). Drop it. Edelman et al. is arXiv **2402.11004**.

### 2.6 Methods we borrow

- QK/OV decomposition and K-composition: Elhage et al. 2021.
- Prefix-matching score: Olsson et al. 2022.
- Activation patching and mean ablation: Zhang & Nanda 2024; Heimersheim & Nanda 2024.
- Clamping during training: Singh et al. 2024.
- Bayesian online changepoint detection, for the oracle's intuition: Adams & MacKay 2007.
- Counting: Behrens et al., *Counting in Small Transformers*, ICML 2025; Yehudai et al., *When Can Transformers Count to n?*, 2024.

---

## 3. Research questions

- **RQ1 (mechanism).** Does a trained induction head behave as a discounted counter, `score(v) = Σ_j exp(β·m_j − λ·d_j)·1[x_{j+1} = v]`? Does Prop. 1 predict the model's decisions from its own fitted λ and β?
- **RQ2 (optimality).** Does the learned λ track the data's switch rate `h` and typo rate `ε` the way the Bayes analysis says it should? Is the model's decision rule closer to the discounted counter (uses `n_c`, linear in distance) or to Bayes (ignores `n_c`, logarithmic in distance)?
- **RQ3 (architecture).** How do depth (1–4 layers), MLPs on or off, and positional encoding (learnable ALiBi, learned absolute, RoPE, NoPE) move the model between the two rules?
- **RQ4 (optional, transfer).** Do pretrained LMs (Pythia family) show the same phase boundary on synthetic conflict prompts? How does it change across training checkpoints?

---

## 4. Theory (the "proving" part)

### 4.1 Setup

The context contains earlier occurrences `j` of the current key or prefix, each followed by a value `x_{j+1}`. Notation:

- `d_j` — distance in tokens from the query to occurrence `j`.
- `m_j` — prefix-match quality; `δ_j` is the number of mismatched prefix tokens.
- `c` — the old value, seen `n_c` times at distances around `d_c`.
- `w` — the new or typo value, seen `n_w` times at distances around `d_w < d_c`.

### 4.2 Proposition 1 — an idealized induction head is a discounted counter

**Assumptions.**

- The head's attention logit on occurrence `j` is `β·m_j − φ(d_j)`.
- Non-matching positions get logit ≤ `−M` with `M` large.
- The OV circuit copies the value token with gain `γ`, the same for all tokens.
- Nothing else writes to the value logits.

**Claim.** The output logit of token `v` is `γ·A(v)/Z`, where

```
A(v) = Σ_{j: x_{j+1} = v} exp(β·m_j − φ(d_j))
```

So the model prefers `c` over `w` **iff `A(c) > A(w)`**.

**Special cases.**

- **Linear kernel (ALiBi), `φ(d) = λ·d`, copies clustered, no prefix typos:**
  **`log(n_c / n_w) > λ·(d_c − d_w)`**.
  General copies: `Σ_c e^{−λ d_i} > Σ_w e^{−λ e_k}`.
- **With a prefix typo** costing `β·δ` on the recent occurrences:
  `log(n_c / n_w) > λ·(d_c − d_w) − β·δ_w`.
- **`λ = 0` (no positional preference):** pure frequency, `n_c > n_w`.
- **`λ → ∞`:** pure recency.

**Proof.** Three lines: softmax normalization `Z` is shared, so the comparison reduces to `A(c)` versus `A(w)`. Write it out fully in the paper.

**Probability form.** `log p(c)/p(w) = (γ/Z)·(A(c) − A(w))`. The 50% boundary is exact; the steepness depends on `γ/Z`.

**Corollary 1 (sums of copying heads are `n_c`-monotone).** Let the value logits be `Σ_h γ_h·A_h(v)/Z_h` with every `γ_h > 0` (several heads, possibly with different slopes `λ_h`). Turning one non-matching context position into an extra old copy of `c` raises `A_h(c)` and `Z_h` by the same amount and leaves `A_h(w)` unchanged, so `A_h(c)/Z_h` rises and `A_h(w)/Z_h` falls in every head that gives it nonzero attention. Hence `logit(c) − logit(w)` is **strictly increasing in `n_c`**. Bayes (Prop. 2) is flat in `n_c`, so reproducing Bayes needs something outside this family: a nonlinearity (MLP), a suppression head (`γ_h < 0`), or extra depth. This makes "MLPs/depth move the model toward Bayes" a prediction with a reason, not a hope. Caveat: a trained attention-only model is not guaranteed to stay inside the family (layer-1 heads can compose in other ways), so this is a prediction to test, not a theorem about trained models.

**Caveats found in the first run (Oct 5):**

- **Cor. 1 assumes the OV circuit is a plain copy.** With structured typos (`w = T(c)`), OV can learn "a copied `w` also votes for `T⁻¹(w) = c`", which is linear and escapes the corollary. At point A (h=0, ε=0.2) even 2-layer attention-only models match Bayes (they never flip to `w`). Point A is therefore a weak test; uniform typos are the cleaner frequency corner.
- **In the probability form, Prop. 1 saturates in `n_c`.** The logit difference is `γ·(A(c) − A(w))/Z`, a difference of attention *fractions*, so it rises with `n_c` and then flattens; it is not `log n_c`. The measured `n_c` effect (strong from 1→4, flat or slightly reversed at 8) fits this better than the log-count form.
- **ALiBi imposes an exponential distance kernel by construction.** A "flat then cliff" log-odds curve in `G` may come from the positional encoding rather than from what the model learned. A NoPE / learned-absolute control is required before claiming it as a general property.

**Caveat on distance form.** A sum of heads with different slopes is a mixture of exponentials in `d`, which can approximate a power law or `log d`. So a logarithmic-looking `G` dependence does **not** by itself show Bayes behaviour; `n_c`-sensitivity is the clean discriminator.

**Connection to statistics.** `A(v)` is exactly the exponentially discounted count, with discount `ρ = e^{−λ}` per token. That is the classical "exponential forgetting" estimator for drifting categorical data (Kulhavý & Zarrop 1993). With a BOS token you also get a pseudo-count, matching D'Angelo et al. 2026.

### 4.3 Proposition 2 — Bayes-optimal rule under switches and typos

**Generative model, per key.**

- A hidden current value `z_t ∈ {1..V}`.
- At each time step, with probability `h` the value is redrawn uniformly (a **switch**: a real change).
- Each observed value is `y = z` with probability `1 − ε`. Otherwise it is a **typo**:
  - *structured typo* (main setting): the fixed confusable variant `T(z)`, like Potter → Potyer;
  - *uniform typo* (ablation): a uniform random other value.

**Configuration.** `n_c` copies of `c`, then a gap of `G` steps, then `n_w` copies of `w = T(c)`, then the query.

**Result (structured typo).** Two hypotheses dominate:

- `H_c`: no switch in the gap, and the `w`'s are typos.
- `H_w`: a switch to `w` in the gap, and the `w`'s are clean.

Every other hypothesis is lower order in `h` and `ε`. Let `s = (1 − h)^G`. Then:

```
log P(z=c)/P(z=w) ≈ log( s/(1−s) ) − n_w·log((1−ε)/ε) + log V
```

**Decision rule.**

```
predict w  ⟺  s/(1−s) < (1/V)·((1−ε)/ε)^{n_w}
small hG:  ⟺  log(h·G) > log V − n_w·log((1−ε)/ε)
```

**Consequences, each testable:**

1. **Independent of `n_c`** once `n_c ≥ 2`. Under both hypotheses the old copies are explained equally well, so piling on more old copies does not help. This is the opposite of Prop. 1.
2. The gap enters as **`log G`**, not linearly. Recency becomes trustworthy once "a switch happened in the gap" is more likely than "`n_w` identical typos".
3. The critical gap **shrinks exponentially in `n_w`**: `G* ∝ V·(ε/(1−ε))^{n_w}/h`.
4. **Uniform typos** (ablation): identical repeated typos are very unlikely, so for `n_w ≥ 2` the rule almost always picks `w`. For `n_w = 1` the boundary is `h·G ≈ ε/(1−ε)`: switches versus typos.
5. **Limits.** `h = 0` gives counting (Dirichlet, Edelman et al.). `ε = 0` gives pure recency.

**Hidden value vs. observed token.** The rule above is about the hidden value `z`. The model predicts the *observed* token `y`, which goes through the typo channel: `p(y = w) = (1−ε)·P(z = w) + ε·P(z = c)` for `w = T(c)`. So the model-vs-oracle comparison must use the oracle's predictive over `y`. In particular, `log p(y=c)/p(y=w)` is capped at `log((1−ε)/ε)` even when `z = c` is certain. Appendix A returns the `z`-posterior; the code's oracle returns the `y`-predictive.

**Numerical check** (exact forward filter, V = 64, ε = 0.1; script in Appendix A):

- `n_c`-independence: log-odds for `n_c = 2, 4, 8, 16` were 2.215, 2.226, 2.227, 2.227.
- Critical gap `G*`, exact versus formula:

| h | n_w | exact | formula |
|---|---|---|---|
| 1e-3 | 1 | 2203 | 2092 |
| 1e-3 | 2 | 591 | 582 |
| 1e-3 | 3 | 85 | 84 |
| 1e-2 | 1 | 220 | 208 |
| 1e-2 | 2 | 58 | 58 |
| 1e-2 | 3 | 8 | 8 |

### 4.4 The contrast the experiments decide

**Primary discriminator: `n_c`-sensitivity** (Cor. 1). The `G`-form row is secondary (see the caveat on distance form).

| Probe manipulation | Discounted counter (Prop. 1) | Bayes (Prop. 2) |
|---|---|---|
| Double `n_c` (old copies) | favours `c` by `log 2` in score space | **no effect** |
| Increase gap `G` | linear: `−λ·ΔG` | logarithmic: `−log G` |
| Add one recent `w` | `+log((n_w+1)/n_w)` | `+log((1−ε)/ε)`, constant per copy |
| Change `h` in training | λ should rise with `h` | the boundary shifts by `log h` |
| Change `ε` in training | λ should fall with `ε` | the slope per `n_w` copy changes |

**Optimal λ for a discounted counter.** Real heads may be constrained to Prop. 1's form. So for each `(h, ε)`, numerically compute `λ*_DC(h, ε)`: the λ that minimizes expected cross-entropy *within the discounted-counter family*. Then compare the learned λ with that value too. This gives a "best achievable by one head" curve, and a gap to Bayes that depth or MLPs might close.

**Optional extra theorem** for a stronger paper: derive `λ*_DC` asymptotically for small `h` and `ε`. My guess is `λ* ≈ c·h/ε` up to log factors, but that has to be derived and checked numerically before claiming it.

---

## 5. Task design (replaces the old generator)

### 5.1 Key–value stream with hidden switching values ("switch-typo KV")

- **Vocabulary:** `K` key tokens, `V` value tokens (disjoint), a BOS token, and optionally `P` multi-token prefixes (§5.3).
- **Sequence:** `BOS, k_1, y_1, k_2, y_2, …` with `L_pairs` pairs (default 256 pairs = 513 tokens).
- **Per sequence and per key:** initial `z_k ~ Unif(V)`. At each pair step `t`, every key's `z_k` independently switches with probability `h`.
- **Pair `t`:** key `k_t ~ Zipf(K, α = 1)` with the key identities randomly permuted per sequence (default), or uniform; value `y_t = z_{k_t}` with probability `1 − ε`, otherwise `T(z_{k_t})` (structured) or uniform (ablation).
  - **Why Zipf is the default:** with uniform keys and K = 16, a key goes 128 pairs unseen with probability (15/16)^128 ≈ 3×10⁻⁴, so probes with G = 64–128 would test extrapolation. With Zipf(α=1) the rarest key has a mean gap of about 54 pairs and gaps ≥ 128 occur about 9% of the time. The per-sequence permutation stops the model from learning which key token is rare.
- **Loss on value positions only.** Keys are unpredictable by design, so mask them. Logits are restricted to the `V` value tokens at value positions, so initial loss ≈ `ln V` and model probabilities compare directly with the oracle's.
- **Defaults:** `K = 16`, `V = 64`, `T(z) = (z + 1) mod V`, `ε ∈ {0, .05, .1, .2, .3}`.
- **`h` is per pair step (global time), not per occurrence of a key.** A frequent key returns every ~3 pairs and a rare one every ~50, so the same `h` gives very different per-occurrence switch rates. Don't use `h ≥ 0.1`: the value then switches between most sightings of a rare key and there's little to learn. Choose `h` with the `G*` formula so the boundaries for `n_w ∈ {1..4}` fall inside 4–200 pairs (see §8.1).
- **Generate fresh data every step, on GPU.** That gives infinite data and no memorization confound.

### 5.2 Bayes oracle (exact, cheap)

- Each key is an independent HMM over `V` states; run the forward filter over that key's occurrences, using the gaps between them.
- Output the exact predictive `p*(y_t | history)` over the **observed** token at every value position (emission applied, see §4.3).
- From that, compute the **Bayes cross-entropy floor** and the **Bayes accuracy**.
- **Unit-test** the oracle against brute-force enumeration for tiny `V` and short sequences.

### 5.3 Multi-token prefixes (for the `β·δ` typo-in-prefix term, later)

- Keys become `k`-token prefixes (k = 2–3).
- With probability `ε_p`, one prefix token in a recent occurrence is corrupted.
- This links to the earlier multi-offset shift-head project and to D'Angelo 2026.

### 5.4 Controlled probes (evaluation only)

- Build sequences where the target key has `n_c` clean copies of `c`, then a gap of `G` pairs (filled with other keys), then `n_w` copies of `w = T(c)`, then the query.
- Grid: `n_c ∈ {1, 2, 4, 8}`, `n_w ∈ {1, 2, 3, 4}`, `G ∈ {2, 4, 8, 16, 32, 64, 128}`, with about 500 random fills per cell.
- Record `log p(c)/p(w)` from both the model and the oracle.

---

## 6. Models and training recipe

**Architecture.**

- Attention-only transformer (main), plus an "with MLP" variant.
- Layers ∈ {1, 2, 3, 4}; d_model = 128; 4 heads; pre-LayerNorm.
- Init std 0.02; untied embedding and unembedding.

**Positional encoding variants.**

- `alibi_learn`: one learnable slope per head, initialized log-uniform in [1/64, 1/4]. **λ̂ is a parameter you can read off.**
- `alibi_fixed`: the standard ALiBi slopes.
- `learned_abs`.
- `rope`.
- `nope`.

**Optimizer.**

- AdamW, lr 1e-3, betas (0.9, 0.98), weight decay 0.01.
- 500 warmup steps, cosine decay, grad clip 1.0.
- bf16 autocast; batch 256 sequences; 20k steps (tune with the gate).

**Sanity checks, enforced in code.**

- Initial loss ≈ `ln V` (±0.3); fail loudly otherwise.
- **Gate:** at `h = 0, ε = 0`, a 2-layer `alibi_learn` model reaches ≥ 99% accuracy **on repeat occurrences of a key** and its mean CE is within 0.02 nats of the Bayes floor.
  - The first sighting of each key in a sequence is unpredictable, so the Bayes floor here is not 0. It is ≈ (first sightings / value positions)·ln V, about 16/256 · ln 64 ≈ 0.26 nats with K = 16, Zipf keys and 256 pairs (the code computes it exactly). Accuracy on first sightings is ~1/V for everyone and is excluded.

**Logging.** Every 500 steps: loss, Bayes floor, excess loss, accuracy versus Bayes accuracy, the ALiBi slopes, and the induction score. Save to JSONL and checkpoints.

---

## 7. Measurements

1. **Excess loss over Bayes,** `CE_model − CE_Bayes`, in-distribution. This is the headline "did it learn" number.
2. **Probe log-odds surfaces** (model versus Bayes versus a fitted discounted counter).
3. **`n_c`-sensitivity index:** the slope of `log p(c)/p(w)` with respect to `log n_c`, at fixed `n_w` and `G`. The counter gives > 0; Bayes gives ≈ 0.
4. **Distance-form test:** fit the boundary `G*` as linear in `G` versus linear in `log G` (compare AIC/R²); also the scaling of `G*` with `n_w`.
5. **λ̂ (learned):**
   - read directly for `alibi_learn`;
   - for other positional encodings, regress the induction head's log-attention on same-key value positions against distance, with fixed effects for match;
   - plot λ̂ against `h` and `ε`, overlaid with `λ*_DC(h, ε)`.
6. **Prop. 1 fidelity:** predict each probe decision from the fitted head (λ̂, β̂) and report the agreement rate with the model's actual argmax.
7. **Mechanistic confirmation:**
   - find the induction head (prefix-matching score) and the previous-token head (K-composition);
   - mean-ablate each head and patch the `w`-block activations from a clean run;
   - check whether the head implementing the counter is causal for the decision.

---

## 8. Experiments (in order, with gates)

### 8.1 Current scope: reduced first run (do this first, nothing else until it's clean)

**Server constraint.** The A6000 is shared with an Ollama service that keeps ~37.6 GB of GPU memory resident (compute mostly idle). About 11.5 GB is free, so run **4 jobs in parallel**, checkpoint often, and let the queue retry runs that fail for lack of memory.

| Step | What | Grid | Runs | Est. time |
|---|---|---|---|---|
| 0 | Oracle + brute-force tests, data, model, train, probes | — | — | — |
| 1 | **Gate** | h=0, ε=0; L2 attention-only; alibi_learn | 2 seeds | < 30 min |
| 2 | **First run** | 4 `(h, ε)` points × {L2 attn-only, L2 + MLP, L3 attn-only} | × 3 seeds = 36 | ~3–4 h at 4 parallel |
| 3 | **Probes + rule identification** | all step-2 models | eval only | < 1 h |

**The four `(h, ε)` points** (V = 64; `G*` from the §4.3 formula, in pairs):

| Point | `(h, ε)` | Role | `G*` for n_w = 1, 2, 3, 4 |
|---|---|---|---|
| A | (0, 0.2) | typos only, no switches | ∞ (Bayes never trusts `w = T(c)`) |
| B | (0.03, 0) | switches only, no typos | 0 (Bayes always trusts the recent copy) |
| C | (0.01, 0.1) | mixed | 208, 58, 8, — |
| D | (0.03, 0.2) | mixed | 93, 53, 23, 7 |

Point A is the sharpest single test: with structured typos and no switches, Bayes says `z = c` with certainty for **any** `n_w`, while a counter with small λ flips to `w` once `n_w` copies outweigh the old ones.

**Hypotheses, written down before running.** Any combination of outcomes is reportable; the point is to tell the two rules apart, not to confirm these.

- **(a)** L2 attention-only has `n_c`-slope > 0 (counter behaviour), as Cor. 1 predicts.
- **(b)** Adding MLPs or a third layer moves the `n_c`-slope toward 0 (toward Bayes).
- **(c)** The induction head's learned slope λ̂ is higher at the switch-heavy points (B, D) than at the typo-heavy point A.

**Success criterion for "we're on the right path":** gate passes; in-distribution excess loss over Bayes is small at all four points; the probe `n_c`-slopes and boundaries are stable across seeds. If that holds, the full plan below is worth running.

### 8.2 Full plan (after §8.1)

| ID | What | Grid | Runs (×3 seeds) | Est. A6000 time |
|---|---|---|---|---|
| **E0** | Gate: clean induction | h=0, ε=0; L ∈ {1,2}; alibi_learn | 6 | < 30 min |
| **E1** | Corners | (h, ε) ∈ {(0, .2), (.05, 0), (.02, .1)}; L=2; 5 PEs | 45 | ~2 h |
| **E2** | **Main grid** | h × ε = 5×5; L=2; alibi_learn + rope | 150 | ~6–8 h, run 6–8 in parallel |
| **E3** | Probes and rule identification | all E1/E2 models | eval only | ~1 h |
| **E4** | Depth and MLP | 3 (h, ε) points; L ∈ {1,2,3,4}; attn-only vs +MLP | 72 | ~4 h |
| **E5** | Mechanism | best E2/E4 models | eval only | ~2 h |
| **E6** | Prefix typos (`β·δ`) | k=2,3 prefixes; ε_p sweep | ~36 | ~3 h |
| **E7** (opt.) | Pythia 70m–1.4b and checkpoints on synthetic probes | forward only | — | ~2 h |

**Notes.**

- Run several small jobs concurrently on the single A6000 (48 GB). Each 2-layer d=128 model uses well under 4 GB. Use a simple job queue (§9).
- Expected total: about 1.5–2 days of GPU time including reruns.

**Outcomes.** Each is a publishable finding:

- **(a) Counter rule.** Models implement the discounted counter, and λ̂ tracks `λ*_DC(h, ε)`. Message: "induction heads learn near-optimal forgetting within their family, but are provably suboptimal versus Bayes, because they count stale evidence."
- **(b) Depth or MLPs move behaviour toward Bayes.** Message: "depth buys the ability to discount frequency" (ties to the earlier depth thread).
- **(c) λ̂ doesn't track `h`.** Report which inductive bias dominates (for example, PE type or causal-mask primacy, per Wu et al. 2025).

---

## 9. Code spec for Claude Code

```
recfreq/
  README.md
  pyproject.toml            # torch, numpy, pandas, pyarrow, matplotlib, tqdm, pyyaml, pytest
  recfreq/
    data.py                 # GPU generator for switch-typo KV streams (+ multi-token prefix option); returns tokens, loss mask, metadata (key ids, hidden z, typo flags)
    oracle.py               # exact Bayes forward filter (batched torch), structured/uniform typo; returns predictive probs + CE floor
    counter.py              # discounted-counter predictor + fit of lambda*_DC(h, eps) by grid/LBFGS
    model.py                # attention-only / +MLP transformer; PE in {alibi_learn, alibi_fixed, learned_abs, rope, nope}; hooks returning attention patterns + per-head outputs; head ablation (zero/mean) + patching hooks
    train.py                # CLI: --h --eps --layers --pe --mlp --seed --steps ...; logs JSONL; asserts initial loss ~ ln V; saves ckpt
    probes.py               # build controlled probe batches (n_c, n_w, G grid), evaluate model/oracle/counter log-odds
    analysis/
      fit_lambda.py         # read slopes / regress attention vs distance
      rule_id.py            # n_c-sensitivity, linear-vs-log boundary fits, Prop.1 fidelity
      mech.py               # induction/prev-token scores, K-composition, ablations, patching
      plots.py              # figures for paper
  configs/                  # yaml grids for E0..E7
  scripts/
    queue.py                # run N jobs concurrently on one GPU (subprocess pool, CUDA_VISIBLE_DEVICES=0), skip completed runs, resumable
  tests/
    test_oracle.py          # vs brute-force enumeration (V=4, short seqs); limits h=0 -> counting, eps=0 -> recency
    test_counter.py         # Prop.1 boundary identity
    test_data.py            # statistics of switches/typos match h, eps
    test_model.py           # shapes, causal mask, alibi slope readout, initial loss ~ ln V
  results/                  # runs/<run_id>/{config.json, log.jsonl, ckpt.pt, probes.parquet}
```

**Conventions.**

- `run_id = f"{exp}_h{h}_e{eps}_L{L}_{pe}_{'mlp' if mlp else 'attn'}_s{seed}"`.
- Everything is deterministic given the seed.
- Fail fast on NaN.
- `queue.py --max-parallel 6`.

**Commands on the server.**

```bash
python -m pytest -q
python scripts/queue.py configs/E0.yaml --max-parallel 2      # must pass gate before anything else
python scripts/queue.py configs/E1.yaml --max-parallel 6
python scripts/queue.py configs/E2.yaml --max-parallel 8
python -m recfreq.analysis.rule_id results/ --out figs/
```

Use `tmux` or `nohup` and log to files.

---

## 10. Timeline (about 3 weeks)

- **Days 1–2:** `data.py`, `oracle.py` with tests, `model.py`, `train.py`. Pass E0.
- **Days 3–4:** E1 corners and probes. Check the theory on 3 points before the big grid.
- **Days 5–8:** E2 main grid, λ̂ versus `(h, ε)`, rule identification.
- **Days 9–11:** E4 depth/MLP and E5 mechanism.
- **Days 12–14:** E6 prefix typos and optional E7.
- **Days 15–21:** write the paper (4–8 page workshop format), make the figures, do a second seed sweep for error bars.

---

## 11. Risks and fallbacks

- **2-layer attention-only can't fit ε > 0 data well.**
  - Report the excess loss and compare with the counter family's best possible loss.
  - Add MLPs, which is RQ3 anyway.
- **λ̂ stuck near its init (ALiBi slope gets no gradient).**
  - Also try an init sweep. If it's still stuck, that's a result about learnability.
  - Use the attention-regression λ̂ for the other positional encodings.
- **Boundaries fall outside the context.**
  - Pick `h` so that `G*` lands in 4–200 pairs. The formula in §4.3 tells you `G*` in advance; compute it before choosing the grid.
- **Models learn Bayes perfectly, so there's no counter behaviour.**
  - Also a result ("transformers ignore stale frequency").
  - Then the depth sweep shows *where* that ability appears (1 versus 2 versus 3 layers).
- **Novelty challenge from D'Angelo 2026.** Our axis is recency and nonstationarity, which they don't study. Cite them for the overlap (`β·δ`) term.

---

## 12. Paper outline (workshop / working paper)

1. **Introduction.** Sir's Potter/Potyer example; recency versus frequency; LLM evidence (Qiao 2026, Naphade 2026).
2. **Setup.** Switch-typo KV task; Bayes oracle.
3. **Theory.** Prop. 1 (head = discounted counter, boundary formula); Prop. 2 (Bayes boundary, `n_c`-independence, log-G); the contrast table.
4. **Results.**
   1. Training reaches near Bayes in the corners.
   2. λ̂ versus `(h, ε)`.
   3. Rule identification (`n_c`-sensitivity, distance form).
   4. Depth and MLP.
   5. Mechanism.
5. **Related work.** §2.
6. **Limitations.** Toy scale; independent keys; uniform switches.

**Figures.**

1. Task schematic and phase diagram (Bayes).
2. Learned λ̂ heatmap over `(h, ε)` with the `λ*_DC` contour.
3. Probe surfaces: model versus Bayes versus counter.
4. `n_c`-sensitivity against depth.
5. Attention-versus-distance fit for the induction head.

**Venues.** Mechanistic-interpretability and ICL/theory workshops at the next ML conferences (ICLR/ICML 2027 workshop cycle). Check the current calls for papers for deadlines and page limits.

---

## 13. Plan B — memorization idea (only if Sir prefers it)

- Rerun the week-4 data-size sweep with a **learning-rate sweep** (1e-4 to 1e-2) and longer training, keeping the lowest-train-loss run per `T`, at m = 2, 5, 10.
- **Test for a first-order transition:**
  - a corner in min-train-loss versus `log T`;
  - a jump in `Σ D_i²/m`;
  - **hysteresis** (start runs from memorizing versus generalizing solutions near `T*`).
- Cite the Anthropic Circuits Update (July 2023) as the motivation. Their middle regime was partly an optimization artifact.

---

## Appendix A — oracle check script (used for §4.3 numbers)

```python
import numpy as np
V = 64
def post(obs, gaps, q, h, eps):           # structured typo: y = z w.p. 1-eps, y = z+1 w.p. eps
    b = np.ones(V) / V
    for i, y in enumerate(obs):
        if i > 0: s = (1-h)**gaps[i]; b = s*b + (1-s)/V
        e = np.zeros(V); e[y] += 1-eps; e[(y-1) % V] += eps
        b *= e; b /= b.sum()
    s = (1-h)**q; return s*b + (1-s)/V
c, w = 0, 1
def LO(nc, nw, G, h, eps, s=8, q=8):
    b = post([c]*nc + [w]*nw, [0] + [s]*(nc-1) + [G] + [s]*(nw-1), q, h, eps)
    return np.log(b[c] / b[w])
# n_c independence:   [LO(n, 2, 40, .002, .1) for n in (2, 4, 8, 16)] -> 2.215, 2.226, 2.227, 2.227
# boundary formula:   s/(1-s) = (1/V) * ((1-eps)/eps)**nw,  s = (1-h)**G
```

---

## Appendix B — prompt to start the Claude Code session

> Read `research_plan_recency_vs_frequency.md`. Build the `recfreq/` package exactly as in §9: oracle and tests first (`tests/test_oracle.py` must pass against brute force), then data, model and train. Run E0 locally on CPU with tiny settings to smoke-test, then give me the exact commands to run E0 and E1 on our A6000 server with `scripts/queue.py`. Enforce the initial-loss and gate checks from §6. Don't start E2 until E0 passes. Keep all results under `results/` as JSONL and parquet, and write `analysis/rule_id.py` so it produces the §4.4 contrast plots.

---

## 14. Run 3 — pre-registered predictions (written 5 Oct 2026, before any run-3 test was executed)

Context: analysis 3 found (i) behaviour is best fit by a Bayes filter with fitted (h′, ε′), h′ ≫ h and ε′ pulled toward ~0.25; (ii) mechanistically, 2–3 "copy heads" each act as a Prop. 1 counter, and their measured attention predicts the decision (R² 0.90–0.99). Run 3 tests these claims. For each test: the prediction, and what result would count **against** us.

1. **Causal ablation of copy heads** (zero their output; control = same number of non-copy heads in the same layers).
   - Predict: probe log-odds collapse toward 0 (|LO| shrinks by > 70% on average) and repeat-sighting loss rises sharply; control ablation changes |LO| by < 20%.
   - Against: copy-head ablation leaves the decision largely intact, or the control does as much damage.
2. **Slope surgery** (ALiBi models; multiply the copy heads' slopes by 0.5 and 2).
   - Predict: ×2 → fitted h′ increases and the switch-over gap G* shrinks; ×0.5 → h′ decreases and G* grows; monotone in all seeds.
   - Against: h′ / G* do not move, or move in the opposite direction.
3. **New probe layouts** (spacing 2 and 8; swapped layout where the old copies are the typo-variant of the recent ones). Fit each description on the original layout only, then predict the new layouts with no refit.
   - Predict: Bayes(h′, ε′) predicts the new layouts better (higher R²) than the best counter description, and its fitted (h′, ε′) on each layout separately stay within ~2× of the original.
   - Against: the counter generalizes better, or h′, ε′ change by much more than 2× between layouts (the description would then be layout-specific).
4. **Content vs distance** of the nearest-copy attention discount (swapped layout decouples "nearest" from "typo-variant").
   - No directional prediction; report whichever effect carries the discount.
5. **Bridge simulation** (CPU): random single counters and random sums of 2 counters, scored with the Bayes(h′, ε′) family.
   - Honest risk: if random counter sums are also fit by Bayes(h′, ε′) with R² as high as the models (~0.9), then "Bayes-shaped" is a generic property of counter sums and is NOT by itself evidence of Bayes-like computation; the informative content would then be the specific (h′, ε′) values. Report either way.
6. **More training / larger width** (C and D; L2; ALiBi and RoPE; 60k steps at width 128, and width 256 × 8 heads at 20k steps; 2 seeds).
   - Predict: h′ moves toward h by less than 2× (miscalibration mostly persists).
   - Against: h′ reaches within 1.5× of h — then the miscalibration is mainly under-training and must be framed as such.

### 14.1 Run 3 results against the predictions above (5 Oct 2026, after all runs finished)

| # | Verdict | Numbers |
|---|---|---|
| 1 | **Failed as pre-registered** (bad control) | copy-head ablation: median −43% |LO| (> 70% in 29% of 84 models); control changed |LO| by a median 32%. The control heads were themselves fast-fading copy-attending heads. Post-hoc: ablating all late heads removes the decision in every model (−96%). Post-hoc: removing a faster head shifts toward the old value in 2-layer models (positive in 83–100% of models), not in 3-layer models. |
| 2 | **Passed** | ALiBi slopes ×0.5/×1/×2 → median h′ 0.030/0.055/0.109, monotone in 56/57 models; G*(n_w=2) monotone in 39/39 with finite G*. Repeat-sighting loss changes only +0.01/+0.02 nats. |
| 3 | **Passed** | Bayes(h′, ε′) fitted on sp4 beats the best counter in 90% of 168 (model, layout) pairs (R² 0.93 on sp2/sp8 vs ≤ 0.78); swapped layout: lower error in 94% of 84 models; refit (h′, ε′) within 2× in 97%. Weaker for 2-layer learned-abs and 2-layer enriched (win rate 58%). |
| 4 | **New finding** | Typo-variant copies get less attention in 79–100% of heads; RoPE/learned-abs heads down-weight duplicate copies (per-copy attention ∝ n^−0.35…−0.68). Link to n_c-insensitivity suggestive only (ρ = 0.40 for copy 1→2 across models; n.s. for 2→4 and within encodings). |
| 5 | **Claim stands** | Random sums of 1–3 counters: counter R² ≈ 0.99, Bayes(h′, ε′) ≈ 0.74. Trained models: Bayes 0.94, counter 0.72. "Bayes-shaped" is not generic. The earlier explanation "sum of counters with different decays → Bayes shape" is refuted. |
| 6 | **ALiBi: persists; RoPE: nearly closes with width** | 3× training changes h′ by ×0.76–1.17; 2× width by ×1.12–1.63. ALiBi h′/h = 2.9–5.3; RoPE + width h′/h = 1.31 (C), 1.54 (D). |

Also measured: in ordinary training data 57–88% of repeat sightings have a conflicting history and Bayes ≠ "most recent value" at 4–14% of positions, so the earlier claim "conflicts are rare in training" was wrong; the better explanation is test 2's flat loss.

## 15. Run 4 — pre-registered predictions (written 5 Oct 2026 ~15:50, before any run-4 code was run)

Context: run 3 found that copy heads give typo-variant copies less attention (content effect) and that the models' behaviour is Bayes-shaped rather than counter-shaped. Run 4 asks where the content signal comes from and whether a content-aware counter closes the gap between mechanism and behaviour.

**A. Path ablation of layer-0 heads** (2-layer attention-only models only: ALiBi, RoPE, learned-abs, enriched). Each layer-0 head's output is mean-ablated separately on the query path, key path, value path of the layer-1 heads, or everywhere.
- A1 (sanity): ablating the previous-token head on the key path breaks induction (repeat-sighting loss rises by > 1 nat).
- A2: in at least 2/3 of models, some single layer-0 head other than the previous-token head, ablated on the query or key path, removes ≥ 50% of the typo-variant attention discount (|b_variant|).
- A3: in those models the same ablation makes behaviour more counter-like: in the swapped layout the log-odds start to depend on the number of old copies (slope of LO vs log2 n_c rises by ≥ 0.2), and the in-sample R² advantage of Bayes(h′, ε′) over the counter shrinks.
- Against: the discount is spread over several heads (no single head removes ≥ 50%), or removing it leaves behaviour unchanged — then the Bayes-like typo handling must come mainly from what heads write (OV), not where they look.

**B. Content-aware counter.**
- B1: a single counter with measured content terms (typo-variant attention offset, recent-block offset, duplicate exponent, OV typo-vote κ; 8 parameters), fitted on the original layout only, predicts spacing 2/8 within 0.03 R² of Bayes(h′, ε′) and has error on the swapped layout no worse than 1.5× Bayes(h′, ε′)'s. Against: it stays clearly worse (gap > 0.05 R²) — the measured content terms would then not explain the Bayes shape.
- B2: a mechanism-built prediction (each copy head's measured attention kernel and gain; only a global scale, an offset and κ fitted) reaches median R² ≥ 0.8 on spacing 2/8. Against: < 0.6.

### 15.1 Run 4 results against the predictions above (5 Oct 2026)

| # | Verdict | Numbers |
|---|---|---|
| A1 | **Passed** | previous-token head on the key path: loss + > 1 nat in 99% of 70 models (median +2.57). |
| A2 | **Passed** | one non-previous-token layer-0 head removes ≥ 50% of the typo-variant discount in 84% of models (83% with loss + < 0.5 nat); key path 51/58, query path 7/58. |
| A3 | **Failed** ("against" branch) | swapped-layout n_c slope rises ≥ 0.2 in 7% (median +0.01); Bayes-minus-counter gap median −0.005; h′ ×0.93. No single layer-0 head on any path makes behaviour counter-like (≥ 0.2 in 10%). |
| B1 | **Failed** | content-aware counter: R² gap to Bayes(h′, ε′) on sp2/sp8 = 0.106; swapped-layout error ratio 4.4. |
| B2 | **Failed** ("against") | mechanism-built prediction: median R² 0.365 on sp2/sp8. Post-hoc: fitted kernels reproduce measured attention splits at R² −0.49, while the same readout on measured attention gives R² 0.97. |

Post-hoc (labelled as such in the report): slow copy heads' attention splits are Bayes-shaped (R² 0.92 vs counter 0.43), fast heads counter-shaped (0.95); slower heads more Bayes-like within 93% of models. Removing the fastest head does not calibrate h′ in ALiBi models (closer in 7% of 42). OV weights show no clearly typo-specific vote (v−1 vs v+1 preferred in only ~50% of heads).

## 16. Run 5 — pre-registered predictions (written 5 Oct 2026 ~16:45, before any run-5 code was run)

Context: run 4 showed that the typo-variant attention discount comes from one layer-0 head on the key path, but removing it does not remove the Bayes-like behaviour; and that slow copy heads split attention between old and recent copies in a Bayes-shaped way that no distance-plus-offsets kernel captures. Run 5 asks how that split is computed. 2-layer attention-only models only (70).

**5A. Combination ablations of layer-0 heads.** Mean-ablate every layer-0 head except the previous-token head, together (on all paths; on the key path only; on the query path only; on the value path only), and every pair of non-previous-token heads on all paths (4-head models only).
- 5A-1: with all non-previous-token layer-0 heads ablated on all paths, induction still works (repeat-sighting loss rises by < 1 nat) in ≥ 2/3 of models.
- 5A-2: in those models the Bayes advantage disappears: the in-sample R² gap Bayes(h′, ε′) − counter falls below 0.05 (from ~0.1–0.3), in ≥ 2/3 of them.
- Against 5A-2: the Bayes advantage survives — then the Bayes shape is computed inside the layer-1 copy heads from the previous-token signal and position alone.

**5B. Score decomposition in slow copy heads** (heads in the slower half of each model's copy heads by fading rate). Split each attention score into the distance (ALiBi) part and the content part q·k, then split the content part by upstream source (token embedding incl. absolute position, previous-token head, other layer-0 heads, LayerNorm bias), once on the key side and once on the query side.
- 5B-1: the content part of the old-minus-recent split changes with the gap G (range > 1 nat over G = 2…64) in ≥ 2/3 of models.
- 5B-2: on the key side, most (≥ 50%) of that G-dependence comes from layer-0 head outputs at the recent copies, not from embeddings.
- Against: the content split is flat in G (then the G-dependence of the split comes only from distance and normalization), or its G-dependence comes mainly from embeddings/position.

### 16.1 Run 5 results against the predictions above (5 Oct 2026)

| # | Verdict | Numbers |
|---|---|---|
| 5A-1 | **Passed** | all non-previous-token layer-0 heads removed: induction works in 89% of 70 models (median +0.22 nats); learned-abs models break (+1.07). |
| 5A-2 | **Failed** ("against") | Bayes advantage < 0.05 in only 8% of working models; for ALiBi/RoPE the gap grows 0.24 → 0.39 (Bayes R² 0.95). The Bayes shape is computed inside layer 1 from previous-token match, position and token identity. |
| 5B-1 | **Failed narrowly** | content split range > 1 nat over G in 64% of models (median 1.53 nats vs 6.7 for distance). |
| 5B-2 | **Borderline** | key-side layer-0 share median 0.58 (≥ 0.5 in 63% of models). ALiBi: other layer-0 heads 0.78; RoPE: previous-token head 0.56 (confounded by rotation). |

Post-hoc correction (P1): against the proper log-ratio counter, slow ALiBi heads' attention splits fit Bayes better held-out (0.75 vs 0.53), slow RoPE heads too (0.84 vs 0.58); mid/fast ALiBi and learned-abs heads are counters. The original P1 numbers (0.92 vs 0.43) used a bounded counter form and overstated the difference.

## 17. Verification audit — pre-registered predictions (written 5 Oct 2026, before any audit check was run)

Snapshot of all results: `~/recfreq/snapshots/2026-10-05/` with `MANIFEST.txt` (sha256 of 647 files). All 123 runs complete; no non-finite values in any training log.

- **V2.3 reproducibility.** Retraining with the original seed gives h′ and ε′ within 15% of the original and a probe-surface correlation > 0.95. New seeds (3, 4) fall inside the range of the original seeds' h′ ± 30%.
- **V2.4 hyperparameters** (lr 3e-4, lr 3e-3, batch 256; C and D; L2 ALiBi and RoPE; 2 seeds). For ALiBi, h′/h > 2 in every run. Bayes(h′, ε′) beats the best counter on held-out gaps in ≥ 90% of runs. Against: h′/h ≤ 1.5 for some setting, which would make the miscalibration a hyperparameter artifact.
- **V4.3 null surfaces.** On generic smooth monotone surfaces (lo = A·tanh(a0 + a1·log2 G + a2·n_w + a3·log2 n_c) + B with random parameters, plus cell noise like the probes'), Bayes(h′, ε′) reaches median in-sample R² < 0.80, against 0.94 on the trained models. On cell-shuffled model surfaces, R² < 0.2.
- **V5.1 predictive law.** Leave-one-setting-out over the 12 grid settings: the laws h′ = α·h^β and logit ε′ = γ + δ·logit ε, fitted on 11 settings, predict the held-out setting's probe surfaces with median R² ≥ 0.75, and better than true Bayes with a fitted scale and offset.
- **V5.2 slope elasticity.** Scaling the ALiBi slopes of all late heads by s ∈ {0.25, 0.5, 0.71, 1, 1.41, 2, 4}: log h′ is linear in log s (median per-model R² > 0.9), with elasticity k in [0.7, 1.3].
- **V1.1 calibration.** Oracle predictive probabilities are calibrated on generator samples (ECE < 0.01), and mean −log p* matches the oracle CE within Monte-Carlo error.
- **V1.1 recovery.** BayesFree recovers known (h′, ε′) from noisy synthetic Bayes surfaces to within 25% (median) wherever the true boundaries fall inside the probe range.

### 17.1 Literature check (audit Phase 6, 5 Oct 2026)
All 17 cited works verified against arXiv / venue pages (titles, authors, IDs). Corrected above: Zhou et al. (title, venue), Makkuva et al. (venue title), Hegazy et al. (exact title, arXiv ID); Rajaraman et al. venue unconfirmed. Confirmed: D'Angelo et al. 2607.02800 (ICML 2026); Selective Induction Heads 2509.08184 (ICLR 2025 proceedings); Edelman et al. 2402.11004; Qin, Jiang & Zhu, ICLR 2026 = arXiv 2604.10946; Dudley et al. 2604.16988; Bajaj et al. 2510.22752; Mistry et al. 2502.06902 (first author Mistry); Bajaj et al. 2604.01094; Ekbote et al. 2508.07208 (NeurIPS 2025); Wu et al. 2502.01951 (ICML 2025); Qiao et al. 2603.12271; Naphade 2601.06189 (quote supported); Crosbie & Shutova, NAACL Findings 2025; Behrens, Biggio & Zdeborová, ICML 2025. Novelty search found no work on conflicting old/recent continuations with a Bayes switch/typo trade-off for induction heads; the closest new papers (2606.12058, 2508.03934) do not overlap.

### 17.2 Audit results against the §17 predictions (5 Oct 2026)

| Prediction | Verdict | Numbers |
|---|---|---|
| V1.1 calibration | **Pass** | ECE < 0.01 and mean NLL = expected entropy at 4 settings |
| V1.1 recovery | **Pass** | median error < 25% where identifiable; fits at noise floor (test criterion revised from absolute R² after diagnosis) |
| V2.3 reproducibility | **Pass** | same seed: h′ within 4%, ε′ within 2%, surface r ≥ 0.9998; new seeds inside range (4/4) |
| V2.4 hyperparameters | **Pass** | ALiBi h′/h 3.3–6.3 (> 2 everywhere); Bayes beats best counter 24/24; no h′/h ≤ 1.5 |
| V4.3 nulls | **Pass** | generic smooth surfaces 0.73 (< 0.80); shuffled 0.07 (< 0.2); models 0.95 |
| V5.1 predictive law | **Pass** | leave-one-setting-out median R² 0.867 vs true Bayes (in-sample a, b) 0.738 |
| V5.2 elasticity | **Pass on the median, narrowly** | log-linear R² 0.93; k median 1.29 (IQR 0.82–2.40); 2-layer attention-only k ≈ 1.1–1.2 |

Also: probes in-distribution (+0.004 nats); bf16 vs fp32 no effect (h′ ×0.995–1.011); 31/31 headline numbers reproduce; fit robust to optimizer (99%) with tight bootstrap CIs; laws h′ = 1.45 [1.29, 1.78]·h^0.74 [0.71, 0.78], logit ε′ = −0.35 [−0.47, −0.04] + 0.44 [0.35, 0.57]·logit ε (point estimates; an earlier version of this line gave the bootstrap medians 1.50 and −0.33, corrected 6 Oct).
Corrections made: per-copy evidence "about a third" → "about half" (median 0.46× exact Bayes); slope-surgery loss changes +0.006/+0.016; run-3 slope surgery strictly fails "all seeds" (56/57); **"the miscalibration is cheap" was wrong** — trained slopes are at the ordinary-data loss minimum in 85/85 ALiBi models and calibrating h′ costs +0.044 nats (median), so the miscalibration is loss-optimal for these heads; 3 citation fixes.

## 18. Law scope (run 6) — pre-registered predictions (written 6 Oct 2026 ~01:00 IST, before any E5 run was trained)

**Question.** Is the formula (Law 1) a real, predictive regularity, and how far does it reach? Law 1 was fitted on 2-layer attention-only ALiBi models, width 128, K = 16 Zipf keys, V = 64, N = 256, structured typos, at 12 settings h ∈ {0.003, 0.01, 0.03} × ε ∈ {0.05, 0.1, 0.2, 0.3}.

**Frozen law (L).** These constants are the point estimates from the 12 setting medians (36 models): α = 1.453, β = 0.736, γ = −0.345, δ = 0.438, a = 0.215, b = 1.433. Note: the paper's "1.50" and "−0.33" are bootstrap medians, not these point estimates; to be fixed in the write-up.
- For a model trained at (h, ε), the law predicts h′ = α·h^β and logit ε′ = γ + δ·logit ε.
- The predicted probe surface is ŷ = a + b·Bayes(h′, ε′) on the 112 cells, computed with that model's own V and typo type.
- Score: per-model R² against its stored probe surface. The law **holds** at a setting when the median over its seeds is ≥ 0.75 (the same threshold as V5.1).
- Models whose surface variance is < 0.05 are listed but excluded from R².

**Runs (89, L2 attention-only unless stated).**
- E5a, new (h, ε), 3 seeds:
  - interpolation: (0.005, 0.07), (0.02, 0.15), (0.007, 0.25);
  - extrapolation: (0.001, 0.1), (0.06, 0.1), (0.01, 0.02), (0.01, 0.4).
- E5n, context length N ∈ {192, 512}, 2 seeds: h ∈ {0.003, 0.01, 0.03} at ε = 0.1; ε ∈ {0.05, 0.2, 0.3} at h = 0.01. The N = 256 baselines exist.
- E5d, task dimensions at C (0.01, 0.1) and D (0.03, 0.2), 2 seeds: V = 32, V = 128, K = 8, K = 32, uniform key frequencies, uniform typos.
- E5r, RoPE at the 10 grid settings not yet run with RoPE, 2 seeds. C and D exist from E2a.

**Law predictions for E5a** (h′, ε′):

| Setting | h′ | ε′ |
|---|---|---|
| (0.005, 0.07) | 0.029 | 0.19 |
| (0.02, 0.15) | 0.082 | 0.25 |
| (0.007, 0.25) | 0.038 | 0.30 |
| (0.001, 0.1) | 0.009 | 0.21 |
| (0.06, 0.1) | 0.18 | 0.21 |
| (0.01, 0.02) | 0.049 | 0.11 |
| (0.01, 0.4) | 0.049 | 0.37 |

**Predictions.**
- **S1 (interpolation).** L holds at each of the 3 interior settings. The median fitted h′ is within ×2 of the law's h′, and the median |logit ε′ error| is < 0.5.
- **S2 (extrapolation).** Graded the same way per setting. Expectation: L holds at h = 0.001 and h = 0.06. We have no confident expectation for ε = 0.02 and 0.4, because only 4 ε values anchor the ε′ law.
- **S3 (form).** In ≥ 80% of the new runs with non-flat surfaces, Bayes(h′, ε′) has lower leave-one-G-out RMSE than both counter families (exp, pow).
- **S4 (context length).** Hypothesis H_N: the finite context sets the exponent. Its idea is that long-range survival beyond N is never observed, so h′ is floored at a scale that falls with N.
  - H_N predicts, at (0.003, 0.1), median h′ at N = 512 ≤ 0.77× that at N = 256, and at N = 192 ≥ 1.1× that at N = 256.
  - It also predicts that the h-exponent from the 3-point sweep at ε = 0.1 satisfies β_512 ≥ β_256 + 0.1 and β_192 ≤ β_256. β_256 on these 3 settings is 0.67.
  - **Supported** if both N = 512 conditions hold. **Refuted** if β_512 ≤ β_256 + 0.03 and the h′ ratio is > 0.9. Otherwise inconclusive.
  - Also reported: whether L holds at each N, and the ε′ slope δ_N from the ε-sweep.
- **S5 (task dimensions).** A variant's constants **transfer** if L holds at both C and D and the median h′ is within ×1.5 of the same-setting baseline (V64, K16, N256). Expectations, not graded:
  - V transfers;
  - K and uniform keys shift h′, direction unknown;
  - uniform typos do not transfer the ε′ law.
- **S6 (RoPE).** RoPE's law has the same form if:
  - log h′ vs log h fits RoPE's 12 setting medians with R² ≥ 0.8;
  - logit ε′ vs logit ε fits them with R² ≥ 0.8;
  - leave-one-setting-out with RoPE's own constants gives a median R² ≥ 0.75.
  
  Expectation: RoPE's h′/h is below ALiBi's at ≥ 9 of 12 settings.
- **S7 (stress tests on existing data, CPU; criteria fixed before computing).**
  - Leave-one-h-level-out: fit on 8 settings, predict the 4 settings of the held-out level. Median R² ≥ 0.75 for each held-out level.
  - Leave-one-ε-level-out: the same, per level.
  - Separability: keep the simple separable law unless cross terms raise the leave-one-setting-out median R² by ≥ 0.03.
  - Noise ceiling: the seed-to-seed surface R² at the same setting. Report the law's R² as a fraction of this ceiling.

**Deviation (6 Oct 2026, ~02:45 IST, before any E5 result was analysed).** The queue was trimmed from 89 to 63 runs so that it ends by about 07:30. Reason: GPU time; the N = 512 runs cost about 3× a normal run, not the 2× assumed.

Dropped:
- the ε-sweeps at N = 192 and N = 512 (12 runs);
- the second seed of the RoPE grid (10 runs);
- the second seed of the uniform-keys and uniform-typo variants (4 runs).

Graded criteria are unchanged:
- **S4** uses only the h-sweep at ε = 0.1, which is kept with 2 seeds at N = 192 and 512. δ_N is no longer reported.
- **S5:** the uniform-keys and uniform-typo variants are graded on 1 seed (noisier).
- **S6:** uses 1 seed at the 10 new RoPE settings, plus the 3 existing seeds at C and D.

Trimmed configs: `configs/E5n_trim.json`, `E5d_trim_a.json`, `E5d_trim_b.json`, `E5r_trim.json`. The originals are kept as the full design.

### 18.1 Run 6 results against the §18 predictions (written as each part finished, 6 Oct 2026)

**E5a, new (h, ε) — finished 02:53.** 21 models. One model at (0.01, 0.4) has a flat surface (variance 0.03) and is excluded from R². Frozen law, medians over seeds:

| Setting | Kind | Law R² | Fitted h′ / law h′ | logit ε′ error (abs.) | Own Bayes fit R² | Verdict |
|---|---|---|---|---|---|---|
| (0.005, 0.07) | interp. | 0.91 | 1.64 | 0.19 | 0.94 | holds |
| (0.02, 0.15) | interp. | 0.94 | 1.04 | 0.10 | 0.95 | holds |
| (0.007, 0.25) | interp. | 0.89 | 0.63 | 0.00 | 0.93 | holds |
| (0.01, 0.02) | extrap. | 0.88 | 1.72 | 0.12 | 0.95 | holds |
| (0.001, 0.1) | extrap. | 0.72 | 1.51 | 0.59 | 0.87 | fails (narrowly) |
| (0.06, 0.1) | extrap. | 0.73 | 0.98 | 0.84 | 0.97 | fails (narrowly) |
| (0.01, 0.4) | extrap. | −2.96 | 0.46 | 0.17 | 0.91 | fails |

- **S1: PASS (3/3).** All three interior settings meet R² ≥ 0.75, h′ within ×2 and logit error < 0.5.
- **S2: the law holds at 1 of 4 extrapolation settings** (ε = 0.02). The pre-stated expectation (holds at h = 0.001 and 0.06) was wrong for both, narrowly (0.72, 0.73).
- **S3: PASS.** Bayes(h′, ε′) beats both counters on held-out gaps in 20/20 new models.
- Over all 20 new models, the frozen law's median R² is 0.87. This equals the leave-one-setting-out estimate made before these runs existed (0.867).

**Post-hoc diagnosis** (`verify/law_scope_posthoc.py`; chosen after seeing the S2 failures):
- **The output scale is the main failure, not the beliefs.** The law fixes the affine scale at b = 1.43.
  - (0.01, 0.4): the models' surfaces are nearly flat (sd 0.36 nats), so R² is unstable. The law's absolute error is ordinary there (RMSE 0.71 nats vs 0.68 on the original grid). With a and b fitted per model, the law's beliefs reach R² 0.79, against 0.86 for the true (h, ε). The models' own scale is 0.87.
  - (0.001, 0.1): the models' scale is 4.8. With a and b fitted, the law's beliefs reach 0.74; true Bayes reaches 0.51.
- **ε′ also depends on h.** At (0.06, 0.1) the fitted h′ matches the law (0.18 vs 0.18), but ε′ is 0.105 against 0.21 predicted. The cross-term law, fitted on the original 12 settings only, so still out of sample, predicts this setting at R² 0.93 with frozen a and b. Over all new models, the cross-term law gives median R² 0.90 (frozen a, b) and 0.94 (a, b fitted), against 0.87 and 0.91 for the separable law. It is worse at h = 0.001 (0.44).
- **Summary.** The believed-parameter part of Law 1 transfers to new settings, including most extrapolations. The fixed output scale does not transfer to the edges (ε = 0.4, h = 0.001). A complete law needs a model of the scale b, and an h-dependence of ε′.

**E5n, context length — finished 04:53.** 12 models: h ∈ {0.003, 0.01, 0.03} at ε = 0.1, N ∈ {192, 512}, 2 seeds. The N = 256 baseline has 3 seeds.

| N | h′ at h = 0.003 | h′ at 0.01 | h′ at 0.03 | β_N (3-point slope) | Frozen-law R² (median) |
|---|---|---|---|---|---|
| 192 | 0.029 | 0.063 | 0.123 | 0.63 | 0.90 |
| 256 | 0.025 | 0.060 | 0.114 | 0.67 | 0.90 |
| 512 | 0.025 | 0.063 | 0.103 | 0.62 | 0.90 |

- **S4: H_N REFUTED by the pre-registered rule.** h′(N = 512)/h′(N = 256) at (0.003, 0.1) is 1.02, not ≤ 0.77. β_512 − β_256 = −0.05, not ≥ +0.1.
- The N = 192 side moved the way H_N said, but only slightly: h′ is 1.17× higher, and β is 0.03 lower.
- **Interpretation:** context length is not what sets h′ or the exponent in 192–512 pairs. h′ is invariant to doubling the context.
- **Positive scope result:** the frozen law transfers across context length (R² 0.90 at every N), and Bayes beats both counters in 12/12 models.

**E5d (part a), vocabulary and number of keys — finished 06:35.** 16 models: V ∈ {32, 128} and K ∈ {8, 32}, at C (0.01, 0.1) and D (0.03, 0.2), 2 seeds each. The frozen law is evaluated with the Bayes rule for the model's own V.

| Variant | Law R² at C, D | h′ / baseline h′ at C, D | Own Bayes fit R² | Verdict |
|---|---|---|---|---|
| V = 32 | 0.92, 0.94 | 0.89, 0.90 | 0.94, 0.96 | transfers |
| V = 128 | 0.91, 0.91 | 1.13, 1.15 | 0.93, 0.95 | transfers |
| K = 8 | 0.89, 0.92 | 1.10, 0.97 | 0.94, 0.96 | transfers |
| K = 32 | 0.92, 0.93 | 0.95, 1.07 | 0.94, 0.96 | transfers |

- **S5 (V, K): the constants TRANSFER for all four variants.** The law holds at both points, and h′ is within ×1.5 of baseline; in fact within 0.89–1.15.
- The expectation "K shifts h′" was wrong: h′ does not depend on the number of keys in 8–32.
- Bayes(h′, ε′) beats both counters in 16/16, and in 48/48 new models so far.
- **Post-hoc reading of the V result.** If the models ignored V, matching them with the V-aware Bayes rule would need h′ to scale roughly like V/64, i.e. ×0.5 at V = 32 and ×2 at V = 128. The observed change is ×0.9 and ×1.14, about a fifth of that. So the models' switch-over shifts with log V much as the Bayes boundary does.

**E5r, RoPE grid — finished 07:26.** 10 new settings × 1 seed (trimmed), plus C and D from E2a (3 seeds each).

- **S6: FAIL by the pre-registered rule.**
  - The h′ power law fits RoPE's 12 setting medians with R² 0.68 (needed ≥ 0.8).
  - The ε′ law fits with R² 0.91 ✓.
  - Leave-one-setting-out with RoPE's own constants gives a median R² of 0.89 ✓.
- **Post-hoc diagnosis of the failure.** One fit is not identifiable: (0.003, 0.3), one seed. It returned h′ = 0.0003 with scale b = 128 and offset −81. Without it, RoPE gives h′ = 1.26·h^0.88 with R² 0.89. The ALiBi grid's h′ law has R² 0.82 by the same computation.
- **Expectation met:** RoPE's h′/h is below ALiBi's at 12/12 settings. RoPE is 1.7–4.4×, excluding the unidentified point; ALiBi is 2.8–13.7×.
- **New:** RoPE's typo belief is almost calibrated, with logit ε′ = 0.01 + 1.02·logit ε. ALiBi's is compressed (slope 0.44).
- Bayes(h′, ε′) is again the best description: own-fit R² 0.93–0.97, and it beats both counters in every RoPE model.
- The frozen ALiBi law does not transfer to RoPE (median R² 0.73), as expected given the different constants.

**E5d (part b), key and typo distributions — finished 07:36.** 1 seed each, at C and D.
- **Uniform key frequencies: constants TRANSFER.** Law R² 0.93 and 0.83; h′ at 0.92× and 0.69× of baseline, inside ×1.5.
- **Uniform typos: constants do NOT transfer.** Law R² 0.17 and 0.34, as expected. The bigger finding: Bayes(h′, ε′) loses to the counters on held-out gaps in both models, and its fitted ε′ sits at the 0.5 cap.
- **Post-hoc check (`verify/uniform_typo_check.py`).** The 0.5 cap is right for structured typos but not for uniform ones, so we refitted all 5 uniform-typo models with ε′ up to 0.95: the 3 E2u models (h = 0, ε = 0.2) and these 2. ε′ moves to 0.72–0.88 and R² rises to 0.88–0.95, but the counters still win on held-out gaps in 4 of 5 (the exception is the E5d model at C).
- **Conclusion:** "Bayes with miscalibrated beliefs" describes models trained with structured typos (w = c + 1). With uniform typos, a discounted counter describes them better.
- **Interpretation (not tested):** run 3 found that the copy heads discount typo-variant copies by content. With uniform typos there is no recognisable variant to discount, so the heads stay counters.

**S3 final:** Bayes(h′, ε′) beats both counters in 60/62 new models with non-flat surfaces; the 2 exceptions are the uniform-typo models. **PASS.**

**Run 6 summary.**
- **The law predicts new models.** Median R² 0.87 over new (h, ε), 0.89–0.94 inside the fitted range.
- **Transfers unchanged:** context length 192–512, V 32–128, K 8–32, uniform key frequencies.
- **Extrapolates partially:** 1 of 4 out-of-range settings hold. The beliefs transfer; the single output scale does not.
- **Positional encoding:** RoPE needs its own constants (nearly calibrated ε′; h′ 1.7–4.4× h).
- **Limit:** the description holds only for structured typos.
- **Refuted:** the context-length explanation of the too-high h′.

## 19. Region of validity and explanation (run 7). Pre-registration started 27 Oct 2026, before any E6 run was trained

**Why.** After the progress talk, sir asked three things:
- Is the law correct, and is the derivation (Prop. 2) correct?
- Where exactly does the law work?
- Why does it work inside that region and fail outside it?

This round is open-ended and iterative. Each round's predictions are added here, dated, before that round's results are read.

**The law under test (frozen, §18):**
- h′ = 1.453·h^0.736
- logit ε′ = −0.345 + 0.438·logit ε
- a = 0.215, b = 1.433

**When does the law "work" at a setting (h, ε)?** The test uses 2-layer ALiBi models, V = 64, K = 16, N = 256, Zipf keys and structured typos. The decision is made on the median over seeds.
- **(i) Error:** the law's RMSE over the 112 probe cells is ≤ 1.25 × the RMSE of the models' own 4-parameter Bayes(h′, ε′) fit.
- **(ii) Measurability:** under the law's (h′, ε′), the exact Bayes switch point (LO = 0) lies inside G ∈ [2, 128] for at least 2 of the 4 values of n_w. Where (ii) fails, the setting is marked "not measurable with these probes" and only RMSE is reported, not R².

**E6 sweep.** 25 new settings × 2 seeds; seed 0 of every setting is trained first.
- h axis at ε = 0.1: h ∈ {0.0003, 0.002, 0.045, 0.1, 0.15}. Run 6 already has 0.001 and 0.06.
- h axis at ε = 0.2: h ∈ {0.0003, 0.001, 0.002, 0.045, 0.06, 0.1, 0.15}.
- ε axis at h = 0.01: ε ∈ {0.01, 0.35, 0.45}. Run 6 already has 0.02 and 0.4.
- ε axis at h = 0.003 and at h = 0.03: ε ∈ {0.01, 0.02, 0.35, 0.4, 0.45}.

**Prior expectation** (from the run-6 pattern, before any E6 result):
- At ε = 0.1, the law works for h from about 0.002 to 0.045 and fails beyond.
- Along ε, it works from 0.02 to 0.3 and becomes unmeasurable above about 0.35.

**Candidate explanation H-lin** (to be tested with a CPU surrogate and survival probes; its numerical predictions will be added here before E6 results are read):
- ALiBi copy heads can only make the log-odds "the old value is still current" linear in distance.
- Bayes needs that log-odds to fall roughly with log-distance.
- The loss-optimal line, fitted over the short repeat gaps of ordinary data, falls too fast at the probe gaps. This gives h′ > h, and the switch points for different n_w get re-spaced, which distorts ε′.

### 19.1 Round 1 results (27 Oct, 02:26–02:48 IST server time; no E6 result read)

**A. Prop. 2 redone** (`verify/prop2_exact.py`, `verify_out/prop2_exact.csv`). The two-path logic is right, but the paper's boundary drops three terms:
- **(a) Same-value redraw.** "No change" has prior s + (1−s)/V, not s.
- **(b) The old copies could all be typos of c−1** after a switch: a factor of 1 + ρ^{n_c}, with ρ = ε/(1−ε).
- **(c) The probes measure observed-token odds.** LO = 0 at P_c/P_w = (1−ε)/(1−2ε), not at 1.

The corrected boundary is s/(1−s) = [R(1+ρ^{n_c})(1−ε)/(1−2ε) − 1]/V, with R = ((1−ε)/ε)^{n_w}.

Against the exact oracle (920 cases with G* ≥ 4 pairs; h 1e-4–0.2, ε 0.01–0.45, n_c 1/2/8, n_w 1–6):

| Version | Median error in G* | Cases within 10% |
|---|---|---|
| Corrected | 0.9% | 91% |
| Paper (exact s) | 47% (median \|log ratio\| 0.38) | 20% |
| Paper (small-hG form) | ×5.8 | 3% |

- **Where the paper's version fails:** at ε ≥ 0.3 its median error is 77% and it is within 10% in only 2% of cases. Even at ε ≤ 0.2 it is within 10% in only 53%.
- **Corrected log-odds on the grid's probe cells:** median difference 0.012 nats, 90th percentile 0.13.
- **Remaining corrected-formula errors:** only where G* < 10, because it ignores switches inside the recent block.

**C1/C2 for H-lin: REFUTED as the main cause.**
- **(5) Direct test (survival probes, E1 C s0):** the model's belief that a single old copy is still current tracks exact Bayes within about 0.08 from d = 1 to 192 pairs. It is not "linear in distance, too fast".
- **The linear-survival surrogate** (Bayes filter with s = σ(c0 − λg), fitted by cross-entropy):
  - gives h′/h of only 1.0–3.7 (models: 3–14);
  - gives a log–log slope of 0.94 (models: 0.69);
  - gets the ε dependence of h′ with the wrong sign (+0.14 vs models' −0.38);
  - Spearman of h′ against the models is 0.80, which meets the criterion, but the magnitudes and the ε sign fail.

**What the probes actually show:**
- For n_w ≤ 3, the models hold a plateau near log((1−ε)/ε) up to G ≈ 16–32, then fall steeply between G = 32 and 128 to a floor of about −4 to −5 nats (−2.4 at ε = 0.3).
- Per extra recent copy, the switch point moves by a factor of about 2 (Bayes: about 9).
- The slowest copy-head ALiBi slope scales as h^0.375 (R² 0.86) and slightly *rises* with ε.

**New hypothesis H-support (post hoc, from existing data).** The models are calibrated where ordinary data contains the probe situation, and deviate where it is rare. We counted generator events matching each cell: a run of n_w sightings of w preceded by c = w−1, with the gap binned like G (`verify/support.py`).
- Over all 532 setting × cell combinations: Spearman(|LO_model − LO_bayes|, log rate) = −0.71.
- **Within each cell, across the 19 settings:** the median Spearman is −0.73, and 26 of 28 cells are negative.
- With cell fixed effects and |LO_bayes| controlled, each 10× increase in rate cuts the deviation by 1.2 nats.
- Cells with rate > 1e-3 per query deviate by 0.05–0.11 nats; cells with rate < 1e-4 deviate by 1.2–1.5 nats.
- This is consistent with E2c, where 20% Bayes-labelled conflict sequences halved the excess abruptness.

## 19.2 Round 2: pre-registered predictions (27 Oct 02:48 IST, before the E6 results and the dose runs are read)

**R2-P1, dose-response (causal test of H-support).**
- **Design:** retrain at C (0.01, 0.1) and D (0.03, 0.2) with conflict fraction f ∈ {0.01, 0.03, 0.1, 0.4} × 2 seeds, as config E7. The existing data are f = 0 (E1/E2b) and f = 0.2 (E2c).
- **Predictions, at both settings over the six doses:**
  - the seed-mean median over cells of |LO_model − LO_bayes| falls with f (Spearman ≤ −0.8);
  - h′/h at f = 0.4 is below h′/h at f = 0;
  - |logit ε′ − logit ε| at f = 0.4 is below its value at f = 0.

**R2-P2, collapse on new settings.**
- **The model:** dev = cell fixed effect + β·log10(rate), with β = −1.22, fitted on the 19 old settings (n_c = 2 cells).
- **Predictions:** it predicts the E6 models' per-cell deviations from their generator rates alone, with Spearman(predicted, observed) ≥ 0.6 over all E6 cells. Refitting β on E6 alone gives a value within ±50% of −1.22.

**Note.** A setting-level support summary (rate of n_w ≥ 2 conflicts) is dominated by typo runs and barely depends on h. So H-support in this form says nothing at the setting level about the h axis. That is a known gap, to be addressed with the per-cell, G-binned rates.

### 19.3 Round 2b: the probes' own coordinates, and an E6 pre-registration (27 Oct 02:55 IST; no E6 or E7 result read)

**Switch points** (`verify/switch_points.py`). For each model and n_w (n_c = 2), G* is the gap where its log-odds cross 0 (log-linear interpolation). We compare it with the exact Bayes G*. On the 19 old settings:
- **Over h:** the model's G*(n_w = 2) scales as h^−0.93 (Bayes: h^−1).
- **Over n_w:** each extra recent copy moves the model's switch point by about 0.5× as much as it moves Bayes's.
- **Over V:** at C, G*(n_w = 2) is 34 / 38 / 42 for V = 32 / 64 / 128 (Bayes: roughly ∝ V).
- **Over K:** the effect is small and inconsistent between C and D.
- **Survival probes** (round 1): the single-sighting time belief is near-Bayes.

**Natural law** (fitted on the 19 old settings, 170 model × n_w crossings inside the window):

  log G*_model = −0.739 + 0.933·log(1/h) + 0.327·log(1/r_B)

with r_B = [R(1+ρ²)(1−ε)/(1−2ε) − 1]/V, the corrected-Prop. 2 threshold for s/(1−s). Since Bayes has log G* ≈ log(1/h) + log(1/r_B), this reads: **the models get the time scale nearly right but weigh the evidence terms at about 1/3 strength.** The evidence terms are the recent copies, the typo channel and the prior 1/V of switching to exactly that value.
- In-sample R² 0.90; leave-one-setting-out median |log error| 0.155 (factor 1.17).
- **Interpretation (post hoc):** the frozen law in Bayes coordinates (h′ ∝ h^0.74, ε′ pulled toward 0.25, ε′ depending on h) is how "right time scale, under-weighted evidence" looks when forced into Bayes's (h′, ε′, a, b).

**R2-P3** (natural law on E6). With the coefficients frozen above, on the E6 models (n_c = 2, n_w = 1–4):
- **(a) Accuracy:** for crossings inside [2, 128], the median |log G*_obs − log G*_pred| is ≤ 0.30.
- **(b) Inside or outside the window:** whether a crossing exists in [2, 128] is predicted correctly for ≥ 80% of (model, n_w) pairs.
- **(c) Comparison with the frozen law:** on the same crossings, the natural law's median error is smaller than that of the frozen Bayes-coordinate law (the crossing of a + b·Bayes(h′_law, ε′_law)).

**R2-P4** (the frozen law's region, prior expectation). On E6, the frozen law meets criterion (i) at fewer than half of the 25 settings. Its error grows at the h and ε extremes. The natural law's error does not grow with h, because h enters with a fixed exponent.

**R2-P5** (dose and the natural law; written 27 Oct 02:55 IST, no E7 result read). Fit the natural law's two slopes separately per conflict fraction f, pooling C and D and seeds; only the intercept is shared within each f. The f values are 0 (E1/E2b, at C and D), 0.01, 0.03, 0.1, 0.4 (E7) and 0.2 (E2c).
- The evidence slope C2 rises with f: Spearman over f ≥ 0.8, and C2(f = 0.4) > 1.5 × C2(f = 0).
- **Caveat:** C1 cannot be estimated from only 2 h values, so the time slope is not tested here.

The idea behind this test: time is learned from abundant ordinary repeats, while evidence-combination is learned from rare conflicts. Extra Bayes-labelled conflicts should therefore mainly fix the evidence weight.

### 19.4 Round 2 interim results (27 Oct 03:10 IST; still no E6 or E7 result read)

**B1 survival probes** (`survival_probes.py`; 33 seed-0 models, `verify_out/survival_summary.csv`).
- **What was measured:** each model's belief that a value seen once d pairs ago still holds, read off as s = (M − 3/V)/(1 − 3/V) with M the mass on c−1, c, c+1.
- **ALiBi:** it tracks exact Bayes in all 19 settings, including the settings where the law fails (h = 0.001, h = 0.06, ε = 0.4). RMSE in s is 0.014–0.068 (median 0.048); the largest gap at any d ≤ 192 is 0.16.
- **RoPE:** RMSE 0.010–0.031.
- **C2.5 (pre-registered direction for H-lin): FAIL.** For ALiBi, logit s is closer to linear in log d than to linear in d in 13 of 19 models.
- **Conclusion:** the models' sense of time is calibrated. The miscalibration is not in time.

**B1 wide window** (G ∈ {192, 224}, n_w ∈ {5, 6}).
- **ε = 0.4:** the models are not flat. They have clear switch points (G* ≈ 140, 80, 51, 35 for n_w = 1–4 at h = 0.01) and fall to about −1.5 nats. Bayes's log-odds there stay within about ±0.41, so a + b·Bayes cannot follow them, whatever the beliefs.
- **h = 0.001:** n_w = 1 never crosses by G = 224 (LO still +1.0); n_w = 2 crosses at G ≈ 149.
- **The natural law on these unseen crossings** (n_w = 5, 6 or G > 128): median |log error| 0.32 (factor 1.38; n = 23), against 0.14 on the crossings it was fitted on.

**Natural-law slopes by encoding** (exploratory). Using log G* = c + a·log(1/h) + b·log(1/r_B):
- ALiBi: a = 0.93, b = 0.33 (R² 0.90).
- RoPE: a = 0.84, b = 0.70 (R² 0.91).
- Learned absolute: a = 0.49, b = 0.28. Only 2 h values, so its a is not identifiable.

RoPE's near-calibration in Bayes coordinates is mainly a larger evidence weight.

**Attention and evidence** (post hoc, `verify/attn_evidence.py`, run-3 attention, short gaps G ≤ 8).
- In ALiBi copy heads, the old/recent attention split moves by 0.41× Bayes's per-copy evidence per extra recent copy (median per model; heads 0.30–0.54). This is close to ALiBi's behavioural evidence weight of 0.33.
- RoPE heads show about the same attention sensitivity (0.36×), yet weigh evidence at 0.70 behaviourally. So attention alone does not explain the encoding difference; the value/output side must contribute. Open.

## 19.5 Round 3: the low-h edge (pre-registered 27 Oct 03:18 IST, before any E8 run)

**Partial E6 read so far:** only wave 1, 8 seed-0 models on the h axis.
- **At h ≥ 0.045,** the natural law's switch points hold (n_w = 1: 18 / 8.0 / 5.3 observed vs 16 / 7.7 / 5.3 predicted at h = 0.045 / 0.1 / 0.15).
- **At h ≤ 0.001,** the models switch much earlier than the law's h^−0.93 scaling: at h = 0.0003, n_w = 4, observed 101 vs predicted 195. The model's clock stops slowing down below h ≈ 0.002.
- **The copy-head slopes still fall with h** (0.044 / 0.032 / 0.022 at h = 0.003 / 0.001 / 0.0003; about 0.002 at h = 0). So this is not a slope floor.

**H-ctx.** At very small h a key switches less than about once per context (hN < 0.5), so the long time scale gets little training signal.

**E8 design:** N = 512 at h ∈ {0.0003, 0.001}, ε = 0.1, seeds 0 and 1. Compare with N = 256: E6 at 0.0003 and E5a at 0.001.

**R3-P1.** If H-ctx holds, the geometric mean over seeds of G*_model at n_w = 3 and 4 rises to at least 1.3× its N = 256 value, at both h. If the ratio stays within [0.8, 1.25] at both h, the edge is not set by context length.

## 19.6 Round 4: what sets the low-h edge, gap support or context length? (pre-registered 27 Oct 05:08 IST)

**Read before writing this:** the region map on 16 of 25 E6 seed-0 models (`verify/region_map.py`) and per-model switch-point spacing (`verify/switch_points.py`). **Not read:** survival probes on E6 (running now), any E7 or E8 model, E6 seed 1.

**What the partial map shows:**
- **h′ extrapolates well.** The frozen h′ law matches the fitted h′ from h = 0.002 to 0.15 (h = 0.045 / 0.06 / 0.1 / 0.15: fitted 0.156 / 0.18 / 0.27 / 0.34 vs law 0.148 / 0.18 / 0.27 / 0.36).
- **Below h ≈ 0.002, h′ levels off at about 0.01** (fitted 0.013 / 0.009 at h = 0.0003, ε = 0.1 / 0.2).
- **ε′ is what fails away from the fitted h range.** It depends on h, which the separable law ignores. At ε = 0.1, ε′ falls from about 0.33 (h ≤ 0.001) to 0.08 (h = 0.15). The run-6 cross-term law (fitted on the original 12 settings, so out of sample here) has half the ε′ error, but still misses the drop at h ≥ 0.06.
- **Evidence weight rises with h.** Per-model evidence weight w = (switch-point spacing in log G per recent copy) / log((1−ε)/ε) rises with h: 0.10 / 0.27 / 0.34 / 0.46 / 0.74 at h = 0.001 / 0.003 / 0.01 / 0.03 / 0.1 (Spearman with log h 0.80, n = 58). The clock slope near the crossing (d LO / d log G) matches Bayes where it can be measured. So "evidence at about 1/3" (§19.3) is an average over an h-dependent weight.

**H-gap (new, competing with H-ctx).** The model's clock is calibrated only over gaps that training data contains often enough. With Zipf keys, K = 16 and N = 256, only 0.3% of repeat gaps are ≥ 128 pairs. At low h, Bayes still trusts a 128-pair-old copy (s ≈ 0.96), but the model has almost no training signal there, so its clock falls away beyond G ≈ 100. This gives the switch near G ≈ 100 seen at h ≤ 0.001. Each key switches at rate h on its own, so K changes long-gap support (P(gap ≥ 128) = 0.0001 / 0.0034 / 0.016 / 0.031 for K = 8 / 16 / 32 / 64) without changing the switches per key per context, hN. H-ctx predicts no effect of K. (N = 512 doubles long-gap support, so E8 alone cannot tell the two apart.)

**R3-P2 (E6 survival probes, H-gap).** For h ≤ 0.001 (the 3 seed-0 models at ε = 0.1 and 0.2):
- the model's single-sighting survival s(d) is within 0.10 of Bayes at d ≤ 32;
- it is at least 0.15 below Bayes on median over d ∈ [128, 248].

For h ≥ 0.045, median |s_model − s_Bayes| over d ≤ 248 is ≤ 0.10.

**E9 design:** K ∈ {8, 32, 64}, Zipf keys, N = 256, h ∈ {0.0003, 0.001}, ε = 0.1, seeds 0 and 1 (12 runs). The K = 16 references are E6 (h = 0.0003) and E5a (h = 0.001). Switch points are read with the standard and wide-window probes.

**R3-P3.** Take the geometric mean over seeds and over n_w = 3, 4 of G*_model. If H-gap holds, then at both h:
- K = 32 gives at least 1.3× the K = 16 value;
- K = 8 gives at most 0.8×.

If H-ctx holds, both ratios lie in [0.8, 1.25]. K = 64 is exploratory.

**R3-P4.** On E6 models not yet read (remaining seed 0 and all of seed 1), Spearman(log h, w) ≥ 0.6.

**R3-P2 verdict (05:17 IST): FAIL.** Survival probes on the low-h E6 models show a calibrated single-sighting clock even far beyond the gaps training contains:

| Model (seed 0) | max \|s_model − s_Bayes\|, d ≤ 32 | median (s_Bayes − s_model), d ∈ [128, 248] |
|---|---|---|
| h = 0.0003, ε = 0.1 | 0.011 | −0.013 |
| h = 0.0003, ε = 0.2 | 0.018 | −0.021 |
| h = 0.001, ε = 0.2 | 0.014 | −0.036 |

The model is slightly *above* Bayes at long range, not 0.15 below. So H-gap, in its clock form, is refuted: a single old memory is trusted correctly out to d = 248 even at h = 0.0003.

What breaks at low h is how an old memory and new copies are combined. At h = 0.0003, ε = 0.1:
- one extra recent copy moves the log-odds by about 0.05 nats at G ≤ 32, but by 1.4 nats at G = 128 (n_w 3 → 4);
- Bayes moves it by about 2.2 nats at every G.

So time and evidence interact in the model, whereas Bayes adds them. R3-P3 (E9) stays as registered, since it is a test about the conflict switch points, not the clock.

**R3-P2, second part (05:30 IST): PASS** for h ≥ 0.045. The median |s_model − s_Bayes| is 0.003–0.011 for all 7 models, against a limit of ≤ 0.10.

Across every seed-0 model with survival probes (49 models; h 0.0003–0.15, ε 0.01–0.45; ALiBi, RoPE and learned absolute positions), the median |s_model − s_Bayes| is ≤ 0.041. This includes every setting where the law fails. **The single-memory clock is calibrated over the whole map**, so the law's failures cannot come from the model's sense of time for one memory.

### 19.7 E6 seed-0 verdicts (27 Oct 06:39 IST; all 25 seed-0 models; seed 1 still training)

**R2-P3 (natural law on E6)**

| Part | Result | Criterion | Verdict |
|---|---|---|---|
| a | median \|log err\| 0.290 | ≤ 0.30 | **PASS** |
| b | inside/outside window correct 0.76 | ≥ 0.80 | **FAIL** |
| c | on 49 shared crossings, natural 0.284 vs frozen law 0.245 | natural < frozen | **FAIL** (the frozen law is better) |

**R2-P2 (support collapse on E6, 25 models, 700 cells)**

| Part | Result | Criterion | Verdict |
|---|---|---|---|
| a | Spearman(predicted, observed deviation) 0.668 | ≥ 0.6 | **PASS** |
| b | β refitted on E6 = −1.03 | within ±50% of −1.22 | **PASS** |

For comparison, the cell effect alone gives Spearman −0.25. So a model's per-cell deviation from Bayes is predicted, on unseen settings, by how often training data contains that conflict configuration.

**The pre-registered "works" criterion (§19) is too strict.** It requires law RMSE ≤ 1.25× the model's own fit. The law passes it in only 4 of its own 12 fitting settings (median ratio 1.43 there), and in 8 of 44 settings overall.

**Post-hoc criterion** (labelled as such): "as good as inside its fitting box", meaning the ratio is ≤ 1.51, the 75th percentile of the in-box ratios. Under it, 8 of 32 outside-box settings qualify:

| Region | Ratio | Qualifies |
|---|---|---|
| h ∈ [0.001, 0.045] at ε = 0.07–0.25 | — | yes, for all except h = 0.002 and 0.045 at ε = 0.1 (1.61, 2.42) |
| ε ≥ 0.35 | 2.7–11.7 | none |
| ε ≤ 0.02 | 1.9–2.9 | none, except (0.01, 0.02) at 1.47 |
| h ≥ 0.06 | 2.6–6.5 | none |
| h = 0.0003 | — | not measurable |

**Evidence weight w at the ε edges** (new models): w = 1.03–1.86 at ε ≥ 0.35 (h = 0.003 and 0.03), and 0.19–0.45 at ε = 0.02. R3-P4 is not decidable on seed 0 alone: only 6 new models have a w, at 2 values of h. It waits for seed 1.

**R3-P1 verdict (27 Oct 07:17 IST): FAIL.** E8, 4 models, with crossings read from the standard and wide-window probes. Geometric mean G* at n_w = 3, 4:

| h | N = 512 | N = 256 | Ratio | Needed under H-ctx |
|---|---|---|---|---|
| 0.0003 | 157 | 117 (E6 seed 0 only; seed 1 pending) | **1.34** | ≥ 1.3 ✓ |
| 0.001 | 86 | 77 (E5a, 3 seeds) | **1.13** | ≥ 1.3 ✗ (inside the no-effect band [0.8, 1.25]) |

H-ctx required ≥ 1.3 at both h, so it fails. The no-effect outcome also did not occur at both h. So context length has at most a small effect, and only at h = 0.0003. N = 512 also doubles long-gap training data (P(gap ≥ 128) from 0.0034 to 0.0065), so this small effect is equally consistent with H-gap's support story. R3-P3 (E9) separates the two.

## 19.8 Round 5: is the low-h edge undertraining? (pre-registered 27 Oct 07:46 IST, before any E10 run)

**What was read before writing this:**
- **R3-P1:** context length has a small effect (×1.34 / ×1.13).
- **R3-P3 interim** (seed 0, wide probes): K = 32 moves G* by only ×1.12 / ×1.16, despite ≈ 4.6× more long-gap training data. So gap support is not the main cause either.
- **Training curves:** the excess loss over Bayes is still falling at 20k steps at every h (h = 0.0003: 0.0067 → 0.0053 → 0.0040 at 10k / 15k / 20k).

**H-signal.** The evidence weight w is learned from switch events. Their number per training step is proportional to h. At low h, 20k steps contain too few of them, so w stays low (0.10–0.17 at h ≤ 0.001, against 0.34 at h = 0.01).

**E10 design:** 80k steps (4×, same LR schedule stretched) at h ∈ {0.0003, 0.001, 0.01}, ε = 0.1, seed 0. w is read with the standard and wide probes, exactly as in `verify/evidence_weight.py`. The 20k-step references are the medians of E6 s0 (h = 0.0003, w = 0.135), E5a (h = 0.001, 0.102) and E1 (h = 0.01, 0.338).

**R5-P1.** If H-signal holds:
- w(80k) / w(20k) ≥ 1.5 at both h = 0.0003 and h = 0.001;
- each of those ratios exceeds the h = 0.01 ratio.

If both low-h ratios lie in [0.8, 1.25], the low-h weighting is what training converges to, not undertraining.

**R3-P3 interim verdict (27 Oct 08:13 IST; seed 0 only, standard + wide probes): FAIL for H-gap.** Ratio of G* (n_w = 3, 4) to K = 16:

| K | Long-gap support vs K = 16 | h = 0.0003 | h = 0.001 |
|---|---|---|---|
| 32 | ≈ 4.6× | 1.12 | 1.16 |
| 64 | ≈ 9× | 1.11 | 1.20 |
| 8 | ≈ 0.03× | 0.65 | 0.83 |

- **H-gap required** K = 32 ≥ 1.3 at both h, and K = 8 ≤ 0.8 at both h. Only K = 8 at h = 0.0003 meets its part.
- **H-ctx's no-effect band** [0.8, 1.25] holds for K = 32 and 64, but not for K = 8 at h = 0.0003.

**Reading:** long-gap support matters only when it is almost absent (K = 8). Beyond the K = 16 level, 9× more long-gap data barely moves the switch point. So the low-h edge is set by neither context length nor long-gap support. Final after seed 1.

### 19.9 Dose verdicts (27 Oct 08:21 IST; E7 seed 0 complete, seed 1 still training; f = 0 and 0.2 have 3 seeds)

**R2-P1** (`verify/dose.py`)

| Setting | a: dev falls with f (Spearman ≤ −0.8) | b: h′/h lower at f = 0.4 | c: ε′ closer to ε at f = 0.4 |
|---|---|---|---|
| C | **FAIL**: Spearman +0.31; dev 0.74 / 0.32 / 0.34 / 0.66 / 0.74 / 0.70 at f = 0 / 0.01 / 0.03 / 0.1 / 0.2 / 0.4 | **PASS**: 5.95 → 4.66 | **PASS**: \|Δlogit\| 1.01 → 0.31 |
| D | **FAIL**: dev *rises*, Spearman +0.83, 0.31 → 0.35 | **PASS**: 3.42 → 2.73 | **FAIL**: 0.36 → 0.57 |

**R2-P5: FAIL.** The evidence slope C2 is 0.35 / 0.48 / 0.54 / 0.47 / 0.48 / 0.50 at f = 0 / 0.01 / 0.03 / 0.1 / 0.2 / 0.4. That is a step up at any nonzero f, not a graded rise: Spearman 0.54, C2(0.4) / C2(0) = 1.41.

**Reading:**
- Extra Bayes-labelled conflict training raises the evidence weight a little (about 0.35 → 0.5) and lowers h′/h by about 20%. More of it does not help further.
- Even at f = 0.4 the models stay far from Bayes (h′/h 2.7–4.7).
- So rarity of conflicts in training is a real but partial cause. It predicts *which* cells deviate on unseen settings (R2-P2 PASS), but adding conflicts does not remove the deviation.
- The rest looks like a limit of what 2-layer ALiBi attention learns to represent: RoPE models reach w ≈ 1 on ordinary data.
