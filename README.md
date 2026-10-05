# Recency vs frequency in induction heads

Nirbhay Sharma and Gathik Jindal · TOLLM course project (Semester 7)

When a context holds a continuation seen often but long ago ("Potter" ×5) and a conflicting one seen once, recently ("Potyer" ×1), which one does an induction head copy? We study this in a synthetic key–value task with two noise sources:
- **switches:** a key's hidden value is redrawn with rate *h*, so recency is right;
- **typos:** a copy is corrupted with rate *ε*, so frequency is right.

The task has an exact Bayes-optimal predictor, so every trained model can be compared with the best possible rule.

## Main findings so far

Trained 2–3 layer transformers (ALiBi, RoPE, learned positions) behave like the Bayes rule evaluated at the wrong parameters. They act as if switches are more common than they are, and they pull the typo rate toward about 0.25.

**The law.** Fitted on 2-layer ALiBi models (point estimates, with seed-resampled 95% CIs):
- h′ ≈ 1.45 · h^0.74, with coefficient CI [1.29, 1.78] and exponent CI [0.71, 0.78];
- logit ε′ ≈ −0.35 + 0.44 · logit ε, with intercept CI [−0.47, −0.04] and slope CI [0.35, 0.57].

**Supporting evidence:**
- **Prediction:** the law predicts the probe behaviour of models trained at a held-out setting at R² 0.87, which is 89% of the seed-to-seed ceiling.
- **Cause:** rescaling the copy heads' ALiBi slopes moves h′ as h′ ∝ s^k with k ≈ 1.1–1.3. The trained slopes minimise ordinary-data loss, so the miscalibration is what training prefers.
- **Mechanism:** the decision is made by 2–3 late heads acting as distance-discounted counters, logit(c) − logit(w) ≈ Σ γ_h (A_c − A_w), with R² 0.90–0.99.

**Verification.** All claims were pre-registered in [the research plan](research_plan_recency_vs_frequency.md), §14–§18, and audited:
- 36 unit tests;
- null models;
- retrains with new seeds, learning rates and batch sizes;
- 31/31 headline numbers recomputed from raw files.

See the audit report for details. Failed predictions are reported alongside the successes.

**Status:** run 6 is in progress (§18, law scope). It tests the law on new (h, ε) values, context lengths N = 192 and 512, V = 32 and 128, K = 8 and 32, uniform keys and typos, and a RoPE grid. Its results will be added here when the runs finish.

## Repository layout

| Path | Contents |
|---|---|
| `research_plan_recency_vs_frequency.md` | Research plan, every pre-registration and every verdict (§14–§18) |
| `paper/` | Workshop paper draft (LaTeX: `main.tex`, `refs.bib`, `figures/`); `paper_overleaf.zip` is the same draft for Overleaf; `paper_draft_preview.docx` is a Word preview |
| `slides/` | Progress deck: `deck.html` (open in a browser; print gives one slide per page), `speaker_notes.md`, slide sources in `source/` |
| `recfreq/recfreq/` | Python package: data generator, exact Bayes oracle, models, training, probes, rule fitting, mechanism analyses |
| `recfreq/recfreq/verify/` | Verification audit and law-scope checks |
| `recfreq/tests/` | Unit tests (`pytest -q`) |
| `recfreq/configs/` | Experiment grids E0–E5 |
| `recfreq/scripts/` | Job queue and run orchestration |
| `recfreq/results/runs/` | One folder per trained model: `config.json`, `done.json` (final metrics, heads), `log.jsonl` (training curve), `probes.parquet` (conflict-probe results), `stdout.log`. Checkpoints (`ckpt.pt`, ~400 MB total) are not in the repo |
| `recfreq/figs*/`, `recfreq/paper_figs/` | Analysis outputs and figures per run |
| `recfreq/verify_out/` | Audit outputs (claims registry, nulls, predictive law, stress tests) |
| `recfreq/report/` | HTML reports: `first-run-report.html` (all runs) and `verification-audit.html` |
| `recfreq/snapshots/` | Frozen copy of run summaries with SHA-256 manifest, used by the audit |
| `generate_induction_datasets.py`, `models.py`, `week2 induction head.ipynb`, `paper2.pdf`, `try-1.pdf`, `plan_eval.md` | Earlier project files |

## Reproducing

```bash
cd recfreq
python -m venv .venv && .venv/bin/pip install torch numpy scipy pandas pyarrow matplotlib pytest
.venv/bin/python -m pytest -q
.venv/bin/python scripts/queue.py configs/E1.json --max-parallel 6
.venv/bin/python -m recfreq.fit_rules results/runs --exp E1 --out figs_fit
.venv/bin/python -m recfreq.verify.law_scope --out verify_out
```

Training one model (20k steps, width 128) takes about 5–6 minutes on an RTX A6000.
