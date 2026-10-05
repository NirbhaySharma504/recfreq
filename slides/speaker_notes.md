# Recency vs Frequency: Progress Deck — speaker notes

## 1. Which memory wins?

Progress update on the recency-vs-frequency project: where the idea came from, what we built, what we tested, what holds up, and whether we are ready to write the paper.

## 2. Old and frequent, or new and rare?

The whole project is one question. A language model sees a name many times long ago and a slightly different spelling once, recently. Whether it should trust the old or the new copy depends on the world: typos favour the old, frequent copy; real changes favour the recent one. Induction heads do this copying in transformers, so we asked what rule they use.

## 3. Two candidate rules, and our bets

Two rules make different predictions. A counter, which is what a single head can compute, keeps getting more confident as old copies pile up. The exact Bayes answer stops caring after two old copies: what decides is how long ago they were (G) and how many recent copies there are. We wrote down three bets before running anything.

## 4. A synthetic world where we know the right answer

We generate our own data so that we know the right answer exactly. Each key has a hidden value. Occasionally it switches to a new value (rate h); occasionally it is shown with a typo (rate ε). Every training step uses brand-new sequences, and an exact Bayes filter tells us what an ideal predictor would say at every position.

## 5. Conflict probes: old evidence against new

The probe puts the two kinds of evidence head to head. A fresh key τ is shown with value c a few times, then other keys for a gap of G pairs, then with value w a few times, then queried. We read off how much the model prefers c over w, and compare it with the exact Bayes answer. Varying the three knobs gives 112 probe types.

## 6. Five rounds of experiments and one audit

In total we trained 155 small transformers across five rounds and a final audit. Each round's predictions were written down before it ran. The audit retrained models, ran null models and recomputed every headline number from the raw files.

## 7. The main tests in plain words

Six kinds of test. The probes measure behaviour. Rule fitting asks which formula explains it. Controls rule out the obvious artifacts. Slope surgery is the causal test. The mechanism work looks inside. The audit checks that everything we report is real.

## 8. Models reason like Bayes, but with the wrong beliefs

Solid lines are trained models, dashed is exact Bayes, colours are the number of old copies. Top row is world C (h = 0.01, ε = 0.1), bottom row world D (h = 0.03, ε = 0.2). The models have Bayes's shape: lines for 2, 4 and 8 old copies nearly overlap, as Bayes says they should. But the switch from old to recent happens at the wrong gap and too sharply. The best description is Bayes run with the wrong h and ε. That fit is much better on real models than on random curves or random sums of counters, so it is not just flexibility.

## 9. The wrong beliefs follow two simple laws

Given only the world's true switch and typo rates, these two formulas predict how a trained model will behave. We tested that by leaving one world out, fitting the laws on the other eleven, and predicting the held-out world's trained models: R² 0.87. Confidence intervals: 1.50 [1.29, 1.78], exponent 0.74 [0.71, 0.78], intercept −0.33 [−0.47, −0.04], slope 0.44 [0.35, 0.57].

## 10. Fading rate sets the belief, and training chose it

This is the causal test. ALiBi gives each head a learned fading rate (slope). After training we scaled it up or down and measured the model's beliefs again. A faster fade makes the model believe switches are more common, in a near power law. On the right: ordinary-data loss is lowest exactly at the slope training found, in every one of 85 models. So the wrong belief is not slack; it is the loss-optimal setting for these heads.

## 11. Copy heads are fading counters that add up

Inside the model the decision is simple: a few copy heads each look back at old and recent copies, their attention fades at the learned ALiBi slope, and their outputs add up. That sum predicts the model's choice almost perfectly. What we cannot yet explain is how the heads' attention ends up Bayes-shaped; it survives removing every layer-0 head except the previous-token head.

## 12. What failed, or changed our story

We report failures as clearly as successes. Several predictions we made did not hold, and two earlier claims had to be corrected by the audit. All of these are in the paper's limitations and appendix; none of them undercut the main result.

## 13. Hypothesis scorecard

Our three original bets came out mixed, which is fine: they were written to be testable, not to be confirmed. The bigger questions came out clearly. Behaviour has the Bayes shape with wrong beliefs, each copy head is a fading counter, and the result survives every control we tried.

## 14. Recommendation: start writing the paper now

Our recommendation is to start writing now. The draft exists and the core claims are verified. The remaining items are small edits. Three optional experiments could run on the GPU while we write: context length (our suspect for the 0.74 exponent), the full grid with RoPE, and a finer look inside the copy heads.
