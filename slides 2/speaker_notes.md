# Recency vs Frequency: Follow-ups — speaker notes

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

## 6. Six rounds of experiments and one audit

In total we trained 218 small transformers across six rounds and an audit. Each round's predictions were written down before it ran. The audit retrained models, ran null models and recomputed every headline number from the raw files.

## 7. The main tests in plain words

Six kinds of test. The probes measure behaviour. Rule fitting asks which formula explains it. Controls rule out the obvious artifacts. Slope surgery is the causal test. The mechanism work looks inside. The audit checks that everything we report is real.

## 8. Models reason like Bayes, but with the wrong beliefs

Solid lines are trained models, dashed is exact Bayes, colours are the number of old copies. Top row is world C (h = 0.01, ε = 0.1), bottom row world D (h = 0.03, ε = 0.2). The models have Bayes's shape: lines for 2, 4 and 8 old copies nearly overlap, as Bayes says they should. But the switch from old to recent happens at the wrong gap and too sharply. The best description is Bayes run with the wrong h and ε. That fit is much better on real models than on random curves or random sums of counters, so it is not just flexibility.

## 9. The wrong beliefs follow two simple laws

Given only the world's true switch and typo rates, these two formulas predict how a trained model will behave. We tested that by leaving one world out, fitting the laws on the other eleven, and predicting the held-out world's trained models: R² 0.87. Confidence intervals: 1.45 [1.29, 1.78], exponent 0.74 [0.71, 0.78], intercept −0.35 [−0.47, −0.04], slope 0.44 [0.35, 0.57].

## 10. Fading rate sets the belief, and training chose it

This is the causal test. ALiBi gives each head a learned fading rate (slope). After training we scaled it up or down and measured the model's beliefs again. A faster fade makes the model believe switches are more common, in a near power law. On the right: ordinary-data loss is lowest exactly at the slope training found, in every one of 85 models. So the wrong belief is not slack; it is the loss-optimal setting for these heads.

## 11. Copy heads are fading counters that add up

Inside the model the decision is simple: a few copy heads each look back at old and recent copies, their attention fades at the learned ALiBi slope, and their outputs add up. That sum predicts the model's choice almost perfectly. What we cannot yet explain is how the heads' attention ends up Bayes-shaped; it survives removing every layer-0 head except the previous-token head.

## 12. How far does the formula reach?

We froze the formula and trained 63 new models in settings it had never seen. Inside the original range it predicts them as well as on the original grid. It also transfers unchanged to other context lengths, vocabulary sizes and numbers of keys. Outside the range, the beliefs still transfer but the fixed output scale does not, which breaks 3 of 4 extrapolations. RoPE models follow the same Bayes shape with their own, much better calibrated, constants. With random instead of +1 typos the models are counter-like, so the Bayes description needs structured typos. Our idea that a finite context causes the too-high switch rate was wrong: doubling the context changed nothing.

## 13. What failed, or changed our story

We report failures as clearly as successes. Several predictions we made did not hold. All of these are in the paper's limitations and appendix; none of them undercut the main result.

## 14. Hypothesis scorecard

Our three original bets came out mixed, which is fine: they were written to be testable, not to be confirmed. The bigger questions came out clearly. Behaviour has the Bayes shape with wrong beliefs, each copy head is a fading counter, and the result survives every control we tried.

## 15. Prop. 2 redone: right logic, three dropped terms

Sir asked whether the derivation is right. The two-hypothesis logic is right, but our boundary formula dropped three terms. First, when a value is redrawn it can land on the same value, so "no change" has prior s plus (1 minus s) over V. Second, after a switch, the old copies could all have been typos of c minus 1. Third, and largest, our probes measure the odds of the observed token, which includes the typo channel, so the switch point is where the posterior odds equal (1 minus eps) over (1 minus 2 eps), not 1. With these three terms the formula matches the exact oracle to 0.9% median error; the paper's version is off by 47% median and almost always wrong at large eps. The qualitative conclusions stay. Also important: the law for h' and eps' is an empirical fit, not a derivation.

## 16. The switch-rate half transfers; the typo-rate half does not

Sir asked for the exact region where the formula works. We froze the formula and trained new models over a cross in the h and epsilon plane, two decades in h and from 0.01 to 0.45 in epsilon. The colour is the formula's error divided by the error of the best Bayes fit to that model; inside the box we fitted on, the ratio is about 1.4, so that is our yardstick. Our pre-registered test, 1.25 times, turned out stricter than the formula manages even on its own fitting grid, so we judge it against how it does there. It works on a band from h of about 0.001 to 0.045 for typo rates from 0.05 to 0.25. We trained every new setting twice, with two seeds, and no setting changed sides, so these edges are stable. The two halves of the formula behave differently. The believed switch rate follows the formula over two decades of h, and only stops below h of 0.002. The believed typo rate is the problem: it is not a function of the typo rate alone, it falls as h grows, and the formula has no h in it. That is the whole failure at large h. At epsilon of 0.35 and above, the models switch far harder than Bayes allows, so no Bayes form fits.

## 17. The clock is right; recent copies are under-counted, by an amount that depends on h

Sir asked why the formula works inside the region. Bayes's decision has two separate parts: how likely the old value is to have survived the gap, which is the time part, and how much each recent copy says that the value changed, which is the evidence part. We measured both directly. For time, we show the model a single old copy and ask what it believes now. In all 124 models, at every switch rate and typo rate, this belief matches exact Bayes, so the models' clock is right. For evidence, we measure how far one extra recent copy moves the switch point compared with Bayes. That weight is well below one at low h and grows with h, reaching about one at h of 0.1. RoPE models reach one at a lower h, but they too fall at very low h. Our formula is a Bayes fit, and a Bayes fit can only express a weak evidence weight as a higher believed typo rate. Inside the box we fitted on, the weight stays within about 0.25 to 0.65, and the typo-rate formula is effectively its average, so it works there. Outside, the weight moves further with h, and a formula without h cannot follow.

## 18. One ALiBi slope does two jobs: the clock and the evidence weight

This is part of the explanation for why the evidence weight depends on h. In an ALiBi copy head, a single learned slope sets how fast attention to an old copy decays with distance. Training tunes that slope so that a single old memory fades exactly as fast as values really change, which is why the clock is calibrated, and so the slope grows with h. But the same slope also decides how much each extra recent copy shifts the balance between old and new. Across our models the evidence weight tracks the slope, and when we edit the slope by hand in 25 models we had not used before, the evidence weight moves with it: down by a third when we halve it, up by about a quarter when we double it. We had predicted at least 30 percent for doubling, so that half just missed. We then repeated the edit on 25 more models from a second seed: halving gave 0.56 and doubling 1.37, the same direction and a bit stronger. Bayes treats time and evidence separately; ALiBi ties them through one number. So in ALiBi the slope is one real cause of the drift. But it is not the whole story: RoPE models, which have no such slope, also give recent copies little weight at very low h. We tested whether this is because weighing old against new rarely matters for the training loss, by adding conflict examples. That raised the weight at moderate h and for RoPE at very low h, but not for ALiBi at h of 0.0003, so that reading is at most part of the answer.

## 19. Every edge is the evidence weight moving; the clock never breaks

Sir asked why the formula fails outside the region. At every edge the clock is still right, and what moves is the weight given to recent copies. At high h, recent copies get almost the full Bayes weight, so the believed typo rate drops, but the formula has no h in its typo part. At very high typo rates, Bayes gives a recent copy almost no weight, under a tenth of a nat, but the models give it several times more, so they switch where no Bayes curve can. This regime barely matters for the training loss, and it shows: at h of 0.01 and epsilon of 0.35, two seeds reach the same loss, but one weighs recent copies like Bayes or more and the other almost ignores them. So the direction is explained, but the exact behaviour there depends on the seed. At very low typo rates it is the other way round, and the misfit is mild. At very low h the picture is less clean: a single memory is still trusted correctly far beyond the gaps seen in training, but recent copies only count once the old copies are far away. Doubling the context length changed the switch point only slightly, so that is not the cause. Giving the model nine times more long gaps in training, by using more keys, barely changed it either. Training four times longer did not change it at the lowest h either. What we can say is that the evidence weight is smallest here, for RoPE models as well as ALiBi. We tested the idea that this is because conflicts rarely matter in training, by adding conflict examples: that lifted RoPE's weight at h of 0.0003 from 0.23 to 0.48, but left ALiBi's at 0.14. For ALiBi at this h, nothing we tried, including doubling the slope, moved the weight. We also tested whether the model counts recent copies by how much attention its copy heads give them: making one copy carry the attention of four reproduced only about a tenth of the effect of four real copies, so the count is computed somewhere else. So we mark this edge as only partly explained.
