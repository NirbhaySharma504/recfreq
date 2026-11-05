# Recency vs Frequency: Where We Are — speaker notes

## 1. Which memory wins?

A short recap of the question and the formula, then what we found and what we hypothesised in the earlier rounds, then this week's experiment, which answers Sir's request for a concrete range and variability, and finally where the project stands.

## 2. Old and frequent, or new and rare?

The question came from Sir's Potter and Potyer example. When an induction head has seen a continuation often but long ago, and a conflicting one rarely but recently, which does it copy? Either answer can be right: if the world only has typos, frequency is right; if the world really changes, recency is right. So we built a task with both: each key's value is redrawn at rate h, and each copy is corrupted at rate epsilon. Because the data comes from known rules, there is an exact best answer, the Bayes predictor, and we can compare every trained model to it.

## 3. Weigh "it really changed" against "the new copies are typos"

Bayes compares two stories. Story one: nothing changed, and the recent copies are typos. Story two: the value switched during the gap, and the recent copies are clean. The ratio gives three terms: time, which is how likely the old value survived the gap; the prior, because a switch only helps the new value if the redraw lands on it; and evidence, where each recent copy multiplies the odds by one minus epsilon over epsilon. The old copies appear in both stories in the same way, so they cancel, which is why Bayes does not care about more than two old copies. When we re-derived this for Sir, the logic held, but three small terms had been dropped; with them the formula matches the exact computation to under one percent.

## 4. Models reason like Bayes, but with the wrong beliefs

Solid lines are trained models, dashed is Bayes, colours are the number of old copies. The lines for 2, 4 and 8 old copies nearly overlap, as Bayes says they should, but the switch happens at the wrong gap. The best description of every model is Bayes run with the wrong switch rate and the wrong typo rate, and those wrong beliefs follow two simple laws. When we say "the formula" in the rest of this deck, we mean this: Bayes with h-prime and epsilon-prime from these two laws. It was fitted on 12 settings, so it is an empirical law, not something we derived.

## 5. The clock is right; the evidence is under-weighted

Six things we knew before this week. The models are Bayes-shaped, and the formula predicts new models well and survives changes in task size, though not changes in encoding or typo structure. When we split Bayes's decision into its two parts, the time part is exactly right in every model we probed, and the whole deviation comes from the evidence part: each recent copy counts for less than in Bayes, and how much less depends on the switch rate. That explains why a formula fitted on one box works there and drifts outside it.

## 6. Why are recent copies under-counted?

Four explanations we tested for why recent copies count too little. The ALiBi slope does tie the clock to the evidence weight: when we edit it by hand, the weight moves. But it cannot be the whole story, because RoPE has no such slope and shows the same drop at low h. The rarity of conflicts in training explains which situations deviate, but adding conflict data only helps part way. Two specific explanations were ruled out: neither context length, nor long-gap data, nor training length moves the low-h edge, and the copy heads do not count recent copies through their attention mass. What sets the evidence weight at the very lowest switch rates is still open.

## 7. Turning "it works in this range" into a tested claim

Sir asked for a concrete claim: the formula works in this range, with this variability. Our range from last time was honest but chosen after looking at the results. So this week we did it properly: we wrote down the range and what would count as passing, and timestamped it, before any model had finished training. Then we trained models at 24 settings nobody had used before, picked at random inside the range, three seeds each, and another set where we changed one training setting at a time. The formula was never refitted. Training three seeds per setting lets us measure how much models vary just from the random seed, which is the yardstick for variability.

## 8. The range holds: every check we committed to passed

The map shows the range as the dashed box. Squares are the settings from earlier rounds; circles are the 24 new ones, blue where the formula does as well as it does on its own fitting data, orange where it does worse. Across the 72 new models the formula, never refitted, explains a median 89 percent of the variation in the models' behaviour, with a typical error of 0.68 nats, and the models' believed switch rate lands within a factor of 0.8 to 1.7 of the formula's value for 90 percent of them. The weaker settings are at the two ends of the switch-rate range, not where we expected.

## 9. The formula's error is a steady bias, about twice the seed noise

This is the variability part of Sir's question. Grey bars are how much two seeds of the same recipe differ: about a quarter of a nat. Orange bars are how far the formula is from the average model, with that seed noise removed: about 0.6 nats. So the formula is not as good as simply rerunning the training. But its miss is the same from seed to seed, so it is a stable, systematic description, which is what we will claim.

## 10. Robust to training settings; the evidence story replicates

Left: changing the learning rate, batch size, training length or making the model narrower leaves the formula's error within 7 percent; the wider model misses our limit by half a percent. Changing the architecture does matter, so we only claim the formula for two-layer attention-only models. Right: we also measured the evidence weight on the new models. The trend from last time comes back more cleanly, and the typical weight sits in the range the typo-rate law was fitted on, which is the explanation for why the formula works inside this range and drifts outside it.

## 11. A claim we can defend, and what is left

This is the claim we propose for the paper. Every number in it comes from models the formula never saw, graded against criteria fixed in advance. The story behind it: the models get the time part of Bayes exactly right and under-weight recent evidence by an amount that depends on the switch rate, and in ALiBi the positional slope is one real cause. The limits are part of the claim: the bias is about twice the seed noise, it is for two-layer attention-only models, typos in our data only go one way, and the very lowest switch rates are not fully explained. We think this is enough to start writing, and would like Sir's view on whether the RoPE test or a two-way typo version should come first.
