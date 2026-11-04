# Recency vs Frequency: Validating the Formula — speaker notes

## 1. Does the formula holdacross the band?

Last time Sir asked for a concrete statement: the formula works in this range, with this much variability. This deck is that test. We fixed the range, the pass criteria and the scoring code before training anything, then trained 148 new models overnight and graded them.

## 2. "It works in this range, with this variability": can we say that and defend it?

The formula has two parts: a believed switch rate that is a power law in the true switch rate, and a believed typo rate that is linear in logit space. We plug those into the exact Bayes rule to predict the probe log-odds. The problem with our last answer was that the band came from a yardstick we chose after looking. So this time everything was written down and hashed before training: the band is exactly the one on the follow-up slides, and the code that grades it was fixed too.

## 3. Fresh random settings, three seeds each, and a replicate as the yardstick

E13 is the main test: 24 points spread at random across the band, not on any grid we had used before, three seeds each so that we can measure seed noise at every point. E14 asks whether the constants depend on the training recipe: at four points we change one thing at a time. Adding MLPs or a third layer changes the architecture, so those two were marked exploratory, with no prediction. The yardstick for variability is a replicate: how far apart two seeds are. That splits the formula's error into seed noise, which no formula can remove, and bias, which is the formula's own miss.

## 4. Every pre-registered test passed: the band holds

The formula was never refitted. On 72 models at 24 settings it had never seen, the median R² is 0.89, and 92% of models are at 0.7 or above. The typical error is 0.68 nats over the probe cells. The believed switch rate lands within a factor of two of the formula in 71 of 72 models. And the yardstick from last time, now used before seeing the data, passes at 19 of 24 settings. P5 is not a test of quality but of what kind of error it is; we predicted it would be mostly bias, and it is. The low tail of the histogram is one setting, the corner at the lowest h and highest epsilon, where the surfaces are so flat that R² says little.

## 5. The 5 weaker settings sit at both ends of the h range

Blue circles pass, orange circles fail the yardstick from last time; the squares are the earlier settings, pale blue if they passed and pale orange if not. The weak points are at the two ends of the switch-rate range, not where we guessed. At the low end, R² drops, but that is partly because there is little structure to explain: the surfaces are nearly flat and the absolute errors are small. Inside the middle of the band, roughly h from 0.004 to 0.03, every setting passes.

## 6. The formula's error is a stable bias, about twice the seed noise

This is the variability statement Sir asked for. Grey is how much two seeds of the same recipe differ; orange is how far the formula is from the average model, after taking that seed noise out. The formula's bias is about 0.6 nats, the seed noise about 0.25, so the formula is not as good as rerunning the training. But the bias is stable: different seeds agree on the believed switch rate within about 9 percent. And for a new model in the band we can now give a 90 percent interval: its believed switch rate will be between 0.8 and 1.74 times the formula's value, and its error under about 0.93 nats.

## 7. The training recipe barely matters; the architecture does

At four points we changed one training setting at a time. Halving or doubling the learning rate, doubling the batch, doubling the training length and halving the width all leave the formula's error within 7 percent of the default. Width 256 misses our pre-registered limit by half a percent, and its believed rates are actually the closest to the formula, so this is a near-miss on the overall error, not a different belief. Changing the architecture, adding MLPs or a third layer, does move the models away from the formula; those were exploratory and we only claim the formula for 2-layer attention-only models.

## 8. On new models, the evidence weight rises with h more cleanly than before

After training we ran the survival and wide-window probes on every new model, so we could measure the evidence weight, how much an extra recent copy counts compared with Bayes. This was not part of the pre-registration. The trend from the follow-up slides shows up again and more cleanly: w rises steadily with h, and its typical value inside the band sits in the range that the typo-rate formula averages over. That is the explanation from last time, now on models it was not derived from.

## 9. What we can now state, and what it does not cover

This is the sentence we propose to put in the paper. Every number in it comes from models the formula never saw, graded against criteria written before training. The limits are part of the claim: the formula has a consistent bias, it is weaker at the two ends of the switch-rate range, it is for two-layer attention-only models, and the very low switch-rate edge is still only partly explained. We think this is enough to start writing, and we would like Sir's view on whether to add the RoPE validation first.
