# Anatomy of the remaining error, after recentring

2026-08-18. Best model = ensemble of the two recenter variants. Joined to per-event cache
features (49% join rate from float-key rounding; 5,372 events, enough for structure).

## 1. The remaining core width is pileup-MAGNITUDE, not mechanism

Splitting each weak bin at the median of window-energy/true-energy (a truth-based diagnostic,
not an input):

| bin | clean half | pileup-heavy half |
|---|---|---|
| 15mm low | 0.0683 | 0.0895 |
| 30mm low | 0.0598 | 0.0977 |
| 120mm low | **0.0370** | **0.0977** |

Every weak bin, including 120mm which was never window-limited or mis-seeded, is a mixture of a
clean population already at 0.04-0.07 and a pileup-heavy population near 0.10. The remaining
lever is not a new mechanism: the inputs that describe pileup magnitude (window/cluster ratio,
occupancy, density) are already features. This is the aleatoric component the epistemic study
measured, now localised.

## 2. The tails are catastrophic and separate from the core

|res| quantile ratio p95/p68 is ~5 in all three bins against 2.0 for a Gaussian: a 5-10%
population with |res| ~ 0.4-0.6 sits far outside the core. sigma_eff (a 68% interval) barely
sees them, so **fixing tails will not move sigma_eff** -- but they matter for physics use and
they are the same events that made trimmed risk backfire. Candidates: wrong-photon matches or
windows dominated by an overlapping cluster. Worth showing the mentors as its own plot.

## 3. Centroid-seed disagreement still marks trouble AFTER recentring

30mm low: events where the centroid and the loudest cell agree within one cell sit at 0.0489;
the rest at 0.0796. The disagreement magnitude is observable at inference and is ALREADY an
input (dxc, dyc in --extra): the model knows, and the gap persists, which again says magnitude
of ambiguity, not missing information.

## 4. Timing availability correlates the right way

Bins with more time-stamped cells do better (15mm low 0.0706 vs 0.0823). Consistent with the
no-time ablation (timing worth 20%); raw timestamps in tokens remain the right delivery, and
engineered pulls remain harmful (measured again on recentred windows: +0.0107).

## Consequences

- The queue's k=10 folds (more training data) are the right next step for the CORE -- data
  smooths the pileup-heavy mixture; no feature can delete it.
- One targeted arm added: `--film` conditioning on top of recenter (never combined) -- if
  per-event pileup magnitude is the story, modulating every block by the measured pileup
  context is the mechanism built for exactly that, and it failed before only in the
  mis-centred coordinate system.
- Stop adding per-cell features aimed at pileup identity; three measurements now say the model
  already extracts what the inputs carry.

## Addendum (same day): why recentring works — the first story was wrong

Pushing the analysis further overturned the causal reading of the recentring gain, before it
reached the mentors:

- The cluster centroid is FARTHER from the true photon than the argmax seed — median 3.08 cells
  against 0.43 at 15mm, missing by >2 cells in 79% of events against 17.6%.
- Even restricted to events where the two disagree, argmax is the closer estimator 80% of the
  time. A hybrid (argmax-on-agreement, centroid-on-disagreement) is therefore WORSE than argmax.
- Yet centring the window on the centroid improves 15mm low-E by 33%.

So the gain is not photon localisation. Centring on the cluster barycentre gives the window
symmetric COVERAGE of the full correlated energy field (photon tail plus the pileup the model
must subtract), and gives the network a coordinate frame anchored to the energy field rather
than to one volatile cell. The "17.6% mis-seeded" measurement stands, but it was a symptom of
dense pileup, not the mechanism of the fix.

Estimators tried against truth and closed without GPU: argmax-iterated log-centroid (fixes the
median, not the tail), hybrid (worse), clustered 3x3 seed (worse — dense pileup aggregates
too). The localisation tail appears genuinely unresolvable from energy alone at 15mm; per-cell
timing is too sparse (~30% of cells) and pull features have failed on three protocols.
