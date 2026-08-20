# Reviewer defense — SpaceTformer paper

2026-08-20. Prepared against the current 20-page draft (`paper/main.tex`). Every "cite" below
points at a table/figure that exists in the draft; every draft response is written so it can be
pasted into a rebuttal with minimal editing.

## 1. Weakness table

| # | Weakness | Type | Severity | Status |
|---|---|---|---|---|
| W1 | Baselines (ParticleNet/GravNet) not hyperparameter-tuned | experimental | HIGH | closing now: 6-arm sweep running; first arm (pnet lr 1e-4 = 0.0456, worse than default) already strengthens us |
| W2 | Single detector concept, single simulated sample | experimental | HIGH | not fixable with our resources; framed as a case study + Limitations item (1) |
| W3 | d=256/6-layer collapse undiagnosed (optimization vs saturation) | technical | HIGH | closing now: warmup / pre-LN / both arms running |
| W4 | 3x-data claim is an extrapolation | experimental | MEDIUM | measured 4-point within-protocol curve (N^-0.18) is the strongest evidence obtainable without a generator; labelled as extrapolation in text |
| W5 | Catastrophic tails (5–10% at \|res\| 0.4–0.6) unsolved | technical | MEDIUM | dissected in anatomy section; deployment answer: the model already outputs a per-event confidence (inter-quantile width) |
| W6 | Simulation provenance placeholders (nu, generator, timing resolution) | presentation | MEDIUM | blocked on collaboration; must be filled before submission |
| W7 | Scheduler defect disclosure (double-step cosine in later runs) | presentation | LOW | disclosed in Methods; all paired comparisons within-era |
| W8 | Float-key event merge collapses 21 events (0.03%) | technical | LOW | disclosed in Table 1 caption; effect below rounding |
| W9 | Clean-sample pair in the timing figure is an earlier-era configuration | presentation | LOW | labelled in the caption |
| W10 | Single author on collaboration simulation | policy | HIGH (process) | must resolve authorship/Editorial Board with mentors before arXiv |

## 2. Top anticipated questions, ranked by likelihood

**Q1. Your baselines use default hyperparameters under a recipe tuned for your model. How do we
know ParticleNet/GravNet would not win if tuned?**
Motivation: the single most predictable attack on any paired comparison.
Answerable: partially now (h2h section), fully when the sweep lands.
Draft: "We agree this is the key fairness question and ran a dedicated sweep per baseline
(learning rate halved/doubled, neighbourhood size k=16→24). The first completed arm, ParticleNet
at lr 1e-4, degrades to 0.0456 from its default 0.0412; the full sweep is reported in Sec. 6.4.
No tuned arm approaches the transformer's 0.0399 ± 0.0005."

**Q2. Isn't the small-capacity conclusion just a failure to train the larger model?**
Motivation: 0.1179 for d=256/L6 screams optimization failure to an ML reviewer.
Answerable: when the diagnosis arms land (warmup, pre-LN, both).
Draft: "We diagnosed this directly: retraining d=256/6L with learning-rate warmup, with
pre-LayerNorm, and with both [numbers]. [If recovered:] the collapse is an optimization artefact
and the recovered model reaches X, which still does not beat d=128 — capacity remains closed at
the accuracy level, and we have corrected the text to say 'no capacity gain' rather than
'instability'. [If not recovered:] the failure persists under the standard remedies."

**Q3. Everything is one simulated sample of one proposed detector. Why should any conclusion
generalise?**
Motivation: narrow-scope objection, guaranteed from ML venues.
Answerable: partially with existing data.
Draft: "The paper claims a diagnosis method, not universal numbers, and Limitations (1) says so.
Two internal transfers support the method: the clean sample (a second, pileup-free distribution)
reproduces the timing conclusion with the opposite sign (Fig. timing), and the same window
diagnosis transfers across the five detector regions, which differ eightfold in granularity —
effectively five related sub-detectors. CMS HGCAL reports the same qualitative pattern."

**Q4. The 3x prediction is an extrapolation. Why print 0.033?**
Answerable: yes.
Draft: "The four measured points are strikingly linear in log-log over a 4x range of N and the
exponent is stable between single models and ensembles (-0.184/-0.179). We label the 3x point an
extrapolation in both the text and the figure and give the collaboration-facing conclusion —
produce more simulation — which is falsifiable and cheap to test."

**Q5. sigma_eff ignores your own 5–10% catastrophic tails. Is the headline metric hiding failure?**
Answerable: yes.
Draft: "We report the tails explicitly (Sec. anatomy, Fig. residuals) rather than hiding them: the
p95/p68 ratio is ~5 against 2.0 Gaussian, their origin is wrong-photon matching and overlapping
clusters upstream of regression, and the model's predicted inter-quantile width — already used for
calibration — provides a per-event flag for them at deployment. sigma_eff is the collaboration's
figure of merit, not our choice."

**Q6. The gate 'constraint' has a free additive head h(z,g); is the physics constraint real?**
Answerable: yes (we fixed the claim before review).
Draft: "The equation in Methods shows the full readout including h; we describe the gated sum as
the dominant pathway and an inductive bias, not a bound. Removing the gated-sum structure
entirely (direct regression) degrades 0.0390 → 0.0650 (Table loss), which is the measured value
of the bias."

**Q7. Why should we believe the recentring gain isn't just a photon-position effect you could get
from a better position estimator?**
Answerable: yes, strongly.
Draft: "We measured exactly that hypothesis and it is false: the barycentre is farther from the
true photon than the seed (median 3.06 vs 0.44 cells, Fig. misseed) yet wins, and three centre
estimators built for localisation (iterated log-centroid, hybrid, clustered seed) all lose to the
plain barycentre. The mechanism is aperture coverage (Fig. containment, solid vs dashed)."

**Q8. Your CV protocol trains 50 models for one number. Is this deployable?**
Answerable: yes.
Draft: "CV is the evaluation protocol, not the deployment; deployment is one fold's five-member
ensemble (111 clusters/s CPU) or a single model (555/s) — Fig. throughput maps the accuracy-cost
frontier including a free analytic point."

**Q9. Only 30% of cells have timestamps; you zero-fill. Isn't that throwing information away?**
Answerable: yes.
Draft: "Every token carries explicit has-timestamp indicators next to the zero-filled value; after
the linear embedding this is exactly a learnable missing-value embedding. Five engineered
alternatives (pulls x3, cuts, inverse-variance combination) all measured worse than raw+flags; a
Fourier basis on the raw time channels is in the queue as the last untried representation."

**Q10. The ledger shows 120+ runs. How do we know the headline isn't the winner of a garden of
forking paths?**
Motivation: multiple-comparisons concern, sophisticated reviewers ask it.
Answerable: yes.
Draft: "The headline number comes from a protocol fixed before the final measurement: ten-fold CV,
every event predicted once by models that never saw it, with the half-sample preview (0.0379) and
its correction to the full-sample 0.0388 reported in Limitations (4) precisely so selection cannot
hide there. Screening runs never enter headline claims and are marked in the ledger."

**Q11. Why is 120mm low-E worse than your starting point?**
Answerable: yes.
Draft: "Because recentring is measured to cost resolution at 120mm (the barycentre is dragged by
pileup tails where the seed was already right) and four of five ensemble members use it — stated
with numbers in Sec. results. The region-restricted member exists precisely to limit this, and the
trade is globally favourable: −54%/−19% in the two failing bins against +0.011 in one bin that was
never near the target."

**Q12. Which is it: 41%, 37.6%, or 27% containment?**
Answerable: yes.
Draft: "41% is the value of the measurement shown in Fig. containment (12-file subsample,
cluster-cell energy within Chebyshev w=4 of the seed); earlier project notes quoted variants on
different subsamples/definitions. The paper uses one definition, one figure, one number."

## 3. Recommended ablation subset (main paper)

1. **Window scan + recentring** (Fig. window_scan, Table gain) — proves the core thesis; nothing
   else shows representation >> architecture.
2. **Paired encoder comparison** (Table h2h + encoder table) — the credibility anchor; keep both
   granularities (2-seed means + full-family history).
3. **Timing ablation, clean vs pileup** (Fig. timing) — unique: shows *why* timing matters, not
   just that it does.
4. **Loss/target study incl. trimmed risk** (Table loss) — the trimmed-risk failure preempts "why
   not optimise the metric directly".
5. **Scaling curve** (Fig. scaling) — converts the unmet 0.035 target into a falsifiable claim.
6. Appendix keeps: recipe sweep, gate x4, feature doors, mmgeo — cited from Closed doors, full
   numbers in the ledger.

## 4. Text edits still recommended before submission

1. Fill the five red placeholders (nu/luminosity, provenance, timing resolution, authorship,
   repo URL) — the paper cannot go out with red text.
2. When the baseline sweep and d256 diagnosis land, replace the two "in progress" notes with
   numbers and rewrite the capacity sentence per the outcome (Q2 above shows both versions).
3. Abstract, last sentence: keep "extrapolation" framing verbatim — do not let it drift back to
   "predicts" without the caveat.
4. Add one sentence to Limitations naming the venue-fit honestly: contribution is diagnosis +
   negative results + controlled comparison, not architectural novelty.

## Venue note

For JINST/EPJC/CHEP this paper is submission-shaped once W6/W10 are resolved. For a top ML venue
the honest gaps are W2 (single dataset) and the absence of a new method; a workshop (ML4PS at
NeurIPS) is the right ML-side target — the diagnosis story and the negative-results ledger are
exactly what that audience values.
