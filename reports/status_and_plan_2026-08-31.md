# Where the three targets stand, what the mentors are still owed, and what is left to try

31 August 2026. Numbers from `reports/predictions/`, seed 0, single model, development
split. Reproduce the table with the snippet at the bottom.

## The three targets you set

**1. The final model must win in every region.** Met, 15 of 15 bins, seed 0.

**2. Every region must be below 0.08.** Met. The worst bin is 30 mm low at 0.0712.

**3. Every region must beat the mentors' graph model.** Met where a comparison exists,
which is the SpaCal-Pb region only. The talk publishes no other region.

| region | low (new / control) | mid | high |
|---|---|---|---|
| 15 mm  | 0.0643 / 0.0767 | 0.0438 / 0.0472 | 0.0291 / 0.0322 |
| 30 mm  | 0.0712 / 0.0768 | 0.0385 / 0.0427 | 0.0297 / 0.0307 |
| 40 mm  | 0.0574 / 0.0636 | 0.0302 / 0.0327 | 0.0228 / 0.0243 |
| 60 mm  | 0.0493 / 0.0516 | 0.0299 / 0.0315 | 0.0229 / 0.0235 |
| 120 mm | 0.0596 / 0.0600 | 0.0341 / 0.0356 | 0.0293 / 0.0298 |
| aggregate | **0.0376 / 0.0402** | | |

Against the GNNMP curve at 30 mm (see `gnn_comparison_2026-08-30.md`), after scaling our
numbers by the shared 3x3 anchor to hand the whole sample difference to them:

| E_T [GeV] | ours, scaled | GNNMP |
|---|---|---|
| 0.5-1.4 |  6.90% | 18.97% |
| 1.4-2.3 |  5.67% |  8.66% |
| 2.3-3.2 |  4.40% |  6.46% |
| 3.2-4.1 |  3.69% |  5.36% |
| 4.1-5.0 |  2.69% |  4.54% |

**The one thing standing between this and the paper is cross-validation.** All ten folds
were launched; the four that finished before the credit ran out read 0.0368, 0.0369,
0.0371, 0.0372 -- out-of-sample, and consistent with the 0.0375 the development split
predicted. Two earlier development-split winners did not survive ten folds, so this is
not a formality, but so far it is holding.

## The 24 August list: what is done and what is not

| # | Mentor asked | State |
|---|---|---|
| 1 | Intro: open on the five regions and their granularity | **done** |
| 2 | Intro: contamination driven by beam-pipe proximity, not cell size | **done** |
| 3 | Intro: name the LHCb Upgrade II baseline as the direct comparison point | **not done** |
| 4 | "external baseline" is wrong for the GBDT -- reword | **done** |
| 5 | Body centre = seed *module*, and it hurts position estimation | **done** |
| 6 | Figure 14: investigate the highest-energy anomaly | **done** (region composition, 1.5 sigma) |
| 7 | **Figure 7: redo with transverse energy on the x-axis** | **not done** |
| 8 | Cite the GNN work | **done** |

Item 7 was misread on my side: a *new* E_T figure was added for the GNN comparison, but
Figure 7 itself still has true photon energy on its x-axis. `plot_resolution.py` already
takes `--x ET`, so this costs nothing but the command.

Item 3 is a decision, not a measurement: the paper compares against a GBDT we trained and
against the standard 3x3 sum, but never states which of these is *the* Upgrade II
baseline. The Run 1 and 2 calorimeter paper is now cited, so the anchor exists.

## Graphs still needed

1. **Figure 7 redone against E_T** -- item 7 above, one command.
2. **Per-region sigma_eff vs E_T, ours against GNNMP**, five panels rather than the
   current two. Only 30 mm can carry their curve; the other four panels show ours with
   the 3x3 anchor, which is what makes them readable at all.
3. **The 15-bin new-vs-control bar chart**, one bar per region-energy cell, so "wins
   everywhere" is a picture rather than a table the reader has to scan.
4. **CV fold spread**, once the folds are back: ten points per configuration, so the
   development-split number and the cross-validated one sit on the same axis. This is the
   figure that answers "did it survive", and it is the one a referee will look for.
5. **Resolution vs E with the 0.08 line drawn**, since that threshold is now a stated
   target and no figure shows it.

## What is left to try on the physics

Ordered by what the error analysis says is actually available, not by novelty.

**a. Supervision volume, the only lever that has moved all session.** Gate labels reach
39% of the training set (28,547 synthetic against 72,554 real). `overlay_b.pkl` and
`overlay_c.pkl` are already generated. If 2x and 3x keep improving, the road to 0.035 is
open; if flat, the honest ceiling on this sample is about 0.037 and we say so. This is
the cheapest remaining experiment and the one with a real prior behind it.

**b. Three seeds per fold rather than one.** The paper's headline is an ensemble; the CV
now running is single-seed. The development split says the ensemble is worth 0.0375 ->
0.0367, about 2%. It costs three times the GPU for a known-size gain.

**c. The 30 mm low bin, which is now the worst cell at 0.0712.** It is the only bin above
0.07 and it sets the "below 0.08" margin. Worth one targeted error analysis: is it
containment, mis-seeding, or pileup? The answer decides whether anything else helps.

**d. Closed already, do not re-open:** topology/graph encoders for the tail, a
missing-energy head, overlay time alignment, 120 mm overlay (physically impossible at
that pitch), EMA stacking, gatesup 10.0, width feedback, q68, gs-balance, per-bin bias
removal, per-region window routing (the ceiling is 1%), selective recentring.

## Can mid and high go below 0.02?

Measured, not guessed. `fit_resolution.py` fits `sigma_eff(E) = a/sqrt(E) + b/E + c` per
region to this model, and every mid/high bin is compared against its own fitted curve at
the bin's median energy.

| region | bin | median E | now | fitted floor | gap |
|---|---|---|---|---|---|
| 15 mm  | mid  | 60.9 | 0.0438 | 0.0367 | 0.0071 |
| 15 mm  | high | 86.0 | 0.0291 | 0.0287 | **0.0004** |
| 30 mm  | mid  | 44.7 | 0.0385 | 0.0372 | 0.0013 |
| 30 mm  | high | 72.9 | 0.0297 | 0.0272 | 0.0025 |
| 40 mm  | mid  | 25.8 | 0.0302 | 0.0307 | -0.0005 |
| 40 mm  | high | 44.2 | 0.0228 | 0.0225 | **0.0003** |
| 60 mm  | mid  | 16.8 | 0.0299 | 0.0279 | 0.0020 |
| 60 mm  | high | 27.1 | 0.0229 | 0.0225 | **0.0004** |
| 120 mm | mid  | 13.1 | 0.0341 | 0.0327 | 0.0014 |
| 120 mm | high | 19.0 | 0.0293 | 0.0289 | **0.0004** |

Fitted terms: a = 0.209 / 0.203 / 0.141 / 0.083 / 0.035 and c = 0 / 0 / 0 / 0.0134 /
0.0241 for 15 / 30 / 40 / 60 / 120 mm.

**Not easy, and in one region not possible at all.** Three things fall out.

- The high bins are already *on* their own curve, four of five within 0.0004. There is no
  slack left to collect at high energy by tuning anything.
- 120 mm has a fitted constant term of 0.0241. A constant term is calibration and leakage,
  independent of energy: if that number is real, no estimator reaches 0.02 at 120 mm at
  any energy whatsoever.
- The mid bins hold the only real slack, and it is at 15 mm: 0.0438 against a floor of
  0.0367, a gap of 0.0071. Even collecting all of it leaves 0.037, not 0.02.

**The honest caveat, which cuts both ways.** That floor curve is fitted to our own
resolution points, so "we sit on the floor" is partly circular. The non-circular question
is whether the fitted `a` is the detector's sampling term or our own residual pileup
error wearing a `1/sqrt(E)` shape. The fit itself flags this: at 15, 30 and 40 mm the
constant term ran onto its bound at zero, which means `a` absorbed everything.

That is testable with no GPU: the PicoCal TDR quotes a design stochastic term per
technology. If the inner regions were built to roughly 10%/sqrt(E) and we fit 0.21, then
half our width at high energy is pileup rather than sampling, and there *is* headroom --
0.10/sqrt(86) is 1.1%, so a clean-shower estimator could in principle reach far below
0.02 at 15 mm high. If instead the design number is near 0.20, the door is shut.

### The design number settles it, and it splits the detector in two

PicoCal's stated target, from the LHCP 2024 proceedings on the detector: **`sigma_E/E =
10%/sqrt(E[GeV]) + 1%`**, quoted for the calorimeter as a whole, with Shashlik described
as already fulfilling it. That is an external number, not one fitted to our own points,
so it breaks the circularity above.

| region | bin | median E | ours | design floor | ratio | can 0.02 exist? |
|---|---|---|---|---|---|---|
| 15 mm  | mid  | 60.9 | 0.0438 | 0.0163 | 2.69x | yes |
| 15 mm  | high | 86.0 | 0.0291 | 0.0147 | 1.98x | yes |
| 30 mm  | mid  | 44.7 | 0.0385 | 0.0180 | 2.14x | yes |
| 30 mm  | high | 72.9 | 0.0297 | 0.0154 | 1.93x | yes |
| 40 mm  | mid  | 25.8 | 0.0302 | 0.0221 | 1.37x | **no** |
| 40 mm  | high | 44.2 | 0.0228 | 0.0181 | 1.26x | yes |
| 60 mm  | mid  | 16.8 | 0.0299 | 0.0264 | 1.13x | **no** |
| 60 mm  | high | 27.1 | 0.0229 | 0.0217 | 1.06x | **no** |
| 120 mm | mid  | 13.1 | 0.0341 | 0.0294 | 1.16x | **no** |
| 120 mm | high | 19.0 | 0.0293 | 0.0250 | 1.17x | **no** |

This corrects what the previous section concluded. Against our own fitted curve every bin
looked finished; against the design curve the detector splits in two.

**The outer regions are blocked by the detector, not by us.** At 60 and 120 mm the design
floor is already 0.022 to 0.029 at those bins' energies, above the 0.02 target. No
estimator reaches 0.02 there, because the calorimeter was not built to. The reason is
geometric: energy and pseudorapidity are tied, so the outer regions never see high energy
in the first place -- 120 mm "high" is 19 GeV, while 15 mm "high" is 86 GeV. And we are
already within 6 to 17 per cent of the design there, which is the more interesting
statement about those regions.

**The inner regions have a factor of two of real headroom.** At 15 and 30 mm the design
floor is 0.0147 to 0.0180, comfortably under 0.02, and we sit at 1.9 to 2.7 times it. The
excess is not sampling -- the detector's sampling term is 0.10 and our fit reads 0.21 --
so it is pileup, in exactly the regions where pileup is worst. That is the same lever as
everywhere else in this project, and the same one that is already half-pulled: gate
supervision reaches 39 per cent of the training set.

**So mid and high below 0.02 is the wrong target as stated, but half of it is right.**
Ask for it at 15 and 30 mm, where physics permits it and we are a factor of two away.
Do not ask for it at 60 and 120 mm, where the honest claim is the opposite and stronger:
we are within a sixth of the detector's design resolution on a pileup-loaded sample.

## What CERN's own pileup work says, and what our sample actually contains

The HL-LHC answer to pileup is timing. Interaction vertices are spread over about
+-150 ps, so 20-30 ps hit timing separates them: CMS HGCAL targets better than 30 ps for
clusters above 5 GeV, the CMS MIP Timing Detector 30-60 ps for 4D vertexing, and PicoCal's
own target is `sigma_t ~ O(10 ps)`. That is the mechanism our detector was designed
around, and it is the natural candidate for the factor of two at 15 and 30 mm.

**The separation is present in our sample, and it is large.** Referencing every cell's
front-face time to the shower core's own time, on 2,000 real 15 mm minimum-bias events:

| cells | n | median offset | IQR |
|---|---|---|---|
| core, d <= 1 | 8,181 | 0.000 ns | **0.077 ns** |
| ring, d = 2-3 | 27,159 | +0.181 ns | 0.300 ns |
| far, d >= 5 | 64,690 | +0.391 ns | **1.641 ns** |

Far cells arrive later and are spread 21 times wider than core cells. The times are
continuous rather than digitised, so this is truth-level timing without detector smearing.
A discriminant this strong should be worth something.

**It is not, and that is measured, not assumed.** The obvious reading of the table is that
the default time reference is wrong: it is the median over *all* cells in the window
(`picocal_data.py:318`), and far cells outnumber core cells eight to one, so the reference
is set by the pileup it is meant to reject. But that fix already exists as `--tpull`,
which references time to the energy-weighted top decile, and it was already run at this
window and recentring. On the same seed:

| bin | `--tpull` | control |
|---|---|---|
| aggregate  | 0.0409 | **0.0402** |
| 15 mm mid  | 0.0532 | **0.0472** |
| 15 mm high | 0.0354 | **0.0322** |
| 30 mm mid  | **0.0425** | 0.0427 |
| 30 mm high | 0.0333 | **0.0307** |

It loses in four of five, including both bins where this section predicted it would win.
The hypothesis is refuted by an experiment that was already on disk.

**So the bottleneck is not identifying pileup cells.** That reading is consistent with the
gate study: a gradient-boosted per-cell estimator reaches correlation 0.945 with the true
photon fraction on the same observables, while the network's gate reaches 0.211 and the
network routes the information around it. Per-cell separability is available and unused;
what fails is converting it into an energy. Any further attempt on the inner regions
should attack the aggregation step, not the input representation, because five timing
constructions and four gate-supervision protocols have now all measured null there while
per-cell information demonstrably exists.

## Re-ranked with risk as the criterion

Asked for on 31 August: no risk. That changes the order, and it changes it a lot, because
**the result is already won and does not need a new architecture.**

On the development split the current model beats the control in 15 of 15 bins, every
region sits under 0.08 with the worst at 0.0712, and it is ahead of GNNMP in all five E_T
bins even after handing them the entire sample difference. Four cross-validated folds read
0.0368 to 0.0372. Nothing on the risky list is needed to make that a paper.

**Do (cannot fail):**

1. The two sentences the paper contradicts -- correctness, no GPU
2. Figure 7 against E_T -- mentor item, one command
3. Name the Upgrade II baseline -- mentor item, a decision
4. The 15-bin bar chart and the design-curve figure -- no GPU
5. Release artifacts: split definition, pinned environment, weights, one command per
   figure -- no GPU, and it is the thing that actually makes a result the one others must
   beat

**Do (a measurement, not a gamble):**

6. Finish the six cross-validation folds. This is the only GPU item with no downside,
   because both outcomes are publishable: it either confirms the development split, and
   the headline improves, or it does not, and the paper reports both -- which is exactly
   what it already does for the two-stage window. The only thing at stake is money.

**Do if credit allows (known size, not a gamble):**

7. Supervision volume 2x/3x. The lever has already worked once on this exact label source;
   scaling it is the most likely of the remaining ideas to pay, and a flat result is itself
   a publishable ceiling statement rather than a wasted run.
8. Three seeds per fold. Worth about 2%, size known in advance.

**Hold:**

9. Object condensation and the ParT pairwise bias. Both are argued in their own reports and
   both may be null -- five representation changes and four supervision schemes already
   were. They are the right ideas for a second paper or for a version of this one that has
   spare GPU and spare weeks. They are not needed for this submission, and running them
   under time pressure risks the thing that is already finished.

## The earlier list, ranked by upside rather than risk

Ranked by what a wrong answer would cost, divided by what it costs to get. Everything in
the first block runs on this laptop.

### No GPU — do these regardless of credit

**1. The two sentences the paper now contradicts.** `main.tex:53` says four
gate-supervision protocols "measured null", and the table at `main.tex:1477` says "neutral
or worse". The best configuration in the project is `--gatesup 5.0`. It is not a flat
contradiction -- the null was measured on an overlay covering only 15 and 30 mm, without
per-cell regression -- but the paper cannot ship saying the opposite of its own best
result. This is a correctness fix, not an improvement, so it ranks first.

**2. Figure 7 against transverse energy.** Mentor item 7 from 24 August, still open.
`plot_resolution.py --x ET`. One command.

**3. Name the Upgrade II baseline.** Mentor item 3, still open. A decision about which of
the GBDT, the 3x3 sum, or the published calorimeter curve is *the* comparison point.

**4. Repair `oracle_ceiling.py`.** Calibrate inside the bin instead of extrapolating
across energy, and score the trained model on the same overlay events so the comparison
stops being cross-sample. This decides whether any further pileup work is worth GPU time,
which is why it outranks the figures below despite being the least presentable.

**5. Two figures the argument now needs**: the 15-bin new-versus-control bars, so "wins
everywhere" is a picture; and resolution versus energy with the 0.08 line and the design
curve `10%/sqrt(E) + 1%` both drawn, which is the figure that makes the outer-region claim
land.

### GPU, when credit returns

**6. Finish the six remaining cross-validation folds.** Everything above and below is
unpublishable without it. Four folds read 0.0368 to 0.0372 and the checkpoints resume, so
this is the cheapest it will ever be.

**7. `--cellsup`: supervise the per-cell energy head.** The one architectural idea the
error analysis actually points at, and it has never been run. `picocal_models.py:322`
records that nothing supervises `rhead` -- `--gatesup` supervises the *gate*, which the
paper itself shows is not a fraction estimator (correlation 0.211 against a
gradient-boosted 0.945 on identical observables), while `cellreg` emits a real per-cell
energy that can exceed the cell's own deposit. The overlay's `sig` is exactly the right
target for it and is already generated. Per-cell separability is demonstrably present and
demonstrably unused; this is the only untried way to force it through the aggregation
step, which the timing and gate studies have now both identified as the bottleneck.

**8. Supervision volume, 2x and 3x.** `overlay_b.pkl` and `overlay_c.pkl` exist. Labels
currently reach 39% of the training set. Flat between 2x and 3x means the lever is spent
and the honest ceiling on this sample is about 0.037.

**9. Three seeds per fold.** The paper's headline is an ensemble and the CV is single-seed.
Worth about 2%, a known quantity, at three times the cost. Last, because it buys a number
rather than an answer.

### Ranked out — do not re-open

Per-region window routing (measured ceiling 1%), selective recentring (wins 2 of 15),
topology and graph encoders for the tail, a missing-energy head, overlay time alignment,
120 mm overlay (impossible at that pitch), EMA stacking, `--gatesup 10.0`, width feedback,
`--q68`, `--gs-balance`, per-bin bias removal, `--tpull` (loses 4 of 5, measured above),
and 0.02 as a target at 40 mm mid, 60 mm and 120 mm, where the detector's own design
resolution is already above it.

## Two questions for the mentors, both blocking

1. The luminosity of our `minimum_bias` sample. The 3x3 anchor lets the comparison happen
   without it, but the number belongs in the paper.
2. Permission to reproduce the digitised GNNMP values, and their tabulated numbers if the
   author will share them. They are someone else's Preliminary results.

## Reproducing the table

```
python - <<'EOF'
import sys, pandas as pd
sys.path.insert(0, 'scripts')
from run_experiments import resolution
t = pd.read_csv('reports/predictions/minbias__SubNetW8CleanAuxExDnGs50RcOvV2CrQdEma.csv')
t = t[t.seed == 0]
e = t.groupby(['true_energy', 'region_name'], sort=False).pred_energy.median().reset_index()
for r in ['15mm', '30mm', '40mm', '60mm', '120mm']:
    s = e[e.region_name == r]
    q = s.true_energy.quantile([1/3, 2/3]).values
    for lab, c in (('low', s[s.true_energy <= q[0]]),
                   ('mid', s[(s.true_energy > q[0]) & (s.true_energy <= q[1])]),
                   ('high', s[s.true_energy > q[1]])):
        print(r, lab, resolution(c.pred_energy.values, c.true_energy.values)['sigma_eff'])
EOF
```
