# Reading the GNNMP numbers off the ICHEP talk, and what they can honestly be compared to

30 August 2026. Reproduce with `python scripts/gnn_compare.py --region 30mm` (and `--region 40mm`,
`--region 30mm,40mm`).

## What was needed

The talk (W. Vetens, ICHEP 2026, 31 July, slide 11, "GNNMP versus Standard Approach") plots
sigma_eff(dE/E) against transverse energy for cluster seeds in the SpaCal-Pb region, on single
photons with minimum-bias clusters, at a luminosity of 1.5e34 cm^-2 s^-1. It is the same metric
this project reports, defined the same way, on the same sample construction -- but the values are
drawn as curves and tabulated nowhere, so nothing could be laid on shared axes.

## The digitised table

`scripts/gnn_compare.py --digitise <slide>.png` calibrates on the axis frame -- 0.5 to 5.0 GeV
across 389 pixels, 0 to 40% across 291 -- and then finds, inside each bin, the row of pixels of
each series' colour that spans that bin. A row is 0.137%, so the read-off error is about +/-0.2%
absolute. The bin edges fall out of the measurement rather than being guessed: the bars are
0.9 GeV wide, starting at 0.5 GeV.

| E_T [GeV] | 3x3 clustering | opt. cluster shapes | GNNMP model |
|---|---|---|---|
| 0.5-1.4 | 37.66% | 23.92% | 18.97% |
| 1.4-2.3 | 19.11% | 13.75% |  8.66%* |
| 2.3-3.2 | 13.20% |  9.48% |  6.46% |
| 3.2-4.1 | 10.31% |  7.70% |  5.36% |
| 4.1-5.0 |  8.38% |  6.74% |  4.54% |

Every value was read on a bar unbroken across its bin except the one marked `*`, where two other
curves cross the GNNMP bar and only a tenth of it survives; it agrees with the eye but carries a
larger uncertainty than the rest.

## The problem this creates, and how it was handled without waiting for anyone

Our own `minimum_bias` sample has no luminosity recorded -- one of the two red `\todo`s in the
paper. Without it, an absolute overlay compares two pileup conditions and calls the difference a
method result.

What makes the comparison possible anyway is that both sides publish a 3x3 cluster sum alongside
their model. Ours is the sum of the nine cells around the seed -- both depth segments, `e` is
exactly `fr + bk` -- calibrated by the power law the rest of the paper uses for raw sums, fitted on
the training events of each region and scored on the held-out ones.

**The ratio between the two 3x3 curves is not a pure sample difference and must not be labelled as
one.** Their "3x3 clustering" is LHCb's standard reconstruction, with per-cell calibration and
position corrections; ours is a bare calibrated sum. Two readings fit the data. The control for
this is the gradient-boosted reference already in the paper, which is a moderately better algorithm
on *identical* events: it moves sigma_eff by 20-30% in **both directions** across these bins
(better than the sum in the lowest bin at 30 mm, worse in the other four). So an algorithm change
of that size is worth about as much as the whole gap -- the two effects are not separable, and the
ratio is labelled "sample and algorithm difference together" everywhere below.

## The anchor, per region

SpaCal-Pb spans two cell sizes in our sample. The two are reported separately, because they behave
differently and because the union is an inference about their selection rather than something the
talk states -- their dataset name, `GNN1_3x3modules_ModulesPb_v4_...`, suggests a window of 3x3
*modules* in the Pb region, which is worth confirming.

| E_T [GeV] | our 3x3, 30 mm | our 3x3, 40 mm | their 3x3 | ratio 30 mm | ratio 40 mm |
|---|---|---|---|---|---|
| 0.5-1.4 | 42.86% | 23.41% | 37.66% | 1.14 | 0.62 |
| 1.4-2.3 | 16.67% | 12.16% | 19.11% | 0.87 | 0.64 |
| 2.3-3.2 | 10.83% |  8.03% | 13.20% | 0.82 | 0.61 |
| 3.2-4.1 |  9.50% |  7.30% | 10.31% | 0.92 | 0.71 |
| 4.1-5.0 |  8.59% |  5.36% |  8.38% | 1.03 | 0.64 |

**30 mm is the honest one.** On the shared estimator our 30 mm events and their SpaCal-Pb events
are equally hard -- ratio 0.82 to 1.14, scattered around one, with the lowest bin actually harder
for us. 40 mm is uniformly easier by a third, so grouping the two would move the anchor and flatter
us. Everything below is 30 mm.

## The answer

Single model, seed 0, no ensemble, 2,186 held-out 30 mm events.

| E_T [GeV] | this work | scaled to their sample | GNNMP | opt. shapes | their 3x3 |
|---|---|---|---|---|---|
| 0.5-1.4 |  7.85% |  6.90% | 18.97% | 23.92% | 37.66% |
| 1.4-2.3 |  4.95% |  5.67% |  8.66% | 13.75% | 19.11% |
| 2.3-3.2 |  3.61% |  4.40% |  6.46% |  9.48% | 13.20% |
| 3.2-4.1 |  3.40% |  3.69% |  5.36% |  7.70% | 10.31% |
| 4.1-5.0 |  2.76% |  2.69% |  4.54% |  6.74% |  8.38% |

"Scaled to their sample" multiplies our number by the ratio of the two 3x3 anchors. It is an
illustration under a stated assumption, not a measurement -- it hands the entire sample *and*
algorithm difference to them. On either reading we are ahead in all five bins, by 1.5x to 2.7x
after scaling.

The improvement each method buys over the 3x3 baseline of its own sample -- the quantity that does
not need the luminosity at all, and the robust comparison here:

| E_T [GeV] | this work / our 3x3 | GNNMP / their 3x3 |
|---|---|---|
| 0.5-1.4 | 5.46x | 1.99x |
| 1.4-2.3 | 3.37x | 2.21x |
| 2.3-3.2 | 3.00x | 2.04x |
| 3.2-4.1 | 2.79x | 1.92x |
| 4.1-5.0 | 3.11x | 1.85x |

## What is still owed before any of this goes in the paper

1. These are someone else's **Preliminary** numbers, measured off their slide. Felipe and Carla
   have to see this table and agree both to the values and to their being reproduced.
2. Their tabulated values, if the author will share them, so the digitisation error disappears
   rather than being carried.
3. The luminosity of our `minimum_bias` sample. The anchor makes the comparison possible without
   it; the number belongs in the paper regardless.
4. Whether their E_T is the true photon E_T. Ours is `E_true * pt / p` from the truth record. If
   theirs is reconstructed, the lowest bin is not the same selection.
5. What their SpaCal-Pb selection is in cell sizes -- one pitch or two -- and whether "3x3 modules"
   in the dataset name means their window is modules rather than cells.
6. The anchor ratio mixes sample difficulty with algorithm quality and cannot separate them; the
   improvement-factor table is the claim that survives, the scaled column is an illustration.
7. The dev-split caveat that applies to every number here as elsewhere: this is the fixed 70/15/15
   split, not the ten folds.
