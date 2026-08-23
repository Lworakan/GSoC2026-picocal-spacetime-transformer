# What we would need to go below 0.037, and why we think so

Worakan Lasudee, 2026-08-23. For Felipe and Carla. Everything below is measured
on the samples you gave us; the scripts and the per-event predictions behind
every number are in the repository.

## Where we are

Aggregate $\sigma_{\rm eff} = 0.0388 \pm 0.0002$, fully out-of-sample under a
ten-fold protocol (72,533 events), against 0.1253 for a tuned gradient-boosted
reference on the same sample — a factor 3.2. Paired re-implementations of
ParticleNet and GravNet lose in every region at every seed, and the margin
survives a per-baseline hyperparameter sweep.

## Why we are asking for data rather than trying more models

We measured 148 configurations under one frozen protocol. The record is not
encouraging for further modelling:

- **Ten encoder families** (ParticleNet, GravNet, pairwise-bias attention,
  time-sliced attention, CNN, ConvNeXt, patch tokens, MLP-Mixer, energy-flow
  network, plain transformer): none beats the plain transformer.
- **The one architectural idea that did win on our development split — a
  two-stage window that points at the photon before measuring — does not
  reproduce under cross-validation.** We ran the check, it failed, and it is in
  the paper as a limitation.
- **Ensembling is exhausted.** Members disagree with each other by 0.87% per
  event against a total error of 3.76%; pooling 31 of them buys 4.7%.
- **Two bounds are closed.** A window centred on the true photon position is
  worth 4% and we already take most of it. A per-cell gate given truth is worth
  10–12% more than the one fitted from observables, and the network already
  routes that information around the gate.

Realistically, everything left on our side is worth about 2% in total.

## Request 1: about three times more minimum-bias simulation

We measured the data-scaling curve directly, training on 25%, 50%, 75% and 100%
of the sample within the same protocol (validation and test untouched):

$$\sigma_{\rm eff} \propto N^{-0.18}, \qquad 0.0524 \to 0.0466 \to 0.0439 \to 0.0402$$

The exponent is the same for single models ($-0.184$) and for three-seed
ensembles ($-0.179$), and the worst region–energy bin scales in parallel. A
threefold sample predicts an aggregate of **≈ 0.033** — larger than everything
the architecture campaign bought put together, by roughly a factor five.

**Questions:** is a threefold minimum-bias production feasible, what would the
queue time be, and is there an existing larger sample we could use instead?

We also checked that we cannot manufacture it ourselves:

- Eight-fold geometric augmentation of the window (D4 transforms) is neutral.
  What the scaling curve buys is diversity of *pileup configurations*, which
  symmetry transforms cannot create.
- Synthetic overlays — clean photons dropped onto pileup — are distinguishable
  from real minimum-bias events by a classifier at AUC 0.93 on cell-level
  observables, and training on them is neutral. Manufactured events do not come
  from the distribution the model has to serve.

## Request 2: per-event truth flags

Eight per cent of events carry twenty-one per cent of the total error: their
residuals sit beyond three $\sigma_{\rm eff}$, and if they were predicted as
well as the rest the aggregate would be 0.0296 instead of 0.0376. The model can
already *flag* them — its own predicted interval width separates them at
AUC 0.93 — but we cannot say what they are.

Our hypothesis is wrong-photon matching and overlapping clusters. To test it we
would need, per event in the tuple:

- the number of generator photons whose shower overlaps the reconstructed
  cluster, and their energies;
- a matching-quality flag for the photon the cluster is assigned to.

These are analysis-only quantities; they would never enter the model as inputs.
With them, the "catastrophic tail" section of the paper turns from a hypothesis
into a measurement, and if the cause is what we think, a targeted fix is worth
more than anything else on the list.

## Request 3: four facts the paper is missing

These are the red placeholders in the draft:

1. the generator and framework the samples were produced with, and versions;
2. the pileup condition — $\nu$ or instantaneous luminosity;
3. the timestamp resolution that was simulated, and whether out-of-time
   spillover is included. Our probes show the network exploits timing down to
   about a quarter of the per-event time spread and loses 19% when it is
   coarsened to half, so this number sets how our timing results should be read;
4. how you would like to be credited, and whether single-author use of
   collaboration simulation needs Editorial Board review or release as an
   LHCb note before anything is submitted.

## What we would do with each

| if we get | we would | expected |
|---|---|---|
| 3× minimum-bias | re-run the ten-fold protocol, turning the extrapolation into a measurement | ≈ 0.033 |
| truth flags | diagnose the 8% tail and attempt a targeted fix | up to 0.034 if the cause is what we think |
| the four facts | fill the placeholders and finish the draft | — |
| nothing | finish at 0.0388 with the falsification record as the contribution | 0.0388 |

The last row is not a bad outcome. But the first two are the only paths below
0.037 that our own measurements support, and neither of them is something we
can do from this side.
