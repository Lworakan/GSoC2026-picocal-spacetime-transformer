# Where the SOTA claim can and cannot be made, and what to run for it

2026-08-17. Written from a literature pass plus this week's measurements. The point is to separate
claims we could defend from claims a referee would reject.

## What the closest published work actually is

**ClusTEX** (arXiv:2603.18172v2, Maidannyk, Couderc, Malcles, Sahin, IRFU/CEA) — a single-step
graph transformer that does candidate selection *and* reconstruction for **overlapping** EM showers,
evaluated on efficiency, position and energy resolution, splitting rate and di-photon mass against
PFClustering. Its stated novelty is a positional encoding that separates the local graph coordinate
from a global detector coordinate.

**This is not a baseline we can beat.** Their task starts from raw readout and produces clusters;
ours starts from a cluster and regresses its energy. Claiming a win over ClusTEX would be comparing
different tasks on different detectors. It is the right paper to position against, not to beat.

What we did take from it is portable and now implemented as `--globpe`: our tokens have only ever
seen local offsets, the region one-hot and the pitch, never where in the detector the cluster sits,
which is what sets the local occupancy. Their Fig. 15 reports the learned global embedding
converging to a cosine of the detector extent, so the Fourier basis is supplied directly.

**arXiv:2607.08175** (Jul 2026) compares six architectures on cell-level ECAL data for collimated
photon identification: cell-level ML beats shower-shape variables, a transformer wins, and an
MLP-Mixer is put forward as the resource-constrained alternative for trigger-level use. Different
task (classification) but it supports our input choice and points at the cheap-encoder direction.

**CMS HGCAL / TICLv5 line** (arXiv:2510.01851 and the Run-3 physics-objects note) reports that
algorithms trained on high pileup gain most **at low energy and in the forward region where pileup
contamination is largest, with marginal gains at high energy**. That is exactly the pattern we
measured independently: the window fix moved 15mm and 30mm low-E hard and left the aggregate flat.
Useful corroboration, and a citation for why a low-energy-bin result is the interesting one.

## What we can honestly claim today

1. **A measured diagnosis, not a tweak.** A 9x9 window captured 37.6% of the 15mm cluster energy;
   the cluster reaches ring 15. Widening it moves 15mm low-E from 0.1697 to 0.1253 at five seeds
   (-26%) and 30mm low-E from 0.1036 to 0.0923 (-11%), with the aggregate unchanged. The optimal
   window is region-dependent, so a single shared window is the wrong design.
2. **A negative result on label pressure, four times over.** H7 (corr 0.971, resolution worse),
   `--gatesup`, `--prior-feat`, `--prior-teach`. Forcing the gate toward the true photon fraction
   makes resolution worse; the gate is a variance-minimising weight, not a fraction estimator. No
   comparable head-to-head exists — weak-supervision HEP work is nearly all classification.
3. **Timing is worth 20% of the aggregate and 24-39% in the weak bins**, measured by ablation on
   two independent base models.

## What blocks a SOTA claim

We have never compared against a published architecture on our own data. `--arch pnet` (ParticleNet
EdgeConv, dynamic kNN) and `--arch gravnet` are now implemented behind the same readout, loss,
splits and seeds, and are queued. That comparison, plus the mentors' promised GNN and standard-reco
outputs, is what settles it. Until then the honest phrasing is "best in our own controlled
comparison", not "state of the art".

Also worth stating plainly: **the transformer backbone is not novel.** A 3-layer d=128 set
transformer over calorimeter cells is standard. The contributions are the diagnosis, the negative
result and the timing measurement.

## Ranked directions, with the reasoning

**1. Cheap encoder at a wide window.** The window gain is real but W10 is 441 tokens against 81, so
attention cost rises ~29x and the throughput figure the mentors asked for collapses. The detector
window is a regular grid, so an O(n) grid convolution can afford W10; `CNNSub` already exists,
shares the physics readout, and has never been scored under this protocol. Queued as phase 9. If a
CNN at W10 matches the transformer at W10, the deliverable becomes "full containment at deployable
cost", which is a better result than either number alone.

**2. Per-region window.** 15mm needs W10+, 30mm saturates near W8, and 60mm and 120mm clusters end
by ring 3 and ring 1, so those regions spend attention on padding. This follows directly from the
measured containment profile and needs no new architecture.

**3. Inverse-variance front+back time combination.** Expected 0.82x on per-cell sigma_t, which
tightens every timing discriminant. Not yet implemented.

## Ruled out, with the measurement

- **Per-cell fraction supervision in any form** — four independent measurements above.
- **Ring-summed halo features as a *substitute* for a wide window** — at W4, `R15` gives only
  -0.0037 on 15mm low-E where a real window gives -0.0608, and `R12` is worse than baseline.

  **Corrected 2026-08-17 (same day):** I read that as "ring sums lose to a real window" and lowered
  expectations for the radial halo architecture. Wrong conclusion from a true observation. Ring sums
  do not REPLACE the window, they EXTEND it, and they only pay once the window already resolves the
  core: **W8 + rings-to-15 reaches 15mm low-E 0.0953 and 30mm low-E 0.0850 — the best arm on both
  target bins, at 289 tokens against W10's 441, which it also beats (0.1019).** Better and cheaper
  at once. That is exactly the core-detail-plus-outer-summary logic of the 2026 patch-hierarchical
  work, and it means the halo-token architecture should be judged on its own measurement rather
  than on my inference from the W4 rows.
- **Generic graph encoders** — measured on this data: Geo 0.1804, GeoQp 0.1684, PairT 0.2852,
  EfnResidual 0.2883, against the W4 champion at 0.1628.
- **DINO-style self-supervised pretraining** — the payoff requires unlabelled data far exceeding
  labelled data, but our labels come free from simulation for every event, and the measured
  bottleneck was containment, not representation quality.
- **Multi-view graph over front/back** — the two longitudinal samples are two scalars per cell, not
  the rich complementary views the multi-view literature assumes, and the information is already
  present: `--depth` was measured at 0.1732 against 0.1628. Reconsider only if phase 7 shows the
  encoder family matters on this data.

## Reading of the pattern across 50+ experiments

Information additions move the metric; re-expressions of the same information do not. The window is
the extreme case — 62% of the cluster energy was being discarded. Before proposing any architecture,
the question to ask is which measurable quantity the model currently cannot see.
