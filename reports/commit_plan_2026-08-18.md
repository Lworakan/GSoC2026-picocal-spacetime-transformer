# Proposed commits for review

Nothing has been committed or pushed. This is the content to read before approving, in the order I
would land it. Seven commits, smallest and most independent first.

## 1. Correct the r = 0.92 per-cell gate claim

`reports/novelty_sota_positioning.md`, `reports/bounds_2026-08-12.md`,
`reports/novelty_gap_check_2026-08-05.md`, `scripts/overlay_bound.py`

The positioning document carried a publication-ready sentence claiming the gate reaches "r = 0.92
with held-out per-cell truth". The paired overlay sample gives 0.211 per cell; 0.92 came from an
aggregate-level, earlier-era measurement. Removed from the claim sentence with a note not to
reinstate it, corrected in the other two documents, and `overlay_bound.py`'s docstring no longer
calls a sum estimator's spread a physics floor.

    docs: correct the per-cell gate correlation from an aggregate-era 0.92 to the measured 0.211

    The novelty sentence in novelty_sota_positioning.md was publication-ready and wrong. The paired
    overlay sample measures 0.211 per cell. The claim does not depend on the number -- it rests on
    bounded fraction semantics, the pileup setting and the absence of per-cell labels -- so the
    sentence stands without it. overlay_bound.py's "perfect pileup removal" row is a sum estimator,
    not a bound, and its docstring said otherwise.

## 2. Measurement tools

`scripts/all_results.py`, `scripts/domain_gap.py`, `scripts/cell_info_ceiling.py`,
`scripts/fit_cell_prior.py`

    tools: one comparable table for every experiment, plus the domain-gap and information-ceiling probes

    all_results.py scores the seed ensemble rather than pooling residuals across seeds, which mixed
    models with different bias and read 0.0867 where the ensemble reads 0.0779 at 30mm low-E, and
    reports the spread across seeds instead of a statistical error that assumed independent test
    events. domain_gap.py measures synthetic-versus-real separability (AUC 0.931 at 15mm).
    cell_info_ceiling.py shows a gradient-boosted per-cell estimator reaching corr 0.945 from
    observables alone against the network gate's 0.211.

## 3. Containment: window, ring sums, patches

`scripts/picocal_data.py`, `scripts/train_picocal.py` (`--rings`, `--patch`, `--halo`,
`--only-region`)

    feat: window extensions after measuring that 9x9 sees 37.6% of the 15mm cluster

    The window had been fixed at 9x9 and never varied. The 15mm cluster reaches ring 15. W7 at five
    seeds gives 15mm low-E 0.1253 against the champion's 0.1654 and 30mm low-E 0.0923 against
    0.0957. --rings extends rather than replaces the window and only pays once the core is resolved.

## 4. Window centring

`scripts/picocal_data.py`, `scripts/train_picocal.py` (`--recenter`, `--rc-regions`)

    feat: centre the window on the cluster centroid, per region

    Against the truth entry point the photon lands more than two cells from the argmax seed in 17.6%
    of 15mm events (p90 8.28 cells) and 0.1% of 120mm events. At seed 0 recentring gives 15mm low-E
    -0.0371 and 30mm low-E -0.0086 but loses at 40mm and 120mm, so the estimator is chosen per
    region from the measured mis-seeding rate.

## 5. Physical geometry units

`scripts/picocal_data.py`, `scripts/train_picocal.py` (`--mmgeo`, `--globpe`)

    feat: express cell geometry in millimetres scaled by the shower size

    The clean sample shows the photon 99.7-100% contained within 120 mm in every region, so the
    shower has one physical width while its width in cells varies eightfold. In cell units the model
    must learn five radial functions for what physics says is one. --globpe adds the global detector
    coordinate that ClusTEX (arXiv:2603.18172) separates from the local one.

## 6. Architectures and baselines

`scripts/picocal_models.py`, `scripts/train_picocal.py`

    feat: spacetime-factorised attention, plus ParticleNet and GravNet baselines

    SpacetimeLayer buckets cells by time pull and attends within a bucket before buckets exchange
    pooled summaries, in the sense of TimeSformer but with the per-cell time as the axis: pileup
    energy has a 13.2 ns time rms at 15mm against 1.6 ns for the photon. PNetEncoder and
    GravNetEncoder are re-implemented behind the same readout, loss, splits and seeds so the encoder
    is the only difference -- comparing against numbers published on other datasets would prove
    nothing.

## 7. Cross-validation and capacity

`scripts/run_experiments.py`, `scripts/train_picocal.py` (`--fold`, `--nfold`, `--dim`, `--layers`)

    feat: k-fold splits and explicit model-size flags

    The fixed 70/15/15 split trains on 50,787 of 72,554 events and tests on 411 at 15mm low-E, where
    one configuration spanned 0.1138-0.1319 over five seeds. k=10 with a 5% validation slice trains
    on 85.5% and tests every event once. --dim/--layers exist to measure the capacity question
    rather than argue it.

## Deliberately not included

- `reports/predictions/*.csv` and `models/*.pt` from screening runs. The five-seed prediction sets
  belong in the repository; the one-seed smoke and screening files do not, and `*_smoke.csv` should
  be added to `.gitignore`.
- Any claim that the architecture is state of the art. ParticleNet and GravNet are implemented but
  have not run, so the honest phrasing remains "best in our own controlled comparison".
