# What the window is actually for — and the correction it forced

2026-08-23. Continues reports/twostage_2026-08-21.md. Development protocol
(fixed 70/15/15 split) throughout; member values, not ensembles, unless stated.

## The claim we had been making

The paper said the resolution ceiling was set by the input window, and explained
the gain as *coverage*: the $9\times9$ window held only 41% of the 15 mm
cluster's energy, so the shower was being cut off. That number is real, but it
counts **cluster** energy — photon plus pileup — and the photon is not what is
spilling out.

## What the photon actually does

Clean sample (no pileup), window on the seed cell, median contained fraction of
the **photon's own** energy:

| region | w=1 | w=2 | w=4 | w=8 |
|---|---|---|---|---|
| 15 mm | 0.944 | 0.983 | **0.998** | 1.000 |
| 30 mm | 0.961 | 0.989 | 0.999 | 1.000 |

A seed-centred $9\times9$ window already holds the whole photon. Widening to
$17\times17$ added no signal at all, yet it is the largest single improvement
in the project (15 mm low-E 0.1654 → 0.1156).

## Why the width still matters: the centre is wrong, not the shower wide

The window is placed on the reconstructed barycentre, which pileup drags a
median 3.1 cells off the photon at 15 mm. On the paired overlay sample, where
each cell's photon content is known, a **barycentre-centred** window at 15 mm
holds:

| w | median photon kept | events keeping < 90% |
|---|---|---|
| 2 | 0.563 | **73%** |
| 3 | 0.974 | 38% |
| 4 | 0.996 | 7.8% |
| 6 | 1.000 | 0% |

So below $w = 6$ the aperture loses the photon for a substantial tail of events,
and the width is buying tolerance to the centre estimate. Above $w = 6$ the
photon is always inside, yet the window scan keeps improving out to $w = 8$–10 —
that part is not coverage. It is the pileup field: at 15 mm the $17\times17$
window carries 3.6× the photon's own energy over a median 160 fired cells.

## Three controls that separate the two roles

1. **A plain calibrated sum, no model.** Widening makes 15 mm low-E monotonically
   worse: 0.36 (w=1) → 0.53 (w=8). The outer cells are not signal to be added.
2. **`--sum-core`, barycentre centring.** Restrict the network's gated sum to a
   central core while the encoder still attends over the whole window:

   | sum region | members | 15 mm low-E | 30 mm low-E |
   |---|---|---|---|
   | full $17\times17$ | 0.0399 | 0.0755 | 0.0746 |
   | core $w=4$ | 0.0404 | 0.0863 | 0.0761 |
   | core $w=2$ | 0.0436 | 0.1270 | 0.0888 |

   The damage tracks the truncation tail above almost exactly — 7.8% of events
   truncated at $w=4$, 73% at $w=2$.
3. **`--sum-core` with the two-stage pointer.** If the failure is centring and
   not the core idea, then pointing first should make the core free. It does:
   with the window centred on the predicted photon position (median error 0.16
   cells), a $w=2$ core scores **0.0389** — equal to the two-stage member mean
   with the full sum (0.0389 ± 0.0002), and far from the 0.0436 the same core
   costs on a barycentre-centred window.

## What this means

- The corrected mechanism: **a wide window is an error-tolerant aperture first
  and a pileup sampler second**, with the crossover measured at $w \approx 6$.
  The paper now says this, with the numbers above.
- Control 3 is a deployment result rather than an accuracy one. Once the centre
  is accurate, the energy sum needs 25 of 289 cells; the other 264 are read by
  the encoder but never summed. Nothing is gained in resolution — the network
  was already ignoring their energy — but the readout becomes a much smaller
  object, which matters for a trigger implementation.
- The signal-to-background argument that motivated the test (S/B in the core is
  6:1 against 1:3.6 across the window) did **not** translate into resolution.
  That is the fourth independent time a hand-designed decomposition matched but
  did not beat what the free network already does.

## Status

Round 11 was cut short by credit exhaustion after the first seed of the
pointer-plus-core arm; the 0.0389 above is one member, and its prediction CSV is
still on the Studio, not downloaded. The `--sum-core 3` arm never ran. Neither
changes the conclusion, but a second seed would be worth having before the
number appears in a table rather than in prose.
