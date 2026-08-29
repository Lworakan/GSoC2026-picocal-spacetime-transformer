# The plan, after the fold discovery: training quality is the lever

Worakan Lasudee, 2026-08-29. Supersedes the framing in
`plan_to_0p035_2026-08-29.md`, which still treated data as the residual term.

## What changed today

Three arms, retrained on identical events with identical flags, came back
4–5% better than the versions the paper is built on. That is not a modelling
gain — it is the same model trained properly. It puts a number on something
the campaign never measured:

| source of variation | size |
|---|---|
| seed-to-seed, same arm same fold | **0.0002 – 0.0004** |
| fold-to-fold, same arm, after repair | 0.0011 (`Rr01`: 0.0380 – 0.0393) |
| bad training vs good training, same everything | **0.0015 – 0.0024** |

**Training quality varies 5–10× more than seeds do.** Every ensemble we have
built averages the small term and leaves the large one untouched. That is why
ensembling buys only 2.6–3.7% across every arm we have, at 5× the inference
cost — and why chasing it further was the wrong direction.

## Where the remaining headroom is

Even after the repair, `Rr01` spans 0.0380 to 0.0393 across folds that are
statistically identical by construction. If every run landed where the best
run lands, that arm would read 0.0380 rather than its 0.0386 mean — **1.5%
from convergence alone**, with no new mechanism and no new data.

The suspected cause is in the training loop and is written down in
`train_picocal.py`: the final weights are the EMA state frozen at the single
epoch that minimised a validation loss computed on **3,264 events** under ten
folds. Choosing the minimum of ~100 noisy numbers selects the epoch that was
lucky on those events. The literature calls the fix weight averaging —
[SWA](https://arxiv.org/abs/2502.10119),
[model soups](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf) —
and our own nb52 already measured the two halves of it: averaging **along one
trajectory** (EMA) won 4%, while averaging **across seeds** failed
catastrophically (0.2610), because independently initialised runs sit in
different basins. We took the win and then gave part of it back at selection
time.

## The ladder

| step | value | status |
|---|---|---|
| published headline | 0.0388 | five members, damaged folds |
| repair folds 5–9 | ≈ 0.0378 | measured on two members at one seed: **0.0382**; third member requeued |
| `--final-ema` — stop selecting on noise | ? | five folds running |
| SWA over the last *k* epochs | ? | next, if `--final-ema` moves |
| gate supervision `--gatesup` | −2.3% | 395/400 on the development split, W4 era; W8 confirmation pending |
| scale the overlay that supervision reads | ? | 9,717 of 72,554 events carry the gate target; we generate it ourselves |

If the repair lands at 0.0378 and gate supervision holds, that is 0.0369
before any convergence work. The convergence terms above are the only
measured place where several more per cent could come from without new data.

## Gates, fixed now

1. **Repair** — the five-member ensemble must improve by roughly the 2% the
   single seed showed. If it does not, the single-seed result was not
   representative and nothing here is published.
2. **`--final-ema`** — must be at least as good in 13 of 15 bins and
   significantly worse in none. It is a variance fix, so a per-bin trade-off
   would mean the diagnosis is wrong.
3. **Gate supervision under cross-validation** — 380/400 or it is dropped.
   This is where the two-stage window died after reading 400/400 on the
   development split.
4. **Overlay scaling** — the `--synaux` control repeated at every volume. The
   gain must come from supervision, not from the extra events.

## What is deliberately not in this plan

More ensemble members. They buy 2.6–3.7% at five times the inference cost,
the mentors asked for throughput, and the term they reduce is the small one.
Single-model numbers become the headline; the ensemble becomes an option.
`Rp` already reads **0.0389 as a single model**, within a whisker of the
five-member ensemble it is compared against.
