# Error analysis of every result, and the experiment that follows from it

Worakan Lasudee, 2026-08-25. All numbers below are recomputed from
`reports/predictions/` under the stated protocol; nothing is quoted from an
older report.

## 1. Where the error is, on the production model

Ten-fold ensemble, 72,533 events, aggregate $\sigma_{\rm eff} = 0.0388$.
"Agg if perfect" replaces that bin's predictions with truth and rescores
everything — the most any improvement to that bin can buy.

| bin | n | share | bin $\sigma_{\rm eff}$ | agg if perfect | gain | median E |
|---|---|---|---|---|---|---|
| 60 mm low | 8313 | 11.5% | 0.0526 | 0.0327 | **0.0061** | 7.5 GeV |
| 40 mm low | 6956 | 9.6% | 0.0580 | 0.0335 | **0.0053** | 12.0 GeV |
| 30 mm low | 4660 | 6.4% | 0.0757 | 0.0346 | 0.0042 | 20.7 GeV |
| 60 mm mid | 8312 | 11.5% | 0.0310 | 0.0354 | 0.0034 | 16.8 GeV |
| 40 mm mid | 6955 | 9.6% | 0.0316 | 0.0358 | 0.0030 | 26.1 GeV |
| 15 mm low | 2789 | 3.8% | 0.0787 | 0.0363 | 0.0025 | 31.7 GeV |
| … | | | | | | |
| 120 mm high | 1461 | 2.0% | 0.0297 | 0.0382 | 0.0006 | 19.1 GeV |

**The headline bin is not the worst bin.** 15 mm low-E has the largest
$\sigma_{\rm eff}$ but only 3.8% of the events; the aggregate is carried by
60 mm and 40 mm at low *absolute* energy. Two error regimes are visible:
$\sigma_{\rm eff}$ falls with absolute energy everywhere except 15 mm and
30 mm low-E, which sit far above the trend — 15 mm low-E is $0.0787$ at
31.7 GeV where 60 mm high-E is $0.0232$ at 27.1 GeV, a factor 3.4 at the same
energy. That gap is the pileup, and it is confined to 10.2% of the sample.

## 2. What is closed, with the measurement that closes it

Every entry is a paired comparison on identical events.

| lever | verdict | evidence |
|---|---|---|
| window size, per region | **closed** | no width beats $w{=}8$ significantly in any of the 5 regions; best case 353/400 ($P{=}0.88$); $w{=}8$ strictly best at 30 mm (0/400) |
| window size, at the high-leverage bins | **closed** | 60 mm low-E: $w4/w6/w7$ all flat (210, 141, 61 of 400). 40 and 30 mm low-E prefer $w{=}8$ outright |
| window centring | **closed** | module centre vs *true photon position*: $+0.0004 \pm 0.0004$, $P{=}0.86$ |
| aperture, constant mm radius | **worse** | $R{=}120$: $+0.0013 \pm 0.0004$, 398/400. $R{=}90$: $+0.0136$, 400/400 |
| aperture, corners only | **null** | $-0.0001 \pm 0.0003$, 149/400 |
| ring/halo extension | **trades regions** | 6/15 bins; loses all three 120 mm bins (ring 15 is 1800 mm there) |
| per-event window routing | **bounded** | oracle 0.0384 vs 0.0388; no threshold wins everywhere |
| loss reweighting `--wlow` | **worse, dose-response** | $\alpha = 0.4/0.5/0.7/1.0 \to$ 0.0399/0.0406/0.0410/0.0431 |
| gate supervision `--prior-*` | **worse, dose-response** | 15 mm low-E $+7.7\% \to +14.8\% \to +25.2\%$ with weight |
| per-bin bias removal | **worse** | 13/15 bins worse; aggregate $0.0388 \to 0.0390$ |
| timing, engineered | **closed** | 5 constructions, all lose to raw timestamps |
| encoder family | **closed** | 10 families, none wins |
| two-stage pointing | **fails CV** | dev-split 0.0378, does not reproduce |
| ensembling | **exhausted** | 28 members, pooled 0.0376 |

Two of these are dose-response curves rather than single nulls. That matters:
a single null can be an underpowered experiment, but a monotone degradation
with the intervention's strength is positive evidence that the intervention is
harmful, and both point the same way — the network's own weighting is better
than any weighting we impose on it.

## 3. What the timing ablation says about the top bin

Timing is a pileup-separation tool, so it should buy most where pileup is
worst. It does:

| | 60 mm | 40 mm | 30 mm | 15 mm | 120 mm |
|---|---|---|---|---|---|
| low-E | 19.8% | 26.3% | **36.1%** | 27.0% | 28.2% |
| high-E | 10.4% | 9.9% | 24.1% | 22.3% | 13.4% |

It buys the **least** at 60 mm — the highest-leverage bin. So 60 mm low-E's
$0.0526$ is not mostly pileup. With sampling at $0.0225$ (18% of the variance)
and the $1/E$ term at $0.0387$ (54%), the dominant component is an
energy-independent $\approx 0.29$ GeV that timing cannot reach and that a
post-hoc correction on window energy makes worse. We do not have a mechanism
for it, and naming one would be a guess.

## 4. The experiment this justifies

No exploratory arm has supporting evidence. Every mechanism with a plausible
story has been measured, and the two with the strongest priors failed with
dose-response curves. Designing another exploratory sweep would be searching
where we have already looked.

What is *not* yet established to publication strength is the optimality claim
itself. The per-region window nulls come from arms trained **without
recentring**, so they are an era behind the production configuration, and they
carry three seeds. The claim "no window beats $w{=}8$ in any region" deserves
to rest on matched-era, five-seed evidence.

### Experiment matrix

Factor: window half-width. Levels: $w \in \{6, 7\}$ against the existing
$w{=}8$. Everything else fixed at the production configuration.

| Run | Factor | Value | Fixed config | Seeds | Predicted outcome |
|---|---|---|---|---|---|
| B | — | $w{=}8$ | `--window 8 --extra --dens --recenter --cleanaux --aux` | 0–4 (exists) | baseline 0.0387 |
| A1 | window | $w{=}7$ | as B | 0–4 | within noise of B in every region; possibly ahead at 15 mm |
| A2 | window | $w{=}6$ | as B | 0–4 | behind at 30 mm; flat elsewhere |

Fixed across all runs: objective `qd`, EMA on, 3 seeds' worth of architecture
(d=128, 4 heads, 3 layers), the same fixed 70/15/15 split, 100 epochs,
patience 15, batch 96, lr 3e-4.

### Resource estimate

| item | value |
|---|---|
| runs | 2 arms × 5 seeds = 10 |
| time per seed | ~35 min measured on the H100 (2117 s, 2046 s, 1811 s observed) |
| **GPU hours** | **≈ 5.8 h** |
| storage | 2 prediction CSVs, ~4 MB |
| API cost | none |

### Execution

```bash
B='--sample minbias --extra --dens --recenter --cleanaux --aux'
python scripts/train_picocal.py $B --window 7 --seeds 0 1 2 3 4
python scripts/train_picocal.py $B --window 6 --seeds 0 1 2 3 4
```

Runs are sequential on one GPU; no dependency between them, so they can be
split across two GPUs if available. Checkpointing is already per-seed and
resumable.

### Analysis plan, fixed before the runs

- **Primary**: per-region $\sigma_{\rm eff}$, five-seed median ensemble, paired
  bootstrap over 400 shared resamples against the $w{=}8$ arm on identical
  events (`scripts/paired_encoders.py` pattern).
- **Secondary**: the same in all fifteen region–energy bins, and the aggregate.
- **Decision rule, set now**: the optimality claim stands if no alternative
  width wins in more than 380 of 400 resamples in any region. If one does, the
  claim is withdrawn and that width becomes the new production setting for
  that region.
- **Failed runs**: a crashed seed is rerun once; if it crashes again the arm is
  reported at the seeds that completed, with the count stated.
- **Plots**: one figure, $\sigma_{\rm eff}$ against $w$ per region with
  bootstrap bands, replacing the current era-mixed window scan.

### What this experiment cannot do

It cannot lower 0.0388. It converts a negative result into a positive claim,
which is the only thing on this sample that is still worth GPU time. The
measured scaling curve, $\sigma_{\rm eff} \propto N^{-0.18}$, still puts a
threefold sample at $\approx 0.033$ — an order of magnitude more than
everything in Section 2 put together, and the reason the data request is the
real experiment plan.
