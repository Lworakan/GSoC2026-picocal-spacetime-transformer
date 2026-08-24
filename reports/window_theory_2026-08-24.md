# Why the wide window wins, and what the theory predicts next

Worakan Lasudee, 2026-08-24. Everything below is measured on arms already in
`reports/predictions/`; no new training was needed to reach the conclusion.

## The observation that started it

The wide window helps the inner low-energy bins enormously and does nothing,
or slightly harms, everywhere else. Taken at face value this looks like a
trade-off to be engineered around — a router, a per-region window, an
ensemble that mixes both. Three of those were tried and none wins everywhere.

They were solving the wrong problem.

## First correction: the original comparison was confounded

The dramatic version of the observation compared
`SubNetW8CleanAuxExDnRcQdEma` against `SubNetW4CleanAuxExDnQdEma`. Those two
arms differ in **two** things: the window half-width, and whether the window
is recentred on the module. With both recentred and five seeds each, the wide
window is ahead in 11 of 15 bins and the four it trails are $+0.0003$ to
$+0.0010$ — inside the $0.002$ noise floor this project uses. Most of the
apparent trade-off was recentring, not size.

## The factorial

Four arms, all `CleanAux ExDn`, scored on the same 10,877 events:

| | w = 4 | w = 8 |
|---|---|---|
| centre = cluster barycentre | 0.1654 | 0.1156 |
| centre = module centre | 0.0761 | 0.0711 |
| centre = **true photon position** | — | **0.0713** |

(15 mm low-E; aggregate 0.0390 / 0.0386 / — / 0.0388 / 0.0383.)

Two things fall out immediately.

**The centring problem is closed.** The module centre and the true photon
position are statistically indistinguishable: $+0.0004 \pm 0.0004$ on the
aggregate, the module centre worse in 343 of 400 paired resamples ($P =
0.86$), and $-0.3\%$ in the 15 mm low-E bin where the difference would show
first. There is no gain left from pointing the window better. This is why the
learned two-stage pointer, which reached a median error of 0.16 cells, bought
nothing under cross-validation: it was competing against a centre that is
already optimal.

**The window's size only matters when the centre is wrong.** Widening from
w = 4 to w = 8 is worth $-30\%$ at the barycentre and $-6.6\%$ at the module
centre in the same bin. The wide window was never buying pileup information.
It was buying tolerance for a centre that pileup had dragged three cells off
the photon.

## The law

Why does widening help at 15 mm and not at 120 mm? Because the window is
defined in **cells** and the shower is a fixed size in **millimetres**. A
half-width of w cells spans $w \times \mathrm{pitch}$, and the pitch varies by
a factor eight across the calorimeter. Measuring the benefit of widening
against the physical size of the starting window, in Molière radii
($R_M \approx 35$ mm):

| region | pitch | w = 4 half-width | in $R_M$ | benefit of w = 8 |
|---|---|---|---|---|
| 15 mm | 15 mm | 60 mm | 1.7 | **−30.1 %** |
| 30 mm | 30 mm | 120 mm | 3.4 | −18.6 % |
| 40 mm | 40 mm | 160 mm | 4.6 | −3.1 % |
| 60 mm | 60 mm | 240 mm | 6.9 | +0.6 % |
| 120 mm | 120 mm | 480 mm | 13.7 | +4.3 % |

Spearman $\rho = +1.000$ across the five regions; Pearson on $\log R_M$ is
$-0.935$. The ordering is perfect. Widening helps exactly where the window is
physically small compared with a shower plus a centring error, and it starts
to *hurt* once the window is already many Molière radii wide, because the
extra cells then contribute pileup variance and no photon.

So there is no per-region trade-off in the physics. There is one quantity —
the window's physical radius — and the cell-based definition sets it wrong in
every region except by accident.

## The prediction

**Define the window in millimetres, not in cells.** Choose one physical
radius $R$ and take $w_{\rm region} = \lceil R / \mathrm{pitch} \rceil$:

| R | 15 mm | 30 mm | 40 mm | 60 mm | 120 mm |
|---|---|---|---|---|---|
| 120 mm | w = 8 | w = 4 | w = 3 | w = 2 | w = 1 |
| 140 mm | w = 10 | w = 5 | w = 4 | w = 3 | w = 2 |

This is one model with one rule, not a router, not an ensemble, not a per-bin
choice fitted to the test set. It gives the inner regions the aperture the
factorial says they need and takes away from the outer regions the cells the
same factorial says are pure variance there. If the law above is the whole
mechanism, this configuration should be at least as good as w = 8 in every
bin and better in the outer ones — the "wins everywhere" result that no
post-hoc combination could produce.

**Falsification.** If a millimetre-constant window is *not* at least as good
as w = 8 in the outer regions, then the extra cells there are carrying
something real — pileup context — and the aperture story is incomplete.
That is a clean test either way.

**Cost.** No architecture change. The window is already zero-padded and
masked to a maximum of 289 tokens, so the rule is implemented by masking
cells beyond $R/\mathrm{pitch}$ at the largest w, and $R$ becomes a single
scan parameter. One training run per value of $R$.

## What this replaces in the paper

Section~\ref{sec:gain} currently gives the wide window two roles: an
error-tolerant aperture below w ≈ 6 and pileup context above it. The
factorial supports the first and contradicts the second — at 60 mm and
120 mm, where a pileup-context term would help most, widening does nothing
($+0.6\%$) or hurts ($+4.3\%$). The second role should be withdrawn and
replaced with the Molière-radius law, which explains the same measurements
with one parameter instead of two mechanisms and a crossover.
