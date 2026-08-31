# The one architecture idea that explains every timing failure, and has not been tried

31 August 2026.

## The observation that started it

Timing has been attacked five ways in this project and every one measured null. That is
strange, because the discriminant is large and it is present in the sample. On 2,000 real
15 mm minimum-bias events, with each cell's front-face time referenced to the shower
core's own time:

| cells | n | median offset | IQR |
|---|---|---|---|
| core, d <= 1 | 8,181 | 0.000 ns | 0.077 ns |
| ring, d = 2-3 | 27,159 | +0.181 ns | 0.300 ns |
| far, d >= 5 | 64,690 | +0.391 ns | 1.641 ns |

Cells far from the seed arrive later and are spread 21 times wider. This is the HL-LHC
pileup signature: interaction vertices are spread over about +-150 ps, which is why CMS
HGCAL targets sub-30 ps hit timing and PicoCal targets O(10 ps). The information is there.

## Why every attempt failed, mechanically

**A per-cell time is meaningless without a reference, and every reference we have is
contaminated by the thing it is meant to reject.**

- The default reference is the median over all cells in the window
  (`picocal_data.py:318`). Far cells outnumber core cells eight to one, so that median is
  set by pileup.
- `--tpull` replaces it with an energy-weighted top-decile time (`picocal_data.py:27`).
  Better in principle, but the top decile still moves when pileup lands near the seed. It
  loses in four of five bins: 15 mm mid 0.0532 against 0.0472, 15 mm high 0.0354 against
  0.0322, 30 mm high 0.0333 against 0.0307, aggregate 0.0409 against 0.0402.

Every construction tried has been *absolute*: a number attached to one cell, whose meaning
depends on a global t0 that cannot be estimated cleanly. That is a single explanation for
five independent null results, which is what makes it worth acting on.

## The fix is relational, and it needs no reference at all

Belonging to a vertex is a property of a **pair** of cells, not of one cell. Two cells from
the same shower have `t_i - t_j ~ 0` whatever the event's absolute time is. `Delta t_ij` is
invariant under a global time shift; a per-cell time is not. So the quantity that carries
the pileup information is pairwise, and the architecture has never been given it.

This is exactly the Particle Transformer construction. ParT embeds pairwise features into
`U in R^(N x N x d')` with `d'` equal to the number of attention heads, and adds the h-th
channel of `U` to head h's scores as a **pre-softmax additive bias**. On jet tagging that
one change beat ParticleNet by a large margin, and it is why ParT displaced it as the
reference architecture.

Two things follow for us.

**It attacks the bottleneck we actually measured.** The gate study showed per-cell
separability is available and unused: a gradient-boosted per-cell estimator reaches
correlation 0.945 with the true photon fraction while the network's gate reaches 0.211 and
the network routes around it. What fails is aggregation. A pre-softmax bias acts directly
on the aggregation weights -- it lets the model learn "attend to cells in time with the
seed and ignore the rest", which is the pileup-rejection operation stated in the only
place where it can be expressed.

**The paper has a hole here anyway.** ParT is cited at `main.tex:119` as related work, but
the re-implemented baselines are ParticleNet and GravNet -- the two architectures ParT
superseded. A referee at any HEP-ML venue asks why the strongest published attention
baseline was cited and not run. Implementing the pairwise bias closes the hole and
supplies the experiment in the same change.

## What the pairwise features should be

For a cell pair (i, j), all reference-free by construction:

1. `Delta t` front-face, the vertex discriminant measured above
2. `Delta t` front-back asymmetry difference, which separates depth from arrival time
3. `log Delta r` in millimetres, scaled by the Molière radius rather than by cell count --
   the window law in this project is written in millimetres, not cells
4. `log(min(E_i, E_j))` and `log(E_i + E_j)`, the standard ParT kinematic pair

## Cost, and the honest risk

`nn.MultiheadAttention` takes a float `attn_mask` that is added before the softmax, so the
bias needs no new attention kernel -- a pointwise convolution stack over the pair tensor
and one extra argument. With `--window 8` the window holds 289 cells, so `U` is
289 x 289 x n_heads, a few hundred thousand floats per event. Memory is not the obstacle;
four trainings already fit in 18.5 GB of a 40 GB card.

The risk is stated plainly: this is the sixth timing attempt after five nulls. What makes
it different is that it is the first *reference-free* one, and the reference is the single
mechanism that explains all five failures. If it also measures null, the conclusion is
strong rather than absent -- the timing in this sample is unusable for energy at
nanosecond truth resolution, and the paper should say so and ask for a sample with the
detector's design 20 ps smearing applied.

## Where it sits in the queue

Below finishing cross-validation, which nothing else can substitute for. Above `--cellsup`
if the goal is a result other groups have to beat rather than a smaller number, because it
brings the missing baseline and the new mechanism in one change. Both are worth running;
this one is worth running second, not fourth.
