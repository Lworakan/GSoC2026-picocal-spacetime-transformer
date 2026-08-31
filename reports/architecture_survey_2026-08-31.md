# Which transformer actually fits photon energy regression, and what it takes to become the baseline

31 August 2026. Two separate questions are answered here, because the second is the one
that was really asked and architecture alone does not settle it.

## The constraint most set architectures get wrong for this task

Energy is **extensive**. The target is a sum over cells of a per-cell quantity, and any
readout that compresses the set into a fixed-size latent vector and regresses from it has
thrown away the one structural fact we are certain of. This is not a preference; it is the
single strongest prior available, and this project has already measured it twice:

- The gated sum `S = sum_i sigma(f_i) E_i` is multiplicative and cannot express an additive
  error. It plateaued.
- `--cellreg` replaced it with `relu(w_i E_i + r_i E_i)` summed over cells -- a genuine
  sum-decomposition where a cell's output may exceed its own deposit -- and it won
  (0.0402 -> 0.0394 alone, 0.0376 with gate supervision).

The current readout, verified in `picocal_models.py:445-462`, is already the right shape:
`tot = sum(cell_e)`, `base = a log1p(tot) + b`, plus a correction head reading an
attention-pooled embedding and the globals. That is Deep Sets **conditioned by**
attention, not replaced by it.

So the question is not "which transformer replaces this" but "which transformer conditions
the per-cell function better".

## The survey, and what each is worth here

**Particle Transformer (ParT) -- take it.** Pairwise features embedded as
`U in R^(N x N x d')` and added to the attention scores pre-softmax, `d'` = number of
heads. It beat ParticleNet by a large margin on jet tagging and displaced it as the
reference. For us it is the only candidate that acts on the aggregation weights, which is
where the gate study located the bottleneck, and the only one that can express
`t_i - t_j`, the reference-free form of the pileup discriminant that five absolute timing
constructions failed to use. Full argument in `pairwise_bias_2026-08-31.md`. It also fixes
a hole: the paper cites ParT at `main.tex:119` and re-implements only ParticleNet and
GravNet, the two architectures it superseded.

**L-GATr, Lorentz-equivariant geometric algebra transformer -- reject, and say why.** It
is the strongest recent HEP architecture and it is the wrong symmetry group for this
problem. L-GATr is equivariant under Lorentz transformations because jet physics is
boost-covariant. We are measuring an energy in a fixed detector frame: a boost changes the
very quantity being predicted, so imposing boost equivariance would impose an invariance
the target does not have. Naming this explicitly is worth a sentence in the paper --
"the fashionable equivariance is the wrong one here" is a stronger position than silence.

**Set Transformer (PMA, ISAB) -- already present, nothing to gain.** Pooling by Multihead
Attention with learned seed vectors is what `pool_attn` at `picocal_models.py:315` does,
and `slot_attn` covers the inducing-point idea. ISAB exists to cut attention from O(N^2);
with a 17x17 window that is 289 tokens, which is not a cost problem -- four trainings fit
in 18.5 GB of a 40 GB card. Nothing here is worth a run.

**Perceiver / Perceiver IO -- reject on the extensivity argument.** A latent bottleneck of
fixed width is exactly the readout that discards the sum structure. It is the right tool
when N is enormous and the target is not a sum. Neither holds.

**Point Transformer / vector attention / local-window attention -- subsumed.** These
inject geometry through neighbourhood restriction. A pairwise bias with `log Delta r` as
one channel expresses the same locality more flexibly and in one mechanism, so running
both would be redundant.

**Recommendation: ParT-style pairwise bias on top of the existing sum-decomposition
readout.** One change, one new mechanism, one missing baseline closed.

## What actually makes a result "the baseline everyone must beat"

Architecture is the smaller half. Looking at why ParticleNet and then ParT became the
things people benchmark against, none of it is the network:

1. **A public dataset with a fixed split.** JetClass exists and everyone reports on it.
   Ours is a private LHCb simulation sample. Until a split is published or the numbers are
   reproducible on something others can obtain, no one can be beaten by us. This is the
   single biggest gap and it is not a modelling problem.
2. **Code and weights released, runnable.** `research-publishing` conventions: pinned
   environment, one command per figure, checkpoints downloadable.
3. **One metric everyone already reports.** We have this and it is undervalued in the
   paper: `sigma_eff` as the half-width of the central 68% interval is *identical* to the
   definition the GNN work on this detector uses, and the E_T binning has now been matched
   too. That is a genuine claim to being the comparable number on this detector.
4. **The strongest published baseline run, not cited.** ParT. Point 4 and the architecture
   recommendation are the same piece of work, which is why it ranks where it does.
5. **A protocol that cannot be argued with.** Ten-fold cross-validation over every event,
   paired bootstrap with a stated decision bar. The paper already has this and it is
   stronger than what most architecture papers do. Finishing the six remaining folds is
   what makes it true of the *current* model rather than the previous one.

Items 1 and 2 cost no GPU and are worth more than another 2% on the number. Item 1 in
particular is a question for the mentors: whether any part of this sample can be released,
or whether the paper should ship a generator configuration so the sample is reproducible
rather than shared.

## Ranked, merged into the existing queue

1. Finish the six cross-validation folds (nothing substitutes)
2. **ParT-style pairwise bias** -- new mechanism plus the missing baseline
3. `--cellsup` -- supervise the per-cell head against overlay truth
4. Release artifacts: split definition, environment, weights, one command per figure
5. Supervision volume 2x/3x
6. Three seeds per fold

The change from the earlier list is that ParT moves above `--cellsup`, and the release
artifacts enter the list at all -- because the goal stated was not a smaller number but a
result other groups have to beat, and a number nobody can reproduce is not that.
