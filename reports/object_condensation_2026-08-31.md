# Better than a pairwise bias: supervise the latent geometry, not the inputs

31 August 2026. Never tried here -- `grep -i "condensation\|contrastive\|repuls\|triplet"`
over `scripts/` returns nothing.

## The failure this is aimed at, restated precisely

The project has one durable, reproducible finding about why it is stuck:

> The information is present in the inputs and the network routes it around the gate.

Measured: a gradient-boosted per-cell estimator reaches correlation **0.945** with the true
photon fraction on exactly the observables the network receives; the network's own gate
reaches **0.211**.

Now look at what has been tried against that, and notice they are all the same kind of move:

- **Five timing constructions** -- changed the *input features*.
- **Four gate-supervision protocols** (direct, auxiliary head, prior feature, distillation)
  -- attached an *auxiliary scalar target* to a per-cell output.
- **`--cellreg`** -- changed the *readout arithmetic*, and this one worked, which is the
  hint.

Nothing has ever changed the **geometry of the latent space the aggregation reads**. An
auxiliary per-cell scalar is easy to ignore: the network can satisfy it in one coordinate
and carry the energy in the others. That is precisely what "routes around the gate"
describes, and it explains four nulls with one mechanism.

## Object condensation changes exactly that

Kieseler's object condensation loss is the HEP-native method for turning calorimeter hits
into objects, and it is what CMS uses for end-to-end reconstruction in HGCAL at 200
pileup -- the same occupancy regime as ours. It has three terms:

- **`L_V`, the potential loss** -- an *attractive* pull binding hits of the same object to
  that object's condensation point, and an explicitly *repulsive* push separating hits of
  different origin in the clustering space.
- **`L_beta`, the condensation score** -- learns which hit is the condensation point.
- **`L_P`, the payload loss** -- carries the object's properties, and in the HGCAL work
  that payload is literally *the energy correction factor*.

The repulsive term is the part nothing here has ever had. It does not ask the network to
*report* a photon fraction; it makes cells of different origin **far apart in the space
the attention operates on**. Separation stops being something the network may route around
and becomes a property of the representation the aggregation is forced to read through.

And `L_P` is not a foreign attachment: our readout is already
`tot = sum_i relu(w_i E_i + r_i E_i)`, a per-cell energy summed over the object. The
payload loss is the supervision that readout was built for and never received --
`picocal_models.py:322` records in its own comment that nothing supervises `rhead`.

## The adaptation, and why it is easier than the published case

Standard object condensation reconstructs an unknown number of objects. Ours is the easy
case: **one object -- the photon -- and background.** OC already treats background
explicitly by driving its condensation score down, so single-object OC reduces to
supervised metric learning with a repulsive term against pileup. Fewer moving parts than
the HGCAL application, not more.

The labels exist and are already validated. The overlay's per-cell `sig` gives each cell's
photon share, so the object assignment is a threshold on a truth quantity we generate. And
this is not a speculative label source: gate supervision on exactly these labels is what
took the model from 0.0402 to 0.0376 once the overlay covered every region.

A workable form, with `q_i` the cell's charge, `x_i` its embedding, `alpha` the
condensation point:

```
L_V = sum_i q_i [ y_i * ||x_i - x_alpha||^2  +  (1 - y_i) * max(0, 1 - ||x_i - x_alpha||)^2 ]
```

`y_i = 1` where the photon dominates the cell, 0 where pileup does; `q_i` weighted by cell
energy so a 10 GeV cell matters more than a 10 MeV one, which is the same scaling `rhead`
already uses.

## Why this ranks above the pairwise bias

Both are worth running and they compose -- one changes the input representation, the other
the latent metric. But if only one runs first:

| | ParT pairwise bias | object condensation |
|---|---|---|
| changes | input representation | geometry of the latent space |
| against the measured failure | indirect: better features to route around | direct: makes separation structural |
| precedent for null | five representation changes already null | none -- untried category |
| needs labels | no | yes, and they exist and already work |
| brings a missing baseline | yes, ParT | no |
| matches the readout we have | no | yes, `L_P` is what `rhead` lacks |

The pairwise bias closes a hole a referee will point at. Object condensation attacks the
thing that has actually been blocking the number for the whole project. If the goal is a
result other groups must beat rather than a citation gap closed, this is the one.

**Best single configuration to run:** `--cellreg` (sum-decomposition, already winning) with
object condensation on the encoder embedding and the payload loss supervising `rhead`
against overlay `sig`, on the region-complete overlay. Pairwise bias second, and it can be
added to the same model without conflict.

## The honest risk

This is a fifth per-cell supervision scheme after four nulls. The distinction that makes it
worth the GPU is mechanical, not hopeful: the four nulls all supervised a *scalar the
network could satisfy in an unused coordinate*, while a repulsive potential constrains the
*distances* in the space the attention itself consumes. If it also measures null, that is a
strong result rather than an absent one -- it would say the per-cell information is not the
binding constraint at all, and the remaining width is containment and sampling, which the
design-floor table already suggests for the outer regions.
