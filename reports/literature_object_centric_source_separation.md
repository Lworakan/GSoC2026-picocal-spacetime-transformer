# Object-centric / source-separation literature for EM-blob deblending (verified 2026-08-02)

Setting: N EM blobs on 9x9 grid, universal lateral shape known, only aggregate signal-energy label. Current: token transformer, sigma_eff 0.0402; nb27 showed free NMF fails.

## Ranked references (all arXiv IDs fetched/verified)

1. **StarNet / BLISS — Liu, McAuliffe, Regier, "Variational Inference for Deblending Crowded Starfields", arXiv:2102.02409, JMLR 2023.**
   Mechanism: amortized variational inference; encoder maps image to posterior over {N sources, positions, fluxes}; generative model = known PSF x flux. Trained on simulations sampled from the generative model (forward-KL / wake-sleep) — never on labeled real blends. Beat MCMC cataloging and DAOPHOT on M2 globular cluster.
   Supervision: none on real data; needs only the forward model (we have it: universal shower shape). Small data: fine, training data is self-simulated. GPU: small CNN encoder, easily 12GB.
   Verdict: **best template.** Replace PSF with the EM lateral shape, add per-blob time; the aggregate signal-energy label supervises a "which blob is signal + its energy" head on top. Closest published match to our exact problem.

2. **scarlet — Melchior et al., "Source separation in multi-band images by Constrained Matrix Factorization", arXiv:1802.10157, A&C 2018.**
   Mechanism: NMF-like A x S factorization made identifiable by proximal constraints: positivity, monotonic-from-peak, symmetry, PSF forward model, per-source centers. This is the direct answer to why nb27's free NMF failed — free NMF has rotational/permutation degeneracy on smooth overlapping blobs; identifiability comes from the shape constraints + known convolution kernel, not the factorization itself.
   Supervision: none; needs detected peak positions as init. GPU: per-event convex-ish optimization, CPU-cheap on 9x9.
   Verdict: **strong classical baseline** — fix S to the universal shape (only amplitudes+centers free) and it becomes a per-event weighted least-squares deblend; worth an afternoon.

3. **MADNESS — Biswas et al. (LSST DESC), arXiv:2408.15236, A&A 2025.**
   Mechanism: VAE trained on *isolated* simulated galaxies + normalizing-flow prior in latent space; deblending = gradient-descent MAP in latent space under the pixel likelihood of the full blended scene. ~29% lower flux residual than scarlet.
   Supervision: isolated-source sims only (we can generate isolated showers trivially). GPU: small VAE, 12GB fine.
   Verdict: high — "learn prior on singles, optimize on blends" sidesteps per-blob labels entirely.

4. **scarlet2 score prior — Sampson, Melchior et al., "Score-matching neural networks for improved multi-band source separation", arXiv:2401.07313, A&C 2024.**
   Mechanism: diffusion/score network trained on isolated sources replaces scarlet's hard proximal constraints; score gradient plugged into the analysis-by-synthesis optimization.
   Supervision: isolated-source sims. Verdict: medium-high; for us the shape is parametric so a score prior is only needed if the "universal shape" has residual variability (it does: containment fluctuation, per nb16 — score prior could model exactly that).

5. **PUMML — Komiske, Metodiev, Nachman, Schwartz, arXiv:1707.08600, JHEP 2017.**
   Mechanism: CNN regresses leading-vertex energy image from total+charged images; HEP pileup subtraction on calorimeter grids.
   Supervision: per-pixel leading-vertex truth from sim. Verdict: medium — same physics, but needs the per-pixel labels we lack unless we build overlay sims (nb28 arc already probes this).

6. **Slot Attention — Locatello et al., arXiv:2006.15055, NeurIPS 2020.**
   Mechanism: iterative competitive attention binds K exchangeable slots; decoder reconstructs per-slot image + alpha masks; trained by reconstruction only.
   Supervision: none. GPU: trivial at 9x9. Verdict: medium — vanilla slots on smooth overlapping Gaussians tend to split by intensity/region, not source; but replacing the free decoder with the *universal shape template* (slot = center, energy, time; decode = template render; sum-reconstruction loss + aggregate-energy loss) turns it into amortized template fitting — essentially a learned StarNet encoder. The hybrid, not the original, is the useful idea.

7. **DINOSAUR — Seitzer et al., "Bridging the Gap to Real-World Object-Centric Learning", arXiv:2209.14860, ICLR 2023.**
   Mechanism: slot attention over DINO self-supervised features instead of pixels. Verdict: low — DINO features are meaningless on 9x9 energy grids; evidence is natural-image only.

8. **SAVi — Kipf et al., "Conditional Object-Centric Learning from Video", arXiv:2111.12594, ICLR 2022.**
   Mechanism: slots conditioned on location cues, tracked through video with optical flow as weak supervision. Verdict: low-medium — no video, but the per-cell *time* dimension is our analog of flow (in-time vs out-of-time blobs); conditioning slots on timing peaks is the transferable idea.

9. **Probabilistic Slot Attention — Kori et al., arXiv:2406.07141, 2024.**
   Mechanism: aggregate mixture prior over slots gives identifiability guarantees without supervision. Verdict: low-medium — theory support that slot decompositions can be identifiable when a mixture prior (for us: the physics shape) is imposed; benchmarks synthetic only.

10. **Arcelin et al., "Deblending galaxies with VAEs", arXiv:2005.12039, MNRAS 2021.**
    Mechanism: VAE prior trained on isolated centered galaxies, second network deblends the centered source from the blend. Verdict: low-medium — assumes source centered in stamp (true for our signal blob!), but regresses images not energies; superseded by MADNESS.

## Synthesis
The astronomy lineage (2 → 1 → 3/4) is the answer to "what makes constrained decomposition work where free NMF fails": (a) known forward template/PSF, (b) positivity + center/shape constraints or a prior learned on *isolated* sources, (c) explicit likelihood-based analysis-by-synthesis. None need per-blob labels. Concrete next experiment: StarNet-style amortized inference — transformer encoder emits K x (center, energy, time) slot parameters, differentiable renderer with the universal shape reconstructs the 9x9 window (reconstruction loss), signal head supervised by the aggregate label; train on overlay sims (nb28 machinery). Fits one 12GB GPU; per-event scarlet fit with frozen shape is the cheap baseline to run first.
