# Proposal v5 vs delivered — gap check

2026-08-20. Line-by-line against `Worakan_Proposal_v5.pdf` (submitted 2026-03-31).

## Delivered and exceeded

| Promise | Status |
|---|---|
| Timing ablation with/without (M4) | DONE, beyond promise: 20% aggregate / 38% weak bins, plus the clean-vs-pileup figure proving timing is a pileup tool |
| Resolution fit a/sqrt(E) + b + c/E decomposition | DONE (Fig. resolution fit, per region) |
| Comparison on same data/metrics across models | DONE far beyond: 123-configuration ledger, paired ParticleNet/GravNet, 10-fold CV |
| GravNet (listed as "optional, if time allows") | DONE, paired, 2 seeds |
| ClusTEX-style variant (stretch) | Measured (patch-token arm screened out; ClusTEX global-coordinate idea measured neutral) |
| Exploration notebook >= 6 diagnostic plots | DONE (nb14 and successors) |
| Stage1->4 dataset progression | Superseded by reality: clean sample = Stage 1, minimum-bias = Stage 2; all five regions + 2 longitudinal segments (Stage 3/4) present from the start |
| Linearity / p68 / robust-interval metrics | DONE (sigma_eff is the promised robust half-width; linearity figure in paper) |
| "Reduce the constant term from the 2D floor" | DONE in spirit: aggregate 0.0388 out-of-sample vs 7.3% 2D floor of the evaluation task |

## Promised but NOT done (real gaps)

1. **Cellular Automaton + Graph Clustering baselines (M1/M2, "primary comparison")** —
   never implemented. Honest reframe: the mentors' matched-cluster data IS the output of the
   standard reconstruction chain, so our calibrated-sum baseline (sigma_eff 0.1837) is the
   production algorithm's energy estimate; re-running CA/Graph Clustering upstream was not
   possible from the provided format. The paper says "standard reconstruction" — should add one
   sentence making this substitution explicit. GNN-production outputs were requested from the
   mentors (meeting ask, 2026-08-18).

2. **Space-time positional encoding MLP(sin(omega*[x,y,z,c*t]))** — the proposal's core novelty
   sentence. Raw timestamps as token features won every measured comparison against engineered
   time, but the *Fourier* form of the proposal was never tested on the time axis until now:
   `--tfour` is literally in the H100 queue at this moment. When it lands, the proposal's central
   idea has been tested rather than dropped — either way the story closes.

3. **Latency-vs-HLT-budget plot (M5 deliverable)** — throughput figure exists (CPU), but the
   HLT1/HLT2 budget line, GPU latency, batch sweep {1,32,128,1024} and peak-GPU-memory curve were
   not produced. GPU benchmark is runnable on the Studio after the sweep. The budget number
   itself needs Felipe (per-event budget in Allen).

4. **Tagged code release on GitHub (M6)** — nothing is committed/pushed yet (by agreed policy,
   pending your review). The paper now cites the repo URL, so this is a blocker.

5. **Presentation slides (M6)** — not made.

6. **Final report in LHCb internal-note format (M6)** — the 22-page paper exceeds the content,
   but the *format* depends on Felipe's answer (note vs journal vs proceedings).

7. **Unit tests for the data pipeline (Phase 1 deliverable)** — not written.

8. **Position (x,y) + PID heads** — the proposal's architecture figure promised them; scope
   narrowed to energy-only with the mentors' framing. Should be acknowledged as a de-scope in
   the report, one sentence.

9. **Kernel (linear) attention O(N)** — deliberately not done, with a measured justification:
   81–289 tokens per event makes full attention exact and cheap (paper Sec. Related work).
   Justified deviation, already documented.

## Recommended order to close

1. Commit + push + tag (needs your review of the commit plan) — unblocks #4 and makes the paper's
   repo link real.
2. GPU benchmark job on the Studio after the sweep finishes (closes most of #3; ask Felipe only
   for the budget number).
3. Slides (~15 slides from the paper's figures; fast).
4. One sentence each in the paper for #1 (baseline substitution) and #8 (de-scope).
5. Unit tests: a minimal pytest for splits/window/geometry invariants (~1 hour) — cheap
   insurance and fulfils the promise literally.
6. LHCb-note reformat: wait for Felipe.
