# ML architecture and training advances 2025-2026, judged against PicoCal energy regression

Survey date 2026-08-12. Every arXiv ID and number below was retrieved from a live search or
fetched page during this survey. Where a work is believed to exist but could not be retrieved,
it is marked **unverified**. Numbers are always given with the dataset size they were measured
at, because that is the whole question here.

## The problem, and how it prices each idea

- Scalar regression (photon energy) from a set of 81 tokens x 16-24 features (9x9 cell grid).
- 70k train / 11k test; metric-critical sub-bins of ~400 examples.
- Metric sigma_eff = half-width of smallest interval holding 68.3% of (E_pred-E_true)/E_true,
  per region and per energy tercile. Current 5-seed ensemble ~0.041.
- Noise structured, heavy, one-sided: photon is ~54% of observed energy (median), 10% in the
  worst 5% of events; ring-ring contamination correlation 0.45-0.63; residual 5/95 quantiles
  -0.26/+0.76; trimming the worst 10% of events cuts sigma_eff by 30%.
- Sixteen architecture/encoding experiments produced no aggregate movement. Only *adding input
  information* has ever helped.
- Data starvation is verified: 3x less data moves 0.041 -> 0.056.

Two consequences that reorder the requested list:

1. **Items 3 and 7 (SSM/linear-attention/DeltaNet/RWKV-7; KAN/GATr/equivariant) are priced to
   fail.** They are the 17th architecture swap. They are surveyed below to document the null,
   not to find a winner.
2. **Items 5 and 6 (test-time compute/transduction; small-data recipes) match the diagnosed
   bottleneck** and get the most depth, together with self-supervised pretraining (item 8),
   which is the only mechanism in the literature that converts *unlabelled* events into
   labelled-data-equivalent.

**Contents.** 1 in-context/PFN models; measurement bar; 2 coordinate encodings; 4 distributional and
robust heads; 3 attention alternatives (null); 7 KAN/GATr/equivariant (null); 5 test-time compute
(strongest item); 6 small-data recipes (mostly null); 8 other 2025-2026 work; ranked shortlist;
consolidated unsupported list. Items 5 and 6 carry the depth of the survey.

**An arithmetic note, stated carefully because it is easy to over-read.** The two measured points
(23k -> 0.056, 70k -> 0.041) give a *local* exponent beta = ln(0.056/0.041)/ln 3 = 0.28, i.e.
sigma_eff ~ N^-0.28 **valid only between 23k and 70k**. It cannot be extrapolated far, for a reason
you already established in nb16: the ~0.056-era floor was diagnosed as *containment fluctuation*, a
physical limit that does not shrink with more training data. So the honest model is
sigma_eff(N) = sigma_inf + a N^-b with **sigma_inf > 0**, a two-point fit cannot identify sigma_inf,
and the local slope necessarily *overstates* the slope beyond 70k. Under the optimistic
zero-floor assumption, 210k events would give ~0.030 — treat that as an **upper bound on the
achievable gain, not a prediction**, and treat anything quoted for 700k as unsupported. What survives
is the qualitative statement: **more training events is the only lever with a directly measured
effect on this problem, and no training trick in this survey has a documented effect approaching
even the conservative version of it.**

---

## 1. In-context / prior-fitted models for small tabular-or-set regression

### (a) Row and feature limits — the "out of range" premise partly fails

Flattened, you are 81x20 = **1,620 features at 70k rows**. Verified limits (PriorLabs official
model table, docs.priorlabs.ai/models):

| Model | Documented limit | Verdict on 70k x 1,620 |
|---|---|---|
| TabPFNv2 (Nature 2025) | 10k rows x 500 feat, 10 classes | Out on both axes |
| TabPFN-2.5 (arXiv:2511.08667v2, 5 Feb 2026) | 50k x 2,000 "recommended"; "models also fit larger datasets but are not built and evaluated" | Features in, rows ~1.4x over |
| TabPFN-2.6 | 100k x 2,000 | **In range** |
| TabPFN-3 / -3-Plus (arXiv:2605.13986, 13 May 2026) | 1M x 200; trade-off points 100k x 2,000 and 1k x 20,000; 160 classes | **In range at the 100k x 2k operating point** |

Beyond-limit behaviour is graceful degradation, not error. TabPFN-2.5's own results are reported on
TabArena datasets up to 100k rows, above its 50k recommendation; each estimator **subsamples to 500
features** internally on wide data. TabPFN-3 reaches 1M rows by row-chunking plus a reduced KV
cache (~5x peak-memory cut, single H100) but explicitly warns its early feature-compression into a
fixed row representation "can become a bottleneck" when rows *and* features are both large — which
is precisely your regime.

Alternative scaling routes, all verified: **TabDPT** (arXiv:2410.18164v3, NeurIPS 2025) is
retrieval-based — top-K kNN context per query row, context 128-2048 — so 70k rows are reachable by
construction rather than by long context. **TabICL** (arXiv:2502.05564, ICML 2025) pretrains to 60k,
handles 500k. **TabICLv2** (arXiv:2602.11139, Feb 2026) reaches million-scale under 50 GB GPU.
**TabFlex** (arXiv:2506.05584, ICML 2025) linear attention, >1M-row poker-hand in 5 s.
**arXiv:2502.17361** scales TabPFN v2 by test-time divide-and-conquer with no retraining.
**TabPFN-Wide** (arXiv:2510.06162) continued-pretrains on a wide prior to >30,000 features with
improved noise robustness — the only work targeting your feature count, but built for
few-observation omics.

### (b) Set-valued rows: NO EVIDENCE FOUND

No PFN or tabular foundation model in this literature accepts set-valued per-row structure. PFN
permutation invariance is over **context rows**, not over tokens within a row. You would have to
flatten 81x20. Closest work: **In-Context Multiple Instance Learning** (arXiv:2606.06458, Jun 2026)
— Perceiver-style ICL pretrained on synthetic bag-structured generators, single forward pass, best
average over 12 MIL benchmarks — but it is bag-level **classification**, no regression head, no n
reported. Also **arXiv:2605.07765** (TFMs as summary networks for neural posterior estimation).
Nothing plug-in exists.

### (c) Regression numbers, with sizes

Almost all of it is **rank aggregates and win rates, not RMSE/R2 deltas** — worth saying out loud.
- TabPFN-2.5: 100% win rate vs default XGBoost (classification, <=10k rows x 500 feat);
  **87% up to 100k rows x 2k feat; 85% for regression** on TabArena.
- TabDPT: rank 1 on CTR23 R2 and correlation critical-difference diagrams (**35 OpenML regression
  datasets**); the paper itself states regression "has much higher uncertainty". Power-law scaling
  in model and data size.
- TabICL v1: **classification only — zero regression evidence.**
- TabICLv2: untuned, surpasses hyperparameter-tuned + ensembled + real-data-finetuned
  RealTabPFN-2.5 on TabArena and TALENT (TALENT: 300 datasets, 100 regression, metric RMSE).
- Mitra (arXiv:2510.21204, NeurIPS 2025): beats TabPFNv2/TabICL on classification and regression;
  **strongest below 5,000 samples and 100 features** — flagged: that is not your regime.
- LimiX (arXiv:2509.03505): claims strongest on all benchmarks including regression.
- PriorLabs docs claim TabPFN-3-Plus regression "up to 20% metric improvements" with no benchmark
  or size — treat as marketing, not evidence.

Does the advantage survive n > 10k? By win-rate yes for 2.5 / 2.6 / 3 / TabICLv2 / TabDPT. **No
per-dataset regression effect size at n > 10k was retrievable.**

### (d) Quantile / distributional output and calibration — the most transferable part

TabPFN-3 uses a **bar-distribution (Riemann) regression head**: arbitrary quantiles are decoded by
inverting the predicted CDF in one forward pass, with no per-level retraining. On a benchmark built
from TabArena regression datasets over q in {0.1...0.9} it ranks 1st on normalised pinball loss —
but its gap to Quantile TabICLv2 is **not statistically significant** (Conover-Friedman, alpha=0.05).
**arXiv:2603.26611** (Mar 2026): 39 datasets, training sizes **50 to 20,000**; TFMs win CDE loss,
log-likelihood and CRPS, but calibration "lags behind task-specific neural baselines at larger
sample sizes" and post-hoc recalibration is recommended. Its SDSS DR18 photometric-redshift case is
your nearest analogue: **TabPFN on 50,000 galaxies beat all baselines trained on the full 500,000.**
**NO EVIDENCE FOUND** for calibration evaluated in ~400-example sub-bins, or for any sigma_eff-like
robust-interval metric.

### (e) Fine-tuning PFNs

**Real-TabPFN** (arXiv:2507.03971): continued pretraining on curated real tables lifts mean
normalised **ROC-AUC 0.954 -> 0.976** on 29 OpenML AutoML datasets — a *classification* metric.
**Real-TabPFN-2.5**: further gain, figure-only, no scalar. **arXiv:2603.08206** shows fine-tuning
TabPFN/TabICL under different proper scoring rules changes the inductive bias and the final metric.

### Blunt verdict

No off-the-shelf model here moves a 400-example sub-bin. Flattening 81x20 destroys the set
structure your model exploits, and your own log says only *adding* information helps — flattening
removes it. Every number above is a rank aggregate over public tables with no analogue of
54%-contamination, spatially correlated one-sided noise, and none is evaluated in sub-bins. Two
*ideas* do transfer, and they are not models: (1) **synthetic-prior pretraining on a domain prior**
(Mitra's prior-mixture recipe, TabPFN-Wide's continued-pretraining recipe) attacks your verified
bottleneck; (2) a **bar-distribution / CDF-inversion head plus post-hoc recalibration** as a
replacement for the 3-quantile head.

Checked and found unsupported or inapplicable: TabICL v1 and TabFlex (classification only),
In-Context MIL (bag classification), TabPFN-v1 (1k x 100), Mitra (<5k sweet spot), TabPFN-Wide
(few-row omics), the "20% regression gain" doc claim (unsourced). IDs seen in listings but not
fetched, so cited as **unverified**: arXiv:2505.16226, 2606.07134, 2509.20950, 2601.21731.

---

## Measurement bar used from here on

At n = 400 the sampling error on a 68.3%-interval half-width is roughly 3-5% relative. **Any
published gain below about 5% relative is invisible in your sub-bins**, no matter how good the
paper. Every verdict below is made against that bar.

---

## 2. Coordinate encodings for a hidden continuous latent — negative, and the logic is the reason

| Work | ID / venue | Mechanism | Number, and where | Transfer |
|---|---|---|---|---|
| Fourier Features | arXiv:2006.10739 | Random Fourier mapping makes the MLP's effective NTK stationary with tunable bandwidth, defeating spectral bias | PSNR tables are PDF-only; **numbers unverified**. Setting: dense low-dim coordinate fitting, 2-3 input dims, effectively unlimited samples | Transfers only to your mm-offset/pitch input channels, and only if the readout needs high-frequency dependence on offset — your smooth log-energy readout suggests it does not. Below the bar. |
| SIREN | arXiv:2006.09661 | Periodic activations for implicit neural representations | Images/video/audio/Eikonal-Poisson-Helmholtz; no generalisation numbers retrieved | Signal fitting, not generalisation from 70k samples. Does not move it. |
| Instant-NGP | arXiv:2201.05989 | Multiresolution hash table of trainable features so a small MLP can express high frequency | Speed/quality on NeRF/SDF/gigapixel — a **memorisation** regime | Actively wrong for you: hash encodings buy capacity to memorise coordinates, the opposite of what a 400-example sub-bin needs. |
| Learnable Fourier features | arXiv:2106.02795 | Position -> trainable Fourier mapping -> MLP modulation | Numbers not retrieved | The one cheap ablation left in this item: apply to the mm-offset channels. Expect sub-bar movement given 16 prior encoding nulls. |
| 2026 INR frequency work | trainable multi-resolution Fourier-pyramid PINN; "Content-Aware Frequency Encoding / Fourier-Chebyshev Features" (CAFE) | — | **unverified — could not retrieve IDs** | PDE/INR fitting, not held-out regression. |

**The logical point, stated plainly.** Every result above concerns a fine coordinate that is a
*supplied, densely sampled input*, with the network fitting a high-frequency function of it. Your
bottleneck is different: the sub-cell impact position is **latent**, and your own measurement (the
learned model beats full-covariance GLS by 1.8-3.6x) says the win comes from *conditioning on the
observed pattern to infer* the latent — not from resolving high frequencies in a given coordinate.
**No retrieved paper claims Fourier features help infer a latent position.** The non-vacuous version
of the question is whether Fourier features on the mm-offset/pitch inputs help; there is no specific
published evidence, and offsets enter the physics smoothly, so the transfer argument is weak.

**Closest genuine analogue, and it validates your framing rather than offering a lever:**
**arXiv:2512.20645** (Dec 2025, revised Jul 2026), "Machine learning methods for subpixel trajectory
reconstruction in discretized position detectors" — Geant4 cosmic muons, **8x8 segmented
scintillator array**, structurally your 9x9 grid. Transformer vs CNN vs linear regression vs
energy-weighted centroid: transformer **1.14 deg angular RMSE, 0.24 cm position MAE, i.e. 2.22x and
6.33x better than centroid**. Dataset size not stated on the abs page. **Cite this as external
confirmation that the latent sub-cell position is learnable and linear estimators are not enough** —
it de-risks your GLS-vs-learned claim for a write-up. Sub-pixel localisation microscopy and
DeepLoco-style work supervise position *directly*; you have no position label, so they do not
transfer.

**Target / label encodings** (the "regression as classification" family):
- **arXiv:2402.13425**, "Investigating the Histogram Loss in Regression", **JMLR 2026** — the key
  negative result: the benefits of learning distributions in this setup come from **"improvements in
  optimization rather than modelling extra information"**. Datasets/sizes not on abs page. If HL's
  gain is optimisation-only and your model already converges with EMA and a coverage-width loss,
  expect nothing.
- **arXiv:2403.03950** "Stop Regressing" (cross-entropy value functions): abstract lists domains
  (Atari + SoftMoE, multi-task Atari ResNet, Q-transformer robotics, chess, Wordle) but **no numbers
  on the abs page**; the widely quoted "30%" / "1.8-2.1x" figures are **unverified — could not
  retrieve**. Setting is bootstrapped RL value regression with non-stationary targets, very unlike
  yours.
- **arXiv:2512.01160**, CVIS 2025, MLIP energies as histograms + cross-entropy: **"absolute error
  performance comparable to regression baselines"** plus free uncertainty via predictive entropy —
  **parity, not a win**, in the nearest available domain (physics scalar energy). Conclusion:
  regression-as-classification buys calibration, not sigma_eff.
- **BEL, arXiv:2212.01927** (multi-bit binary label encodings, claims lower error than direct
  regression, no numbers on abs page); **RLEL, arXiv:2303.02273** (learns the label encoding
  end-to-end). Worth one cheap ablation; no retrieved number clears the 5% bar.
- **Fourier-encoded targets with quantified regression gains: NO EVIDENCE FOUND.**

---

## 4. Distributional / generative regression — and the reframe that matters

### Does modelling p(y|x) beat quantiles?

**CARD, arXiv:2206.07275, NeurIPS 2022** — diffusion conditional generator plus a pretrained mean
estimator. Its own scope condition is "especially when the conditional distribution of y given x is
**multi-modal**". The abs page carries no RMSE/NLL/QICE numbers; search prose claimed SOTA on 9/10
RMSE, 8/10 NLL, **5/10 QICE** across the 10 UCI benchmarks — **treat as unverified**. Setting: UCI
tabular, n ~500 (Boston) to ~500k (Year), low-dimensional, roughly homoscedastic. **No evidence that
full-density modelling shrinks a central 68.3% width, and QICE — the interval-quality metric — being
its weakest result (5/10) is a warning.**

**Diffusion / flow regression at n < 100k:** UCI is the only sub-100k evidence retrievable, and it is
flat tabular. For **set-structured scalar regression at n ~ 70k: NO EVIDENCE FOUND.** Flow-matching
searches returned only generative/trajectory work (Posterior-Augmented Flow Matching
arXiv:2605.00825, Preconditioned FM arXiv:2603.02337, Discriminative FM via Local Generative
Predictors arXiv:2603.13928 — IDs from listings, not fetched, none reporting scalar-regression sigma
metrics).

**Heavy one-sided / skewed targets specifically: NO EVIDENCE FOUND** with numbers. Nothing retrieved
quantifies distributional-head gains under asymmetric, spatially correlated contamination.

### The reframe — where the only real lever in this item sits

sigma_eff is a *central* interval width and trimming 10% of events cuts it 30%. That is a
robustness problem, not a tail-modelling problem. The literature that actually matches:

| Work | ID / venue | Mechanism | Number, and where | Transfer |
|---|---|---|---|---|
| Statistical Robustness of Interval CVaR Based Regression | arXiv:2601.11420 (Jan 2026) | **In-CVaR** = average of losses ranked between the alpha- and beta-quantiles, a *two-sided trimmed risk* dropping both extreme and trivially-small losses; lineage LMS/LTS (Rousseeuw 1984) | Experiments and sizes not on abs page — **no fetched numbers** | The closest published formalisation of "optimise the loss you are actually scored on". Your own 30% trimming measurement *is* the effect size, and 30% is far above the 5% bar. Honest risk: trimming at train time on a contaminated *input* distribution (not label noise) can teach the model to ignore pileup-heavy events rather than handle them — validate on untrimmed sigma_eff. |
| Large width penalization for NN prediction-interval estimation | arXiv:2411.19181 | Penalises the *average of large* PI widths more heavily on top of a coverage-width criterion (PICP/PINRW) | Synthetic + solar irradiance forecasting; "significantly reduce[d] large PI width while maintaining PICP" — **no numbers on abs page** | Direct upgrade path for your existing coverage-width loss; an asymmetric width penalty is close to a one-line change. Plausible but unquantified. |
| Square Root Loss / Smooth MAE | arXiv:2606.22068 (Jun 2026) | Infinitely differentiable convex/quasiconvex losses closer to \|e\| than Huber or log-cosh, framed as outlier-sensitive learning-rate modulation | "Superior performance" on many benchmarks, **no numbers on abs page** | Cheap, low ceiling. Your GateHuber history says Huber-family swaps give sub-bar movement. |
| ConFrag | arXiv:2502.17771, NeurIPS 2024 | Noisy-label *regression* sample selection via contrasting label fragments and neighbourhood agreement across expert extractors; introduces Error Residual Ratio | Six curated benchmarks (age, price, music year); "outperforms fourteen SOTA baselines" — **numbers PDF-only** | Noise model is symmetric/Gaussian *label* noise, the opposite of your one-sided structured *input* contamination. Reuse the selection machinery, do not adopt the assumption. |
| Beyond the Mean: Distribution-Aware Loss Functions for Bimodal Regression | arXiv:2603.22328 (Mar 2026) | Mixture head trained with normalized RMSE plus **Wasserstein / Cramer** distances instead of MDN NLL, avoiding mode collapse | **Wasserstein variant reduces Jensen-Shannon Divergence by 45%** on complex bimodal datasets while keeping MSE-level optimisation stability; datasets/sizes not on abs page | The mechanism is a genuinely untried structural option for your two physical modes (photon-dominant vs pileup-dominant). But JSD is distributional fidelity, not sigma_eff — no evidence it narrows a central interval. |

MDN mode-collapse background: arXiv:1906.03631 (winner-takes-all sampling + iterative grouping,
ID verified), arXiv:2510.25001 (BNN vs MDN comparison, Oct 2025, not fetched).

### Checked and unsupported for item 4

- **EBM regression heads** 2025-2026 with quantified tail/central-interval numbers: **NO EVIDENCE
  FOUND** — searches returned only generative/policy EBMs (2309.05803, 2503.07021, BoltzNCE).
- **Conformal + distributional hybrids** (Distribution-Aware CP arXiv:2605.26569, CONTRA
  arXiv:2605.08561, Flow-Based Conformal Predictive Distributions arXiv:2602.07633, SpeedCP
  arXiv:2509.24100 — IDs from listings, not fetched): all target coverage validity and interval
  efficiency, none changes the residual distribution. **Structurally cannot lower sigma_eff**, which
  is computed from residuals rather than from your intervals. Do not spend time here.
- Minimum-covariance-determinant deep learning: **NO EVIDENCE FOUND.**
- "Learning with trimmed risk" as a named deep-learning line with numbers: only In-CVaR and the LTS
  lineage; no fetched numbers.
- Unretrievable IDs, cited as **unverified**: beignet (Fourier-pyramid PINN), CAFE, "Deep Neural
  Expected Shortfall Regression with Tail-Robustness".

---

## 3. Attention alternatives 2025-2026 — documented null

| Work | ID / venue | Mechanism | Number, and where | Transfer |
|---|---|---|---|---|
| Mamba-3 | arXiv:2603.15569, ICLR 2026 oral | SSM-principled recurrence (trapezoidal discretisation, complex state, MIMO) | Perplexity comparable to Mamba-2 at *half* the state size; LLM pretraining, long context | None. The whole value proposition is constant-memory inference at long context; at 81 tokens there is no memory problem. Its own text concedes recent linear models "trade off model quality and capability for algorithmic efficiency". |
| RWKV-7 "Goose" | arXiv:2503.14456 | Delta-rule dynamic state evolution | Multilingual LM benchmarks, 0.19B-2.9B params, web-scale corpora | None. Web-scale LLM pretraining only. |
| Kimi Linear | arXiv:2510.26692 | Hybrid linear/full attention | Performance and speed **comparable to MLA at 4k-16k tokens**; 2.3x faster only at 512k, 2.9x at 1M | This is the citable proof the efficiency argument dies at short length. Parity at 4k means exactly zero at 81. Your model is FFN/readout-bound anyway; the clusters/sec benchmark already shows throughput is not the constraint. |
| Revisiting associative recall in modern recurrent models | arXiv:2508.19029 (Okpekpe & Orvieto) | Re-examines the MQAR gap | The AR gap is driven substantially by **learning rate and optimisation dynamics, not expressivity**; single-layer attention cannot solve AR at all. Exact accuracies **unverified — could not retrieve** | Cuts both ways: it also removes "SSMs can't recall" as a reason to reject them. Either way MQAR is a long-context phenomenon; at 81 tokens with d=128, recall capacity is not binding. |
| Pamba | arXiv:2406.17442 | Mamba over serialised point clouds | Explicit: "Mamba is designed to process the causal sequence... Different orders of input points can result in different outputs." ScanNet200 val (~148k points/cloud): single Hilbert 34.4 mIoU, single Z-order 34.3, Hz 35.1, Hz+swap 36.3 — **~2 mIoU is the price of buying permutation invariance back via multi-path serialisation** | Decisive against SSMs here. You would pay a measured ~2-point tax to remove a property (permutation invariance) that attention gives you free. |

Retrieved but with no usable numbers: Mambular arXiv:2408.06291 (claims "significant performance
improvement", **no numbers, no dataset sizes retrievable**), State-Space Models for Tabular PFNs
arXiv:2510.14573, TabFlex arXiv:2506.05584, Jet Reconstruction with Mamba arXiv:2506.18336
(closest HEP analogue, content unverified), ATLAS arXiv:2505.23735. Gated DeltaNet, Titans and
Hymba: **unverified — could not retrieve** quantified short-sequence results, so nothing is cited.

**Bottom line for item 3: no paper found, in either direction, showing an attention alternative
beating softmax attention at set length under 100 tokens or at label budgets under 100k outside
LLM pretraining. The efficiency motivation is actively contradicted. Do not spend a week here.**

---

## 7. KAN, geometric-algebra and equivariant transformers — evidence still absent at this scale

| Work | ID / venue | Mechanism | Number, and where | Transfer |
|---|---|---|---|---|
| KAN or MLP: A Fairer Comparison | arXiv:2407.16674 | Param- and FLOP-matched KAN vs MLP | Tabular: **MLP wins 6 of 8**. Vision (8 datasets): KAN "consistently fell short" at matched params *and* FLOPs. Audio: MLP wins both. Symbolic regression: KAN wins 7/8 on params, but under FLOP matching drops to roughly even (2 wins, 1 loss), and the advantage traces to the B-spline activation — transplanted into an MLP it matches or beats KAN. Continual learning (class-incremental MNIST): KAN accuracy on tasks 1-2 fell to **0** vs MLP retaining usable accuracy | Direct negative. Your target is non-symbolic scalar regression, the exact regime where KAN loses. |
| KANs for Computer Vision: An Experimental Study | arXiv:2411.18224v2 | Empirical study at the right data scale (MNIST 70k, CIFAR-10 60k, Fashion-MNIST 70k) | Conv-KAN **98.90% at 94.2k params** vs standard CNN **99.10% at 157k**; EfficientKAN [784,64,10] 0.973 on MNIST with *more* params than the MLP. Conclusion: standalone KANs "fundamentally unsuitable for practical computer vision"; hybrids gave "marginal advantages" | 60-70k examples is your scale exactly. The answer there is no. |
| GATr | arXiv:2305.18415, NeurIPS 2023 (algebra follow-up arXiv:2311.04744) | E(3)-equivariant transformer over 16-d projective-geometric-algebra multivectors | Claims wins in error, data efficiency, scalability on n-body / arterial wall-shear-stress / robotic planning. **No numbers verified this session** | Unproven for you, and the symmetry argument does not apply. |
| L-GATr | arXiv:2405.14806 (NeurIPS 2024); arXiv:2411.00446 | Lorentz-equivariant geometric-algebra transformer | Amplitude regression (scalar target from 4-momenta): Z+4g MSE **≈0.001 vs ≈0.003** for DSI; Z+5g still wins at **4x10^4 training points** — inside your regime. But data scaling is explicit: JetClass top tagging **2M jets 0.839 vs ParT 0.836; 10M 0.859 vs 0.850; 100M 0.866 vs 0.861** — small everywhere, and the paper frames the gain as helping "particularly for small training sets" | The strongest-looking evidence in this slice, and it argues against you. L-GATr exploits an exact continuous 6-parameter Lorentz symmetry over 4-momenta. Your 9x9 grid has only a finite discrete D4, which your train-time augmentation already covers and which returned nothing at 70k. There is no continuous symmetry left to exploit. |
| LorentzNet | arXiv:2201.08187 | Minkowski dot-product-attention equivariant GNN | Competitive at **0.5% of training samples** (a few thousand jets); per-fraction table **unverified — PDF not extractable** | Sample-efficiency story on the axis where you are not architecture-constrained. |
| PELICAN | arXiv:2211.00454 / 2307.16506, JHEP 03 (2024) 113 | Permutation-equivariant, Lorentz-invariant | Claims lower complexity and high sample efficiency. **No numbers verified** | Same. |

**Bottom line for item 7: still absent. No paper since 2024 shows a KAN or geometric-algebra
regression win over a well-tuned transformer at 10^4-10^5 examples on a non-symbolic target; the
KAN evidence at exactly 60-70k examples is negative; and every HEP equivariance win is
continuous-Lorentz-shaped with margins that narrow as data grows.**

---

## 5. Test-time compute and adaptation — the strongest item in the survey

Local code check first: `scripts/train_picocal.py` has `--d4aug` and `d4_apply()` used **only inside
the training loop** (lines 154-155). There is no inference-time D4 averaging anywhere in the eval
path; `scripts/benchmark_inference.py` only *budgets* for it (`cps / 40`). So test-time D4 averaging
is genuinely untried.

| Work | ID / venue | Mechanism | Number, and where | Transfer |
|---|---|---|---|---|
| Group Averaging for Physics Applications: Accuracy Improvements at Zero Training Cost | arXiv:2511.09573, NeurIPS 2025 ML4PS workshop | Average a trained model's predictions over a symmetry group at inference, making it exactly equivariant with no retraining | **"up to 37% improvement in VRMSE"**, "always decreases the average evaluation loss", cost proportional to group size, and "under mild conditions, the group-averaged model will have provably better prediction accuracy than the original". Dataset names/sizes **not retrievable** | Strongest single lead for the D4 idea, and mechanistically distinct from train-time D4 (Jensen/variance reduction on the output, not a data prior). Caveat: the guarantee assumes the group is an *exact* symmetry. Your cell-pitch regions, incident direction and physics readout break D4, so blind 8-fold averaging trades variance for bias — the honest version is a weighted or selected subset of the 8 transforms. |
| TTA for RNN surrogates of composites | arXiv:2409.02478, Eng. Appl. Artif. Intell. | Random 3D rotations of input tensors, predict, rotate predictions back, average | **~19% mean-relative-error reduction at 200 rotations** vs a single prediction; uniaxial cyclic loading MaRE **0.121 -> 0.0791 (45%)**; plateau at N~200 (checked to 100,000). Training set **547** FE/FFT samples after 40k mean-field pretraining | Proof the mechanism gives double-digit relative gains on physics regression at zero training cost — but 547 samples and an *exact* rotation symmetry. Scale the 19% down hard for 70k samples and an approximate D4. |
| Q-PART | arXiv:2503.04131, CVPR 2025 | Test-time training that minimises prediction variance across K augmentations of each test case | **Theorem 1: E[L_reg] <= 2 E[L_var]/K** — minimising augmentation-variance provably bounds regression error. Ablation (pre-school cohort) MAE **7.842 -> 7.283** with variance minimisation -> 7.235 full model. EchoNet-Dynamic+Pediatric, 10,749 videos, cohorts of 831 / 914 / 1,539 | ~7% relative MAE from the variance-minimisation component alone at n~900 per cohort. The closest published analogue to "optimise the model per test event to be consistent under transforms", and the theorem is the thing to cite. |
| Test-time local training of neural networks for tabular data | *Scientific Reports* 2025, doi 10.1038/s41598-025-31491-3 (PMC12804743) | Retrieve k nearest training rows per test point, fine-tune *all* parameters on them, predict | Regression: **RMSE reduction >10% on 3 of 9 datasets**; best average rank 1.56+/-0.68; Bikesharing (17,379 rows) **0.2344 -> 0.2242 (4.4%)**; k=100, T=100 iterations. Neutral or worse on Cpusmall and CTscan | Directly applicable: 70k train rows, 11k test points, and a metric evaluated in sub-bins where a local model is exactly the right object. 100 gradient steps per test event is affordable. Not reliable — 3 of 9 got >10% — so expect it to pay in sparse detector regions and do nothing in dense ones. |
| TTA improves efficiency in conformal prediction | arXiv:2505.22764 | Learned augmentation weights on logits before conformal scoring; 12-augmentation policy | RAPS+TTA-Learned set-size reduction: ImageNet (50k) **16.9/13.3/9.2%** at alpha=0.01/0.05/0.10; iNaturalist (100k) 13.4/16.4/9.8%; CUB-Birds (5,794) **8.2/1.3/7.3%**, explicitly weaker because its calibration set is 2,827 vs 25k-50k. **Zero regression experiments** | The CUB result is the transfer warning: TTA-as-interval-shrinker degrades when the calibration/evaluation set is small — exactly your 400-example sub-bin. |
| I Can't Believe TTA Is Not Better: When TTA Hurts Medical Image Classification | arXiv:2604.09697 (Apr 2026) | Systematic TTA evaluation, 3 MedMNIST v2 benchmarks x 4 architectures (21k-11M params) | **TTA "consistently degrades accuracy"**, drops up to **31.6 percentage points** (ResNet-18, pathology); exactly **one** improvement (+1.6%, ResNet-18 dermatology); worse as more views are added. Cause: "distribution shift between augmented and training-time inputs — amplified by **batch normalization statistics mismatch**"; intensity-only augmentations preserve more performance than geometric transforms; including the original image partially mitigates | The mandatory counterweight. Two mitigating facts for you: your net uses LayerNorm not BatchNorm, so the identified mechanism does not apply; and you trained *with* D4, so the transformed inputs are in-distribution by construction. Their own conclusion is the right protocol: "TTA should not be applied as a default post-hoc improvement but must be validated on the specific model-dataset combination." |
| TTT-on-nearest-neighbours | arXiv:2305.18466; 2025 reproduction arXiv:2511.16691 | Fine-tune on retrieved neighbours at test time | ~20% relative perplexity reduction with ~20 neighbour updates — **numbers unverified, not fetched**. Language modelling | Lineage citation only. |
| In-Place Test-Time Training | arXiv:2604.06169 | Fast-weight MLP projections updated during inference | Qwen3-4B RULER-16k **6.58 -> 19.99**; language, next-token objective | **NO EVIDENCE** for regression transfer. |
| Ensembling economics | arXiv:2005.07292; arXiv:2506.04677 | Fixed-budget ensembling; cost of ensembling in forecasting | 16 thin WideResNets **82.52%** vs one WRN-28-10 **80.6%** on CIFAR-100 (image classification, label as such). Forecasting regression on M5 (28,298 series) and VN1 (15,053): **2-3 members capture most of the gain**, diminishing returns after | Your 5-seed ensemble is already past the knee. Seeds 6-10 will not move sigma_eff. |

**Deep unfolding / LISTA-style learned iterative solvers for scalar regression at small n: NO
EVIDENCE FOUND.** Retrieved hits (arXiv:2606.02661 precipitation, arXiv:2605.27245 symbolic
regression, arXiv:2505.19148 DISTA-Net) are structured-output inverse problems, not scalar targets.
**TENT (arXiv:2006.10726) and "Learning to (Learn at Test Time)" (arXiv:2310.13807)** rely on
entropy or self-supervised proxies; your quantile head could supply a label-free signal, but no
regression numbers exist.

---

## 6. Small-data training recipes 2025-2026 — mostly negative or unmeasurable

| Work | ID / venue | Mechanism | Number, and where | Transfer |
|---|---|---|---|---|
| Singh, Mobahi, Agarwala, Dauphin — SAM analysis | arXiv:2502.02407, ICML 2025 (PMLR v267:55702-55719) | Decomposes what SAM's perturbation actually regularises | C4 language modelling, Nanodo transformers: **SAM eval loss worse than AdamW at every scale** — 23.9M 3.59 vs 3.57; 42.5M 3.46 vs 3.45; 117.9M 3.29 vs 3.28; 1.2B NaN — "even worse than AdamW while being 2x computationally expensive". Diagnosis: outside vision, SAM's perturbation is absorbed by logit-statistic regularisation rather than function curvature. Their Functional-SAM fix recovers 0.02-0.06 loss. Experiments cover only ImageNet-1k/JFT/IN-21k ViTs and C4 LMs | **NO EVIDENCE FOUND for SAM on regression, and none at 70k scale.** SAM costs 2x training compute and its known failure mode is "non-vision task where the loss is not cross-entropy over logits" — your loss is quantile + coverage. Do not spend the week here. |
| When, Where and Why to Average Weights? | arXiv:2502.06761v3 | Systematic study of weight averaging (LAWA, EMA, SWA) | AlgoPerf: NadamW 612 GPU-h -> LAWA 550 -> EMA 541; both reach target in 82% of baseline steps; 124M transformer on 5B tokens, 3,200 -> 2,240 (LAWA) / 2,200 (EMA) steps. **Generalisation gains are small**: WMT 31.157 -> 31.429 BLEU, OGBG 0.3012 -> 0.3129 mAP, others negligible. Critically: "when the learning rate is fully annealed to zero, WA converges closely to the annealed model" | You already run EMA with annealing. LAWA/SWA/soup on top buys speed, not accuracy. |
| Model soups | arXiv:2203.05482 | Average weights of independently fine-tuned models | Souped **different-hyperparameter** runs, not same-recipe seeds | Souping 5 same-recipe seeds is not the paper's setting. Honest expectation: soup ~= your 5-seed ensemble at 1/5 the inference cost — a compute win, not a sigma_eff win. **Soup-on-top-of-EMA-plus-ensemble: NO EVIDENCE FOUND.** |
| C-Mixup | arXiv:2210.05775, NeurIPS 2022 | Mix pairs weighted by label closeness | RMSE ERM -> C-Mixup: Airfoil **2.901 -> 2.717 (6.3%)**, NO2 0.537 -> 0.509 (5.2%), Exchange-Rate 0.0236 -> 0.0203 (14.0%), Electricity 0.0581 -> 0.0570 (1.9%), Echocardiogram 5.402 -> 5.177 (4.2%); headline **6.56% in-distribution**. Airfoil/NO2 are **~1-2k-row tabular sets** | The big deltas live where any variance reduction looks heroic. |
| CEMS | arXiv:2506.06853, ICML | Manifold-curvature-based example synthesis | Airfoil RMSE 2.901 (ERM) -> 2.717 (C-Mixup) -> 2.360 (ADA) -> **1.455** (CEMS); NO2 0.537 -> 0.507; Electricity flat at 0.058. No small-sample ablation | Airfoil 2.901 -> 1.455 will not replicate at 70k. |
| RC-Mixup | arXiv:2405.17938, KDD 2024 | Robust C-Mixup under noisy data | Spectrum RMSE 10.044/12.125/13.074 (C-Mixup) -> **7.442/7.471/7.941** (RC-Mixup) under low/med/high Gaussian noise. From a search snippet, not a fetched table — **provisional** | The one whose *premise* matches yours, and additive contamination makes mixing two calorimeter events less physically absurd than mixing two images. But its wins are largely tail/MSE wins and sigma_eff already discards the worst 31.7%. |
| Anchor Data Augmentation | arXiv:2311.06965, NeurIPS 2023 | Causal anchor-regression replicas | "Competitive with C-Mixup"; tables **unverified — PDF not extractable** | Same regime. |
| Investigating the Histogram Loss in Regression | arXiv:2402.13425v3, JMLR 2026 | HL-Gauss: KL to a Gaussian-smoothed target histogram | Insensitive to bin count for k>30; HL-OneBin much worse (discretisation bias); linear architectures *underperform* with HL-Gauss on time series; **transformer results on ETD mixed/tied**. Train-set sizes **unverified**. Headline finding: benefits come from "improvements in optimization rather than modelling extra information" | Negative for swapping your quantile head. |
| HL-Gauss on molecular energies | arXiv:2512.01160, CVIS 2025 | Soft-binned cross-entropy for MLIP energies | OMol25, UMA-S 150M: MAE-regression **0.0091** meV/atom vs HL-Gauss 128-bin **0.0122**, 256-bin **0.0153** — classification **lost**, "consistently underperformed" across all four chemistry categories | Second independent negative. Do not replace the quantile head. |
| Semi-Supervised Regression with Heteroscedastic Pseudo-Labels | arXiv:2510.15266 | Bi-level learning of per-sample pseudo-label uncertainty | UTKFace **526 labelled / 9,992 unlabelled**: MAE 6.135 -> 5.639 (8.1%); IMDB-WIKI 9,575 / 181,925: 10.172 -> 9.177 (9.8%); STS-B 260 / 4,940: MSE 1.746 -> 1.540 | Every gain comes from a **10-20x unlabelled:labelled ratio**. You have 70k labelled and ~11k unlabelled test inputs — a 0.16x ratio. Either reframe as transductive use of the test set (expect a fraction of these numbers) or generate unlabelled events from simulation without truth-matching. |
| Ensemble Knowledge Distillation for MLIPs | arXiv:2503.14293 | N teachers trained on energies generate *forces*; one student trained on energies + ensemble-averaged forces | Claims SOTA on COMP6 from ANI-1ccx; MAE tables **unverified — not retrieved** | The interesting transfer is not compression but **auxiliary-target generation**: your 5-seed ensemble could emit a per-cell contamination estimate as a dense auxiliary target for a single student. |
| Less Data, Faster Training | arXiv:2605.20314 | Smaller datasets induce sampling biases acting as layer-wise LR adjustment | Sparse-parity (20,6) MLP converges in ~1,500 steps at N=2^14 vs >2,000 at N=2^20, "100x speedup in compute"; random-label controls confirm the mechanism | About *speed*, on synthetic parity. Says nothing about final error and does not contradict your 0.041 -> 0.056 at 3x less data. |

Born-Again Networks is arXiv:1805.04770 (ID confirmed); **no small-n regression numbers retrieved.**
**Stochastic depth / dropout schedules / weight-decay tuning for small transformers on regression:
NO EVIDENCE FOUND** — retrieved 2026 items (Explicit Dropout arXiv:2604.20505; arXiv:2603.17811) are
image/audio/action classification.

**Bottom line for item 6: after applying the 5% measurement bar, nothing in this item survives.
The mixup family's real deltas are at 1-2k rows; SAM is actively negative outside vision;
weight averaging on top of EMA buys wall-clock, not accuracy; regression-as-classification has two
independent negatives; semi-supervised regression needs an unlabelled pool you do not have.**

---

## 8. Other 2025-2026 work I judge relevant, with numbers

### 8a. Self-supervised pretraining on unlabelled events — the only literature that converts unlabelled data into labelled-data-equivalent

This is the mechanism that matches your verified bottleneck. Two retrieved numbers:

- **Is Tokenization Needed for Masked Particle Modelling? arXiv:2409.12589v2** (Oct 2024). Masks
  particles in a *set* and predicts them; several backbones (regression, k-means, CNF, flow).
  JetClass accuracy vs labelled-set size, pretrained vs random init:
  **1k jets ~68% (flow) vs ~48% random; 10k ~75% vs ~60%; 100k ~82% vs ~76%; 100M 85.0 vs 84.3.**
  Verbatim: "The Flow-backbone achieves the same performance with 10k jets as the randomly
  initialized network with 1M." Encoder 512-dim, 8 layers; ~50 constituents per jet (BTag capped at
  15). A direct L1-regression pretraining objective went **48.9% -> 79.2%** with a transformer
  decoder. **The gain shrinks monotonically with labels: +15 points at 10k, +6 at 100k, +0.7 at
  100M.** At your 70k you sit right in the +6-points-of-accuracy band — a real but not enormous gain,
  and it is classification, not sigma_eff.
- **Towards foundation-style models for energy-frontier heterogeneous neutrino detectors via
  self-supervised pre-training, arXiv:2604.07037** (Apr 2026). Sparse ViT, masked-autoencoder
  reconstruction (75% of *occupied calorimeter patches* masked, voxel-level occupancy and charge)
  plus a relational objective, two-phase schedule. Downstream tasks include **momentum regression**
  and vertex reconstruction. Verbatim: "with roughly 10^3 labelled events, the pre-trained encoder
  already matches the flavour-classification performance of a randomly initialised model trained on
  an order of magnitude more data" — i.e. **~10x label efficiency**. Regression resolution values are
  not on the abs page. This is the closest architectural and detector analogue to your grid.
- Background: **Masked Particle Modeling on Sets, arXiv:2401.13537** (Mach. Learn. Sci. Technol.
  2024) — VQ-VAE-token masked prediction on sets, "transfer efficiently with small fine-tuning data
  sets", **no numbers on abs page**. **Machine-learned particle flow as a foundation model,
  arXiv:2606.14373** (Jun 2026) — MLPF latents reused for jet flavour ID, **jet energy regression**
  and missing-momentum regression; only retrievable number is a linear probe beating the baseline for
  missing-momentum regression with **~35x fewer parameters**; no data-efficiency curve on the abs
  page.

**Blunt: this is the highest-ceiling item in the whole survey, because it is the only one that
addresses data starvation without new labels. It requires an unlabelled corpus much larger than 70k
events, which for a simulation-driven study means simply generating more events — at which point you
should compare against just training supervised on those events instead.**

### 8b. The scaling-law comparison every proposal must beat

**Neural Scaling Laws for Deep Regression, arXiv:2509.10000** (Mach. Learn. Sci. Technol. 7 025011,
2026). Fits power laws of loss against training-set size and model capacity for a physics parameter
estimation task, across fully connected nets, ResNets and ViTs. Verbatim: exponents **"range from 1
to 2, with specific values depending on the regressed parameters and model details"**, and "the
performance of deep regression models can improve substantially with increasing data size".
**Your own measured point implies sigma_eff ~ N^-0.28, so 3x more data -> ~0.030 and 10x -> ~0.0215.
Nothing else in this survey has a documented effect of that size. If you can generate more simulated
events, that is the experiment.**

### 8c. HEP transformer reconstruction, for framing rather than a lever

- **ClusTEX, arXiv:2603.18172** (Mar 2026, rev Jul 2026; Maidannyk, Couderc, Malcles, Sahin).
  Graph transformer doing candidate selection and reconstruction in one inference stage, with a
  **novel positional encoding that separates local coordinates within the graph from global detector
  coordinates**. Claims improved energy resolution, reduced splitting, and retained di-photon mass
  reconstruction for boosted pi0 where standard algorithms fail — **no numerical resolution values on
  the abs/HTML pages retrieved**. The local-vs-global positional-encoding split is the one encoding
  idea here that your 16 experiments may not have covered exactly; note you already have region
  one-hot plus mm offsets, which is close.
- **HistoAE, arXiv:2511.22246** (Nov 2025, rev Jun 2026). Unsupervised autoencoder with a
  histogram-based loss forcing a 2-d latent space that corresponds to particle charge and impact
  position. Silicon microstrips: **charge resolution 0.25 e, position resolution 3 um, "comparable to
  the conventional approach"** on beam-test data. **A documented null on exactly your idea of forcing
  a latent to be the impact position: comparable, not better.**
- **Understanding Energy Dependent Hadronic Calorimeter Response from a Machine Learning
  Perspective, arXiv:2606.10960** (Jun 2026) — retrieved as a title/abstract listing only,
  **numbers unverified**.

### 8d. Learning using privileged information — mechanism sound, numbers absent for regression

Simulation knows the per-cell truth (true contamination, true impact position). A teacher trained
with privileged inputs, distilled into a student that sees only the real observables, is an
architecture/training move rather than a new inference-time feature. Retrieved 2025-2026 work is
**entirely LLM-shaped**: pi-Distill / privileged-information distillation (arXiv:2602.04942;
OpenReview uWlyzJOm3B, FbJu6NEBQR) and Rethinking On-Policy Distillation (arXiv:2604.13016). Their
own caveats are relevant — success "depends heavily on the properties of the privileged information,
such as the utility or KL between student and teacher", and privileged-context self-distillation can
*degrade* the student. **NO 2025-2026 REGRESSION NUMBERS FOUND for LUPI.** Mechanism is attractive
and directly targets your latent variable; evidence is absent, so treat as a research bet, not a
recipe.

### 8e. Retrieval-augmented tabular deep learning

**TabR, arXiv:2307.14338, ICLR 2024** — a feed-forward net with a kNN-like attention component in
the middle that retrieves training objects (features *and labels*) to improve a prediction. Reported
as best average among tabular DL models and beating GBDT on a "GBDT-friendly" benchmark; **no RMSE
deltas or dataset sizes retrieved this session, and no 2025-2026 successor surfaced.** Relevant
because retrieval gives the model access to similar training events at inference, which is a direct
attack on data starvation in sparse sub-bins, and because TabDPT (item 1) shows the same mechanism
works inside a PFN. **Untried here; unquantified for set inputs.**

---

# Ranked shortlist — at most 5 things worth implementing next

Ordered by (documented effect size above the 5% sub-bin noise floor) x (probability the mechanism
transfers) / (implementation cost).

### 1. Test-time D4 group averaging, with learned or selected per-transform weights
**Cite:** arXiv:2511.09573 (group averaging, "up to 37% VRMSE", provable improvement under exact
symmetry), arXiv:2409.02478 (19% MaRE at 200 rotations, physics regression, exact symmetry),
arXiv:2604.09697 (the counterweight: TTA "consistently degrades" in MedMNIST, up to -31.6 pp, via
BatchNorm statistics mismatch — inapplicable to your LayerNorm net, and your model was trained *with*
D4 so the transforms are in-distribution).
**Expected effect:** the honest range is 0 to -0.003 sigma_eff (0.041 -> ~0.038). Reasoning: the two
positive papers both exploit *exact* symmetries at n<1000, where variance dominates; your D4 is
broken by cell-pitch regions, incident direction and the physics readout, and your 5-seed ensemble
already removes some of the same variance. Do not expect 19-37%.
**Cost:** hours. `d4_apply()` already exists; you need it in the eval path and an 8-way average, plus
a per-transform weight fit on validation. Zero training cost.
**Why first:** it is the cheapest untried thing in the repo, and it is the one mechanism whose
absence from the code I verified directly (`d4_apply` is called only at lines 154-155, inside the
training loop; `benchmark_inference.py` already budgets `cps/40` for it).

### 2. An In-CVaR / trimmed-risk training objective aligned to sigma_eff
**Cite:** arXiv:2601.11420 (In-CVaR = mean of losses ranked between the alpha- and beta-quantiles, a
two-sided trimmed risk; LMS/LTS lineage), plus arXiv:2411.19181 (asymmetric penalty on *large*
prediction-interval widths on top of a coverage-width criterion) as the cheaper variant that plugs
straight into your existing loss.
**Expected effect:** the largest plausible in the survey, -0.005 to -0.010 (0.041 -> 0.031-0.036),
because your own measurement — trimming the worst 10% cuts sigma_eff by 30% — *is* the effect size,
and 30% is six times the sub-bin noise floor. Discount it because trimming at train time on a
contaminated *input* distribution can teach the model to ignore pileup-heavy events rather than
handle them.
**Cost:** 1-2 days. A ranked-loss mask inside the existing loss, plus a beta/alpha sweep. Must be
validated on **untrimmed** sigma_eff, per region and tercile, or the result is circular.
**Caveat:** arXiv:2601.11420 is a statistical-robustness paper; **no fetched experimental numbers**.
The justification is your own measurement, not theirs.

### 3. Test-time local fine-tuning on the k nearest training events
**Cite:** *Sci. Rep.* 2025, doi 10.1038/s41598-025-31491-3 (k=100 neighbours, T=100 steps, all
parameters; RMSE reduction **>10% on 3 of 9** regression datasets, Bikesharing 17,379 rows
0.2344 -> 0.2242); arXiv:2503.04131 Q-PART for the principled version (Theorem 1:
E[L_reg] <= 2 E[L_var]/K; MAE 7.842 -> 7.283 from variance minimisation alone at n~900);
arXiv:2307.14338 TabR and arXiv:2410.18164 TabDPT for the "retrieval attacks data starvation" logic.
**Expected effect:** -0.002 to -0.006 **concentrated in the sparse regions and the sub-bins you care
about**, and plausibly zero in the dense ones. This is the only item that targets the 400-example
sub-bin directly rather than the aggregate.
**Cost:** 2-4 days. Needs a neighbour index over an embedding (use your encoder's pooled
representation), a per-test-point fine-tune loop, and careful leak-free evaluation. Inference cost
goes up by ~100 gradient steps per event — check it against the clusters/sec budget.
**Blunt:** 3 of 9 datasets got >10% and two got nothing, so this is a coin-flip with a good payoff.

### 4. Self-supervised pretraining on a large unlabelled event pool, then fine-tune on the 70k
**Cite:** arXiv:2409.12589v2 (masked particle modelling on sets: at **10k labels 75% vs 60%** random
init, at **100k labels 82% vs 76%**, at 100M 85.0 vs 84.3 — "the Flow-backbone achieves the same
performance with 10k jets as the randomly initialized network with 1M"); arXiv:2604.07037 (sparse
ViT, 75% of occupied calorimeter patches masked, momentum regression downstream: "**with roughly 10^3
labelled events, the pre-trained encoder already matches** the flavour-classification performance of
a randomly initialised model trained on an order of magnitude more data").
**Expected effect:** the highest ceiling in the survey, because it is the only mechanism that
converts unlabelled events into labelled-data-equivalent, and data is your verified lever. But the
retrieved numbers are classification accuracy and the gain **shrinks monotonically with labels**
(+15 pts at 10k, +6 at 100k, +0.7 at 100M) — at 70k you are in the modest band. Guess: -0.002 to
-0.005 if the unlabelled pool is >=10x the labelled set.
**Cost:** 1-2 weeks, plus generating or obtaining the unlabelled corpus.
**The honest objection:** masked pretraining is what you do when *labels* are the scarce thing and
events are not. If you can generate more events *with* truth labels, train supervised on them
instead — that is the only lever with a directly measured effect here (arXiv:2509.10000 for the
scaling-law framing; your own local exponent for the size, with the caveat in the header that
extrapolation beyond ~2x is not licensed by two points and a known containment floor). Check which of
labels or events is actually scarce for PicoCal before committing a fortnight.

### 5. A two-component mixture head trained with a Wasserstein/Cramer distance
**Cite:** arXiv:2603.22328 (Mar 2026; normalized RMSE + Wasserstein/Cramer replacing MDN NLL;
**Wasserstein variant cuts Jensen-Shannon Divergence 45%** on complex bimodal data while keeping
MSE-level optimisation stability); arXiv:1906.03631 for MDN mode-collapse mitigation.
**Expected effect:** uncertain, plausibly -0.001 to -0.004. The mechanism matches a real physical
structure you have measured (photon-dominant vs pileup-dominant events, residual 5/95 quantiles
-0.26/+0.76), and it is the one *structural* option in the survey that your 16 experiments did not
cover. But the reported gain is distributional fidelity (JSD), **not interval width** — there is no
evidence it narrows a central 68.3% interval.
**Cost:** 2-3 days. Replace the 3-quantile head with 2 components x (weight, location, scale), keep
the physics readout, take the higher-weight mode or a weight-gated combination as the point estimate.

**Deliberately not in the top 5, but cheap enough to run as a one-day control:** learnable Fourier
features on the mm-offset/pitch input channels (arXiv:2106.02795) — a clean way to close out item 2
with a documented null rather than an untested assumption.

---

# Popular ideas checked and found unsupported for this setting

Consolidated. Every entry below was searched for this survey; "NO EVIDENCE FOUND" means the search
was run and returned nothing applicable, not that it was skipped.

**Architecture (the 17th swap):**
1. Mamba-2/3, gated linear attention, DeltaNet/Gated DeltaNet, RWKV-7, Titans, Hymba beating softmax
   attention at set length <100 — **no paper found in either direction.**
2. Any attention alternative evaluated at <100k labels on scalar regression — **no evidence.**
3. The efficiency motivation for linear attention — **actively contradicted**: Kimi Linear
   (arXiv:2510.26692) is at *parity* with full attention at 4k-16k tokens; the 2.3-2.9x speedups
   appear only at 512k-1M. At 81 tokens the gain is exactly zero.
4. MQAR/associative-recall as a reason to *reject* SSMs — also unsupported; arXiv:2508.19029
   attributes the gap to optimisation, and it is a long-context effect.
5. SSMs on permutation-invariant sets — **documented negative**: Pamba (arXiv:2406.17442) spends
   ~2 mIoU on multi-path serialisation to buy back invariance you already have free.
6. KANs on non-symbolic regression — **documented negative at exactly your data scale**:
   arXiv:2407.16674 (MLP wins 6/8 tabular; KAN's symbolic edge traces to the B-spline activation),
   arXiv:2411.18224 (60-70k-example vision, "fundamentally unsuitable").
7. Equivariant / geometric-algebra transformers where the symmetry is discrete and already
   augmented — **no evidence.** Every HEP win (L-GATr arXiv:2405.14806/2411.00446, LorentzNet
   arXiv:2201.08187, PELICAN arXiv:2211.00454) exploits an exact continuous Lorentz group and is a
   *data-efficiency* argument whose margin narrows as data grows (JetClass 2M 0.839 vs 0.836;
   100M 0.866 vs 0.861).

**In-context / foundation models:**
8. Any PFN or tabular foundation model that accepts **set-valued** rows — **NO EVIDENCE FOUND.** All
   of them would force you to flatten 81x20 = 1,620 features, destroying the structure your model
   exploits. (Row/feature limits are *not* the blocker: TabPFN-2.6 does 100k x 2,000 and TabPFN-3
   arXiv:2605.13986 has a 100k x 2,000 operating point.)
9. TabICL v1 and TabFlex — **classification only**, zero regression evidence.
10. Mitra (arXiv:2510.21204) — real, but its sweet spot is **<5,000 samples and 100 features.**
11. PFN calibration in ~400-example sub-bins, or against any sigma_eff-like robust-interval metric —
    **NO EVIDENCE FOUND.**
12. The "TabPFN-3-Plus regression up to 20% improvement" claim — documentation only, no benchmark or
    size. Marketing, not evidence.

**Encodings and heads:**
13. Fourier / random-Fourier / SIREN / hash features improving regression where the fine coordinate
    is **latent** rather than a supplied input — **NO EVIDENCE FOUND.** Instant-NGP-style hash
    encodings are actively wrong here: they buy memorisation capacity.
14. Fourier-encoded **targets** for regression with numbers — **NO EVIDENCE FOUND.**
15. Regression-as-classification / HL-Gauss / soft-binned targets replacing your quantile head — two
    independent negatives: arXiv:2402.13425 (JMLR 2026: the benefit is "improvements in optimization
    rather than modelling extra information"; transformer results mixed/tied) and arXiv:2512.01160
    (physics energies, MAE 0.0091 -> 0.0122/0.0153, "consistently underperformed").
16. Diffusion / flow-matching heads for scalar regression at n<100k outside UCI — **NO EVIDENCE
    FOUND.** CARD (arXiv:2206.07275) scopes itself to multi-modal targets and its **weakest** metric
    is QICE, the interval-quality one (5/10).
17. Energy-based regression heads with quantified tail or central-interval numbers — **NO EVIDENCE
    FOUND**; searches return only generative/policy EBMs.
18. Conformal and conformal-distributional hybrids (arXiv:2605.26569, 2605.08561, 2602.07633,
    2509.24100) — these change *interval validity*, not the residual distribution, and sigma_eff is
    computed from residuals. **Structurally cannot lower your metric.** Do not spend time here.
19. Distributional-head gains under heavy **one-sided** noise specifically — **NO EVIDENCE FOUND.**
20. Minimum-covariance-determinant deep learning — **NO EVIDENCE FOUND.**

**Training recipes:**
21. SAM and all variants (ASAM, GSAM, mSAM, friendly-SAM, efficient-SAM) — **NO EVIDENCE FOUND** on
    regression or at <=100k examples, and the one rigorous 2025 study (arXiv:2502.02407, ICML 2025)
    shows SAM **losing to AdamW at every scale on non-vision tasks at 2x compute** (3.59 vs 3.57;
    3.46 vs 3.45; 3.29 vs 3.28; NaN at 1.2B). Its failure mode is precisely "loss is not
    cross-entropy over logits", which describes your quantile+coverage loss.
22. Model soups on top of EMA + a 5-seed same-recipe ensemble — **NO EVIDENCE FOUND.** Soups
    (arXiv:2203.05482) average *different-hyperparameter* runs; arXiv:2502.06761 provides no
    ensemble-interaction analysis and shows WA converges to the annealed model when the LR anneals to
    zero. Expect a compute win, not a sigma_eff win.
23. SWA/LAWA vs EMA at this scale — arXiv:2502.06761: **speed, not accuracy** (612 -> 550 -> 541
    GPU-h; BLEU 31.157 -> 31.429).
24. Deep-ensemble scaling past ~3 members — arXiv:2506.04677 (forecasting regression, 28,298 and
    15,053 series): gains saturate at 2-3. **Seeds 6-10 are wasted compute.**
25. C-Mixup / CEMS / Anchor Data Augmentation as published — real deltas but at **1-2k-row tabular
    scale** (Airfoil 2.901 -> 2.717 -> 1.455); nothing shows they survive at 70k. RC-Mixup's
    noise-robustness premise is the only one that matches, and its numbers here are **provisional**
    (search snippet, not a fetched table).
26. Stochastic depth, dropout schedules, weight-decay tuning for small transformers on regression —
    **NO EVIDENCE FOUND**; the 2026 hits are image/audio/action classification.
27. Semi-supervised regression / pseudo-labelling as published — every gain (arXiv:2510.15266)
    requires a **10-20x unlabelled:labelled ratio**; you have 0.16x.
28. Self-distillation / Born-Again Networks at small n on regression — **NO EVIDENCE FOUND** with
    numbers. (The reusable idea from arXiv:2503.14293 is auxiliary-target generation, not
    compression.)
29. Deep unfolding / LISTA-style learned iterative solvers for **scalar** regression at small n —
    **NO EVIDENCE FOUND**; all hits are structured-output inverse problems.
30. TENT, TTT layers, In-Place TTT — classification and language modelling only; **NO EVIDENCE
    FOUND** for scalar regression.
31. Learning-using-privileged-information / teacher-with-truth distillation — mechanism is attractive
    and matches your latent variable, but **NO 2025-2026 REGRESSION NUMBERS FOUND**; retrieved work
    (arXiv:2602.04942, arXiv:2604.13016) is entirely LLM-shaped and reports that privileged-context
    distillation can *degrade* the student. Research bet, not a recipe.
32. Forcing a latent to be the impact position via unsupervised structure — **documented null**:
    HistoAE (arXiv:2511.22246) learns a 2-d latent of charge and impact position and lands
    "comparable to the conventional approach" (0.25 e, 3 um).

**Unverified — could not retrieve** (named for completeness, cited nowhere as evidence): Gated
DeltaNet / Titans / Hymba short-sequence numbers; ATLAS arXiv:2505.23735; beignet Fourier-pyramid
PINN; CAFE Fourier-Chebyshev encoding; "Deep Neural Expected Shortfall Regression with
Tail-Robustness"; the "Stop Regressing" (arXiv:2403.03950) 30% / 1.8-2.1x figures; LorentzNet and
PELICAN per-fraction tables; ConFrag (arXiv:2502.17771) and Anchor-DA tables; TTT-NN
(arXiv:2305.18466 / 2511.16691) perplexity numbers; a claimed "TabPFN test-time training cuts
required sample size 3-5x" for which no arXiv ID exists in any retrieved result;
arXiv:2606.10960 hadronic-response paper; arXiv:2505.16226; arXiv:2606.07134; arXiv:2509.20950;
arXiv:2601.21731.

