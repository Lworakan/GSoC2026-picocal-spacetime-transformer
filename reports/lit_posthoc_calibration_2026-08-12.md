# Post-hoc calibration and bias correction for a compressed energy regression

Survey date 2026-08-12. Setting: `width_binned_calibration` in `scripts/picocal_models.py`
fits `polyfit(q50, log E_true, 1)` inside three terciles of `q75 - q25`, then exponentiates.
Metric: sigma_eff of `(E_pred - E_true)/E_true`, reported per region and per **true**-energy
tercile (confirmed in `scripts/score_regions.py:25`).

## Headline: the premise of the question is half wrong, and I can prove it

The brief asserts the positive low-energy bias is "regression attenuation ... an OLS slope is
biased low and predictions shrink toward the mean". The diagnosis is correct. The implied
remedy is not. **OLS shrinkage is not an error — it is the variance-minimising use of a noisy
predictor.** Un-shrinking the slope flattens the per-true-energy bias and *raises* sigma_eff.

I verified this with a synthetic replica of our setting (steeply falling `E^-3` spectrum on
1-100 GeV, heteroscedastic log-noise 0.06-0.20, width observable correlated with the noise;
`n_val = 10k`, `n_test = 100k`). Scripts: `sim_attenuation.py`, `sim_estimators.py`,
`sim_v3.py` in the session scratchpad.

**All numbers in this report marked "(synthetic)" come from that replica, not from PicoCal data.**
Nothing here has been run on `reports/predictions/*.csv` yet.

De-attenuation sweep **(synthetic)**, slope moved from the OLS value `a` toward 1 by fraction `f`
(`f = 1` is the fully un-attenuated / structural slope, the Deming/York limit):

| f | sigma_eff | vs baseline | median bias T0 (low E) | T1 | T2 (high E) |
|---|---|---|---|---|---|
| 0 (current OLS) | 0.1677 | -- | +0.0620 | +0.0127 | -0.0664 |
| 0.25 | 0.1690 | **+0.80%** | +0.0460 | +0.0093 | -0.0510 |
| 0.50 | 0.1718 | **+2.47%** | +0.0303 | +0.0056 | -0.0347 |
| 0.75 | 0.1765 | **+5.26%** | +0.0153 | +0.0021 | -0.0179 |
| 1.00 | 0.1835 | **+9.47%** | +0.0005 | -0.0018 | -0.0003 |

The bias goes to zero exactly as sigma_eff degrades by 9.5%. This is a monotone trade, not a
free lunch. The size of the penalty depends on how heteroscedastic the predictor noise is: with
noise drawn independently of energy (`s ~ U(0.06, 0.20)`) the same sweep costs only +0.27% /
+0.97% / +2.00% / **+3.61%** at `f = 0.25 / 0.5 / 0.75 / 1`. So the honest range for the cost of
fully removing the bias is **+3.6% to +9.5% sigma_eff**, and our mild measured bias puts us at the
low-`f`, low-cost end either way. Explicit Deming with `lambda = 1` (orthogonal/TLS) reproduces an intermediate
point: in the same generator it cut the low-bin bias from +0.0327 to +0.0177 (-46%) while
sigma_eff went 0.1731 -> 0.1736 (+0.3%); on the steeper spectrum, +0.0535 -> +0.0296 (-45%)
for 0.1760 -> 0.1775 (+0.9%).

Note also the top row of the same tables: **uncalibrated raw q50 has near-zero per-tercile bias
(+0.0016 / -0.0011 / +0.0013) and 8.6-9.5% worse sigma_eff.** The bias we measure is
*manufactured by our own calibration step*, and it is the price of the sigma_eff we already
banked. Our observed +0.0132 / +0.0085 is roughly 5x smaller than the sim's +0.062, i.e. our
attenuation is mild, so the recoverable bias is small and the sigma_eff cost of recovering it
is proportionally the same. Do not pay it.

---

## 1. Errors-in-variables / attenuation

**Deming, orthogonal/TLS, York.** York 1968 (*Earth Planet. Sci. Lett.* 5, 320) and Deming 1943
minimise perpendicular (or error-weighted) distance to estimate the **structural** slope
between two latent variables. Requires `lambda = var(err_y)/var(err_x)`; `lambda = 1` is
orthogonal regression, `lambda -> inf` recovers OLS.

**How lambda is estimated with heteroscedastic predictor noise.** In practice three routes, none
clean: (i) per-point weights `w_i = 1/(s_yi^2 + a^2 s_xi^2)` iterated on `a` — York's own
iterative scheme, and the reason York is the astronomy standard rather than Deming; (ii) replicate
measurements, `lambda` from within-subject variance components (the clinical-chemistry route,
Linnet 1990, *Stat. Med.* 9:1463, weighted Deming); (iii) a structural model with a Gaussian-mixture
prior on the latent predictor and full MCMC — **Kelly 2007, arXiv:0705.2774** (ApJ 665, 1489,
`linmix_err`), which handles heteroscedastic *and* correlated errors, intrinsic scatter,
non-detections and Malmquist selection in one likelihood.

Kelly is also the cleanest statement of the distinction that decides this question: the
structural slope and the *predictive* slope are different estimands, and the attenuated OLS
slope is the correct one when the goal is predicting y from a noisy x. We can estimate our
`lambda` for free: `s_x^2` is the per-event quantile width (`(q75-q25)/2.698)^2`), so
`lambda_hat = sigma_intrinsic^2 / mean(s_x^2)`. In my generator `lambda_true` ran 0.013-0.145.

**HEP calibration that explicitly corrects this: Cukierman & Nachman, arXiv:1609.05195**,
*Mathematical properties of numerical inversion for jet calibrations*, NIM A (2017) — volume and
page numbers not confirmed by anything I retrieved, do not copy them into a bibliography unverified.
This is the closest published analogue to our exact procedure. ATLAS/CMS jet calibration
parameterises average response vs `E_true` then inverts it to a function of `E_reco`
("numerical inversion"). The paper proves numerical inversion is **inherently biased** and that
"calibrated reconstructed jets are not guaranteed to be centered around the corresponding
particle-level jet", and that the usual approximations **over-estimate the resolution**. It
proposes extensions that reduce the bias. I could not extract per-number figures — the arXiv
abstract page carries the qualitative claims and the PDF would not convert. Treat the
percentages as unverified; the structural claim is verified.
*Applicability:* directly on point, and it is the citation to use when a mentor asks why the
per-tercile bias is nonzero. It also warns that our per-region sigma_eff may be optimistically
biased for the same reason.

*Blunt applicability for the whole item:* implementing Deming/York/TLS in our calibration will
make the per-tercile bias table look better and the headline metric worse. Only do it if bias
becomes a reported deliverable in its own right.

## 2. Astronomy's version: Eddington / Malmquist

**Eddington bias** (Eddington 1913): with a steeply falling source count, noise up-scatters more
faint objects into a flux bin than it down-scatters bright ones, so objects observed at a given
flux have preferentially lower true flux. This is our spectrum exactly (median 24 GeV, p95 80).
**Malmquist bias** is the selection-limited cousin.

*Numbers found.* Surveys correct **counts**, not objects: in LoTSS/GMRT-class analyses only the
lowest flux bin is significantly affected, with measured differential counts over-estimating the
input model by **21%**, applied as a divide-by-1.21 to that bin. Per-object "flux deboosting"
factors from mm-wave point-source catalogues (SPT-SZ, arXiv:1306.3470) run **1/1.194 at
5 <= SNR < 6 down to 1/1.052 at 9 <= SNR < 10** — i.e. a 19% down-correction at low SNR falling
to 5%.

Note the crucial detail: deboosting is a **posterior-median/mean shrinkage toward the steeply
falling prior**, so it moves the estimate in the *same direction as OLS shrinkage already does*.
Eddington is arguably the better *framing* of our bias than attenuation (the brief's suspicion is
right), but it prescribes **more** shrinkage, not less, and our OLS already applies it implicitly.
*Blunt applicability:* correct framing, zero new action. A formal Bayesian deboost with our
measured spectrum as prior would land very close to what `polyfit` already gives, and the 19%
figures are for SNR ~5 sources, far noisier than our 0.03-0.17 regime.

## 3. HEP unfolding / response matrices, per-event

**Iterative Bayesian unfolding** (D'Agostini 1995, NIM A 362:487), **OmniFold**
(Andreassen et al., arXiv:1911.09107, PRL 124:182001), **unbinned-unfolding practical guide**
(arXiv:2507.09582, 2025), **OmniFold for neutrino cross-sections** (arXiv:2504.06857, 2025).

Mechanism: OmniFold iteratively reweights simulation with classifiers, alternating detector-level
and particle-level steps. Its output is a **per-event weight** on simulated events, not a
corrected value for a measured event.

**This cannot improve a per-event metric, and that is a definite answer, not a hedge.** Unfolding
is a deconvolution of the *distribution*; the map from one observed event to its true energy is
one-to-many and unfolding never claims to invert it. The reported gains are distribution-level:
the neutrino application found *no strict chi^2 improvement* over conventional IBU, achieving
"similar chi^2 with less bias and smaller uncertainties"; iterations were capped at ~5 because
detector resolution limits what is recoverable. There is no sigma_eff-equivalent number to quote
because the methods do not produce a per-event point estimate.
*Blunt applicability:* none. Do not spend time here. Worth one sentence in the thesis to
pre-empt the reviewer question.

## 4. Conditional calibration for regression: which ones move the point estimate?

| Method | ID | Moves point estimate? | Quantified point-estimate effect |
|---|---|---|---|
| Kuleshov et al., *Accurate Uncertainties for Deep Learning Using Calibrated Regression* | arXiv:1807.00263, ICML 2018 (PMLR v80:2796) | **Yes, indirectly** — isotonic recalibration of the CDF; the median of the recalibrated CDF is a shifted point estimate | Paper reports calibration error and NLL only. **No point-estimate error numbers.** |
| Song, Diethe, Kull, Flach, *Distribution Calibration for Regression* | arXiv:1905.06023, ICML 2019 (PMLR v97:5897) | **Yes** — GP-Beta link recalibrates the full conditional CDF per instance, so its median/mean move | Reports distribution- and quantile-level calibration gains. **No sigma_eff-analogue or point-error numbers.** |
| Conformalised quantile regression (Romano, Patterson, Candès) | arXiv:1905.03222, NeurIPS 2019 | **No** — adds a scalar `Q(E)` to both endpoints | Interval length / coverage only. Point estimate untouched by construction. |
| Mondrian conformal regressors (Boström & Johansson) | PMLR v128:114-133, COPA 2020; extended PMLR v152:24-38 (2021) | **No** — group-conditional interval calibration | Interval informativeness only. |
| Isotonic regression on the conditional mean | classical | **Yes** — monotone step fit replaces the linear one | **-10.7% to -11.9% sigma_eff (synthetic, item 5)** |

*Blunt applicability:* CQR and Mondrian conformal are the wrong tool — they are the right answer
to a question we are not being scored on. Kuleshov/Song are the only two that touch a point
estimate, and neither paper gives a point-estimate number, so adopting them would be a research
bet, not an import. The cheap version of their idea — recalibrate monotonically, then take the
median — is item 5, and it works.

## 5. Binning choice — the actual win

Our calibration bins on predicted width only. Width is a proxy for noise; **it carries no
information about where on the response curve the event sits.** Synthetic replica,
`n_val = 10k`:

| calibration | sigma_eff | vs current | T0 sigma_eff | T1 | T2 |
|---|---|---|---|---|---|
| raw q50, no calibration | 0.1835 | +9.5% | 0.1929 | 0.1858 | 0.1719 |
| **width x3 OLS (current)** | **0.1677** | -- | 0.1690 | 0.1574 | 0.1520 |
| width x3, q50 x5 (joint) | **0.1455** | **-13.2%** | 0.0876 | 0.1281 | 0.1625 |
| width x3, q50 x3 | 0.1460 | -12.9% | 0.0865 | 0.1286 | 0.1618 |
| width x5, q50 x5 | 0.1456 | -13.1% | 0.0884 | 0.1283 | 0.1622 |
| width x10, q50 x10 (100 cells) | 0.1470 | -12.3% | 0.0900 | 0.1281 | 0.1637 |
| q50 x5 only, no width bins | 0.1466 | -12.6% | 0.0879 | 0.1288 | 0.1591 |
| isotonic `log E ~ q50`, unbinned | 0.1477 | -11.9% | 0.0888 | 0.1290 | 0.1607 |
| width-only, x5 / x10 / x20 | 0.1675 / 0.1671 / 0.1668 | -0.5 / -0.7 / -0.9% | | | |
| linear in `(q50, width, q50*width)` | 0.1675 | -0.5% | | | |
| per-width-bin direct sigma_eff-optimal `(a,b)` | 0.1549 | -7.6% | 0.1548 | 0.1432 | 0.1415 |

**Leakage check (important).** In the generator above the noise level `s` is a function of energy,
so `q50` leaks information about `s` and part of the gain could be recovered heteroscedasticity
rather than response-curve curvature — which would not transfer, because in the real detector the
noise level is carried by region/pitch, not by energy. Rerunning with `s ~ U(0.06, 0.20)` drawn
**independently of energy**: joint width x q50-5 gives **0.1055 vs 0.1194 baseline = -11.6%**,
isotonic in q50 alone -10.7%, width-only refinements still under 1%. The gain survives, so the
mechanism is genuine — the shrinkage-optimal slope varies along the response curve because the
steeply falling spectrum makes the prior pull energy-dependent. The T0/T2 asymmetry in the table
above (T0 -48%, T2 +7%) *is* the leakage component and does not survive; the aggregate ~11-12%
does.

Readings:
- **Binning on the predicted value is worth ~11-13%. Refining the width binning is worth <1%.** We
  have been tuning the axis that does not matter. Almost the entire gain is available from a
  single monotone/piecewise fit in q50; width adds ~0.6% on top.
- **No overfitting at 10k.** 3x5 = 15 cells x 2 params and even 10x10 = 100 cells stay within
  0.15% of the 100k-event result. The ranking is identical at both sizes. Overfitting is not the
  binding constraint; still cross-fit, because it costs nothing.
- Adding width as a *regressor* instead of a bin label is worth only 0.5% — the brief's
  suggestion, and it does not pay. The nonlinearity in q50 is what matters, not extra linear terms.
- **Directly minimising sigma_eff is worse than conditioning (-7.6% vs -13.2%) and it is
  dangerous.** Unconstrained Nelder-Mead on sigma_eff found a degenerate global optimum:
  drive `E_pred -> 0`, every relative residual equals exactly -1, sigma_eff = 0. Any
  metric-direct fit **must** be constrained (slope on a bounded grid, intercept profiled).
  Even constrained, it bought less than binning and wrecked the bias (T2 -0.140).

Smooth alternatives (monotone splines, GAMs) are worth trying but the isotonic result says the
ceiling is the same -11.9%; the residual 1.3% comes from letting the *slope* vary, which a
monotone fit on q50 alone cannot do.

## 6. Log-space target and the Jensen gap

`exp(a*q50 + b)` estimates the conditional **median** of `E_true`. sigma_eff and median bias are
quantile functionals. **The pipeline is already correct for this metric and the Jensen correction
is the wrong sign.** `exp(sigma^2/2)` would inject a positive median bias of +0.08%
(sigma_log = 0.04), +0.50% (0.10), +2.02% (0.20) — largest exactly in the low-energy/high-noise
bins where we already measure positive bias.

Cost of applying it (synthetic):

| correction | sigma_eff | vs current |
|---|---|---|
| none | 0.1677 | -- |
| `+ 0.04^2/2` | 0.1735 | +2.8% |
| `+ 0.10^2/2` | 0.1742 | +3.3% |
| `+ 0.20^2/2` | 0.1768 | +4.8% |

The standard references (`BC = e^(sigma^2/2)`; Duan's 1983 smearing estimator as the
distribution-free version) are right about the *mean* and irrelevant to us. There is a real
residue: `r = exp(delta) - 1` is right-skewed, so the shortest 68.3% interval is not centred at
`median(delta) = 0`, and a small per-bin multiplicative offset chosen to minimise sigma_eff
directly is legitimate. My grid search found the optimal global offset gained only 1.9-2.3% and
ran to the edge of the search window while destroying the bias table (T2 -0.115) — so it is
real but small, non-robust, and it trades exactly the thing item 1 was worried about.
*Blunt applicability:* report the Jensen numbers to show the pipeline was checked; change nothing.

## 7. Other post-processing levers

- **Region conditioning** (promoted to rec-1). sigma_eff varies 3x within one energy tercile, and
  `width_binned_calibration` never sees the region. Width is only a proxy for cell pitch /
  occupancy. Fitting `(a, b)` per region x q50-bin is the same mechanism as item 5 applied to
  the axis where we already know a 3x spread exists. No literature number; the mechanism is the
  one that paid ~12% in simulation.
- **Test-time augmentation** already exists in `scripts/eval_tta.py`; averaging q50 over
  augmentations reduces predictor noise, which *reduces the attenuation at its source* and is
  strictly better than correcting for it downstream.
- **Cross-fitting the calibration.** Currently `(a, b)` come from the validation split and are
  applied to test — fine. But if calibration params are ever reported on the same events that set
  them, K-fold cross-fitting is mandatory.

---

## Ranked list: at most 5 post-hoc changes worth implementing

**1. Condition the calibration on region.**
`width_binned_calibration` never sees the region, yet sigma_eff varies 3x within a single energy
tercile in our **real** data — that is the one observation here that is not synthetic. Fit `(a, b)`
per region, i.e. `region x width` at minimum. This is ranked first because the evidence for it
comes from our own measurements, not from a generator I wrote.

**2. Bin the calibration jointly on q50 as well as width. -11.6% to -13.2% sigma_eff (synthetic,
unverified on our data).**
Replace `width_binned_calibration` with: cut `w = q75 - q25` into 3 quantile groups; inside each,
cut `q50` into 5 quantile groups; in each of the 15 cells fit `polyfit(q50, log E_true, 1)`;
fall back to the parent width-group fit when `n_cell < 30`. Apply test events by the *validation*
cut edges. Survives the leakage check, so the mechanism should transfer, but the magnitude on
PicoCal data is unmeasured. Combine with rec-1 as `region x q50 x width`, using 5 q50 bins per
region and dropping to 3 if any region has under ~600 validation events.

**3. Replace the piecewise-constant slope with a monotone smooth in q50.**
`sklearn.isotonic.IsotonicRegression(out_of_bounds='clip').fit(q50_val, logE_val)` inside each
width group, or a monotone I-spline / GAM with 5-8 knots. Measured -11.9% on its own; it buys
robustness rather than extra sigma_eff, and removes the bin-edge discontinuities.

**4. K-fold cross-fit the calibration coefficients (K = 5).**
No sigma_eff gain expected; it makes the -13% claim defensible when a reviewer asks whether 15
cells x 2 params overfit 10k events. My 10k-vs-100k comparison says they do not (within 0.15%),
and cross-fitting is how you show it.

**5. Report the bias/sigma_eff frontier instead of trying to remove the bias.**
Compute the de-attenuation sweep on real data with `f in {0, 0.25, 0.5, 0.75, 1}`,
`a_f = a_OLS + f*(1 - a_OLS)`, `b_f = mean(logE) - a_f*mean(q50)` per bin. Publish the table.
It converts an apparent defect ("+0.0132 bias") into a deliberate, quantified choice, which is
the scientifically honest framing and costs one loop.

Estimator for anyone who wants the structural slope anyway (Deming with variance ratio lambda):
`a = [S_yy - lambda*S_xx + sqrt((S_yy - lambda*S_xx)^2 + 4*lambda*S_xy^2)] / (2*S_xy)`,
`b = mean(y) - a*mean(x)`, with `lambda = sigma_intrinsic^2 / mean(((q75-q25)/2.698)^2)`.

## Explicitly checked, does NOT help a per-event point-estimate metric

- **Deming / orthogonal / TLS / York regression** — flattens per-true-energy bias, costs +3.6% to
  +9.5% sigma_eff to remove it fully (synthetic, two noise models). Monotone trade. Not a fix.
- **Kelly 2007 `linmix_err` MCMC** — the right tool for a structural slope with heteroscedastic
  errors; the structural slope is not what a predictive metric wants.
- **Eddington / Malmquist / flux deboosting** — corrects populations; per-object deboosting
  shrinks *toward* the falling prior, the same direction OLS already goes. 19%-to-5% SNR-dependent
  factors are for SNR~5 sources, not our 0.03-0.17 regime.
- **Iterative Bayesian unfolding, OmniFold, all 2024-2026 unbinned ML unfolding** — produce
  per-event *weights* on simulation, never a corrected per-event value. Cannot move sigma_eff.
  No number exists to quote because the estimand is a distribution.
- **Conformalised quantile regression (arXiv:1905.03222)** — adds a constant to both interval
  endpoints; point estimate provably unchanged.
- **Mondrian / group-conditional conformal (PMLR v128:114)** — group-conditional *interval*
  calibration; point estimate unchanged.
- **Jensen / lognormal mean correction `exp(sigma^2/2)`, and Duan smearing** — costs +2.8% to
  +4.8% sigma_eff and injects positive median bias of +0.08% to +2.0%, worst in exactly the bins
  that already read positive. Wrong sign for a quantile metric.
- **Refining the width binning (5, 10, 20 groups)** — worth 0.5-0.9%. We have been tuning the
  wrong axis.
- **Width as an extra linear regressor, with interaction** — 0.5%. The missing structure is
  nonlinearity in q50, not extra linear terms.
- **Unconstrained direct sigma_eff minimisation** — degenerate optimum at `E_pred -> 0` where
  every relative residual is -1 and sigma_eff is exactly 0. Constrained, it still loses to joint
  binning (-7.6% vs -13.2%) and ruins the bias table.
