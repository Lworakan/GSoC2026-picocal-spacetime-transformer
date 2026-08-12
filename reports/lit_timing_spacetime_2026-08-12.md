# Per-cell timing and spacetime constraints under pileup: what the literature actually shows

Survey date 2026-08-12. Setting: LHCb PicoCal, photon energy regression from a 9x9 cell window, two longitudinal samples per cell (front, back), each with an energy, one timestamp and a validity flag. No usable reconstructed vertex.

Rules used: every external number carries the document actually read and a verbatim sentence. Where a document could not be read (CDS serves a proof-of-work wall for the TDRs), the entry says so and the number is not used to support a conclusion. WebFetch's summariser returned false negatives on several PDFs that did contain the target sentences; the surviving quotes come from ar5iv HTML or from `pdftotext` on the saved bytes.

## 0. Our own numbers, recomputed for this survey

Five measurements on the cached samples (`.scratch/cache/clean-aux_100.pkl` 30303 events, `minbias_94.pkl` 72554 events; 9x9 window, cells above the 2.49 MeV threshold). The literature comparison hinges on them, so they were recomputed rather than quoted.

**(a) The reference time is not the bottleneck.** Inverse-variance combination of all cell front-times in the window, using our measured sigma_t(E):

| sample | seed-cell sigma_t | inverse-variance combined | seed energy fraction | fraction of cells below 300 MeV |
|---|---|---|---|---|
| clean | 38 ps (median) | **28 ps** (p10 21, p90 37) | 0.68 | 0.85 |
| min-bias | 38 ps | **21 ps** (p10 12, p90 32) | 0.39 | 0.62 |

A common t0 for the window is therefore known to 20-30 ps, 7-10x smaller than the ~200 ps of in-time separation. The failure of our time-pull feature was **not** a failure of the reference. It is the per-cell term.

*Caveat, and it is a real one.* This combination assumes the per-cell time errors are independent. CMS HGCAL measures a same-module cell-time correlation of rho ~ 0.8, which means sqrt(N) aggregation does **not** hold there and would not hold for us if our cells share a front-end clock or a common light path. The honest reading of the table is therefore an *optimistic* bound: the true combined t0 lies between 21-28 ps (independent) and ~38 ps (fully correlated, i.e. no better than the seed). Even the pessimistic end is 5x smaller than 200 ps, so the conclusion — the reference is not the bottleneck — survives either way. Measuring the actual cell-time correlation matrix on our data is a one-hour job and should precede any claim that leans on the 21 ps number.

**(b) Per-cell significance of in-time pileup is below 1 sigma by construction.** For a 200 ps offset:

| cell energy | our sigma_t | dt/sigma_t at 200 ps |
|---|---|---|
| 2-10 MeV | 756 ps | 0.26 |
| 30-100 MeV | 435 ps | 0.46 |
| 100-300 MeV | 263 ps | 0.76 |
| 300-1000 MeV | 156 ps | 1.28 |

Under pileup every row degrades 1.7-2x, so the ratio falls to 0.13-0.64. Since 62-85% of window cells sit below 300 MeV, the cells carrying the pileup are exactly the cells whose timestamps cannot see it. Three sigma would need roughly (3/0.76)^2 ~ 16 independent 300 MeV cells belonging to the *same* contaminating vertex, and we have no per-cell labels with which to group them.

**(c) The in-time signal is mostly a common mode, not a per-cell differential.** The luminous-region z spread displaces the whole photon cluster by one common amount; it does not spread the cells of one photon against each other. A per-cell pull against a window-level reference is by construction blind to a common shift. This is the mechanical reason both our timing attempts returned nothing.

**(d) The absolute window time is information we currently destroy.** `scripts/picocal_data.py` builds `tfc = tf - t0f` and `tbc = tb - t0b`, subtracting a per-event median, so the absolute time never reaches the network. Measured on the absolute inverse-variance window time:

| sample | mean t0 | RMS of t0 | our measurement error on t0 | implied physical spread | p99 |
|---|---|---|---|---|---|
| clean | 86.22 ns | **0.840 ns** | 0.038 ns | 0.839 ns | 88.05 ns |
| min-bias | 86.75 ns | **3.223 ns** | 0.035 ns | 3.223 ns | **95.41 ns** |

The truth arrival time (`sig_flux_timing`, already carried as `ev['at']` and already used as an auxiliary target) has mean 42.47 ns and RMS 0.343 ns. Correlation of the measured t0 with it is 0.66 on clean and only 0.10 under pileup. Ratio sigma_t/spread = 0.038/0.343 = **0.11 at cluster level** — that is the CMS MTD regime, reached at the cluster and not at the cell.

**How strong a discriminant is t0 actually? Measured, not inferred.** The large population RMS above is mostly the natural arrival-time spread plus energy-dependent walk, not pileup, so it is *not* a significance. Scored properly on the full samples (30198 clean, 72045 min-bias):

| quantity | value |
|---|---|
| median t0, clean / min-bias | 86.284 / 86.593 ns, shift **0.310 ns** |
| AUC of absolute t0 alone, clean vs min-bias | **0.606** |
| min-bias fraction above the clean 95th percentile (87.215 ns) | 18.1% (3.6x enrichment) |
| min-bias fraction above the clean 99th percentile (88.028 ns) | **6.0% (6.0x enrichment)** |
| min-bias fraction above the clean 99.9th percentile (93.490 ns) | 1.23% (12.3x enrichment) |

So the honest statement is: a *fraction* of events, about 6%, carry multi-ns late energy that clean events essentially never show, and absolute t0 flags them at roughly 6-12x purity. It is a tail handle, not a mean shift, and its standalone AUC is 0.61 — modest. Caveat inherited from `bounds_2026-08-12.md` section 2: clean and min-bias are independently generated samples, so part of this separation could be sample difference rather than contamination. A paired overlay sample would settle it, and that ask is already the top item there.

**(e) Front-back time difference carries no depth information in our data.** Seed cell:

- tf - tb has a fixed offset of **-1.234 ns**, identical in clean and min-bias: a calibration constant (light path), not physics.
- spread about it: **0.844 ns** clean, **3.63 ns** min-bias.
- correlation of (tf - tb) with our existing depth proxy log((E_front+1)/(E_back+1)): **+0.026** clean, **-0.020** min-bias. Zero.
- inverse-variance combination of the front and back timestamps of the same cell yields sigma = **0.82x** the better of the two, in both samples. An 18% per-cell timing gain we currently do not take.

---

## 1. Single-origin / vertex-time fits

**ATLAS HGTD, track-time-to-vertex association.** CERN-THESIS-2023-175 (V. Raskina), fetched `https://inspirehep.net/files/91e3516ca0c9cfee824039ecf8f77f40`.
- Spatial compatibility, verbatim: *"where z0 is the track longitudinal impact parameter, sigma_z0 is the per-track resolution on the longitudinal impact parameter which depends on the track pseudorapidity and pT, and s is a significance cut (typically 2.5 or 3)."* Requires reconstructed tracks/vertex: **yes**.
- Ambiguity being resolved, verbatim: *"Having an average vertex density of 1.8 vertices/mm at z = 0 (at <mu> = 200), a forward low pT track can be compatible with up to 9 near-by vertices on average."*
- The time-compatibility chi2 form itself is deferred to the TDR (CDS unreachable): **no quantified number found**.
- Quantified rejection, verbatim: *"The combined approach enhances the rejection of pile-up jets in the forward region with 30 < pT < 50 GeV by approximately 1.5 times at a signal efficiency of 85%."* and *"For the high pT jets, the performance is approximately 20% better when using HGTD."*
- Feasibility for us: not transferable. Every variant is a track-to-vertex test and we have neither.

**The one vertex-free single-origin test found in the whole sweep.** Same thesis, eq. 6.5, verbatim: *"where thit 1 and thit 2 are the times measured for the first and the second hits respectively and alpha is a time difference parameter that is set to two after optimisation"* — a pairwise pull |t1-t2| / sqrt(sigma1^2 + sigma2^2) < 2, with a chi2 < beta extension for three or more hits. It compares hits **to each other**, not to a vertex. Quantified, verbatim: *"The HGTD performance without any cleaning shows that 19.12% of tracks do not get the time assigned, 53.50% of tracks have the correct time assigned to them and 27.38% have misassigned times."* and *"The TDR cleaning does enhance the performance, bringing the ratio of correctly assigned track-times to 49.45% and misassigned to 12.71%."* Misassignment 27.38% -> 12.71%.
- Feasibility for us: **directly implementable**, and it is the correct formalisation of what we already tried. Section 0(b) says the per-cell version of this test has at most 0.76 sigma of leverage against in-time pileup, which is why our pull feature was flat. It has many sigma of leverage against the multi-ns tails in 0(d).

**CMS MTD 4D vertexing — the exact fit.** Two objectives, both located.
- Per-track space-time compatibility chi2 (CHEP 2025 / IOP C01048 eq. 2.4) carries a time-residual term of the form `(t_PCA(h) - t_vtx)^2 / sigma^2(t_PCA(h))` added to the spatial chi2 — a **residual against a fitted common origin time**, exactly the object item 1 asks about. Requires tracks and a vertex: **yes**.
- The vertex finding itself is a deterministic-annealing free energy (eq. 2.5) extended to four dimensions, with an explicit outlier/noise term Z0 and per-track mass priors alpha_h = 0.7 / 0.2 / 0.1 (pion / kaon / proton) used to convert the measured time into a t0 at the vertex. The Z0 term is the mechanism by which an incompatible hit is *softly downweighted* rather than cut.
- Quantified: vertex merging **15% -> 1%**; vertex time resolution **8.7 -> 7.1 ps**; **-30%** CPU for the newer algorithm at equal performance; **-40%** pileup jets; **+10%** missing-ET performance.
- Feasibility for us: the *shape* transfers — a soft outlier term with a fitted common t0 is exactly the estimator our data supports at cluster level (0(a), 0(d)). The mass priors and track extrapolation do not.

arXiv:1810.00860, fetched `https://arxiv.org/pdf/1810.00860`.
- Verbatim: *"In the time domain, pileup collisions at the HL-LHC will occur with an RMS spread of approximately 180-200 ps, constant during the fill and uncorrelated with the line spread along the beam line. Slicing the beam spot in consecutive 30 ps exposures effectively reduces the number of vertices down to current LHC conditions"* — sigma_t/Delta_t = 30/190 = **0.16**.
- Verbatim: *"According to simulation, instances of vertex merging are reduced from 15% in space to 1% in space-time."*
- Verbatim: *"The addition of track-time information with 30 ps precision reduces the wrong associations to a level comparable to that of the current LHC"*; quoted gains +20% isolation efficiency, +30% diphoton vertex, +30% VBF tagging, 40% reduction of missing-ET tails. Requires tracks/vertex: **yes**.
- Feasibility: no. It is a tracker algorithm.

**ATLAS HL-LHC expected performance.** arXiv:1809.02181, fetched `https://arxiv.org/pdf/1809.02181`. Verbatim: *"In order to ensure a timing resolution of 30 ps per track over the whole HL-LHC period, four layers of active sensors will be built."* and *"A significant improvement in performance of up to a factor of 4 higher pile-up jet rejection at constant efficiency is achieved with the use of timing information."* The "factor 4" is ttbar, a specific pT bin and working point; the forward VBF number at a stated efficiency is the ~1.5x above. The "6x" figure circulating in talks is the ratio 180/30 ps, not a measured rejection — do not cite it as a result.

**CMS HGCAL TICL time linking.** The construction, in order:
1. **Layer-cluster time = resolution-weighted mean of the hit times, preceded by a densest-time-window outlier rejection**, requiring N >= 3 timed hits. This step is **vertex-free** and is the closest published analogue to what we can compute in a 9x9 window.
2. **Trackster time**: the layer-cluster times are TOF-corrected to the trackster barycenter plane, then combined by a resolution-weighted average.
3. **At link time a 3 sigma time-compatibility cut** is applied between a track and a trackster — this step *is* vertex-dependent.
Quantified pileup-rejection efficiency for TICL specifically: **not found** — and the reason is that the source is behind the CDS bot challenge, so this is *inaccessible*, not absent.
- Feasibility for us: step 1 is implementable today and is essentially idea 1 + idea 3 of section 7; the densest-time-window rejection is a more robust variant of the top-decile reference we already tried. Steps 2 and 3 need geometry and a vertex we do not have.

**Generic 4D / space-time calorimeter clustering.** The only calorimeter-only *quantified* pileup result found in the entire sweep is arXiv:2310.16497 (item 6, out-of-time). Also identified: arXiv:2209.02932 (CEPC cluster timing), arXiv:2312.14622, arXiv:2005.13324, and arXiv:2203.01317 as the closest methodological precedent at our resolution.

## 2. The time-of-flight geometry term

**It is applied everywhere and ablated nowhere.** CERN-THESIS-2023-175, verbatim: *"The times from individual HGTD hits are aligned using the Time of Flight (TOF) correction, calculated by dividing the path length of the particle's track by the speed of light."* and *"The path length is assumed to be a straight line between the hit's position and the particle's origin x = (0,0,z0)."* Requires a reconstructed track for z0: **yes**. Quantified effect of the TOF correction in isolation, across all four search strands: **no quantified number found**. The only substantive published remark about it is that it *hurts*, verbatim: *"The contributions to these tails in this case come from the z0 mismeasurement and from the TOF overcorrection (which drastically increases the right shoulder in contribution)."*

The same TOF alignment appears in CMS MTD, CMS HGCAL/TICL and CEPC cluster timing, in each case as an unquestioned preprocessing step with no ablation. The cleanest published statement of the form is arXiv:2209.02932 (CEPC): the corrected hit time is `T = t - L/c`, with L the path length from the origin to the hit. In TICL the same correction is applied to bring layer-cluster times onto the trackster barycenter plane.

**The vertex-free variant is quantitatively dead for us, by arithmetic.** The only path-length difference we can compute without a vertex is the transverse one, cell to cell inside the window: extra path = r^2/(2L) with L ~ 12.5 m. At the outermost ring of the 60 mm region, r = 240 mm gives 2.3 mm = **7.7 ps**; in the 15 mm region, r = 60 mm gives 0.14 mm = **0.5 ps**. Against per-cell sigma_t of 263-756 ps this is a 1-3% correction, i.e. nothing. The part of the TOF term that actually matters is the vertex-z piece, sigma_z = 35-40 mm giving **117-133 ps** — and that is a *common* shift on the whole cluster, which is exactly quantity 0(d), not a per-cell correction.

Verdict on item 2: including a geometric TOF term measurably improves nothing that we can compute, and no published work quantifies its benefit even where a vertex is available.

## 3. Nested time-window energies

**arXiv:2107.10207** (Akchurin, Cowden, Damgov, Hussain, Kunori; JINST 16 (2021) P12036), fetched `https://arxiv.org/abs/2107.10207` and the PDF.
- Mechanism, verbatim: *"We feed into the GNN a series of cell energy readouts having increasing integration times. In this way, the series of cell energies represent cumulative effects of time rather than distinct time bins."* Integration ends at 10 ns; windows illustrated at 0-15 ps, 0-30 ps, 0-50 ps, 0-200 ps, 0-1 ns, 0-10 ns. Verbatim example of the nesting: *"the timing precision of 0.5 ns includes time intervals (0-0.5 ns), (0, 4 ns), (0, 1 ns) and (0, 0.5 ns) and plotted at 0.5 ns."*
- Quantified gain: **no quantified number found.** The text gives only *"We observed a time precision dependence of energy resolution - better timing, better resolution - roughly comparable to that of the CNN. The GNN's energy resolution surpasses the CNN's below ~100 ps."* and *"the energy resolution improved somewhat for the two cases (30 and 100 GeV pions) we analyzed."*
- **The times are GEANT4 truth, unsmeared**, verbatim: *"We record the time of any simulated energy deposition as t = t_G4 - z/c where t_G4 is the time when the energy is deposited as reported by GEANT4 and z/c is the travel time of light in vacuum to cover the longitudinal depth."* and *"Although this study does not include the simulation of detector effects, such as electronic noise..."* Timing "precision" is emulated only by window coarseness, never by smearing.
- Needs waveforms or many hits per cell: **yes, structurally** — the binning requires the time of each deposition inside a cell. With one timestamp per readout sample, the cumulative windows collapse to a single threshold indicator. Requires a vertex: no.
- Feasibility for us: the method as published does not transfer. Its operating point is 15-200 ps window granularity on noiseless truth times; ours is 400-1300 ps, 10-100x coarser, with one timestamp.

**arXiv:2203.01317**, "Time-assisted energy reconstruction in a highly-granular hadronic calorimeter" (CALICE AHCAL, JINST 17 P10001), fetched `https://arxiv.org/pdf/2203.01317`. **This is the realistic-resolution analogue and the only quantified precedent at our sigma_t.**
- Verbatim: *"For the time measurements a gaussian smearing with a resolution of 1 ns is applied to the generator-level time stamp of the hit, which is given by the time of the first energy deposit in the cell."* One timestamp per cell, 1 ns sigma — our regime.
- Verbatim: *"The addition of nanosecond-level time resolution is found to result in significant improvement of the energy resolution by approximately 3 % to 4 % for local software compensation compared to the software compensation based on local energy density alone."* Global-only timing variant: *"improvements of up to 25 % over the standard reconstruction"* against 35% for energy-density alone.
- Binning, verbatim: *"uses only two bins in time"*, split at 3 ns, i.e. the variable is E_dep(t_hit > 3 ns). Waveforms: **no**. Vertex: **no**.
- **Number-conflict warning.** The Snowmass timing white paper cites this result as a 10-15% improvement. The primary paper says 3-4% for the local variant and up to 25% for the global one. Quote the primary; 3-4% is the number that corresponds to a per-cell time input.
- Feasibility for us: **implementable as written.** Two bins, one threshold, one timestamp per sample. Caveat: in a hadronic calorimeter the late energy is the slow neutron component of the shower itself, so the 3-4% buys hadronic compensation, not pileup rejection. For us the same variable measures late/out-of-time contamination, which section 0(d) says is present at 3.2 ns RMS with a 9 ns tail.

**arXiv:2203.07286** (Snowmass, precision timing calorimetry), fetched. Shower depth from timing in unsegmented dual-readout fibre, *"assuming a 100 ps sampling rate"*, correlation plot only: **no quantified number found**; requires the full waveform.

## 4. Two-timestamp geometry

**Front-versus-back longitudinal Delta t as a depth or direction estimator: documented null.** Five targeted searches (front/back time difference and shower depth; photon pointing with longitudinal segmentation; SpaCal double-sided readout; PicoCal front/back timestamps; depth-from-timing with a network) returned nothing. arXiv:2504.03088 (LHCb-PUB-2025-002) was fetched and contains no such discussion. The existing SpaCal two-cell literature motivates combining front and back **only** for time-resolution improvement (item 5). Our own measurement, 0(e), independently kills the idea: correlation of seed (tf - tb) with the front/back energy ratio is 0.03.

**What is quantified is the double-*ended* case — same depth, opposite ends of a bar.** arXiv:2104.07786 (CMS BTL sensor prototypes, JINST 16 P07023), fetched `https://arxiv.org/pdf/2104.07786`.
- Position from the difference: verbatim *"this sensor layout is also capable of providing a measurement of the impact point with few millimetres resolution"*, with sigma_x = **3.99 +- 0.06 mm** (3x3x57 mm3, HPK) and **2.80 +- 0.02 mm** (3x4x57 mm3, FBK) from t_left - t_right; slope ~15 ps/mm.
- Resolution from the average: verbatim *"The combination of the two SiPM measurements in t_average improves the time resolution by about sqrt(2) with respect to the individual SiPM, since the dominant stochastic fluctuations from photostatistics are uncorrelated between the two ends."* Measured t_average sigma = **30.5 +- 0.5 ps** (HPK), **24.6 +- 0.2 ps** (FBK). Waveforms: yes (DRS4 at 5.12 GSa/s). Vertex: no.
- **KLOE Pb/scintillating-fibre ECAL**, fetched `https://arxiv.org/abs/2208.04872`: verbatim *"Each cell is read out at both ends by photomultipliers. The energy deposits are obtained from signal amplitudes, the arrival times of particles and their position along the fibres are determined from the signals at the two ends."*; *"The cluster spatial resolution is sigma_par = 1.4 cm/sqrt(E (GeV)) along the fibres and sigma_perp = 1.3 cm in the orthogonal direction"*; sigma_t = 54 ps/sqrt(E) (+) 100 ps.
- ATLAS Tile (two PMTs per cell) and SDHCAL: no document found where the inter-PMT Delta t is used as a geometric estimator — **no quantified number found**.
- Feasibility for us: the geometric half does not transfer (our two samples are at different depths along the same axis, not two ends of one bar, so Delta t has no lever arm against a transverse or longitudinal coordinate — confirmed by 0(e)). The *averaging* half transfers exactly and is item 5's recommendation.

## 5. Time-resolution scaling and weighting

**arXiv:2205.02500** (W + GAGG SpaCal for LHCb; NIM A 1045 (2023) 167629), fetched `https://ar5iv.labs.arxiv.org/html/2205.02500`. This is the paper behind the "18.5 ps at 5 GeV" claim, and reading it changes what the number means.
- Verbatim: *"The time resolution reaches down to (18.5 +- 0.2) ps at 5 GeV using as timestamp the weighted average of the timestamps of the 2 cells and as weights the inverse of their variance at that energy, i.e. 1/sigma_t^2(E)."*
- Fig. 6 caption, verbatim: *"The three set of resolutions are obtained using the front cell, the back one, and the inverse-variance weighted average of the two timestamps."*
- Geometry, verbatim: *"The time measurements were carried out instrumenting 1 front and 1 back cell"*; *"Each section is divided into 9 cells of 15x15 mm2."* Reference detector subtracted quadratically, *"The typical resolution was 14 ps."*
- Coupling dependence, verbatim: *"With the light guides, the resolution is degraded by a factor 1.5 to 2 ... being close to 30 ps at 5 GeV."*
- So 18.5 ps is **not** a per-cell number: it is the inverse-variance combination of a front and a back cell, reference-subtracted, best-case dry optical contact, at 5 GeV *beam* energy. Front-alone and back-alone values are in the figure only: **no quantified number found**. No fitted a/b functional form is given: **no quantified functional form found**.

**PicoCal design targets.** PoS(ICHEP2024)970, "Scintillating sampling ECAL technology for the LHCb PicoCal" (C. Zhang), read directly. Verbatim: *"several design modifications will be implemented, including a high-granularity structure with integrated timing information to address pile-up challenges."*; *"The inner region of the ECAL will be equipped with SpaCal modules featuring double-sided readout."*; *"After LS4, both the inner and outer regions of the ECAL will be capable of measuring time information, and it is likely that both will incorporate double-sided readout."* Table 2 column E_20ps, footnote verbatim *"Energy above which time resolution is smaller than 20 ps"*: **5 GeV** (W+garnet crystal), **40 GeV** (W+polystyrene), **20 GeV** (Pb+polystyrene), **40 GeV** (Shashlik). The 20 ps specification therefore applies only above 5-40 GeV of *cluster* energy; nothing is published at the MeV per-cell scale.
Front-end: arXiv:2512.17355 (SPIDER), verbatim *"covering the range between 2 and 20 GS/s"* and *"a time resolution below 15 ps rms above 5 GeV"*. No TDC LSB stated; LHCb-PUB-2025-002 and the CDS PicoCal records give no functional form: **no quantified functional form found**.

**CMS HGCAL, the standard cell-level form.** Fetched `https://ar5iv.labs.arxiv.org/html/2005.13324`, verbatim: *"sigma_t = sigma_noise (+) sigma_floor, where sigma_noise = A/(S/N), with a noise term A of 1.5 ns/fC, a constant term sigma_floor of 20 ps."* Single-cell test board: *"resulted in a constant term of 50 ps"*. Cluster level: *"clusters with pT > 5 GeV should have a timing resolution better than 30 ps."*

**Are our numbers sane? Yes, and the shape tells us something.** Log-log slope of our five sigma_t(E) points is **-0.45** overall; the interior closed bins 65 -> 650 MeV give **-0.45 and -0.44** (the two edge bins, -0.23 and -0.78, are bin-centre artefacts). That is 1/sqrt(E), i.e. **photostatistics-limited**, not noise-limited: HGCAL's noise term goes as 1/(S/N), slope -1. Nor is there a digitisation plateau — a fixed TDC term would flatten the high-energy end and ours is still falling at 3 GeV. A uniform quantiser giving 756 ps would need w = sqrt(12)*756 ps = 2.6 ns, unrelated to the 25 ns clock (25/sqrt(12) = 7.2 ns). Against the published module numbers, our 38 ps per cell above 3 GeV of deposit sits reasonably beside 18.5 ps (two cells combined, dry contact, 5 GeV beam) and 30 ps (same with light guides), and comfortably beside the 5 GeV E_20ps specification.
Consequence worth stating: because the low-energy end is photostatistics-limited and not floored by a quantiser, resolution weighting is the *correct* thing to do there — our sigma_t(E) model is well founded. The pull feature failed for the reason in 0(b), not because the weights were wrong.

**Resolution weighting is established practice, and 2205.02500 is a direct precedent on our own detector** (quote above, weights 1/sigma_t^2(E)). CMS does the same: `https://indico.cern.ch/event/1339557/papers/5917851/files/13623-CALOR_HGCALtime_review.pdf`, verbatim *"The layer cluster time is computed as the resolution-weighted average of the hits time."* and *"The trackster ToA is computed based on a time-resolution weighted average."* Numerical gain versus energy weighting or a plain average: **no quantified number found** anywhere.

## 6. The honest bound

**No published work demonstrates useful in-time (same bunch crossing) pileup rejection at sigma_t / Delta_t > 1.**

- Every demonstrated gain sits at ratio **0.13-0.17**: CMS MTD 30 ps against a 180-200 ps beamspot time spread (arXiv:1810.00860); ATLAS HGTD 30 ps per track (arXiv:1809.02181); arXiv:2603.10762 (Cartiglia, *A Brief History of Timing*), verbatim *"The ALTIROC ASIC (130 nm CMOS) provides a target time resolution of ~35 ps per layer, read out with two sensor layers per disk for a total per-track resolution of ~25 ps after combination."*
- The only resolution scan that exists is arXiv:1809.02181 Fig. 7(b), pileup-jet rejection *"for different time resolution (sigma(t)) values"*; the per-point numbers live only in the figure and its source (HGTD Technical Proposal, CDS) is unfetchable. **No published scan extends past ~100 ps.**
- No source states an explicit sigma_t/Delta_t limit theorem: **no quantified number found**. A statistical soft weight degrades smoothly, it does not switch off — so this is a strong empirical absence, not an impossibility proof.
- The closest published approach to ratio 1 is LHCb's own pessimistic arm: PoS(LHCP2024)301, verbatim *"Time resolutions on the K*0 decay vertex of 20 and 100 ps are assumed."* against a derived LHCb vertex-time spread of ~130 ps (from arXiv:1110.2866, verbatim *"values observed for the transverse beam sizes are close to 50 um and 55 mm for the bunch length"*, with sigma_z ~ 35-40 mm giving sigma_t = 117-133 ps). Ratio ~0.8, and it publishes **no quantified benefit** — only a mass-spectrum figure.
- **LHCb's own documents do not credit calorimeter timing with in-time vertex separation.** PoS(LHCP2024)301, verbatim: *"the improved spatial and angular resolutions must be complemented with accurate time information on the arrival time of particles to further suppress background from pile-up interactions. Time information provided by the tracking detectors will also help the association of electromagnetic clusters to charged-particle tracks and primary vertices."* — in-time association is attributed to granularity plus **tracking** time. arXiv:2603.10762 Table 6 lists the PicoCal timing role as *"~20 ps per cluster ... Shower separation"*, and *"The goal is to separate overlapping electromagnetic showers in the high-occupancy central region"*. arXiv:2504.03088 says only *"Timing capabilities of order 20 ps precision are crucial for pile-up mitigation in the PicoCal."* without distinguishing in-time from out-of-time.
- **Where calorimeter timing demonstrably works is out-of-time, where Delta_t = 25 ns.** arXiv:2310.16497, Eur. Phys. J. C 84 (2024) 455, fetched: a *"cell time within +-12.5 ns"* cut *"reduce[s] the out-of-time pile-up jet multiplicity by ~50% for jet pT ~20 GeV and by ~80% for jet pT >= 50 GeV ... improving the jet energy resolution by up to 5% for 20 < pT < 30 GeV ... reducing the overall event size on disk by about 6%"*. The same paper states the failure mode verbatim: *"At lower energy significance the time resolution is poorer, and the secondary peaks cannot in general be distinguished."* — which is why the cut is gated at significance > 4. That is precisely our situation one level down.
- **The escape route the literature does point at is aggregation, not per-cell resolution.** "Cluster time measurement with CEPC calorimeter" (identified via `https://arxiv.org/html/2209.02932`, quoted with that caveat): *"the time resolution of muons ... is nearly 1/5 the intrinsic time resolution of an individual one MIP hit"*. Our own section 0(a) confirms the same lever on our data: 38 ps seed -> 21-28 ps combined, i.e. sigma_t/Delta_t = 0.11 at cluster level against the 0.343 ns truth arrival-time spread. Any timing claim we make must be defended on cluster-aggregated sigma_eff, never on per-cell sigma_t — and, per the 0(a) caveat, with the cell-time correlation measured rather than assumed, since HGCAL's rho ~ 0.8 within a module shows sqrt(N) is not guaranteed.

## 7. Ranked implementable ideas

**1. Inverse-variance combination of the front and back timestamps into one cell time.** Inputs: `cell_times_front`, `cell_times_back`, `cell_energies_front`, `cell_energies_back`, the existing sigma_t(E) model, and the fixed -1.234 ns front-back offset from 0(e). Precedent: arXiv:2205.02500 does exactly this on this detector and its headline 18.5 ps is the combined number. Measured effect on our data: per-cell sigma_t x **0.82**, plus one fewer missing-value channel (a cell with only one valid stamp still yields a time). Expected effect on energy resolution: **unmeasured, and there is no basis for an estimate** — the only defensible claim is 18% better per-cell time precision. Cheapest thing on this list and the only one with a same-detector precedent.

**2. Absolute cluster time and its dispersion, as out-of-time contamination features.** Inputs: absolute `cell_times_front/back` (i.e. *stop* subtracting the per-event median), plus the inverse-variance window t0, the pull-weighted RMS about it, and the fraction of window energy later than a fixed threshold. Precedent: arXiv:2203.01317's two-bin late-energy fraction at 1 ns smearing, worth **3-4%** relative resolution, and ATLAS's +-12.5 ns cell-time cut, worth **50-80%** out-of-time jet-multiplicity reduction and up to **5%** jet energy resolution. Evidence it applies to us: 0(d) — absolute t0 has AUC **0.606** on its own against min-bias, a **0.310 ns** median shift, and a tail that flags **6.0%** of min-bias events above the clean 99th percentile at 6x enrichment (1.2% at 12x above the clean 99.9th). Modest but real, and currently thrown away entirely because the pipeline subtracts a per-event median before the network sees anything. This is the one to run.

**Two controls this idea must pass, or the gain will be misread.**
(i) corr(absolute t0, log E_true) = **-0.47 on the clean sample**, where there is no pileup. That is detector time-walk, and it is a partly-direct channel to the label. So train with absolute t0 on **clean only** first: if the metric improves there too, the gain is walk calibration, not contamination rejection — still real, but a different claim, and it must be reported as such.
(ii) Because clean and min-bias are separate samples, any min-bias-only gain must be cross-checked once a paired overlay sample exists.

**3. Pairwise time-consistency pull as a soft per-cell weight, alpha = 2.** Inputs: cell times, sigma_t(E), no vertex. Precedent: CERN-THESIS-2023-175 eq. 6.5, misassigned times **27.38% -> 12.71%**. Formalises what we already tried, with the reference now taken as the 21-28 ps inverse-variance t0 rather than a top-decile mean. Honest expectation: against in-time pileup, nothing — 0(b) caps the per-cell leverage at 0.26-0.76 sigma. Against the multi-ns tails of 0(d) it will fire hard, which makes it a redundant, more expensive version of idea 2. Run only if idea 2 works and you want per-cell attribution rather than a cluster-level flag.

Nothing else survives. The list stops at three.

## 8. Verdict

**Timing has one thing left to give, and it is not what we were looking for.**

The 20% we already have is real and the survey explains where it comes from: our sigma_t(E) is photostatistics-limited with a clean 1/sqrt(E) slope (-0.45 measured), our per-cell numbers sit sanely beside the published SpaCal module numbers once you correct for the fact that 18.5 ps is a two-cell combination at 5 GeV beam energy and not a per-cell figure, and resolution weighting is the right formalism — the same one LHCb's own SpaCal paper and CMS HGCAL both use.

But the specific thing we were chasing — using per-cell time to identify in-time pileup — is quantitatively closed. The physical separation is 200 ps; the cells that hold the pileup have sigma_t of 263-1500 ps under pileup; the per-cell significance is 0.13-0.64 sigma; and the in-time signal is largely a *common* shift of the whole cluster that a per-cell pull cannot see by construction. No published method operates at sigma_t/Delta_t > 1, no resolution scan in the literature even extends past 100 ps, LHCb's own documents attribute in-time vertex association to the tracker and give the calorimeter timing the job of shower separation and spillover, and ATLAS states in print that at low energy significance the timing peaks cannot be distinguished. Two independent negative results of ours are therefore expected, not surprising, and a third attempt of the same shape will also return nothing.

What is open is out-of-time, and it is a modest opening, not a large one. Our absolute window time has a multi-ns late tail under pileup that clean events do not show: AUC 0.606 standalone, 6% of min-bias events flagged at 6x enrichment above the clean 99th percentile. It sits unused because the pipeline subtracts the per-event median before the network sees anything. The published analogue at our resolution (CALICE, 1 ns smearing, two time bins) is worth 3-4% relative resolution — quote the primary, not the Snowmass white paper's 10-15%. Expect that order: single-digit percent on the aggregate, most likely concentrated in the pileup-dominated inner low-energy bins where the contamination lives, and only after the clean-only control rules out the walk leak. That is worth one experiment. It is not worth a fourth in-time attempt.

---

### Reproduction

Own numbers from `/tmp/.../scratchpad/{tcalc,tfb,t0abs}.py` against `.scratch/cache/clean-aux_100.pkl` and `minbias_94.pkl`; sigma_t(E) from `SIGT`/`SIGT_E` in `scripts/picocal_data.py`; the median-subtraction that discards absolute time is `picocal_data.py` lines 124-129.

### Documents that could not be read

CERN-LHCC-2021-012 / LHCB-TDR-023 (Framework TDR for LHCb Upgrade II), CMS-TDR-020 (MTD TDR), the ATLAS HGTD TDR and TP, the TICL performance note, and the CDS PicoCal records: CDS serves an Anubis proof-of-work wall. TDR-internal studies — in particular the HGTD time-resolution scan, the TICL pileup-rejection efficiency, and any PicoCal in-time-versus-out-of-time breakdown — remain **unverified rather than absent**, and must not be reported as absence of evidence. Note also that **no PicoCal TDR exists yet**; the PicoCal design numbers available are conference proceedings and public notes only.
