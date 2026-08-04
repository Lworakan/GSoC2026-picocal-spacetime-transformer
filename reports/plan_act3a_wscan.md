# Act 3a + Clean W-scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline — GPU scheduling and resume logic). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Execute the first two experiments of `reports/spec_road_to_0p0235.md`: Act 3a (sigma_eff-direct calibration, inference-only, free) and Act 1 (nb50: clean window scan W=4/6/8 with the quant stack).

**Architecture:** Act 3a extends nb47 (which already loads the five nb44 checkpoints and infers val/test quantiles) with a direct-metric calibration fit. The W-scan is a new notebook nb50 cloned from the proven generator pattern, importing `scripts/picocal_data.py`, training SubNetFQ (ng=5) per window on clean only.

**Tech Stack:** torch 2.6 (conda `LCHb-lab`), nbformat generators in `.scratch/`, scipy Nelder-Mead for the 2-parameter fits.

## Global Constraints

- Six-stage intro cell; pre-registered criterion >0.002; 2 seeds minimum; val-only selection; no code comments; smoke (CPU) before full (GPU); delete smoke artifacts before full (checkpoints share names).
- GPU: verify idle/locked clock reads 210 or 900 MHz (never 1665) before every launch. **W=6 (169 tokens) and W=8 (289 tokens) exceed the proven-safe 81-token class** — treat as a new crash-class test: batch 48 for W=6, 24 for W=8, watch the first minutes of the first W=6 job; fall back to `NB50_DEVICE=cpu` for W=8 if anything looks unstable. Plain attention (no pair tensors) is expected safe, but unproven at these sizes.
- Anchors: clean best 0.0397 (MeanResidual, nb19); floors per spectrum in the spec; nb44 record 0.0425.

---

### Task 1: Act 3a — sigma_eff-direct calibration in nb47

**Files:**
- Modify: `.scratch/gen_nb47.py` (append two cells before the writer footer)
- Regenerate + execute: `notebooks/47_Quantile mixture pooling.ipynb` (full, GPU, inference-only ~15 min)

**Interfaces:**
- Consumes: `QV`, `QT` (S×N×3 arrays), `yva = D['y'][kva]`, `Et_te`, `width_binned_calibration` — all already defined in nb47.
- Produces: printed comparison polyfit-vs-direct on val AND test; `reports/predictions/nb47_pred_directcal.npy`.

- [ ] **Step 1: Append the Act 3a cells to gen_nb47.py** (insert before `nb['cells'] = c`):

```python
md("""## Act 3a - sigma_eff-direct calibration (spec section 2)

The final linear calibration is fit by least squares, which optimizes MSE - not our quantile-core metric. Here the same 2 parameters per width group are fit by directly minimizing sigma_eff on the validation set (precedent: Belle II selects by FWHM on validation, arXiv:2306.04179; ATLAS fits scale/smearing on holdout, arXiv:2309.05471). Criterion: adopt if val and test move in the same direction; test reported once.""")

code("""from scipy.optimize import minimize
Ev_va = np.exp(yva)
def direct_wcalib(qv, qt, yva_, Ev):
    wv = qv[:, 2] - qv[:, 0]; wt_ = qt[:, 2] - qt[:, 0]
    cuts = np.quantile(wv, [1/3, 2/3])
    gv = np.digitize(wv, cuts); gt = np.digitize(wt_, cuts)
    pe_t = np.empty(len(qt)); pe_v = np.empty(len(qv))
    for g in range(3):
        mv = gv == g
        a0, b0 = np.polyfit(qv[mv, 1], yva_[mv], 1)
        def obj(p):
            pred = np.exp(p[0] * qv[mv, 1] + p[1])
            return resolution(pred, Ev[mv])['sigma_eff']
        res = minimize(obj, [a0, b0], method='Nelder-Mead',
                       options=dict(xatol=1e-4, fatol=1e-6, maxiter=400))
        a, b = res.x
        pe_v[mv] = np.exp(a * qv[mv, 1] + b)
        pe_t[gt == g] = np.exp(a * qt[gt == g, 1] + b)
    return pe_v, pe_t
pv_ls = np.empty(len(QV[0])); _ = None
pe_v_dir, pe_t_dir = direct_wcalib(qv_mean, qt_mean, yva, Ev_va)
pe_v_ls = width_binned_calibration(qv_mean, qv_mean, yva)
print(f'val : polyfit {resolution(pe_v_ls, Ev_va)["sigma_eff"]:.4f} -> direct {resolution(pe_v_dir, Ev_va)["sigma_eff"]:.4f}')
print(f'test: polyfit {resolution(pe_mean, Et_te)["sigma_eff"]:.4f} -> direct {resolution(pe_t_dir, Et_te)["sigma_eff"]:.4f}')
np.save(OUT / 'nb47_pred_directcal.npy', pe_t_dir)""")
```

Note the val-side least-squares reference uses `width_binned_calibration(qv_mean, qv_mean, yva)` — calibrating val onto itself with the LS fit, the honest same-sample comparator for `pe_v_dir`.

- [ ] **Step 2:** `python .scratch/gen_nb47.py` → expect `wrote ... 9 cells`.
- [ ] **Step 3:** Verify GPU clock (210/900 only), then execute nb47 full: `NB47_MODE=full REPO_DIR=... jupyter nbconvert --to notebook --execute --inplace "47_Quantile mixture pooling.ipynb"` (background, log `reports/nb47.log`).
- [ ] **Step 4:** Read the printed val/test lines. Adopt-or-reject by the stated criterion; record the outcome in the final report (no commit).

### Task 2: nb50 generator — clean window scan

**Files:**
- Create: `.scratch/gen_nb50.py`
- Produces: `notebooks/50_Clean window scan.ipynb`; CSV `reports/predictions/nb50_clean_wscan{TAG}.csv` (columns `W,seed,sigma_eff,elapsed`); preds `nb50_pred{TAG}_W{w}_s{s}.npy`; checkpoints `.scratch/ckpt/nb50_W{w}_s{s}.pt`.

**Interfaces:**
- Consumes: `build_grid, make_windows, splits_for, THRESH, NC` from `scripts/picocal_data.py`; `resolution, PITCH, EPS, split` from `run_experiments`.
- Env: `NB50_MODE` smoke/full, `NB50_DEVICE`.

- [ ] **Step 1: Write gen_nb50.py.** Structure identical to gen_nb46 with these differences (everything else verbatim from the nb46 pattern — data cell, SubNetFQ-style model with `NG=5`, pinball, wcalib, checkpointed train loop, CSV resume):
  - Intro cell (six stages): error analysis = containment floor nb16 corr 0.977 + clean-side gap 0.0397 vs floor 0.0235; hypothesis = wider window on clean recovers containment with no pileup cost; research = window scans only ever ran on minbias (nb32); criterion = any W beats clean-W4 (same notebook, same seeds) by >0.002 overall; anchors = clean 0.0397, floor table.
  - Data: `CE = build_grid(sorted((REPO/'data'/'full').glob('matched_*.root'))[:4 if smoke], 'clean')` — clean ONLY, no minbias.
  - Per-W loop: for `W in (4, 6, 8)`: `rows, keep = make_windows(W, CE)`; `ktr,kva,kte = splits_for(keep, len(CE))`; prep arrays inline (nb46 pattern, `NG=5`, globals `[log1p(sumE), log1p(seedE), log(n), fbr, lat]`); store per-event containment `cont_ev = sumE/(1000*Etrue)` for the verdict diagnostic.
  - `BATCH = {4: 96, 6: 48, 8: 24}[W]` and jobs `[(W, s) for W in (4, 6, 8) for s in (0, 1)]` (smoke: `[(4,0), (6,0)]`, 2 epochs).
  - Verdict cell: per-W mean±spread, 2-seed ens, per-bin table against the floor row `sqrt((0.10/sqrt(E))**2+0.01**2)` at bin medians, and `np.corrcoef(r_ens, cont_ev[kte])[0,1]` per W — widening works only if |corr| drops with W while sigma_eff improves.
- [ ] **Step 2:** Generate + CPU smoke: `NB50_MODE=smoke NB50_DEVICE=cpu ... nbconvert ...`. Expect: 2 smoke jobs complete, verdict prints, corr finite.
- [ ] **Step 3:** Delete smoke artifacts: `rm reports/predictions/nb50_*smoke* .scratch/ckpt/nb50_*.pt`.

### Task 3: nb50 full run (GPU, crash-class caution)

- [ ] **Step 1:** `uptime && nvidia-smi --query-gpu=clocks.sm --format=csv,noheader` — 210/900 only; if 1665 stop and ask for `sudo nvidia-smi -lgc 210,900`.
- [ ] **Step 2:** Launch full in background (log `reports/nb50.log`). Job order is W=4 first (safe class), then W=6, then W=8.
- [ ] **Step 3:** When the first W=6 job starts (CSV shows both W=4 rows), check the machine is alive after ~10 minutes (uptime unchanged, CSV/log growing). If the machine hard-resets during W=6/W=8: record the crash class in memory, relaunch with `NB50_DEVICE=cpu` — every job resumes from its checkpoint.

### Task 4: Verdict, memory, next trigger

- [ ] **Step 1:** Read the verdict cell. Determine: best W*, whether containment correlation dropped, per-bin distance to floors.
- [ ] **Step 2:** Update memory (`nb28-overlay-supervision-path.md`): W-scan outcome + which Act 1b hypothesis the error analysis triggers (containment-corr persists → aux containment head; front/back ratio correlates → compensation weights; wide-W wins but low-E worsens → pyramid).
- [ ] **Step 3:** Report to Worakan: numbers, honest verdict, and the triggered next hypothesis. No commits.

## Self-review

- Spec coverage: Act 3a → Task 1; Act 1/nb49(=nb50 here, numbering shifted past existing nb48) → Tasks 2-3; trigger logic for Act 1b → Task 4. Act 4/2 intentionally out of this plan (next plan).
- Placeholders: none — Task 2 Step 1 references the nb46 pattern by name for unchanged code and specifies every delta exactly.
- Type consistency: `direct_wcalib` consumes the same (qv, qt, yva) shapes `width_binned_calibration` uses; nb50 CSV column `W` matches the verdict-cell reader.
