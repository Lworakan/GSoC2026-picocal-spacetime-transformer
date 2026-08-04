# Research Proposal → ICLR 2027

**Target:** ICLR 2027 main track · **Abstract deadline ~Sep 19, 2026 · Paper deadline Sep 24, 2026** · Conf Apr 2027 (West Coast US)
**Status:** aggressive (~2 months). Fallbacks: ML4PS/EuCAIFCon workshop (now), ICLR 2028 (comfortable).

---

## 1. The reframe — from application to general method

**Current framing (rejected at ICLR):** "A spacetime transformer for LHCb PicoCal photon energy reconstruction beats a BDT."
→ reviewers: *"nice application, no new ML."*

**ICLR framing (the pitch):**
> **Learning on time-stamped point sets contaminated by a time-separable background.**
> We introduce **in-time gated pooling (ITGP)** — a general, permutation-invariant aggregation that learns to down-weight background points using their timestamps — and show it beats attention/Deep-Sets/EFN across multiple domains where signal and background are separable in time.

Calorimeter pileup becomes **one instance** of a general problem, not the whole paper.

### Title candidates
- *In-Time Gated Pooling: Timestamp-Aware Aggregation for Point Sets with Separable Background*
- *Learning on Contaminated Point Sets via Time-Gated Aggregation*

---

## 2. The general problem (formalization)

Input: a set `{(x_i, t_i, z_i)}` — feature `x_i`, timestamp `t_i`, weight/energy `z_i`. The set mixes **signal** points (a coherent time cluster) and **background** points (out-of-time contamination). Task: predict a signal-level target (regression/classification) robust to the background fraction.

Standard set models (Deep Sets, Set Transformer, EFN, ParT) treat all points symmetrically or via feature-only attention — they don't exploit that **background is time-separable**.

---

## 3. Core contribution (method)

1. **In-time gated pooling (ITGP):** pooling weight `w_i = z_i · σ(g(Δt_i, valid_i))` — a learned gate on each point's time relative to the signal anchor, so the aggregation suppresses out-of-time points. (Generalizes EFN's energy-weighted sum.)
2. **Time-aware pairwise bias (optional):** `Δt_ij` in a ParT-style pre-softmax attention bias.
3. **Analysis / theory (the ICLR differentiator):** why gating beats feature-only attention when background is time-separable — an IRC-safety-style argument + a controlled synthetic study where contamination fraction and time-separation are dialed.

---

## 4. Novelty vs literature (from our gap analysis)
- Timing+ML for calorimeters exists — but **waveform-level** or "add a timing feature," **not a general gated-pooling mechanism with analysis**. (HIGH-confidence gap on the general method.)
- Set/point-cloud learning is mature — but **no timestamp-gated aggregation for separable contamination** as a named, analyzed method.

---

## 5. Experiment matrix (the hard part)

| Domain | Dataset (public) | Signal vs time-separable background |
|---|---|---|
| **HEP calorimeter (ours)** | LHCb min-bias *(needs LHCb OK to publish)* | photon vs pileup (out-of-time) |
| **Jet tagging w/ pileup** | JetNet / top-tagging + pileup overlay | jet constituents vs pileup |
| **Event cameras** | DVS/N-Caltech (async timestamped events) | object events vs noise events |
| **LiDAR / point tracking** | nuScenes/KITTI temporal | moving object vs background returns |
| **Synthetic (controlled)** | our own generator | dial contamination % + Δt separation |

**Baselines:** Deep Sets, EFN, Set Transformer, PointNet, ParT, plain-attention. **Ablations:** gate on/off, time feature on/off, loss (MSE/Huber), pooling variants. **Metric:** task metric vs contamination level (the money plot: gap widens with pileup).

Minimum for a credible ICLR paper: **synthetic (full control) + 2 real domains** (calorimeter + one of jets/event-cameras).

---

## 6. What we have vs need
**Have:** ITGP prototype (nb17), calorimeter results, EFN/ParT/Huber infra, containment analysis.
**Need:** (a) synthetic contaminated-set benchmark + generator, (b) ≥1 second real domain, (c) the analysis/theory section, (d) 5+ seeds everywhere, (e) LHCb data-publication permission.

---

## 7. Two-month timeline (aggressive)
```
Wk 1  : synthetic generator + ITGP vs baselines on synthetic (the controlled story)
Wk 2  : 2nd real domain (jets-with-pileup — public, closest to ours)
Wk 3  : analysis/theory + ablations, 5+ seeds
Wk 4-5: write 9 pages (OpenReview), main figures
Wk 6-7: internal review (Felipe/advisor), polish, submit Sep 24
```

## 8. Risks (honest)
- **2 months for multi-domain + theory is high-risk.** Most likely realistic outcome: strong workshop paper now + ICLR 2028 full paper. Treat ICLR 2027 as the stretch goal.
- **LHCb data permission** is a hard blocker for using it as a headline dataset — synthetic + public jets de-risk this.
- Reviewers may want more theory — keep the synthetic study rigorous.

## 9. This week
1. Confirm with Felipe/advisor: pursue ICLR reframe? LHCb data publishable?
2. Build the **synthetic contaminated-set generator** (fastest path to the general-method story).
3. Finish nb18 (containment) — feeds the calorimeter section.
