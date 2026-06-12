# knowledge-base.md — Cross-substrate transfer findings

> Append-only knowledge accrued from running predictors and anchor strategies on
> real P9000 RGB data. Each entry links to the EXPERIMENTS row(s) that produced it.
> When you discover a new fact, add a section here BEFORE the next commit.

---

## 1. Optical Brightening Agents (OBA / FWA)

### 1.1 Physics summary

OBA = fluorescent dye in modern papers. Absorbs UV (≤ ~410 nm), re-emits in
blue (~430–450 nm). Under D50 / M0 measurement (illuminant contains UV), the
spectrophotometer counts both reflected light and fluorescent emission, so
measured `R(440 nm) > 1.0` is physically possible and indicates strong OBA.

### 1.2 Per-substrate OBA range in our dataset

Empirical dump over 8 P9000 substrates (EXPERIMENTS row "OBA range across
8 P9000 substrates", 2026-05-23):

| Substrate | R(380) | R(440) | OBA score = R(440)/R(550) |
|---|---|---|---|
| DecorMatte CanvasMatte | 0.117 | 1.089 | **1.20** (heavy OBA) |
| 17MGloss CanvasSatin | 0.193 | 1.097 | ~1.18 |
| 600MT WCRW | 0.311 | 1.029 | ~1.11 |
| Crystalline CanvasSatin | 0.392 | 0.904 | ~1.02 |
| VibranceGloss PGPP | 0.620 | 0.942 | ~1.10 |
| PhotoPeelGloss PGPP | 0.635 | 0.867 | ~1.04 |
| Lyve CanvasMatte | 0.663 | 0.888 | **1.02** (low OBA) |
| PuraSmooth WCRW | 0.819 | 0.865 | **0.99** (no OBA) |

Key observation: **substrate-class name does NOT predict OBA loading.**
DecorMatte (heavy OBA) and Lyve (no OBA) are both labelled CanvasMatte.

### 1.3 Per-ink UV absorption

| Ink | UV absorption (380–410 nm) | Effect on OBA fluorescence |
|---|---|---|
| Cyan (C) | low — mostly transparent to UV | minimal — does NOT shut OBA off |
| Magenta (M) | moderate (absorbs into 400–450 nm) | partial OBA shut-off |
| Yellow (Y) | **high** — strongest UV absorber of CMY | **strongest OBA shut-off** |
| Black (K) | total | total OBA shut-off |

This is *the* reason a cyan ramp fails to capture OBA: along a cyan ramp,
UV throughput stays high at every coverage level, so the OBA emission at
440 nm stays roughly constant → the spectrum at short λ barely changes
→ per-λ curve at 380 nm is undetermined. A yellow ramp would be the
opposite: every coverage level changes OBA emission, so the per-λ curve
at 380 nm is well-sampled.

### 1.4 Linearity claim

**OBA shut-off is approximately linear in UV-absorbing ink coverage** within
single-ink regimes. For a mixture `(c, m, y, k)` of coverages:

```
total_UV_absorbed ≈ c·α_UV(C) + m·α_UV(M) + y·α_UV(Y) + k·α_UV(K)
fluorescent_emission(λ) ≈ baseline_emission(λ) · max(0, 1 − total_UV_absorbed)
```

Linear superposition holds for **physically thin** ink layers; non-linear
saturation appears when total ink exceeds the substrate's transport limit
(separately tracked by the ink-limit machinery).

### 1.5 What a predictor must know to handle OBA correctly

Per-λ predictors that only see `A(λ) → B(λ)` (e.g. C7) **cannot** correctly
handle OBA when two patches with the same `A(λ)` have different ink mixtures.
Concrete failure mode: at 380 nm, paper has `A = 0.117` (DecorMatte); a patch
with `(c=0.5, m=0, y=0)` and a patch with `(c=0, m=0, y=0.3)` may both end
up at `A(380) ≈ 0.10` on the reference substrate but have very different
OBA-killing on the target — C7 predicts the same `B(380)` for both → wrong.

**Two architectural paths address this**:

#### Path 1: Per-ink coverage covariate predictor (`C9` — planned)

Add ink coverage as an explicit covariate:

```
B(λ, RGB) ≈ a(λ) · A(λ, RGB) + b(λ) + Σ_ink γ_λ(ink) · coverage(ink, RGB)
```

For 3 RGB→CMY channels: `2 + 3 = 5` free params per λ. Needs ≥ 5 anchors
spanning coverage of each channel for OLS to be well-conditioned.

#### Path 2: Pre-processing OBA separation (`D7` — planned, PREFERRED) ★

**OBA can be extracted analytically from the paper spectrum alone, with no
extra measurement.** The fluorescence bump at 420–450 nm sits on top of the
substrate's smooth base reflectance. Algorithm:

1. **Estimate substrate_base shape** by fitting a smooth (polynomial / smoothing
   spline) curve to `R_paper(λ)` over the OBA-free range, λ ∈ [460, 730].
2. **Extrapolate substrate_base back to λ ∈ [380, 450]**.
3. **Per-λ in the OBA band**:
   `OBA_emission(λ) = max(0, R_paper(λ) − substrate_base(λ))`.

Per-patch OBA scaling (two options, both work without extra anchors):

- **(a) UV-block proxy from existing spectra**:
  `UV_block(patch) = max(0, 1 − R_patch(380) / R_paper(380))`
  → `OBA_factor(patch) = max(0, 1 − UV_block(patch))`
- **(b) 1-anchor calibration**: one yellow solid on B empirically pins
  the maximum-block coefficient; interpolate linearly for other patches.

Subtract from EVERY measured patch on both substrates:

```
R_clean(patch, λ) = R_measured(patch, λ) − OBA_factor(patch) · OBA_emission(λ)
```

Run ANY predictor on `R_clean_A` → `R_clean_B`. At the end, ADD OBA back to
the prediction using the target's `OBA_emission_B` and the same `OBA_factor`
(which depends on the patch's RGB, not on the substrate):

```
R_pred(patch, λ) = R_pred_clean(patch, λ) + OBA_factor(patch) · OBA_emission_B(λ)
```

**Why this is preferred**: structurally removes OBA from the cross-substrate
problem, so even simple predictors (A3 / C7) work on clean spectra. The
catastrophic S3-cyan failure mode (k=5, median 9.28) is expected to disappear
because the cyan ramp's R_clean at 380 nm DOES vary with cyan coverage once
the OBA bump is gone. C9's 5-params-per-λ overhead is also avoided.

**Designated as predictor wrapper candidate `D7` — not yet implemented.**
The user observed (2026-05-24) that since both substrates are fully measured
in the research dataset, OBA can be computed by analysis alone — no extra
field needed.

### 1.6 OBA-mismatch impact on existing predictors

Measured on DecorMatte ↔ Lyve (OBA mismatch 0.179):

| Predictor | Median ΔE00 | Notes |
|---|---|---|
| A3 (per-λ affine) | 1.54 | per-λ slope absorbs first-order OBA scaling |
| D1 (paper-ratio + PCA residual) | 1.45 | clamp activates at 380/390 nm |
| B3 (pool-PCA, 7-profile pool) | 2.17 | diagonal score map underspecified |
| C7 (per-λ monotone curve) | 1.28 | best of the four — captures OBA-driven curvature |
| **C7 + S3 neutral (k=5)** | **1.06** | substrate transform shared across inks WHEN ramp visits all reflectance levels |
| C7 + S3 cyan (k=5) | 9.28 | catastrophic — cyan transparent at OBA bands |

---

## 2. Predictors (current and planned)

### 2.1 Implemented

| ID | Name | Params | Strengths | Weaknesses |
|---|---|---|---|---|
| **A3** | Per-λ affine `B = a(λ)·A + b(λ)` | 72 (2·L) | trivial closed-form OLS, robust baseline | linear-only, no per-ink coverage info |
| **D1** | Paper-ratio + PCA residual | 36 + p·k | graceful at small k, paper anchor cheap | ratio blows up on OBA-disparate pairs (clamped to [0.3, 3.0]); single-paper anchor limits first-order |
| **B3** | Pool-PCA + per-PC diagonal map | 12 (2p) | exploits cross-substrate pool basis | needs ≥ 5 pool profiles; diagonal can't capture cross-PC coupling; underperforms on small pools |
| **C7** | Per-λ piecewise-linear monotone curve | up to k·L | captures non-linear saturation curves per λ; beats A3/D1 at same k | sees only `A(λ)`, no ink-mix info → OBA-blind in non-neutral anchor sets |

### 2.2 Planned

| ID | Name | Why interesting |
|---|---|---|
| **C9** | Per-λ affine + per-ink coverage covariate | First predictor with explicit per-ink awareness; needed for OBA on non-neutral anchor sets (§1.5) |
| **B3-fullM** | Pool-PCA + full M (not diag) score map | Adds cross-PC coupling at 6× param cost; expected to fix B3's saturated-patch outliers |
| **Hybrid D1+B3** | Paper-ratio first-order + pool-PCA residual | Combines D1's cheap paper anchor with B3's basis structure |

---

## 3. Anchor strategies

### 3.1 Implemented

| ID | Name | k | When it works |
|---|---|---|---|
| **S1** | Forced heuristic (paper + 8 RGB corners + 5 neutrals) | 13 | Default — broad coverage of the RGB cube |
| **S2** | Greedy adaptive (S1 seed + worst-ΔE patch per iter) | variable | When you have a specific ΔE budget and want minimum k |
| **S3** | Single-channel ramp (paper + N levels on one channel) | 1 + N | Confirms / falsifies "substrate transform shared across inks" hypothesis |

### 3.2 S3 channel-specific behaviour

| Channel | Works for OBA bands? | Why |
|---|---|---|
| neutral (R=G=B) | **YES** | touches all inks proportionally → visits all (A, B) per λ |
| Yellow (Y) | YES (UV-strong) | Yellow absorbs UV → varies A at 380 nm strongly |
| Magenta (M) | partial | Magenta absorbs 400–450 nm partially |
| Cyan (C) | **NO** | Cyan transparent at UV → fails at 380–410 nm |

If you must use a single-channel ramp, pick Yellow or Neutral. Cyan is the
worst possible choice for OBA-disparate pairs.

### 3.3 Planned

- **S4** — Solids-only: paper + 100 % C + 100 % M + 100 % Y + 100 % K
  (k = 5). Spans extreme reflectance at every λ via the strongest
  per-ink absorption. Pairs naturally with C9 predictor.
- **S5** — Hybrid ramp: paper + 4 neutrals + 1–2 "blue boost" patches
  (e.g. 50 % B or dark blue) to anchor curve at OBA bands. Cheap
  workaround for C7 when neutral ramp alone isn't enough.

---

## 4. Open questions / hypothesis pipeline

See `docs/RESEARCH_HYPOTHESIS.md` for falsifiable statements:

- **H4** — k ≤ 15 anchors achieve median ΔE00 ≤ 1.5 on ≥ 22/27 substrate
  pairs via any predictor (data from S1 path).
- **H8** — D1+clamp beats A3 on ≥ 60 % of OBA-mismatched pairs.
- **H9** — Substrate transform shared across inks (S3-neutral works on
  ≥ 60 % of pairs, fails on cyan-only).

All three need the **Phase 7 batch runner** (702 directed pairs through every
predictor × strategy combo) for definitive verdict. Single data points
(currently DecorMatte ↔ Lyve) are anecdotal.

---

## 5. UI affordances available

- Profile-list OBA score in dropdown labels (`profile_name (OBA 1.20)`).
- OBA-mismatch tile (red / yellow / green by severity) above head-to-head.
- D1 clamped-bands count in detail block.
- B3 pool-size + basis-rank in detail block.
- C7 trajectory + greedy-S2 trajectory cards.

Planned (queued):

- UI warning when S3 channel ≠ neutral and OBA mismatch is high.
- Per-λ curve plot for C7 (D3 chart at user-chosen wavelengths).
- Per-λ residual heatmap N×L for any predictor.

---

## 6. Lamination prediction (Predict_lamination repo)

### 6.1 Problem

Predict laminated spectra `L(λ)` from unlaminated `U(λ)` for the same CMYK patch. Four paired datasets (3× R2 substrate, 1× R3) from P9000 printer. Target: minimum number of laminated anchors for median ΔE00 ≤ 1.0, P95 ≤ 2.0.

### 6.2 Best model: OLS rank-5 with U+U²

- Encode Δ = L − U via global SVD (all 4 datasets pooled)
- Rank-5 basis covers ~99% of variance
- Features: `[U(λ), U(λ)²]` (72 dims) → predict 5 c-values → reconstruct L
- Per-dataset 80/20 evaluation: median 0.50–0.54, P95 2.1–2.5

### 6.3 ResNet beats OLS on tail metrics

**Architecture**: 72 → 64(BN+ReLU+DO)×3 → 32(BN+ReLU+DO) → 5; Adam 1e-3, batch 64, 200 epochs.

First model to outperform OLS on P95/P99/max (9–63% improvement), though median is 0.58–0.61 (slightly higher than OLS).

| Metric | OLS | ResNet | Improvement |
|---|---|---|---|
| Median | 0.50–0.54 | 0.58–0.61 | −8–15% (worse) |
| P95 | 2.1–2.5 | 1.77–2.16 | 9–19% better |
| P99 | 3.8–5.0 | 2.69–3.56 | 25–42% better |
| Max | 4.9–10.1 | 3.5–4.8 | 30–63% better |

### 6.4 Rejected models

| Approach | Median | P95 | Failure mode |
|---|---|---|---|
| Autoencoder (36→5→36 + OLS) | 4–5 | 17–20 | Latent doesn't preserve spectral info |
| PCA scores → OLS | 0.8–1.0 | 3.0–3.7 | PCA decorrelates but doesn't help OLS |
| Random Forest (50 trees, d=5) | 2.5–3.4 | 16–22 | Overfits / can't extrapolate |
| Gaussian Process (RBF, 300 pts) | ~1.0 | ~3.3 | ~10 min/rep, worse than OLS |
| Neugebauer / Yule-Nielsen | — | — | Per-ink models underfit |
| Savitzky-Golay / 3-NN denoising | — | — | No improvement |

### 6.5 D-optimal anchor selection

Greedy D-optimal (Sherman-Morrison update, Ridge λ=1e-6):
- For N ≤ 50 anchors: 1.5–2.8× better P95 than random selection
- At N ≥ 100: random catches up
- At N = 200: P95 ≈ 2.6–3.6 (vs all-data P95 2.0–2.3)
- At N = 500: P95 ≈ 2.5–3.1

### 6.6 Known issues

- Cross-dataset R2→R3 degraded (P95 3.2–3.7) — different paper substrate
- Outlier filtering (OLS ΔE > mean+3σ) removes ~2% patches but doesn't improve P95
- GPU unavailable — `@tensorflow/tfjs-node-gpu` has no pre-built binary for Node 24
