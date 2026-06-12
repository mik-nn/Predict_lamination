# Predict Lamination

Predict laminated spectra from unlaminated spectra using 4 paired CGATS datasets (R2/R3 substrates, P9000 printer).

## Goal Achieved 🎯

**1 i1 strip (25 sequential patches) + Frozen ResNet + Ridge(5→5) residual adapter → P95 ΔE00 ≤ 1.8 on any substrate.**

## Key Result: Transfer Learning

| Target | Frozen ResNet | + 25 anchors (1 strip) | + 64 anchors (2 strips) | Target (P95 ≤ 2.0) |
|--------|:-----------:|:---------------------:|:---------------------:|:----------------:|
| R2_11-4-23 (reference) | 1.48 | **1.45** | **1.43** | ✅ |
| R2_27-10-23 | 1.88 | **1.70** | **1.61** | ✅ |
| R2_13-02-24 | 2.16 | **1.78** | **1.73** | ✅ |
| R3_23-4-24 (diff substrate) | 2.27 | **1.68** | **1.64** | ✅ |

### Production Protocol

1. **Pre-train**: ResNet (72→64×3→32→5) on reference material with full U+L measurement
2. **Deploy**: measure unlaminated (full chart) + laminated (1 strip = ~25 sequential patches)
3. **Adapt**: fit Ridge(5→5, λ=0.01) on residual = c_actual − c_frozen for anchor patches
4. **Predict**: c = c_frozen + residual_prediction → L = U + V·c → DeviceLink

**Why it works**: The 5→5 residual learns only the *substrate shift* (25 params from 25 anchors = well-regularized), while U→c mapping stays frozen. Direct 32→5 or 37→5 adapters overfit.

## Baselines

| Model | P95 | Notes |
|---|---|---|
| **Frozen ResNet + Ridge 5→5 (25 anchors)** | **1.45–1.78** | Best — works on all substrates |
| Frozen ResNet (no anchors) | 1.48–2.27 | R2 same-substrate P95=1.48–1.88 |
| OLS (U+U², rank-5) all data | 2.03–2.31 | Upper bound without ML |
| ResNet per-dataset (80/20) | 1.77–2.16 | Beats OLS on tail metrics |
| D-optimal + OLS (N=200) | 2.6–3.6 | Anchor selection doesn't help much |
| PCA + OLS / RF / GPR / AE | 3–22 | All rejected |

## Data

- 4 paired CGATS files: `R2_11-4-23` (1617 patches), `R2_27-10-23`, `R2_13-02-24`, `R3_23-4-24` (1485 each)
- Unlaminated + laminated pairs per patch
- SVD of Δ = L − U: σ₁ = 19.99 (92.16% of variance), rank-5 captures ~99%

## Pipeline

1. CGATS parse → CMYK-matched unlaminated/laminated pairs
2. Global SVD of Δ = L − U → basis vectors V_k
3. Encode: U → c-values (c = V^T · Δ)
4. Pre-train: ResNet(U+U² → c) on reference data
5. Deploy: ResNet → frozen c, Ridge(5→5) → residual Δc
6. Reconstruct: L' = U + V · (c_frozen + Δc)
7. Evaluate: spectral→XYZ→Lab→ΔE00

## Project Structure

```
scripts/
├── resnet-transfer.ts      Transfer learning: ResNet + Ridge adapter (FINAL)
├── resnet-test.ts          ResNet per-dataset (beats OLS on P95)
├── resnet-filtered-cross.ts Outlier filter + cross-dataset ResNet
├── honest-eval.ts          OLS 80/20 evaluation (baseline)
├── dopt-rf.ts              D-optimal anchor selection + Random Forest
├── gpr-test.ts             Gaussian Process Regression
├── autoencoder-test.ts     Autoencoder attempt
├── svd-analysis.ts         SVD decomposition analysis
├── tail-analysis.ts        Worst-patch identification
├── model-comparison.ts     All predictors benchmark
├── filter-dark.ts          Dark-patch filtering test
├── sg-exclude-combined.ts  Savitzky-Golay smoothing test
├── neugebauer-*.ts         Neugebauer model tests
├── yule-nielsen-check.ts   Yule-Nielsen model
└── cross-validation.ts     Cross-dataset evaluation
src/
├── cgats-parser.ts         CGATS.17 parser
├── color-math.ts           Spectral→XYZ→Lab→ΔE00
├── types.ts                TypeScript types
└── anchor-strategies/      Anchor selection strategies
```

## GPU

RTX 3080 16GB, CUDA 13.2 available but `@tensorflow/tfjs-node-gpu` has no pre-built binary for Node 24 (NAPI v8). Use `tfjs-node` CPU or downgrade to Node 22.

## Links

- [Knowledge Base](Docs/knowledge-base.md) — Cross-substrate transfer findings
- [Structure](Docs/structure.md) — Full project layout
