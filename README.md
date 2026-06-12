# Predict Lamination

Predict laminated spectra from unlaminated spectra using 4 paired CGATS datasets (R2/R3 substrates, P9000 printer).

## Goal

Find the minimum number of laminated patches needed to accurately predict laminated ΔE00, targeting median ≤ 1.0 and P95 ≤ 2.0.

## Key Results

| Model | Dataset | Median ΔE00 | P95 | P99 | Max |
|---|---|---|---|---|---|
| OLS (rank-5, U+U²) | per-dataset | 0.50–0.54 | 2.1–2.5 | 3.8–5.0 | 4.9–10.1 |
| **ResNet** (64×3+32, 200ep) | **per-dataset** | **0.58–0.61** | **1.77–2.16** | **2.69–3.56** | **3.5–4.8** |
| D-optimal + OLS (N=200) | per-dataset | — | 2.6–3.6 | — | — |
| OLS (rank-5, U+U²) | cross-dataset R2→R2 | — | 2.7–3.6 | — | — |
| OLS (rank-5, U+U²) | cross-dataset R2→R3 | — | 3.2–3.7 | — | — |

**ResNet is the first model to beat OLS on tail metrics** (P95/P99/max) by 9–19% / 25–42% / 30–63%, though median is slightly higher.

## Rejected Approaches

- Autoencoder (36→5→36 latent → OLS): median 4–5, P95 17–20
- PCA scores → OLS: worse than raw U+U² at every rank
- Random Forest (50 trees, depth 5): median 2.5–3.4, P95 16–22
- Gaussian Process (RBF, 300 inducing pts): median ~1.0, P95 ~3.3, ~10 min/rep
- Interpolation (primaries, farthest-point, CMYK grid)
- Neugebauer per-ink n, Savitzky-Golay, outlier exclusion, 3-NN denoising

## Data

- 4 paired CGATS files: `R2_11-4-23` (1617 patches), `R2_27-10-23`, `R2_13-02-24`, `R3_23-4-24` (1485 each)
- Unlaminated + laminated pairs per patch
- SVD of Δ = L − U: σ₁ = 19.99 (92.16% of variance), rank-5 captures ~99%

## Pipeline

1. CGATS parse → CMYK-matched unlaminated/laminated pairs
2. Global SVD of Δ = L − U → basis vectors V_k
3. Encode: U → c-values via OLS (c = V^T · Δ)
4. Predict: U → c' (OLS or ResNet)
5. Reconstruct: L' = U + V · c'
6. Evaluate: spectral→XYZ→Lab→ΔE00

## Project Structure

```
scripts/
├── honest-eval.ts          OLS 80/20 evaluation (baseline)
├── resnet-test.ts          ResNet per-dataset (beats OLS on P95)
├── resnet-filtered-cross.ts Outlier filter + cross-dataset ResNet
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

RTX 3080 16GB, CUDA 13.2 available but `@tensorflow/tfjs-node-gpu` has no pre-built binary for Node 24 (NAPI v8).

## Links

- [Knowledge Base](Docs/knowledge-base.md) — Cross-substrate transfer findings
- [Structure](Docs/structure.md) — Full project layout
