# GLOSSARY.md — Project Terminology

> Companion to `docs/ONTOLOGY.md`. ONTOLOGY captures the data model; this file
> captures the **vocabulary** — terms, abbreviations, and short definitions used
> across the codebase, experiments, and the eventual article. Add a new entry
> the first time a term is used in a doc or commit.

---

## 1. Color science fundamentals

| Term | Definition |
|---|---|
| **Reflectance R(λ)** | Fraction of incident light a patch returns at wavelength λ. Our spectra are 36 bands, 380–730 nm at 10 nm step (the X-Rite eXact / i1Pro convention). Stored normalised to [0, 1]. |
| **CIE XYZ** | Standardised tristimulus values from spectral integration with a CMF (Colour-Matching Function) and an illuminant. We use the **CIE 1931 2° observer + D50 illuminant** everywhere. |
| **CIE Lab (L\*a\*b\*)** | Perceptually-uniform colour space derived from XYZ via a non-linear lightness curve. `L*` is lightness [0, 100], `a*` is green↔red, `b*` is blue↔yellow. Used for all colour-difference numbers in this project. |
| **ΔE76** | Euclidean distance between two Lab points: √((ΔL)² + (Δa)² + (Δb)²). Legacy metric — easy but doesn't track perception well at high chroma. |
| **ΔE00 (CIEDE2000)** | Modern perceptual colour difference (ISO 11664-6). Adjusts for hue, chroma, and lightness non-linearities. **Default metric** in this project; implemented in `lib/colormath.ts:deltaE00`. |
| **D50** | "Daylight at 5000 K" illuminant. Standard reference white in graphic arts. |
| **2° observer** | The narrow CMF (~2° foveal field). The other common choice is 10° (whole-field). We pin 2° because it matches industry/print-shop convention. |
| **White point (WP)** | The tristimulus of the chosen reference white (paper white in this project, captured from the brightest patch). Used to chromatically adapt Lab; the "paper-relative" mode swaps WP for the substrate's own paper instead of D50. |
| **Gamut** | The set of colours the device can produce. Limited by ink set, substrate, and ink-loading limits. |
| **Patch** | One device value + measurement pair. A "chart" is the ordered patch set printed and measured for a profile. |

---

## 2. Spectral file formats

| Term | Definition |
|---|---|
| **ICC** | Industry standard colour profile (`.icc` extension). Header (128 B) + tag table. We read the spectral data X-Rite embeds in a private tag. |
| **ICM** | Windows variant of ICC (`.icm`). Same byte layout. |
| **CxF / CxF3** | "Colour eXchange Format" (ISO 17972). XML container for spectral and colorimetric measurement data. CxF3 = the M0 spectral version we parse. |
| **ZXML** | Zlib-compressed XML. X-Rite stores CxF3 inside an ICC profile as a `'CxF '`-signed tag whose data-type is `'ZXML'`. `lib/iccTagScanner.ts` skips 12 bytes (4 type + 4 reserved + 4 unknown), then `pako.inflate`s. |
| **CGATS.17** | ASCII tabular format for measurement data: header lines, then `BEGIN_DATA_FORMAT … BEGIN_DATA`. MOAB ICC profiles embed CGATS in the `targ` ICC `text` tag. Parsed by `lib/parsers/cgatsParser.ts`. |
| **M0 / M1 / M2** | ISO 13655 measurement conditions. **M0** = unfiltered tungsten (legacy, common). **M1** = D50 illuminant (UV-included). **M2** = UV-cut (suppresses OBA fluorescence). Our primary dataset is M0; CAE_D7_M1 trains on M1 for OBA-handling comparison. |
| **SAMPLE_ID** | Per-patch identifier. We use `R{row}C{col}P{page}` (e.g. `R5C12P1`). Two profiles built on the same chart share IDs → cross-profile patch matching is exact. |

---

## 3. Hardware and printing context

| Term | Definition |
|---|---|
| **Epson SureColor P9000** | The printer this dataset targets. 10-ink (CMYKOG + light variants), RGB-fronted from an ICC workflow. All current profiles are for this device. |
| **Epson media preset** | The driver-level paper setting (e.g. "Canvas Matte", "Premium Luster", "USFA"). The **canonical print-mode** in our taxonomy; two different vendors building profiles for the same media preset share device ink-laydown rules. |
| **Print mode** | Synonym for Epson media preset in our taxonomy. Mapped via `utils/printMode.ts:canonicalPrintMode`. |
| **Breathing Color (BC)** | First-party canvas/paper vendor. 27 P9000 profiles in the dataset (filename prefix `BC_`). Uses abbreviations for the preset (`CanvasMatte`, `PLPP260`, `WCRW`, …). |
| **MOAB** | Vendor (subsidiary of Legion Paper). 18 P9000 profiles. Uses Epson-ish full names (`Prem Luster`, `Exh Canvas Matte`, `USFA`). |
| **Ink mode (mk / pk)** | **mk** = matte black; **pk** = photo black. Chosen automatically by the driver per media preset. The 905-patch BC chart and the ~2033-patch MOAB chart both encode this in the device RGB; we honour it via filename parsing. |
| **Chart** | The ordered list of device RGB values printed and measured for a profile. BC = 905 patches, integer RGB grid. MOAB = ~2033 patches, near-regular 12-level lattice with fractional RGB. |

---

## 4. Substrate physics

| Term | Definition |
|---|---|
| **OBA (Optical Brightening Agent)** | Fluorescent dye in modern white papers. Absorbs UV (≤ 400 nm), re-emits in blue (~ 440 nm). Under D50 (which contains UV), an OBA-loaded paper's reflectance is depressed at 380–410 nm and **boosted above 1.0** at 440 nm. |
| **OBA score** | `R_paper(440) / R_paper(550)`. > 1.05 = OBA present; ≈ 1.0 = no OBA. The diagnostic in `lib/predict/oba.ts:detectOBA`. |
| **OBA mismatch** | `|score_A − score_B|` between two papers. ≥ 0.10 = the predictors that assume multiplicative substrate (e.g. D1) will hit huge ratios at 380 nm. |
| **OBA emission** | The fluorescent excess of `R_paper` over the substrate base. We extract it analytically (degree-2 polynomial fit on `R(λ)` over 460–730 nm, extrapolated back into 380–450 nm; `lib/predict/obaSeparator.ts:extractOBAEmission`). |
| **Paper-relative** | A colorimetry mode where Lab is computed against the substrate's own paper white instead of D50. Removes the trivial "paper looks different" contribution from cross-substrate ΔE. |
| **Device response** | The substrate-independent half of the print model — ink mixing, dot gain, Yule-Nielsen optics. The hypothesis H11 says two profiles built for the same Epson preset share device response once paper white + OBA are factored out. |

---

## 5. Spectral / colorimetric models

| Term | Definition |
|---|---|
| **Neugebauer primary** | One of the 2ⁿ corner spectra for an n-ink subtractive process: paper white, the n primaries, and all 2-, 3-, …-ink overprints. With 3-ink CMY (K = 0) there are 8 primaries. |
| **Demichel** | Weighting scheme for Neugebauer primaries; `w_v = Π c_i^{c_i} · (1 − c_i)^{(1 − c_i)}` per axis. |
| **YNSN / Yule-Nielsen Spectral Neugebauer** | Optical model that raises each primary's reflectance to the power `1/n` before linear mixing, then back to power `n`. `n ≈ 2` for typical inkjet on paper. Captures multi-band light scatter that linear Neugebauer misses. |
| **CYNSN / Cellular YNSN** | A YNSN variant where the primaries are replaced by an interpolated cell grid for more local accuracy. We implemented CYNSN-2 (paper-anchor cell); the dependent research line (H1) was **withdrawn 2026-05-29**. |
| **Dot gain / ink spreading** | Effective ink coverage as a non-linear function of the requested coverage. `lib/analyzers/spreading.ts` parameterises it as a 1-DOF polynomial per channel. |

---

## 6. Predictors (cross-substrate transfer)

| ID | Name | One-liner |
|---|---|---|
| **A3** | Per-λ affine | Independent `(slope, intercept)` regression per wavelength on anchor pairs. 72 parameters for 36 bands. Robust but no structural assumption. |
| **D1** | Paper-ratio + PCA residual | `B(λ) ≈ r(λ) · A(λ) + residual`, where `r(λ) = B_paper / A_paper` (clamped) and the residual is reconstructed from a low-rank PCA basis fitted to anchor errors. |
| **B1 / B3** | PCA pool transfer | Build the basis from a single reference (B1) or the whole pool (B3). Anchor coefficients map source PC scores into target PC scores via a diagonal (or full) transform. |
| **C7** | Per-λ monotone curve | Piecewise-linear monotone fit per wavelength from anchor pairs `(A(λ), B(λ))`. Beats A3 on smooth substrate transforms. |
| **D7** | OBA-separation wrapper | Pre-clean both reference and target spectra by subtracting analytically-extracted OBA emission, run any base predictor on the clean side, then add the target's emission back to the output. Predictor-agnostic. |
| **CAE_RAW** | Conditional Autoencoder, raw spectra | 16-MK pool, 11/5 train/test split. Substrate encoder + ink encoder + decoder. First neural baseline. |
| **CAE_D7** | CAE on OBA-cleaned spectra | Same architecture trained on D7-cleaned inputs. Median-of-medians 1.66 ΔE00, 40 % of held-out pairs achieve median ≤ 1.5. Current best k = 0 predictor. |
| **CAE_D7_M1** | CAE on M1 measurements | Variant for the UV-included condition. Decisively worse than M0. |

---

## 7. Anchor strategies

| ID | Strategy | What it picks |
|---|---|---|
| **S1** | Heuristic forced (default) | Paper white + 6 RGB corners + black + 5 neutrals (R = G = B) along the diagonal. k = 13. |
| **S2** | Greedy adaptive | Start from S1, iteratively add the worst-residual patch. Trims median modestly, big win on P95. |
| **S3** | Single-channel ramp | Paper + 4 patches along one channel (e.g. neutral, cyan-only). k = 5. Best when neutral; catastrophic on UV-transparent channels. |
| **S4** | Lab-saturation | Paper + high-chroma patches at well-separated hue angles. Experimental. |

---

## 8. Hypotheses

> Pre-registered in `docs/RESEARCH_HYPOTHESIS.md`. **Never edited retroactively.**

| ID | Title | Status |
|---|---|---|
| **H1** | Device-substrate separation via CYNSN/affine | **Withdrawn 2026-05-29** — the CYNSN within-profile track never met its gate. |
| **H2** | DeviceSpace RGB↔CMYK invariance | **Withdrawn 2026-05-29** — dependent on H1. |
| **H3** | Single-profile compression (K ≤ 80 patches → ΔE00 ≤ 1.5 median) | Active. |
| **H4** | Cross-substrate transfer (k ≤ 15 anchors → ΔE00 ≤ 1.5 median, ≤ 3.0 P95, on ≥ 80 % of 702 pairs) | Active; single-pair confirmations exist, batch run pending. |
| **H5** | Low-rankness (rank ≤ 4 captures 99 % Frobenius) | **Rejected 2026-05-29** — median rank 5, max 9 across 461 pairs. |
| **H6** | Pool-PCA vs reference-only PCA on OBA-disparate pairs | Active. |
| **H8** | OBA-aware D1 beats A3 on OBA-mismatched pairs | First-pair confirmation; batch pending. |
| **H9** | Per-λ substrate transform shared across inks (C7 + S3-neutral, k = 5) | First-pair confirmation (DecorMatte ↔ Lyve: 1.06 ΔE00); batch pending. |
| **H10** | CAE cross-trained on MK predicts held-out substrate from paper alone | **Rejected** in its initial form (no anchor fine-tune); H10b planned. |
| **H11** | Cross-vendor same-mode device-response equivalence | **Confirmed 2026-05-29** for the three overlapping presets after substrate normalisation. |

---

## 9. Interpolation methods (RGB → spectrum)

| Method | Where | Behaviour |
|---|---|---|
| **k-NN IDW** (Inverse Distance Weighting) | `lib/interp/rgbInterp.ts:buildInterpolator` | Per-band: weighted average of the k nearest neighbours' spectra, weight `1 / d^power`. Simple, deterministic, **biased toward the local mean** → misses gradients. |
| **PCA-score** | `lib/interp/pcaInterp.ts:buildPcaInterpolator` | Project spectra onto top-k PCA components, IDW the scores, reconstruct. **Negative result** in this project: truncation removes signal more than it removes noise. Opt-in via `MODE_INTERP=pca`. |
| **Local-linear WLS** | `lib/interp/wlsInterp.ts:buildWlsInterpolator` | Fit a hyperplane `R(λ) ≈ β₀ + β·rgb` per band from the k weighted neighbours; solve via Cholesky on the 4×4 normal equations. **Best method** for this project — drops the BC chart's interpolation noise floor from ~1.8 ΔE00 to ~0.7. Default for `modeCompare`. |
| **RBF / thin-plate** | Not implemented | Considered; rejected on cost (per-band N×N solve over 2033 points is too heavy). |

---

## 10. Statistics, algebra, algorithms

| Term | Definition |
|---|---|
| **Median** | Middle of a sorted distribution. We report **median ΔE00** as the headline metric (robust to outliers). |
| **P95** | 95th percentile. The worst-case patches that survive into the visible part of a print. Target: P95 ≤ 3.0 for cross-substrate transfer. |
| **R²** | Coefficient of determination from least-squares regression. We report it per λ for A3. |
| **Pearson r** | Linear correlation. Used to compare same-chart cross-profile spectra at matched patches. |
| **Frobenius energy** | `Σ σᵢ²` of a matrix's singular values. "rank-r captures X %" = `(σ₁²+…+σᵣ²) / Σ σᵢ²`. |
| **Effective rank** | Smallest r whose rank-r truncated SVD captures a given energy fraction (we use 99 %). |
| **SVD** | Singular value decomposition. For an N × L matrix, `σᵢ²` = the eigenvalues of `MᵀM`. We compute eigenvalues only (the 36 × 36 Gram matrix). |
| **Jacobi eigensolver** | Iterative max-element rotation method for symmetric eigenvalue problems. `lib/interp/pcaInterp.ts:jacobiEigen`. |
| **Cholesky decomposition** | `A = L · Lᵀ` for symmetric positive-definite A. Used inside the WLS interpolator to solve the 4×4 normal equations per query. |
| **Nelder-Mead simplex** | Derivative-free numeric optimiser. Used by the (retired) CYNSN training loop. |
| **MAD outlier detection** | Median Absolute Deviation. Planned for the spectral cleaning pipeline. |
| **Savitzky-Golay** | Local polynomial smoothing. Planned for the cleaning pipeline. |
| **IDW power** | Exponent on distance in inverse-distance weighting. We use 2 (`w = 1 / d²`). |

---

## 11. Pipeline + dataset terminology

| Term | Definition |
|---|---|
| **DDD loop** | Document-Driven Development. Idea → hypothesis → spec → code + tests → experiment → log → roadmap update → commit. Enforced by `.githooks/pre-commit`. |
| **Held-out / test set** | Patches the predictor did NOT see during fitting. Every reported metric is on held-out patches. |
| **Anchor / anchor set** | The k measured target patches the predictor is allowed to use. Choosing anchors is what S1–S4 do. |
| **Common RGB grid** | A regular RGB lattice (default 11³ = 1331 points) onto which interpolators resample two profiles built on different charts. The cross-set ΔE00 lives on this grid. |
| **Cross-chart alignment** | Special path in `lib/dataset/matrix.ts:alignByDeviceGrid` for profiles that share no SAMPLE_IDs (e.g. BC vs MOAB). Activated automatically in `TransferView` when `< 50` shared IDs. |
| **Interpolation noise floor** | The held-out interpolation ΔE00 a profile achieves against itself. The bound below which cross-profile ΔE00 cannot be trusted to mean substrate difference. |
| **Substrate normalisation** | The H11-proper comparison: `R_clean = R − OBA_emission`, then `T(λ) = R_clean / R_paper_clean`, then re-apply a common reference paper. `MODE_NORM=device` in `modeCompare`. |
| **Mode comparison** | The H11 experiment shipped in `scripts/experiments/modeCompare.ts`. |
| **Paper anchor** | The patch at `RGB = (255, 255, 255)` — the substrate alone, no ink. Used as the paper-white reference. |
