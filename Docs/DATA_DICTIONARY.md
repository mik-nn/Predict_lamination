# DATA_DICTIONARY.md

> Authoritative reference for the data types in `frontend/src/types/index.ts`.
> See `docs/ONTOLOGY.md` for the conceptual model these types represent.

> **Primacy of data:** the project is built around **spectral reflectance**. The 36-band
> `spectra` is the *primary measurement*. `Lab` is a *derived* colorimetric projection
> (D50 / 2°) and is informational only — never the source of truth for analysis. Any
> analyser claim must rest on spectral or XYZ-linear quantities; Lab and ΔE00 are reported
> for human readability.

---

## 1. ProfileMetadata

Metadata extracted from the profile filename.

| Field | Type | Description |
|---|---|---|
| `full_name` | `string` | Filename without extension (e.g. `BC_Lyve_P9000_mk_CanvasMatte`). |
| `brand` | `string` | Brand prefix (usually `BC`). |
| `series` | `string` | Material or ink series (`Lyve`, `VibranceLuster`, `600MT`, …). |
| `printer` | `string` | Printer model (`P9000`, …). |
| `ink` | `string` | Ink type / mode (`mk`, `pk`, `PLPP260`, …). |
| `substrate` | `string` | Substrate name (`CanvasMatte`, `WCRW`, `VibranceLuster`, …). |
| `parsed_at` | `string` (ISO 8601) | When the filename was parsed. |

## 2. Measurement — one patch

### 2.1 Identity

| Field | Type | Range / unit | Description |
|---|---|---|---|
| `SAMPLE_ID` | `string` | `R{row}C{col}P{page}` | Stable cross-profile patch identifier. Join key. |

### 2.2 Primary data: spectral reflectance

| Field | Type | Range / unit | Description |
|---|---|---|---|
| `spectra` | `number[]?` | 0–1, length 36 | **Primary measurement.** Reflectance per wavelength, 380–730 nm, 10 nm step. Source: CxF3 `<cc:SpectralData StartWL="380" Increment="10">`. M0 measurement condition for 24 of 27 P9000 ICMs; M2 (UV-cut) for the remaining 4. |
| `wavelengths` | `number[]?` | nm | `[380, 390, …, 730]`. Companion to `spectra`. |

A `Measurement` without `spectra` is partially useful (device + Lab only) but cannot
feed CYNSN, ink-ratio, or spectral predictor analyses. The `ProfileData.has_spectral`
flag indicates whether at least one patch in the profile is fully usable.

### 2.3 Device values (current dual representation)

> **Migration status:** the codebase is moving toward a single
> `device: { space: 'rgb' \| 'cmyk', values: number[] }` discriminator (see
> `docs/ONTOLOGY.md` §3.1). Until the migration completes, both legacy field sets coexist
> and the active field set depends on the source file.

| Field | Type | Range | Filled by |
|---|---|---|---|
| `RGB_R`, `RGB_G`, `RGB_B` | `number?` | 0–255 (device addressing) | RGB ICM parser, RGB CxF parser. |
| `CMYK_C`, `CMYK_M`, `CMYK_Y`, `CMYK_K` | `number` | 0–100 (nominal coverage) | CMYK CxF parser; CMYK ICM (legacy synthetic path — deprecated). |

For RGB profiles, the CMYK fields are present but set to `0` and must not be used.

The standard CMY mapping for RGB profiles (used inside CYNSN and limits analyser):

```text
C = (255 - R) / 255    ∈ [0, 1]
M = (255 - G) / 255    ∈ [0, 1]
Y = (255 - B) / 255    ∈ [0, 1]
```

### 2.4 Derived colorimetric values

| Field | Type | Source | Description |
|---|---|---|---|
| `LAB_L` | `number` | Derived | CIE L\* (D50 / 2°). Computed via `spectraToLab(spectra, 380)` when not present in source; otherwise taken from CxF `<cc:ColorCIELab>`. |
| `LAB_A` | `number` | Derived | CIE a\*. |
| `LAB_B` | `number` | Derived | CIE b\*. |

**Important:** Lab values from CxF and Lab values computed from spectra can disagree
slightly (different XYZ integration tables, illuminant precision). The codebase prefers
**recomputed** Lab to keep the colorimetric pipeline consistent — this is enforced
inside the analyser layer, not the parser. If you compare `LAB_*` across profiles,
verify both sides came from the same conversion path.

### 2.5 Optional derived fields

| Field | Type | Description |
|---|---|---|
| `deltaE00` | `number?` | CIEDE2000 vs a reference patch (informational; perceptual). |
| `is_outlier` | `boolean?` | Reserved for the future MAD outlier detector. |

## 3. ProfileData

| Field | Type | Description |
|---|---|---|
| `metadata` | `ProfileMetadata` | Filename-derived metadata. |
| `raw` | `Measurement[]` | Raw patches as parsed. |
| `clean` | `Measurement[]` | After cleaning pipeline (currently `=== raw`, placeholder for MAD + Savitzky-Golay). |
| `has_spectral` | `boolean` | At least one patch has `spectra`. Gates spectral analysers. |
| `patch_count` | `number` | `raw.length`. |
| `average_deltaE_raw_clean` | `number?` | Mean ΔE00 between raw and clean (cleaning quality). |
| `wavelengths` | `number[]?` | Convenience copy from the first spectral patch. |

## 4. MatchedPatchPair

| Field | Type | Description |
|---|---|---|
| `sampleId` | `string` | Common `SAMPLE_ID` between two profiles. |
| `ref` | `Measurement` | Patch from the reference profile. |
| `target` | `Measurement` | Patch from the target profile. |

## 5. Analyser result types

See `frontend/src/types/index.ts` for the full surface. Key types and which fields of
`Measurement` they consume:

| Result type | Consumes | Notes |
|---|---|---|
| `InkRatioResult` | `spectra` (paper + ink) | `T(λ) = R_ink/R_paper` per channel. Spectral-first. |
| `PatchGroupResult` | `spectra`, optional Lab | Group breakdown; spectral Pearson r and per-patch R² are primary. Lab deltas reported as secondary. |
| `SpectralPredictionModel` / `Evaluation` / `Comparison` | `spectra` | Per-λ polynomial / YN / XYZ-affine. Pure spectral pipeline. |
| `LinearityResult` | `spectra` (primary), Lab (secondary) | Legacy CMYK-fuzzy path collapsing to RGB join. To be ported to DeviceSpace. |
| `CYNSNEvaluation`, `CYNSNComparisonResult` (in `lib/analyzers/cynsn.ts`) | `spectra`, device values | Forward CYNSN spectral prediction; ΔE00 vs measured spectra. |

## 6. Conventions

- All numerical fields are SI / dimensionless unless documented otherwise.
- Wavelengths always start at 380 nm with a 10 nm step (36 values). If a future dataset
  diverges, add an explicit `startWL` / `step` to the structure rather than silently
  re-interpreting.
- D50 / 2° observer throughout. Bradford adaptation only when explicitly named in a
  function.
- Reflectance is **linear**, never log. Range strictly 0–1.
- `0` reflectance = perfect absorber; `1` = perfect diffuse reflector.
- Any analyser that produces a metric on Lab must also produce the same metric on
  spectra or XYZ (so the hypothesis test does not depend on the perceptual projection).
