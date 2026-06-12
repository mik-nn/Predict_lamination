# SKILLS.md — Domain expertise reference

> What the team / agent needs to know to contribute meaningfully. For the Claude Code
> skill set, see [`docs/AGENTS.md`](AGENTS.md).

---

## Core expertise

### Color science & management (high)

- ICC profiles, A2B / B2A tables, CxF format.
- Spectral colour (ISO 13655 measurement conditions M0 / M1 / M2 / M3).
- Models: Neugebauer, Yule-Nielsen, Murray-Davies, ink-spreading polynomials.
- ΔE metrics: ΔE76, ΔE94, ΔE00 (CIEDE2000 — ISO 11664-6).
- Media white point handling; Bradford / CAT02 chromatic adaptation.

### ICC profile specifications

- ICC.1 v4 (ISO 15076-1:2022): profile format v4.4.0.0.
- Header: 128 bytes; tag count at byte **128**; tag table at byte **132**;
  each tag record = signature (4) + offset (4) + size (4) = 12 bytes.
- Private tag embedding: X-Rite `CxF` tag → `ZXML` data (zlib-compressed XML).

### CxF (Color eXchange Format) specifications

- CxF/X3 (ISO 17972-3) — output target data for printer characterisation. **The format we use.**
- CxF/X4 (ISO 17972-4) — spot colour characterisation.
- ZXML embedding inside ICC: locate `'CxF '` tag, skip 12 bytes
  (4 data-type `'ZXML'` + 4 reserved + 4 unknown), then `pako.inflate`.
- Spectral block: `<cc:SpectralData StartWL="380" Increment="10">…</cc:SpectralData>` —
  36 floats, 380–730 nm.

### Data science & analysis

- Correlation (Pearson / Spearman) and regression diagnostics.
- Residual analysis.
- Savitzky-Golay smoothing.
- Outlier detection (MAD).
- Linear / canonical correlation analysis.

### Programming & engineering

- TypeScript / React / D3 (current focus).
- Document-Driven Development.
- ETL pipelines with explicit invariants.
- Numerical optimisation (Nelder-Mead, simulated annealing, BFGS).
- Scientific visualisation of medium-large datasets (tens of thousands of patches).

### Advanced (under development)

- Variational Autoencoders (β-VAE).
- Disentangled representation learning.
- Transfer learning between domains (substrate adaptation).

---

## Required team / agent level

- **Research scientist:** PhD-level grasp of CIE colour models and spectral physics.
- **Data engineer:** strong TypeScript + ETL design.
- **Visualisation engineer:** expert D3 + scientific plotting.
