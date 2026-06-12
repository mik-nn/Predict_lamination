# Tech.md — Technology Stack

## Frontend (Phase 0–2)

- **TypeScript 5.6** (strict mode)
- **Vite 8** (build tool)
- **React 18.3** + React DOM
- **TailwindCSS 3.4**
- **D3 v7** (visualisation)
- **Zustand 4.5** (state management)
- **@tanstack/react-query 5** (data lifecycle, lightly used so far)
- **pako 2.1** (zlib for ZXML decompression)

Numerical helpers are written inline (Nelder-Mead, Pearson r, etc.) rather than pulled
from `math.js` / `simple-statistics` — the project keeps zero numeric dependencies on
purpose for transparency.

## Data formats

- **Input:** `.icm` (ICC v4 RGB with embedded CxF3 in private `CxF`/`ZXML` tag), `.cxf`
  (standalone CxF3 XML).
- **Intermediate:** typed JSON via `types/index.ts` — `Measurement`, `ProfileData`,
  `MatchedPatchPair`, `LinearityResult`, CYNSN evaluation types.
- **Export:** CGATS.17 ASCII (`lib/cgatsExport.ts`).
- **Future archive:** Parquet, when a Python backend appears.

## Future stack (Phase 3+)

- Python FastAPI backend
- `colour-science` + `scipy` + `scikit-learn`
- PyTorch (for the β-VAE prototype)
- ArgyllCMS integration (for ICC profile builds)

## Coding standards

- TypeScript strict mode; no implicit `any`.
- ESLint + Prettier; commit hooks installed via `npm install` in `frontend/`.
- Functional React with hooks; Zustand for shared state.
- Feature-grouped folder layout (`lib/parsers/`, `lib/analyzers/`, …).

## Local tooling (run from `frontend/`)

| Command | Purpose |
|---|---|
| `npm install` | Install deps + auto-install `.githooks/` via `postinstall`. |
| `npm run dev` | Vite dev server (preceded by `vitest run`). |
| `npm run build` | Production build. |
| `npm run lint` | ESLint (TypeScript). |
| `npm test` | Vitest one-shot. |
| `npm run test:watch` | Vitest watch mode. |
| `npx prettier . --check` | Format check. |
| `npx prettier . --write` | Apply formatting. |

## Node version

- **Required:** Node ≥ 18 (engines field enforces this).
- **CI:** Node 20 on GitHub Actions.
- **WSL2 default:** Node 12 is too old; install Node 20 via `nvm`.

## Versions of record

See `frontend/package.json` for the authoritative list. Bump and document in
`docs/progress-log.md` with the reason for the bump.
