# structure.md — Current tree (June 2026)

> Predict_lamination repo. Snapshot — keep in sync with reality.

```text
Predict_lamination/
├── README.md                            Project overview + results
├── package.json                         Node deps (tfjs, ml-matrix, tsx)
├── esbuild.config.mjs                   esbuild config (src/ → public/dist/)
├── tsconfig.json                        TypeScript config
├── results.json                         130k lines of experiment results
├── outliers.csv                         Top 5% outlier patches per dataset
├── .gitignore
├── Data/
│   ├── CGATS/                           4 paired unlaminated/laminated files
│   └── Icc/                             ICC profiles (untracked)
├── Docs/
│   ├── AGENTS.md                        Claude Code subagents + skills map
│   ├── DATA_DICTIONARY.md               Type reference
│   ├── GLOSSARY.md                      Domain glossary
│   ├── knowledge-base.md                Cross-substrate + lamination findings
│   ├── SKILLS.md                        Domain expertise
│   ├── structure.md                     This file
│   └── Tech.md                          Stack info
├── public/                              Web app (static, served by scripts/serve.cjs)
│   ├── index.html                       3-step wizard UI (Tailwind)
│   ├── tf.min.js                        Local TF.js (no CDN dependency)
│   ├── dist/app.js                      esbuild bundle (26.9kb)
│   ├── dist/app.js.map                  Source map
│   ├── model/model.json + weights       Frozen ResNet model
│   └── baked-params.json                SVD basis + normalization params
├── src/                                 Shared library (scripts + web app)
│   ├── app.ts                           Browser entry point + all re-exports
│   ├── cgats-parser.ts                  CGATS.17 parser → Patch[]
│   ├── color-math.ts                    Spectral→XYZ→Lab→ΔE00 pipeline
│   ├── color-math-debug.ts              Debug version
│   ├── icc-writer.ts                    Binary ICC v2 lut8Type DeviceLink builder
│   ├── ridge.ts                         Ridge regression (Cholesky, browser-safe)
│   ├── strip-matcher.ts                 Row computation + CGATS subset + verify
│   ├── types.ts                         Shared type definitions
│   ├── predictors/                      (legacy experiment predictors)
│   │   ├── a3.ts
│   │   ├── c7.ts
│   │   └── d1.ts
│   └── anchor-strategies/               (legacy anchor selection)
│       └── s2.ts
├── scripts/                             Research experiments + tooling
│   ├── resnet-transfer.ts               Transfer learning: ResNet + Ridge
│   ├── export-baked-params.ts           Export model + SVD → public/
│   └── serve.cjs                        Static file dev server
└── workers/                             CloudFlare Workers (optional)
    ├── api.ts                           CF Worker entry
    ├── wrangler.toml                    CF config
    └── d1-schema.sql                    D1 database schema
```
