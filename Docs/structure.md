# structure.md — Current tree (June 2026)

> Predict_lamination repo. Snapshot — keep in sync with reality.

```text
Predict_lamination/
├── README.md                            Project overview + results
├── package.json                         Node deps (tfjs, ml-matrix, tsx)
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
├── src/
│   ├── cgats-parser.ts                  CGATS.17 parser with CMYK matching
│   ├── color-math.ts                    Spectral→XYZ→Lab→ΔE00 pipeline
│   ├── color-math-debug.ts              Debug version
│   ├── types.ts                         Shared type definitions
│   ├── predictors/                      Predictor implementations
│   │   ├── a3.ts
│   │   ├── c7.ts
│   │   └── d1.ts
│   └── anchor-strategies/
│       └── s2.ts
└── scripts/                             22 experiment scripts (see README)
```
