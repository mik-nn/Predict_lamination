# structure.md — Current tree (May 2026)

> Snapshot — keep in sync with reality. Update when you add a top-level file or directory.

```text
Color_Modeling/
├── CLAUDE.md                            Operating manual (read first).
├── AGENTS.md                            Repo guidelines for AI/contributors.
├── README.md                            Public-facing overview.
├── TODO.md                              Open work.
├── .githooks/                           Pre-commit hook (installed via `npm install`).
│   └── pre-commit                       Blocks feat:/fix: without progress-log change.
├── docs/
│   ├── ONTOLOGY.md                      Domain model + mermaid diagram.
│   ├── DATA_DICTIONARY.md               TypeScript type reference.
│   ├── RESEARCH_HYPOTHESIS.md           Falsifiable hypothesis + metrics.
│   ├── ROADMAP.md                       Phased plan + cross-cutting epics.
│   ├── IMPLEMENTATION.md                Per-module code map.
│   ├── Tech.md                          Stack + build/test commands.
│   ├── EXPERIMENTS.md                   Append-only experiment log.
│   ├── progress-log.md                  Per-session changelog.
│   ├── cynsn-pipeline.md                CYNSN flow + known bugs.
│   ├── AGENTS.md                        Claude Code subagents + skills map.
│   ├── workflow.md                      The DDD loop expanded.
│   ├── SKILLS.md                        Domain expertise reference.
│   ├── PROMPTS.md                       LLM prompt templates.
│   └── structure.md                     This file.
├── data/                                Local sample data (gitignored).
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   ├── scripts/
│   │   └── install-hooks.sh             Symlinks .githooks → core.hooksPath.
│   └── src/
│       ├── App.tsx                      Entry, layout.
│       ├── main.tsx                     React mount.
│       ├── styles.css                   Tailwind base.
│       ├── global.d.ts                  Ambient types.
│       ├── types/
│       │   └── index.ts                 Single source of truth for data types.
│       ├── store/
│       │   └── useProfileStore.ts       Zustand store.
│       ├── utils/
│       │   ├── filenameParser.ts        Filename → ProfileMetadata.
│       │   └── filenameParser.test.ts
│       ├── lib/
│       │   ├── colormath.ts             Spectra → XYZ → Lab, CIEDE2000.
│       │   ├── colormath.test.ts
│       │   ├── cgatsExport.ts           CGATS.17 exporter.
│       │   ├── dataLoader.ts            File-type dispatch + ProfileData wrap.
│       │   ├── iccTagScanner.ts         ZXML tag locate + pako inflate.
│       │   ├── parsers/
│       │   │   ├── icmParser.ts         ICC header → ZXML CxF → Measurement[].
│       │   │   ├── cxfParser.ts         cc:CxF XML → Measurement[].
│       │   │   ├── cxfParser.test.ts
│       │   │   └── index.ts
│       │   └── analyzers/
│       │       ├── limitsAnalyzer.ts    Per-channel ink limits (YN n=2).
│       │       ├── linearityAnalyzer.ts Cross-substrate Pearson/R² (legacy).
│       │       ├── linearityAnalyzer.test.ts
│       │       ├── groupAnalyzer.ts     Patch-group breakdown.
│       │       ├── inkRatioAnalyzer.ts  T(λ) = R_ink / R_paper.
│       │       ├── spectralPredictor.ts Per-λ polynomial / YN / XYZ-affine.
│       │       ├── spreading.ts         Polynomial dot-gain.
│       │       ├── spreading.test.ts
│       │       ├── optimizer.ts         Nelder-Mead simplex.
│       │       ├── optimizer.test.ts
│       │       ├── cynsn.ts             3D CYNSN model.
│       │       ├── cynsn.test.ts
│       │       └── index.ts
│       ├── components/
│       │   ├── ComparisonView.tsx       The UI hub.
│       │   ├── ProfileUploader.tsx
│       │   ├── ProfileList.tsx
│       │   ├── InkLimitSection.tsx
│       │   ├── GroupBreakdownTable.tsx
│       │   ├── InkRatioTable.tsx
│       │   ├── LabScatterPlot.tsx
│       │   ├── SpectralCurves.tsx
│       │   ├── PatchCorrelationScatter.tsx
│       │   └── PredictionAccuracyView.tsx
│       └── test/
│           └── setup.ts                 Vitest jsdom setup.
└── .github/
    └── workflows/
        └── ci.yml                       Node 20: npm test + npm run build.
```

Legacy / pending deletion:

- `frontend/src/lib/cxFParser.ts` — duplicate of `lib/parsers/cxfParser.ts`.
- `frontend/src/utils/cxfParser.test.ts` — duplicate of `lib/parsers/cxfParser.test.ts`.

Removed (May 2026):

- `.specify/` (Spec-Kit scaffolding — unused template).
- `.kilo/`, `.lingma/` (leftover AI-tool configs).
