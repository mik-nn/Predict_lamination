# docs/AGENTS.md — Claude Code Subagents & Skills Map

> This file maps each phase of work in this repo to the Claude Code subagent or skill that
> handles it. Read [`/CLAUDE.md`](../CLAUDE.md) for the operating rules these agents follow.

The legacy abstract-roles version (Research Scientist Agent / Data Engineer Agent / …) was
removed in May 2026 — see `docs/progress-log.md`. The concrete agents below replace them.

---

## 1. Subagents (delegated work)

| Subagent | When to use | Notes |
|---|---|---|
| **`caveman:cavecrew-investigator`** | "Where is X defined?", "What calls Y?", mapping a directory before edits. | Read-only. Caveman-compressed output keeps the main context lean. |
| **`caveman:cavecrew-builder`** | Surgical 1–2 file edits: typo fix, single-function rewrite, format-preserving tweak. | Refuses 3+ file scope. |
| **`caveman:cavecrew-reviewer`** | "Review this diff / PR / file." | One line per finding. |
| **`Explore`** | Broader codebase searches that need 2–4 queries. | Use instead of `cavecrew-investigator` when the question is more open-ended. |
| **`Plan`** | Designing a multi-step refactor or new feature before touching code. | Returns a step-by-step plan; does not edit. |
| **`feature-dev:code-architect`** | Architectural design with specific files-to-create/modify, data flows. | For larger features, not single-file edits. |
| **`feature-dev:code-explorer`** | Deep trace of an existing feature path. | Useful before extending CYNSN or limits pipeline. |
| **`feature-dev:code-reviewer`** | Bug/quality review with confidence-filtered findings. | High-priority issues only. |
| **`general-purpose`** | Open-ended research / multi-step tasks that don't fit a specialist. | Default fallback. |

**Rule of thumb:** if the task fits a specialist, use the specialist. Spawning costs a cold
context, so only delegate when (a) the work is genuinely independent or (b) you need to
keep the main context window lean.

---

## 2. Skills (loaded into the main thread)

| Skill | Use case in this repo |
|---|---|
| **`colormanagement`** | CGATS.17 parsing, CIE XYZ/Lab conversion, spectral integration, ICC byte layout, white-point adaptation, MAD outliers, Savitzky-Golay, ink-limit curve generation. **First stop for any colour-science question.** |
| **`typescript`** | Implement / fix / refactor TS code in `frontend/`. Add types to `types/index.ts`. Wire pipeline functions into React components. Debug `tsc` errors. |
| **`superpowers:test-driven-development`** | Before writing any new analyser. Write failing test first. |
| **`superpowers:systematic-debugging`** | When a bug surfaces (unexpected ΔE, optimiser stalling, parser mismatch). Use before guessing. |
| **`superpowers:verification-before-completion`** | Before claiming "done" or committing. Run `npm test` (or push to CI) and confirm output. |
| **`superpowers:writing-plans`** | When the user asks for a multi-step task without a written plan. |
| **`superpowers:executing-plans`** | When a plan exists and we're working through it with checkpoints. |
| **`superpowers:brainstorming`** | Before creating a new feature, before designing a new experiment. |
| **`firecrawl:firecrawl-search`** / `WebSearch` | Looking up ISO standards, X-Rite docs, ICC spec. |
| **`update-config`** | Adding hooks (e.g. enforce DDD loop), permissions, env vars to `settings.json`. |

---

## 3. Slash commands available

| Command | Effect |
|---|---|
| `/loop <interval> <prompt>` | Run a task on a recurring cadence. Useful for "every 30 min, regenerate CYNSN comparison and append to EXPERIMENTS". |
| `/schedule` | Create cron-scheduled remote agents. |
| `/verify` | Launch the app and confirm a change works in real UI. |
| `/simplify` | Review changed code for reuse / quality, apply simplifications. |
| `/security-review` | Security pass on pending changes. |
| `/review` | Review the current PR. |
| `/caveman <lite\|full\|ultra>` | Switch caveman communication mode. |

---

## 4. Standard workflows

### 4.1 Adding a new analyser

1. Skill: `superpowers:brainstorming` — clarify the inputs/outputs/metric.
2. Update `docs/RESEARCH_HYPOTHESIS.md` if this measures something new.
3. Write a spec into `docs/specs/<name>.md` (optional for small additions).
4. Skill: `superpowers:test-driven-development` — write failing test first in
   `lib/analyzers/<name>.test.ts`.
5. Skill: `typescript` — implement.
6. Skill: `colormanagement` — sanity-check the maths.
7. Run on real data via Dev UI; append `docs/EXPERIMENTS.md` row.
8. Append `docs/progress-log.md` entry.
9. Commit (pre-commit hook validates).

### 4.2 Reviewing a PR / diff

1. Subagent: `caveman:cavecrew-reviewer` for a quick, terse pass.
2. If deeper: `feature-dev:code-reviewer` or `superpowers:requesting-code-review`.

### 4.3 Diagnosing an unexpected metric

1. Skill: `superpowers:systematic-debugging`.
2. Subagent: `cavecrew-investigator` to map data flow.
3. Skill: `colormanagement` to check formula correctness.
4. Record findings in `docs/EXPERIMENTS.md` even if the result is "no bug".

### 4.4 Locating code

- Single known symbol → `grep` in Bash.
- 2–4 related queries → `Explore`.
- Many related queries with caveman-compressed return → `cavecrew-investigator`.

---

## 5. Anti-patterns

- Spawning a subagent for a single-file lookup. Just use `Read`/`Grep`.
- Letting a subagent commit on your behalf. Subagents return findings; the main thread
  edits, runs tests, and commits.
- Running an experiment without recording it. The result is worthless if it cannot be
  reproduced from `EXPERIMENTS.md`.
