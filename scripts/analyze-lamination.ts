import { parseCgatsFile, matchPatches } from '../src/cgats-parser.js';
import { A3Predictor } from '../src/predictors/a3.js';
import { C7Predictor } from '../src/predictors/c7.js';
import { D1Predictor } from '../src/predictors/d1.js';
import { runS2 } from '../src/anchor-strategies/s2.js';
import { Predictor, S2Result } from '../src/types.js';
import { writeFileSync } from 'node:fs';

const DATASETS = [
  { name: 'R2_11-4-23', unlam: 'Data/CGATS/R2_11-4-23.txt', lam: 'Data/CGATS/R2_11-4-23_lam.txt' },
  { name: 'R2_27-10-23', unlam: 'Data/CGATS/R2_27-10-23.txt', lam: 'Data/CGATS/R2_27-10-23_lam.txt' },
  { name: 'R2_13-02-24', unlam: 'Data/CGATS/R2_13-02-24.txt', lam: 'Data/CGATS/R2_13-02-24_lam.txt' },
  { name: 'R3_23-4-24', unlam: 'Data/CGATS/R3_23-4-24.txt', lam: 'Data/CGATS/R3_23-4-24_lam.txt' },
];

function createPredictors(): Predictor[] {
  return [new A3Predictor(), new C7Predictor(), new D1Predictor()];
}

interface ResultRow {
  predictor: string;
  dataset: string;
  minK: number;
  medianAtK: number;
  p95AtK: number;
  maxAtK: number;
  converged: boolean;
}

function fmt(v: number): string { return v.toFixed(3); }

async function main() {
  console.log("=== Lamination Anchor Analysis ===");
  console.log("Thresholds: median dE00 <= 1.0, P95 dE00 <= 2.0\n");

  const allResults: S2Result[] = [];
  const rows: ResultRow[] = [];

  for (const ds of DATASETS) {
    console.log("Loading " + ds.name + "...");
    const unlamPatches = parseCgatsFile(ds.unlam);
    const lamPatches = parseCgatsFile(ds.lam);
    console.log("  Unlaminated: " + unlamPatches.length + " patches");
    console.log("  Laminated: " + lamPatches.length + " patches");

    const pairs = matchPatches(unlamPatches, lamPatches);
    console.log("  Matched: " + pairs.length + " pairs");

    if (pairs.length === 0) {
      console.log("  WARNING: No matches for " + ds.name + ", skipping");
      continue;
    }

    const allUnlam = pairs.map(p => p.unlam);
    const allLam = pairs.map(p => p.lam);

    const predictors = createPredictors();

    for (const predictor of predictors) {
      console.log("  Predictor: " + predictor.name + "...");
      const result = runS2(predictor, ds.name, allUnlam, allLam);
      const lastIter = result.iterations[result.iterations.length - 1];

      allResults.push(result);
      rows.push({
        predictor: predictor.name,
        dataset: ds.name,
        minK: result.converged ? result.minK : -1,
        medianAtK: lastIter.medianDeltaE,
        p95AtK: lastIter.p95DeltaE,
        maxAtK: lastIter.maxDeltaE,
        converged: result.converged,
      });

      console.log("    k=" + result.minK + ", median=" + fmt(lastIter.medianDeltaE) + ", P95=" + fmt(lastIter.p95DeltaE) + ", converged=" + result.converged);
    }
  }

  console.log("\n=== Results Summary ===\n");
  console.log("Predictor | Dataset       | minK | median dE00 | P95 dE00 | max dE00 | Converged");
  console.log("-------------------------------------------------------------------------------");
  for (const r of rows) {
    const mk = r.minK >= 0 ? String(r.minK) : "N/A";
    console.log(r.predictor.padEnd(10) + " | " + r.dataset.padEnd(13) + " | " + mk.padStart(4) + " | " + fmt(r.medianAtK).padStart(10) + " | " + fmt(r.p95AtK).padStart(9) + " | " + fmt(r.maxAtK).padStart(8) + " | " + (r.converged ? "YES" : "NO "));
  }

  console.log("\n=== Cross-Dataset Analysis ===");
  const predictorNames = ["A3", "C7", "D1"];
  for (const pName of predictorNames) {
    const predRows = rows.filter(r => r.predictor === pName);
    const allConverged = predRows.every(r => r.converged);
    if (allConverged) {
      const maxK = Math.max(...predRows.map(r => r.minK));
      const avgMedian = predRows.reduce((s, r) => s + r.medianAtK, 0) / predRows.length;
      const avgP95 = predRows.reduce((s, r) => s + r.p95AtK, 0) / predRows.length;
      console.log(pName + ": k_all = " + maxK + " (avg median=" + fmt(avgMedian) + ", avg P95=" + fmt(avgP95) + ")");
    } else {
      const notConv = predRows.filter(r => !r.converged).map(r => r.dataset);
      console.log(pName + ": DID NOT CONVERGE on all 4. Failed on: " + notConv.join(", "));
    }
  }

  writeFileSync("results.json", JSON.stringify(allResults, null, 2));
  console.log("\nDetailed results written to results.json");
}

main().catch(err => {
  console.error("Fatal error:", err.message);
  process.exit(1);
});
