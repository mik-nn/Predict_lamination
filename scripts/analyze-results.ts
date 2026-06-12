import { readFileSync } from "fs";

function analyze() {
  const raw = readFileSync("results.json", "utf-8");
  const results = JSON.parse(raw);

  const thresholds = [
    { name: "vTight", med: 0.5, p95: 1.0 },
    { name: "tight", med: 1.0, p95: 2.0 },
    { name: "default", med: 1.5, p95: 3.0 },
    { name: "relaxed", med: 2.0, p95: 4.0 },
  ];

  const summary: any[] = [];

  for (const r of results) {
    const predName = r.predictorName;
    const dsName = r.datasetName;

    const convThresholds: string[] = [];
    for (const t of thresholds) {
      let foundK = -1;
      for (const iter of r.iterations) {
        if (iter.medianDeltaE <= t.med && iter.p95DeltaE <= t.p95) {
          foundK = iter.k;
          break;
        }
      }
      convThresholds.push(t.name + ":" + (foundK >= 0 ? foundK : "N/A"));
    }

    const lastIter = r.iterations[r.iterations.length - 1];
    summary.push({
      predictor: predName,
      dataset: dsName,
      finalK: lastIter.k,
      finalMedian: lastIter.medianDeltaE,
      finalP95: lastIter.p95DeltaE,
      converged: r.converged,
      thresholds: convThresholds,
    });
  }

  console.log("=== Convergence by Predictor and Dataset ===\n");

  console.log("Predictor | Dataset       | k_final | median | P95   | vTight | tight | default | relaxed");
  console.log("-".repeat(100));
  for (const s of summary) {
    console.log(
      s.predictor.padEnd(10) + " | " +
      s.dataset.padEnd(13) + " | " +
      String(s.finalK).padStart(7) + " | " +
      s.finalMedian.toFixed(3).padStart(6) + " | " +
      s.finalP95.toFixed(3).padStart(5) + " | " +
      s.thresholds[0].padStart(7) + " | " +
      s.thresholds[1].padStart(5) + " | " +
      s.thresholds[2].padStart(7) + " | " +
      s.thresholds[3].padStart(7)
    );
  }

  console.log("\n\n=== Minimum k across all 4 datasets (by threshold) ===\n");
  const predNames = ["A3", "C7", "D1"];
  const threshNames = ["vTight", "tight", "default", "relaxed"];
  for (const tIdx in thresholds) {
    console.log("Threshold: " + thresholds[tIdx].name + " (median <= " + thresholds[tIdx].med + ", P95 <= " + thresholds[tIdx].p95 + ")");
    for (const pn of predNames) {
      const pResults = summary.filter(s => s.predictor === pn);
      const ks = pResults.map(s => {
        const match = s.thresholds[tIdx].match(/:(\d+|N\/A)/);
        if (!match) return -1;
        const v = match[1];
        return v === "N/A" ? -1 : parseInt(v);
      });
      const allOk = ks.every(k => k > 0);
      if (allOk) {
        console.log("  " + pn + ": k_all = " + Math.max(...ks));
      } else {
        const fails = pResults.filter((_, i) => ks[i] <= 0).map(s => s.dataset);
        console.log("  " + pn + ": NOT ALL CONVERGED (fails: " + fails.join(", ") + ")");
      }
    }
    console.log();
  }

  console.log("\n=== Best C7 k to reach near-threshold (median <= 1.15, P95 <= 2.5) ===\n");
  for (const r of results) {
    if (r.predictorName !== "C7") continue;
    for (const iter of r.iterations) {
      if (iter.medianDeltaE <= 1.15 && iter.p95DeltaE <= 2.5) {
        console.log("  " + r.datasetName + ": k=" + iter.k + " (median=" + iter.medianDeltaE.toFixed(3) + ", P95=" + iter.p95DeltaE.toFixed(3) + ")");
        break;
      }
    }
  }
}

analyze();
