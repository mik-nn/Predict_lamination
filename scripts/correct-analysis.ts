import { readFileSync } from "fs";
import { Matrix, SVD } from "ml-matrix";
import { parseCgatsFile } from "../src/cgats-parser.js";
import { spectraToXyz, xyzToLab, deltaE00 } from "../src/color-math.js";

const D50_WP: [number, number, number] = [96.42, 100, 82.49];

const DATASETS = [
  { name: "R2_11-4-23", unlam: "Data/CGATS/R2_11-4-23.txt", lam: "Data/CGATS/R2_11-4-23_lam.txt" },
  { name: "R2_27-10-23", unlam: "Data/CGATS/R2_27-10-23.txt", lam: "Data/CGATS/R2_27-10-23_lam.txt" },
  { name: "R2_13-02-24", unlam: "Data/CGATS/R2_13-02-24.txt", lam: "Data/CGATS/R2_13-02-24_lam.txt" },
  { name: "R3_23-4-24", unlam: "Data/CGATS/R3_23-4-24.txt", lam: "Data/CGATS/R3_23-4-24_lam.txt" },
];

// Match unlaminated → laminated by CMYK key within each pair
function matchByCMYK(unlam: any[], lam: any[]): { u: any; l: any; key: string }[] {
  const lamMap = new Map<string, any[]>();
  for (const p of lam) {
    const key = p.cmyk.join(",");
    if (!lamMap.has(key)) lamMap.set(key, []);
    lamMap.get(key)!.push(p);
  }
  const pairs: { u: any; l: any; key: string }[] = [];
  for (const pu of unlam) {
    const key = pu.cmyk.join(",");
    const matches = lamMap.get(key);
    if (matches && matches.length > 0) {
      pairs.push({ u: pu, l: matches[0], key });
      matches.shift();
    }
  }
  return pairs;
}

function computeDE(lab1: [number, number, number], lab2: [number, number, number]): number {
  return deltaE00(lab1, lab2);
}

function spectraToLab(s: Float64Array): [number, number, number] {
  return xyzToLab(spectraToXyz(s), D50_WP);
}

function main() {
  // 1. Load and match all data by CMYK
  const allPairs: { u: any; l: any; ds: string; key: string }[] = [];
  const byDataset: { name: string; pairs: { u: any; l: any; key: string }[] }[] = [];

  for (const ds of DATASETS) {
    const unlam = parseCgatsFile(ds.unlam);
    const lam = parseCgatsFile(ds.lam);
    const pairs = matchByCMYK(unlam, lam);
    console.log(ds.name + ": " + unlam.length + " unlam, " + lam.length + " lam, matched " + pairs.length + " by CMYK");
    for (const p of pairs) allPairs.push({ ...p, ds: ds.name });
    byDataset.push({ name: ds.name, pairs });
  }

  const N = allPairs.length;
  console.log("\nTotal CMYK-matched pairs: " + N);

  // 2. Build D = L - U matrix (N x 36)
  const Dmat = new Matrix(N, 36);
  for (let i = 0; i < N; i++) {
    for (let w = 0; w < 36; w++) {
      Dmat.set(i, w, allPairs[i].l.spectra[w] - allPairs[i].u.spectra[w]);
    }
  }

  // 3. SVD of D
  console.log("Computing SVD of " + N + " x 36 matrix...");
  const svd = new SVD(Dmat, { autoTranspose: true });
  const s = svd.diagonal;
  const totalE = s.reduce((sum, v) => sum + v * v, 0);

  console.log("\n=== Singular Values of D = L - U ===");
  let cumE = 0;
  for (let i = 0; i < Math.min(10, s.length); i++) {
    cumE += s[i] * s[i];
    console.log("  σ" + (i + 1) + " = " + s[i].toFixed(4) + "  (" + (s[i] / s[0] * 100).toFixed(2) + "% of σ1, cumE=" + (cumE / totalE * 100).toFixed(4) + "%)");
  }
  console.log("  σ1/σ2 = " + (s[0] / s[1]).toFixed(1));

  // 4. v(λ) = first right singular vector
  const V = svd.V;
  console.log("\n=== v(λ) — Spectral Signature of Lamination ===");
  for (let w = 0; w < 36; w++) {
    const wl = 380 + w * 10;
    console.log("  " + wl + " nm: " + V.get(w, 0).toFixed(6));
  }

  // 5. c(i) distribution per dataset
  const Umat = svd.U;
  let idx = 0;
  for (const ds of byDataset) {
    const n = ds.pairs.length;
    const cVals: number[] = [];
    for (let i = 0; i < n; i++) {
      cVals.push(Umat.get(idx + i, 0) * s[0]);
    }
    idx += n;
    cVals.sort((a, b) => a - b);
    const min = cVals[0], max = cVals[n - 1];
    const med = cVals[Math.floor(n / 2)];
    const q1 = cVals[Math.floor(n * 0.25)];
    const q3 = cVals[Math.floor(n * 0.75)];
    const mean = cVals.reduce((a, b) => a + b, 0) / n;
    console.log("\n=== c(i) for " + ds.name + " (n=" + n + ") ===");
    console.log("  mean=" + mean.toFixed(4) + " median=" + med.toFixed(4) + " Q1=" + q1.toFixed(4) + " Q3=" + q3.toFixed(4));
    console.log("  min=" + min.toFixed(4) + " max=" + max.toFixed(4));

    // c(i) by ink coverage
    const byK: { [k: string]: number[] } = {};
    for (let i = 0; i < n; i++) {
      const k = Math.round(ds.pairs[i].u.cmyk[3]);
      if (!byK[k]) byK[k] = [];
      byK[k].push(cVals[i] || 0);
    }
    for (const [k, vals] of Object.entries(byK).sort((a, b) => +a[0] - +b[0])) {
      vals.sort((a, b) => a - b);
      console.log("    K=" + k + " (n=" + vals.length + "): median=" + vals[Math.floor(vals.length / 2)].toFixed(4));
    }
  }

  // 6. Rank-1 model: L_pred = U + c * v(λ)
  console.log("\n=== Rank-1 Model Error (ΔE00) ===");
  idx = 0;
  for (const ds of byDataset) {
    const n = ds.pairs.length;
    const deVals: number[] = [];
    for (let i = 0; i < n; i++) {
      const c = Umat.get(idx + i, 0) * s[0];
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) {
        pred[w] = ds.pairs[i].u.spectra[w] + c * V.get(w, 0);
        if (pred[w] < 0) pred[w] = 0;
        if (pred[w] > 1) pred[w] = 1;
      }
      const actualLab = spectraToLab(ds.pairs[i].l.spectra);
      const predLab = spectraToLab(pred);
      deVals.push(computeDE(predLab, actualLab));
    }
    deVals.sort((a, b) => a - b);
    const med = deVals[Math.floor(n / 2)];
    const p95 = deVals[Math.floor(n * 0.95)];
    const maxDE = deVals[n - 1];
    console.log("  " + ds.name + ": median=" + med.toFixed(3) + " P95=" + p95.toFixed(3) + " max=" + maxDE.toFixed(3) + " (per-patch c)");
    idx += n;
  }

  // 7. Cross-dataset validation: use global median c for each dataset
  console.log("\n=== Rank-1 Model (fixed median c per dataset) ===");
  idx = 0;
  for (const ds of byDataset) {
    const n = ds.pairs.length;
    const cVals: number[] = [];
    for (let i = 0; i < n; i++) cVals.push(Umat.get(idx + i, 0) * s[0]);
    cVals.sort((a, b) => a - b);
    const medC = cVals[Math.floor(n / 2)];
    const deVals: number[] = [];
    for (let i = 0; i < n; i++) {
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) {
        pred[w] = ds.pairs[i].u.spectra[w] + medC * V.get(w, 0);
        if (pred[w] < 0) pred[w] = 0;
        if (pred[w] > 1) pred[w] = 1;
      }
      deVals.push(computeDE(spectraToLab(pred), spectraToLab(ds.pairs[i].l.spectra)));
    }
    deVals.sort((a, b) => a - b);
    console.log("  " + ds.name + ": median=" + deVals[Math.floor(n / 2)].toFixed(3) + " P95=" + deVals[Math.floor(n * 0.95)].toFixed(3) + " (fixed c=" + medC.toFixed(4) + ")");
    idx += n;
  }

  // 8. Predict c from unlaminated spectrum: c = f(mean(U))
  console.log("\n=== Predicting c(i) from Unlaminated Spectrum ===");
  const cAll: number[] = [];
  const meanUAll: number[] = [];
  for (let i = 0; i < N; i++) {
    cAll.push(Umat.get(i, 0) * s[0]);
    let sum = 0;
    for (let w = 0; w < 36; w++) sum += allPairs[i].u.spectra[w];
    meanUAll.push(sum / 36);
  }

  // Linear regression: c = β0 + β1 * mean(U)
  let sumU = 0, sumC = 0, sumUU = 0, sumUC = 0;
  for (let i = 0; i < N; i++) {
    sumU += meanUAll[i]; sumC += cAll[i]; sumUU += meanUAll[i] * meanUAll[i]; sumUC += meanUAll[i] * cAll[i];
  }
  const b1 = (N * sumUC - sumU * sumC) / (N * sumUU - sumU * sumU);
  const b0 = (sumC - b1 * sumU) / N;
  console.log("  c = " + b0.toFixed(4) + " + " + b1.toFixed(4) + " × mean(U)");

  // 9. Full model: predict L from U alone (zero anchors!)
  console.log("\n=== Rank-1 Model (c from U, ZERO additional anchors!) ===");
  for (const ds of byDataset) {
    const allDE: number[] = [];
    for (const p of ds.pairs) {
      let meanU = 0;
      for (let w = 0; w < 36; w++) meanU += p.u.spectra[w];
      meanU /= 36;
      const cPred = Math.max(-0.2, Math.min(0.05, b0 + b1 * meanU));
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) {
        pred[w] = p.u.spectra[w] + cPred * V.get(w, 0);
        if (pred[w] < 0) pred[w] = 0;
        if (pred[w] > 1) pred[w] = 1;
      }
      allDE.push(computeDE(spectraToLab(pred), spectraToLab(p.l.spectra)));
    }
    allDE.sort((a, b) => a - b);
    const med = allDE[Math.floor(allDE.length / 2)];
    const p95 = allDE[Math.floor(allDE.length * 0.95)];
    const maxDE = allDE[allDE.length - 1];
    console.log("  " + ds.name + ": median=" + med.toFixed(3) + " P95=" + p95.toFixed(3) + " max=" + maxDE.toFixed(3) + " (k=0)");
  }

  // 10. Determine minimum anchors: find which patches give best c estimate
  console.log("\n=== Minimum Anchor Analysis ===");
  // Strategy: pick k patches, estimate their c from the rank-1 model,
  // use median c of those k patches for all predictions
  for (let k = 0; k <= 5; k++) {
    for (const ds of byDataset) {
      const n = ds.pairs.length;
      const allDE: number[] = [];
      for (let pi = 0; pi < n; pi++) {
        let cEst: number;
        if (k === 0) {
          let mu = 0;
          for (let w = 0; w < 36; w++) mu += ds.pairs[pi].u.spectra[w];
          cEst = Math.max(-0.2, Math.min(0.05, b0 + b1 * (mu / 36)));
        } else {
          // Simple: use k nearest patches in mean(U) space as "anchors"
          let muTarget = 0;
          for (let w = 0; w < 36; w++) muTarget += ds.pairs[pi].u.spectra[w];
          muTarget /= 36;

          // Find k patches with closest mean(U) and average their c
          const dists: { idx: number; d: number }[] = [];
          for (let j = 0; j < n; j++) {
            let mu = 0;
            for (let w = 0; w < 36; w++) mu += ds.pairs[j].u.spectra[w];
            mu /= 36;
            dists.push({ idx: j, d: Math.abs(mu - muTarget) });
          }
          dists.sort((a, b) => a.d - b.d);
          let sumC = 0;
          for (let j = 0; j < Math.min(k, n); j++) {
            const cj = Umat.get(idx + dists[j].idx, 0) * s[0];
            sumC += cj;
          }
          cEst = sumC / Math.min(k, n);
        }
        const pred = new Float64Array(36);
        for (let w = 0; w < 36; w++) {
          pred[w] = ds.pairs[pi].u.spectra[w] + cEst * V.get(w, 0);
          if (pred[w] < 0) pred[w] = 0;
          if (pred[w] > 1) pred[w] = 1;
        }
        allDE.push(computeDE(spectraToLab(pred), spectraToLab(ds.pairs[pi].l.spectra)));
      }
      allDE.sort((a, b) => a - b);
      const med = allDE[Math.floor(allDE.length / 2)];
      const p95 = allDE[Math.floor(allDE.length * 0.95)];
      if (k === 0) console.log("  " + ds.name + " k=" + k + ": median=" + med.toFixed(3) + " P95=" + p95.toFixed(3));
    }
  }
  idx = 0;

  // 11. Final: what if we use global median c for all datasets?
  const globalCAll: number[] = [];
  for (let i = 0; i < N; i++) globalCAll.push(Umat.get(i, 0) * s[0]);
  globalCAll.sort((a, b) => a - b);
  const globalMedC = globalCAll[Math.floor(N / 2)];
  console.log("\n=== Global Model: single c = " + globalMedC.toFixed(6) + " for ALL datasets ===");
  idx = 0;
  for (const ds of byDataset) {
    const n = ds.pairs.length;
    const deVals: number[] = [];
    for (let i = 0; i < n; i++) {
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) {
        pred[w] = ds.pairs[i].u.spectra[w] + globalMedC * V.get(w, 0);
        if (pred[w] < 0) pred[w] = 0;
        if (pred[w] > 1) pred[w] = 1;
      }
      deVals.push(computeDE(spectraToLab(pred), spectraToLab(ds.pairs[i].l.spectra)));
    }
    deVals.sort((a, b) => a - b);
    console.log("  " + ds.name + ": median=" + deVals[Math.floor(n / 2)].toFixed(3) + " P95=" + deVals[Math.floor(n * 0.95)].toFixed(3) + " max=" + deVals[n - 1].toFixed(3));
  }

  console.log("\nDone.");
}

main();
