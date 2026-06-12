import { readFileSync } from "fs";
import { Matrix, SVD, solve } from "ml-matrix";
import { parseCgatsFile } from "../src/cgats-parser.js";
import { spectraToXyz, xyzToLab, deltaE00 } from "../src/color-math.js";

const D50_WP: [number, number, number] = [96.42, 100, 82.49];
const DATASETS = [
  { name: "R2_11-4-23", unlam: "Data/CGATS/R2_11-4-23.txt", lam: "Data/CGATS/R2_11-4-23_lam.txt" },
  { name: "R2_27-10-23", unlam: "Data/CGATS/R2_27-10-23.txt", lam: "Data/CGATS/R2_27-10-23_lam.txt" },
  { name: "R2_13-02-24", unlam: "Data/CGATS/R2_13-02-24.txt", lam: "Data/CGATS/R2_13-02-24_lam.txt" },
  { name: "R3_23-4-24", unlam: "Data/CGATS/R3_23-4-24.txt", lam: "Data/CGATS/R3_23-4-24_lam.txt" },
];

function matchByCMYK(unlam: any[], lam: any[]) {
  const lamMap = new Map<string, any[]>();
  for (const p of lam) { const k = p.cmyk.join(","); if (!lamMap.has(k)) lamMap.set(k, []); lamMap.get(k)!.push(p); }
  const pairs: { u: any; l: any }[] = [];
  for (const pu of unlam) { const k = pu.cmyk.join(","); const m = lamMap.get(k); if (m && m.length > 0) { pairs.push({ u: pu, l: m[0] }); m.shift(); } }
  return pairs;
}

function s2lab(s: Float64Array) { return xyzToLab(spectraToXyz(s), D50_WP); }
function clamp(v: number) { return Math.max(0, Math.min(1, v)); }

function savGol(y: Float64Array, win: number, order: number): Float64Array {
  const h = Math.floor(win / 2), n = y.length, out = new Float64Array(n);
  const A: number[][] = [];
  for (let i = -h; i <= h; i++) { const r: number[] = []; for (let p = 0; p <= order; p++) r.push(i ** p); A.push(r); }
  const m = order + 1;
  const At = A[0].map((_, c) => A.map(r => r[c]));
  const AtA = At.map(r => A[0].map((_, c) => r.reduce((s, v, k) => s + v * A[k][c], 0)));
  const aug = AtA.map((r, i) => [...r, ...Array.from({ length: m }, (_, j) => i === j ? 1 : 0)]);
  for (let col = 0; col < m; col++) {
    let pv = col; while (pv < m && Math.abs(aug[pv][col]) < 1e-15) pv++;
    if (pv >= m) continue;
    [aug[col], aug[pv]] = [aug[pv], aug[col]];
    const d = aug[col][col];
    for (let j = col; j < 2 * m; j++) aug[col][j] /= d;
    for (let r = 0; r < m; r++) if (r !== col) { const f = aug[r][col]; for (let j = col; j < 2 * m; j++) aug[r][j] -= f * aug[col][j]; }
  }
  const inv = aug.map(r => r.slice(m));
  const sg = Array(win).fill(0);
  for (let k = 0; k < win; k++) for (let j = 0; j < m; j++) sg[k] += inv[0][j] * A[k][j];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = -h; k <= h; k++) { const idx = i + k; s += sg[k + h] * (idx < 0 || idx >= n ? y[Math.max(0, Math.min(n - 1, idx))] : y[idx]); }
    out[i] = s;
  }
  return out;
}

function buildX(rows: number, getU: (i: number, w: number) => number, quad: boolean): Matrix {
  const nF = quad ? 72 : 36;
  const X = new Matrix(rows, nF);
  for (let i = 0; i < rows; i++) {
    let col = 0;
    for (let w = 0; w < 36; w++) X.set(i, col++, getU(i, w));
    if (quad) for (let w = 0; w < 36; w++) { const u = getU(i, w); X.set(i, col++, u * u); }
  }
  return X;
}

function ols(X: Matrix, y: number[]) {
  return solve(X.transpose().mmul(X), X.transpose().mmul(Matrix.columnVector(y))).to1DArray();
}

function predict(u: Float64Array, betas: number[][], quad: boolean): number[] {
  return betas.map(b => { let v = 0, c = 0; for (let w = 0; w < 36; w++) v += b[c++] * u[w]; if (quad) for (let w = 0; w < 36; w++) v += b[c++] * u[w] * u[w]; return v; });
}

function computeDE(pairs: { u: Float64Array; l: Float64Array }[], betas: number[][], V: Float64Array[], quad: boolean): number[] {
  const de: number[] = [];
  for (const p of pairs) {
    const c = predict(p.u, betas, quad);
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < betas.length; k++) d += c[k] * V[k][w]; pred[w] = clamp(p.u[w] + d); }
    de.push(deltaE00(s2lab(pred), s2lab(p.l)));
  }
  return de;
}

// Load all
const allData: { name: string; pairs: { u: any; l: any }[] }[] = [];
let allPairs: { u: any; l: any; ds: string }[] = [];
for (const ds of DATASETS) {
  const pairs = matchByCMYK(parseCgatsFile(ds.unlam), parseCgatsFile(ds.lam));
  allData.push({ name: ds.name, pairs });
  for (const p of pairs) allPairs.push({ ...p, ds: ds.name });
}

// Global SVD for basis V
const fullD = new Matrix(allPairs.length, 36);
let gRow = 0;
for (const dd of allData) {
  for (let i = 0; i < dd.pairs.length; i++) {
    for (let w = 0; w < 36; w++) fullD.set(gRow, w, dd.pairs[i].l.spectra[w] - dd.pairs[i].u.spectra[w]);
    gRow++;
  }
}
const svdFull = new SVD(fullD, { autoTranspose: true });
const sVals = svdFull.diagonal;
const V = svdFull.rightSingularVectors;
const RANK_MAX = 5;
const allV: Float64Array[] = [];
const allCvals: number[][] = Array.from({ length: RANK_MAX }, () => []);
const Umat_svd = svdFull.leftSingularVectors;
for (let k = 0; k < RANK_MAX; k++) {
  const vk = new Float64Array(36);
  for (let w = 0; w < 36; w++) vk[w] = V.get(w, k);
  allV.push(vk);
  for (let i = 0; i < allPairs.length; i++) allCvals[k].push(Umat_svd.get(i, k) * sVals[k]);
}

// ========================================
// PART 1: HONEST PER-DATASET EVALUATION (train on 80%, test on held-out 20%)
// ========================================
console.log("=== PART 1: HONEST PER-DATASET (train 80%, test 20%) ===\n");
console.log("We split each dataset: train on 80% random, test on held-out 20%");
console.log("This shows what P95 you'd actually get on new measurements of the same material\n");

function shuffleAndSplit(arr: any[], trainPct: number) {
  const idx = [...Array(arr.length).keys()];
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nTrain = Math.floor(arr.length * trainPct / 100);
  return { train: idx.slice(0, nTrain), test: idx.slice(nTrain) };
}

const NREPS = 5;
const models = [
  { name: "OLS (U only, rank-5)", quad: false, smooth: false, excl: false },
  { name: "OLS (U+U², rank-5)", quad: true, smooth: false, excl: false },
  { name: "SG5 + OLS (U+U²)", quad: true, smooth: true, excl: false },
  { name: "SG5 + OLS + excl5%", quad: true, smooth: true, excl: true },
];

for (const mod of models) {
  console.log(`\n--- ${mod.name} ---`);
  console.log("Dataset        median    P95       P99       max");
  for (const dd of allData) {
    const pairs = mod.smooth
      ? dd.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }))
      : dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));

    // Find offset for cvals
    let dsOff = 0;
    for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }

    const medians: number[] = [];
    const p95s: number[] = [];
    const p99s: number[] = [];
    const maxs: number[] = [];

    for (let rep = 0; rep < NREPS; rep++) {
      const { train: trIdx, test: teIdx } = shuffleAndSplit(dd.pairs, 80);
      const trPairs = trIdx.map(i => pairs[i]);
      const tePairs = teIdx.map(i => pairs[i]);

      // Train
      let finalBetas: number[][];
      if (mod.excl) {
        // Find outliers in training set
        const Xtr = buildX(trPairs.length, (i, w) => trPairs[i].u[w], mod.quad);
        const bAll: number[][] = [];
        for (let k = 0; k < RANK_MAX; k++) bAll.push(ols(Xtr, trIdx.map(i => allCvals[k][dsOff + i])));
        const deTr = computeDE(trPairs, bAll, allV, mod.quad);
        const nEx = Math.max(1, Math.floor(trPairs.length * 5 / 100));
        const sTr = [...Array(trPairs.length).keys()].sort((a, b) => deTr[b] - deTr[a]);
        const excl = new Set(sTr.slice(0, nEx));
        const keep = [...Array(trPairs.length).keys()].filter(i => !excl.has(i));
        const Xk = buildX(keep.length, (r, w) => trPairs[keep[r]].u[w], mod.quad);
        finalBetas = [];
        for (let k = 0; k < RANK_MAX; k++) finalBetas.push(ols(Xk, keep.map(i => allCvals[k][dsOff + trIdx[i]])));
      } else {
        const Xtr = buildX(trPairs.length, (i, w) => trPairs[i].u[w], mod.quad);
        finalBetas = [];
        for (let k = 0; k < RANK_MAX; k++) finalBetas.push(ols(Xtr, trIdx.map(i => allCvals[k][dsOff + i])));
      }

      // Test
      const deTe = computeDE(tePairs, finalBetas, allV, mod.quad);
      deTe.sort((a, b) => a - b);
      medians.push(deTe[Math.floor(deTe.length / 2)]);
      p95s.push(deTe[Math.floor(deTe.length * 0.95)]);
      p99s.push(deTe[Math.floor(deTe.length * 0.99)]);
      maxs.push(deTe[deTe.length - 1]);
    }

    const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    console.log(`  ${dd.name}: median=${avg(medians).toFixed(3)} P95=${avg(p95s).toFixed(3)} P99=${avg(p99s).toFixed(3)} max=${avg(maxs).toFixed(3)}`);
  }
}

// ========================================
// PART 2: CROSS-DATASET (train on one, test on another) — honest eval
// ========================================
console.log("\n\n=== PART 2: CROSS-DATASET (train on one, test on another) ===\n");

// Best model (SG5 + U+U² + excl5%) for cross-dataset
console.log("Model: SG5 + OLS (U+U²) + excl 5%\n");
console.log("Train → Test                    median    P95       target≤2.0");
console.log("─".repeat(55));

for (let ti = 0; ti < DATASETS.length; ti++) {
  const trainDs = allData[ti];
  const trainPairs = trainDs.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
  let trOff = 0;
  for (let d = 0; d < ti; d++) trOff += allData[d].pairs.length;

  // Train with exclude
  const Xtr = buildX(trainPairs.length, (i, w) => trainPairs[i].u[w], true);
  const bAll: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) bAll.push(ols(Xtr, allCvals[k].slice(trOff, trOff + trainPairs.length)));
  const deTr = computeDE(trainPairs, bAll, allV, true);
  const nEx = Math.max(1, Math.floor(trainPairs.length * 5 / 100));
  const sTr = [...Array(trainPairs.length).keys()].sort((a, b) => deTr[b] - deTr[a]);
  const excl = new Set(sTr.slice(0, nEx));
  const keep = [...Array(trainPairs.length).keys()].filter(i => !excl.has(i));
  const Xk = buildX(keep.length, (r, w) => trainPairs[keep[r]].u[w], true);
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xk, keep.map(i => allCvals[k][trOff + i])));

  for (let testIdx = 0; testIdx < DATASETS.length; testIdx++) {
    if (testIdx === ti) continue;
    const testDs = allData[testIdx];
    const testPairs = testDs.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
    const deTe = computeDE(testPairs, betas, allV, true);
    deTe.sort((a, b) => a - b);
    const med = deTe[Math.floor(deTe.length / 2)];
    const p95 = deTe[Math.floor(deTe.length * 0.95)];
    console.log(`${DATASETS[ti].name.padEnd(12)} → ${DATASETS[testIdx].name.padEnd(14)} ${med.toFixed(3).padStart(7)} ${p95.toFixed(3).padStart(7)} ${p95 <= 2.0 ? "✅" : "❌"}`);
  }
}

// ========================================
// PART 3: R2-ONLY CROSS-DATASET (same substrate)
// ========================================
console.log("\n\n=== PART 3: R2→R2 (same substrate, different batch) ===\n");
console.log("Train → Test                    median    P95       target≤2.0");
console.log("─".repeat(55));

const r2Data = DATASETS.filter(d => d.name.startsWith("R2"));
for (let ti = 0; ti < r2Data.length; ti++) {
  const trainDs = allData.find(d => d.name === r2Data[ti].name)!;
  const trainPairs = trainDs.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
  let trOff = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === r2Data[ti].name) break; trOff += allData[d].pairs.length; }

  const Xtr = buildX(trainPairs.length, (i, w) => trainPairs[i].u[w], true);
  const bAll: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) bAll.push(ols(Xtr, allCvals[k].slice(trOff, trOff + trainPairs.length)));
  const deTr = computeDE(trainPairs, bAll, allV, true);
  const nEx = Math.max(1, Math.floor(trainPairs.length * 5 / 100));
  const sTr = [...Array(trainPairs.length).keys()].sort((a, b) => deTr[b] - deTr[a]);
  const excl = new Set(sTr.slice(0, nEx));
  const keep = [...Array(trainPairs.length).keys()].filter(i => !excl.has(i));
  const Xk = buildX(keep.length, (r, w) => trainPairs[keep[r]].u[w], true);
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xk, keep.map(i => allCvals[k][trOff + i])));

  for (let testIdx = 0; testIdx < r2Data.length; testIdx++) {
    if (testIdx === ti) continue;
    const testDs = allData.find(d => d.name === r2Data[testIdx].name)!;
    const testPairs = testDs.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
    const deTe = computeDE(testPairs, betas, allV, true);
    deTe.sort((a, b) => a - b);
    const med = deTe[Math.floor(deTe.length / 2)];
    const p95 = deTe[Math.floor(deTe.length * 0.95)];
    console.log(`${r2Data[ti].name.padEnd(12)} → ${r2Data[testIdx].name.padEnd(14)} ${med.toFixed(3).padStart(7)} ${p95.toFixed(3).padStart(7)} ${p95 <= 2.0 ? "✅" : "❌"}`);
  }
}

// ========================================
// PART 4: R2→R3 cross-substrate
// ========================================
console.log("\n\n=== PART 4: R2→R3 (different substrate) ===\n");
console.log("Train → Test                    median    P95       target≤2.0");
console.log("─".repeat(55));

for (let ti = 0; ti < r2Data.length; ti++) {
  const trainDs = allData.find(d => d.name === r2Data[ti].name)!;
  const trainPairs = trainDs.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
  let trOff = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === r2Data[ti].name) break; trOff += allData[d].pairs.length; }

  const Xtr = buildX(trainPairs.length, (i, w) => trainPairs[i].u[w], true);
  const bAll: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) bAll.push(ols(Xtr, allCvals[k].slice(trOff, trOff + trainPairs.length)));
  const deTr = computeDE(trainPairs, bAll, allV, true);
  const nEx = Math.max(1, Math.floor(trainPairs.length * 5 / 100));
  const sTr = [...Array(trainPairs.length).keys()].sort((a, b) => deTr[b] - deTr[a]);
  const excl = new Set(sTr.slice(0, nEx));
  const keep = [...Array(trainPairs.length).keys()].filter(i => !excl.has(i));
  const Xk = buildX(keep.length, (r, w) => trainPairs[keep[r]].u[w], true);
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xk, keep.map(i => allCvals[k][trOff + i])));

  const r3Data = DATASETS.filter(d => d.name.startsWith("R3"));
  for (const r3 of r3Data) {
    const testDs = allData.find(d => d.name === r3.name)!;
    const testPairs = testDs.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
    const deTe = computeDE(testPairs, betas, allV, true);
    deTe.sort((a, b) => a - b);
    const med = deTe[Math.floor(deTe.length / 2)];
    const p95 = deTe[Math.floor(deTe.length * 0.95)];
    console.log(`${r2Data[ti].name.padEnd(12)} → ${r3.name.padEnd(14)} ${med.toFixed(3).padStart(7)} ${p95.toFixed(3).padStart(7)} ${p95 <= 2.0 ? "✅" : "❌"}`);
  }
}
