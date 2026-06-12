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

const WAVES = 36;

function s2lab(s: Float64Array) { return xyzToLab(spectraToXyz(s), D50_WP); }
function clamp(v: number) { return Math.max(0, Math.min(1, v)); }

function savGol(y: Float64Array, win: number, order: number): Float64Array {
  const h = Math.floor(win / 2), n = y.length, out = new Float64Array(n);
  const A: number[][] = [];
  for (let i = -h; i <= h; i++) { const r: number[] = []; for (let p = 0; p <= order; p++) r.push(Math.pow(i, p)); A.push(r); }
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

function buildXnlin(rows: number, getU: (i: number, w: number) => number): Matrix {
  const X = new Matrix(rows, 72);
  for (let i = 0; i < rows; i++) {
    let col = 0;
    for (let w = 0; w < 36; w++) X.set(i, col++, getU(i, w));
    for (let w = 0; w < 36; w++) { const u = getU(i, w); X.set(i, col++, u * u); }
  }
  return X;
}

function ols(X: Matrix, y: number[]): number[] {
  return solve(X.transpose().mmul(X), X.transpose().mmul(Matrix.columnVector(y))).to1DArray();
}

function predictFromQuad(u: Float64Array, betas: number[][]): number[] {
  return betas.map(b => { let v = 0, c = 0; for (let w = 0; w < 36; w++) v += b[c++] * u[w]; for (let w = 0; w < 36; w++) v += b[c++] * u[w] * u[w]; return v; });
}

function computeDE(pairs: { u: Float64Array; l: Float64Array }[], betas: number[][], V: Float64Array[]): number[] {
  const de: number[] = [];
  for (const p of pairs) {
    const c = predictFromQuad(p.u, betas);
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
let row = 0;
for (const dd of allData) {
  for (let i = 0; i < dd.pairs.length; i++) {
    for (let w = 0; w < 36; w++) fullD.set(row, w, dd.pairs[i].l.spectra[w] - dd.pairs[i].u.spectra[w]);
    row++;
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

// Train on one dataset (with SG + exclude), test on all
console.log("=== CROSS-DATASET: TRAIN ON 1, TEST ON ALL (SG5 + excl5%) ===\n");
console.log("Training dataset: one per row");
console.log("┌──────────────┬─────────────────────────────────────────────────────────────────────────────┐");
console.log("│ Train on     │ R2_11-4-23     R2_27-10-23    R2_13-02-24    R3_23-4-24                     │");
console.log("├──────────────┼─────────────────────────────────────────────────────────────────────────────┤");

type Result = { med: number; p95: number; p99: number; max: number };
const allResults: { train: string; test: string; sg5: Result; raw: Result }[] = [];

for (let ti = 0; ti < DATASETS.length; ti++) {
  const trainDs = allData[ti];
  const trainPairs = trainDs.pairs;
  const nTrain = trainPairs.length;
  let dsOff = 0;
  for (let d = 0; d < ti; d++) dsOff += allData[d].pairs.length;

  // --- Model A: SG5 smoothed + exclude 5% ---
  const smoothTrain = trainPairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));

  // Train on all smoothed to find outliers
  const Xall = buildXnlin(nTrain, (i, w) => smoothTrain[i].u[w]);
  const betasAll: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betasAll.push(ols(Xall, allCvals[k].slice(dsOff, dsOff + nTrain)));
  const deAll = computeDE(smoothTrain, betasAll, allV);
  const sorted = [...Array(nTrain).keys()].sort((a, b) => deAll[b] - deAll[a]);
  const nEx = Math.max(1, Math.floor(nTrain * 5 / 100));
  const excl = new Set(sorted.slice(0, nEx));
  const keep = [...Array(nTrain).keys()].filter(i => !excl.has(i));
  const Xkeep = buildXnlin(keep.length, (r, w) => smoothTrain[keep[r]].u[w]);
  const betasSmooth: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) {
    const vals = keep.map(i => allCvals[k][dsOff + i]);
    betasSmooth.push(ols(Xkeep, vals));
  }

  // --- Model B: Raw (no smoothing) + exclude 5% ---
  const rawTrain = trainPairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const XallR = buildXnlin(nTrain, (i, w) => rawTrain[i].u[w]);
  const betasAllR: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betasAllR.push(ols(XallR, allCvals[k].slice(dsOff, dsOff + nTrain)));
  const deAllR = computeDE(rawTrain, betasAllR, allV);
  const sortedR = [...Array(nTrain).keys()].sort((a, b) => deAllR[b] - deAllR[a]);
  const exclR = new Set(sortedR.slice(0, nEx));
  const keepR = [...Array(nTrain).keys()].filter(i => !exclR.has(i));
  const XkeepR = buildXnlin(keepR.length, (r, w) => rawTrain[keepR[r]].u[w]);
  const betasRaw: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) {
    const vals = keepR.map(i => allCvals[k][dsOff + i]);
    betasRaw.push(ols(XkeepR, vals));
  }

  // Test on all datasets
  const linePartsS: string[] = [`${DATASETS[ti].name.padEnd(12)}`];
  const linePartsR: string[] = [`${DATASETS[ti].name.padEnd(12)}`];

  for (let testIdx = 0; testIdx < DATASETS.length; testIdx++) {
    const testDs = allData[testIdx];
    const testPairs = testDs.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
    const deS = computeDE(testPairs, betasSmooth, allV);
    deS.sort((a, b) => a - b);
    const rS = { med: deS[Math.floor(deS.length/2)], p95: deS[Math.floor(deS.length*0.95)], p99: deS[Math.floor(deS.length*0.99)], max: deS[deS.length-1] };

    const testPairsR = testDs.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
    const deR = computeDE(testPairsR, betasRaw, allV);
    deR.sort((a, b) => a - b);
    const rR = { med: deR[Math.floor(deR.length/2)], p95: deR[Math.floor(deR.length*0.95)], p99: deR[Math.floor(deR.length*0.99)], max: deR[deR.length-1] };

    linePartsS.push(`${rS.p95.toFixed(3)}`);
    linePartsR.push(`${rR.p95.toFixed(3)}`);
    allResults.push({ train: DATASETS[ti].name, test: DATASETS[testIdx].name, sg5: rS, raw: rR });
  }
  console.log("│ SG5+excl5  │ " + linePartsS.join("      ").padEnd(68) + " │");
  console.log("│ Raw+excl5  │ " + linePartsR.join("      ").padEnd(68) + " │");
  if (ti < DATASETS.length - 1) console.log("├──────────────┼─────────────────────────────────────────────────────────────────────────────┤");
}
console.log("└──────────────┴─────────────────────────────────────────────────────────────────────────────┘");

// Per-dataset (train and test on same) for comparison
console.log("\n=== PER-DATASET (train & test on same, SG5+excl5%) ===");
let pdRow = 0;
const perDs: { name: string; med: number; p95: number }[] = [];
for (const dd of allData) {
  const pairs = dd.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
  const n = pairs.length;
  const Xa = buildXnlin(n, (i, w) => pairs[i].u[w]);
  const ba: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) ba.push(ols(Xa, allCvals[k].slice(pdRow, pdRow + n)));
  const de = computeDE(pairs, ba, allV);
  const srt = [...Array(n).keys()].sort((a, b) => de[b] - de[a]);
  const en = Math.max(1, Math.floor(n * 5 / 100));
  const ex = new Set(srt.slice(0, en));
  const kp = [...Array(n).keys()].filter(i => !ex.has(i));
  const Xk = buildXnlin(kp.length, (r, w) => pairs[kp[r]].u[w]);
  const bk: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) {
    const vs = kp.map(i => allCvals[k][pdRow + i]);
    bk.push(ols(Xk, vs));
  }
  const de2 = computeDE(pairs, bk, allV);
  de2.sort((a, b) => a - b);
  const m = de2[Math.floor(de2.length/2)];
  const p95 = de2[Math.floor(de2.length*0.95)];
  perDs.push({ name: dd.name, med: m, p95 });
  console.log(`  ${dd.name}: median=${m.toFixed(3)} P95=${p95.toFixed(3)}`);
  pdRow += n;
}

// Detailed comparison table
console.log("\n=== DETAILED COMPARISON (P95 ΔE00) ===");
console.log("Train → Test                    SG5+excl5%   Raw+excl5%   Per-dataset");
console.log("─".repeat(70));
for (const r of allResults) {
  const pd = perDs.find(p => p.name === r.test)!;
  const sgStr = r.sg5.p95.toFixed(3) + (r.sg5.p95 <= 2.0 ? " ✅" : " ❌");
  const rawStr = r.raw.p95.toFixed(3) + (r.raw.p95 <= 2.0 ? " ✅" : " ❌");
  const pdStr = pd.p95.toFixed(3) + (pd.p95 <= 2.0 ? " ✅" : " ❌");
  console.log(`${r.train.padEnd(12)} → ${r.test.padEnd(14)}  ${sgStr.padEnd(14)} ${rawStr.padEnd(14)} ${pdStr}`);
}

// Summary: best single dataset to use as universal predictor
console.log("\n=== BEST SINGLE TRAINING DATASET FOR UNIVERSAL PREDICTOR ===");
for (const trainName of DATASETS.map(d => d.name)) {
  const results = allResults.filter(r => r.train === trainName);
  const avgP95 = results.reduce((s, r) => s + r.sg5.p95, 0) / results.length;
  const worstP95 = Math.max(...results.map(r => r.sg5.p95));
  const worstDataset = results.find(r => r.sg5.p95 === worstP95)!.test;
  console.log(`  Train on ${trainName}: avg P95=${avgP95.toFixed(3)} worst=${worstP95.toFixed(3)} (${worstDataset})`);
}

// R2-only prediction
console.log("\n=== TRAIN ON R2 (all 3 combined), TEST ON R3 ===");
const r2Pairs = allPairs.filter(p => p.ds.startsWith("R2"));
const r3Pairs = allPairs.filter(p => p.ds.startsWith("R3"));
const r2Smooth = r2Pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
const r3Smooth = r3Pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));

// Find R2 row offsets
const r2Offsets = DATASETS.map((d, i) => allData.slice(0, i).reduce((s, dd) => s + dd.pairs.length, 0)).filter((_, i) => DATASETS[i].name.startsWith("R2"));

// Build combined R2 model with SG + exclude
const nr2 = r2Smooth.length;
const Xr2 = buildXnlin(nr2, (i, w) => r2Smooth[i].u[w]);
const br2All: number[][] = [];
let r2Coffset = r2Offsets[0];
for (let k = 0; k < RANK_MAX; k++) {
  const vals: number[] = [];
  for (let d = 0; d < allData.length; d++) {
    if (!allData[d].name.startsWith("R2")) continue;
    for (let i = 0; i < allData[d].pairs.length; i++) vals.push(allCvals[k][r2Coffset + i]);
    r2Coffset += allData[d].pairs.length;
  }
  br2All.push(ols(Xr2, vals));
}
const deR2all = computeDE(r2Smooth, br2All, allV);
const srtR2 = [...Array(nr2).keys()].sort((a, b) => deR2all[b] - deR2all[a]);
const exR2 = new Set(srtR2.slice(0, Math.max(1, Math.floor(nr2 * 5 / 100))));
const kpR2 = [...Array(nr2).keys()].filter(i => !exR2.has(i));
const Xr2k = buildXnlin(kpR2.length, (r, w) => r2Smooth[kpR2[r]].u[w]);
const br2: number[][] = [];
r2Coffset = r2Offsets[0];
for (let k = 0; k < RANK_MAX; k++) {
  let vals: number[] = [];
  for (let d = 0; d < allData.length; d++) {
    if (!allData[d].name.startsWith("R2")) continue;
    const subVals = kpR2.filter(i => {
      let accum = 0;
      for (let dd = 0; dd < d; dd++) if (allData[dd].name.startsWith("R2")) accum += allData[dd].pairs.length;
      return i >= accum && i < accum + allData[d].pairs.length;
    }).map(i => allCvals[k][r2Coffset + i - accum]);
    vals = vals.concat(subVals);
    r2Coffset += allData[d].pairs.length;
  }
  br2.push(ols(Xr2k, vals));
}
const deR3 = computeDE(r3Smooth, br2, allV);
deR3.sort((a, b) => a - b);
console.log(`  R2 → R3: median=${deR3[Math.floor(deR3.length/2)].toFixed(3)} P95=${deR3[Math.floor(deR3.length*0.95)].toFixed(3)} P99=${deR3[Math.floor(deR3.length*0.99)].toFixed(3)} max=${deR3[deR3.length-1].toFixed(3)}`);

// Per-R3 dataset breakdown
console.log("\n  Per R3 dataset:");
for (const dd of allData) {
  if (!dd.name.startsWith("R3")) continue;
  const testP = dd.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
  const deR3ds = computeDE(testP, br2, allV);
  deR3ds.sort((a, b) => a - b);
  console.log(`    ${dd.name}: median=${deR3ds[Math.floor(deR3ds.length/2)].toFixed(3)} P95=${deR3ds[Math.floor(deR3ds.length*0.95)].toFixed(3)}`);
}

// Best possible: train on one R2, test on other R2 (same substrate)
console.log("\n=== WITHIN-R2 CROSS-DATASET (train on one R2, predict another R2) ===");
const r2Datasets = DATASETS.filter(d => d.name.startsWith("R2"));
for (let ti = 0; ti < r2Datasets.length; ti++) {
  const trainName = r2Datasets[ti].name;
  const trainData = allData.find(d => d.name === trainName)!;
  const nTr = trainData.pairs.length;
  let trOff = 0;
  for (let d = 0; d < allData.length; d++) { if (allData[d].name === trainName) break; trOff += allData[d].pairs.length; }

  // SG + exclude on training
  const trSm = trainData.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
  const Xta = buildXnlin(nTr, (i, w) => trSm[i].u[w]);
  const bta: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) bta.push(ols(Xta, allCvals[k].slice(trOff, trOff + nTr)));
  const deTr = computeDE(trSm, bta, allV);
  const sTr = [...Array(nTr).keys()].sort((a, b) => deTr[b] - deTr[a]);
  const eTr = new Set(sTr.slice(0, Math.max(1, Math.floor(nTr * 5 / 100))));
  const kTr = [...Array(nTr).keys()].filter(i => !eTr.has(i));
  const Xtk = buildXnlin(kTr.length, (r, w) => trSm[kTr[r]].u[w]);
  const btk: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) btk.push(ols(Xtk, kTr.map(i => allCvals[k][trOff + i])));

  console.log(`  Train: ${trainName}`);
  for (const testName of r2Datasets.map(d => d.name).filter(n => n !== trainName)) {
    const testData = allData.find(d => d.name === testName)!;
    const teSm = testData.pairs.map(p => ({ u: savGol(p.u.spectra, 5, 2), l: savGol(p.l.spectra, 5, 2) }));
    const deTe = computeDE(teSm, btk, allV);
    deTe.sort((a, b) => a - b);
    console.log(`    → ${testName}: median=${deTe[Math.floor(deTe.length/2)].toFixed(3)} P95=${deTe[Math.floor(deTe.length*0.95)].toFixed(3)}`);
  }
}
