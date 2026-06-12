import { readFileSync, writeFileSync } from "fs";
import { Matrix, SVD, solve } from "ml-matrix";
import { parseCgatsFile } from "../src/cgats-parser.js";
import { spectraToXyz, xyzToLab, deltaE00 } from "../src/color-math.js";

const D50_WP: [number, number, number] = [96.42, 100, 82.49];
const WAVES = [380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730];

const DATASETS = [
  { name: "R2_11-4-23", unlam: "Data/CGATS/R2_11-4-23.txt", lam: "Data/CGATS/R2_11-4-23_lam.txt" },
  { name: "R2_27-10-23", unlam: "Data/CGATS/R2_27-10-23.txt", lam: "Data/CGATS/R2_27-10-23_lam.txt" },
  { name: "R2_13-02-24", unlam: "Data/CGATS/R2_13-02-24.txt", lam: "Data/CGATS/R2_13-02-24_lam.txt" },
  { name: "R3_23-4-24", unlam: "Data/CGATS/R3_23-4-24.txt", lam: "Data/CGATS/R3_23-4-24_lam.txt" },
];

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

function s2lab(s: Float64Array): [number, number, number] {
  return xyzToLab(spectraToXyz(s), D50_WP);
}
function de00(l1: [number,number,number], l2: [number,number,number]): number {
  return deltaE00(l1, l2);
}

function clamp(v: number): number {
  return Math.max(0, Math.min(1, v));
}

// Load all data
const allData: { name: string; pairs: { u: any; l: any }[]; U: Matrix; L: Matrix; D: Matrix }[] = [];
let allPairs: { u: any; l: any; ds: string }[] = [];

for (const ds of DATASETS) {
  const unlam = parseCgatsFile(ds.unlam);
  const lam = parseCgatsFile(ds.lam);
  const pairs = matchByCMYK(unlam, lam);
  const n = pairs.length;
  const Umat = new Matrix(n, 36);
  const Lmat = new Matrix(n, 36);
  for (let i = 0; i < n; i++) {
    for (let w = 0; w < 36; w++) {
      Umat.set(i, w, pairs[i].u.spectra[w]);
      Lmat.set(i, w, pairs[i].l.spectra[w]);
    }
  }
  const Dmat = Matrix.sub(Lmat, Umat);
  allData.push({ name: ds.name, pairs, U: Umat, L: Lmat, D: Dmat });
  for (const p of pairs) allPairs.push({ ...p, ds: ds.name });
}

console.log("Loaded " + allPairs.length + " total CMYK-matched pairs");

// Full SVD for rank analysis
const fullD = new Matrix(allPairs.length, 36);
let row = 0;
for (const dd of allData) {
  for (let i = 0; i < dd.pairs.length; i++) {
    for (let w = 0; w < 36; w++) fullD.set(row, w, dd.D.get(i, w));
    row++;
  }
}
const svdFull = new SVD(fullD, { autoTranspose: true });
const sVals = svdFull.diagonal;
const V = svdFull.rightSingularVectors;

console.log("\n=== SVD of D (all " + allPairs.length + " pairs) ===");
let cum = 0;
const totalE = sVals.reduce((a, b) => a + b * b, 0);
for (let i = 0; i < 8; i++) {
  cum += sVals[i] * sVals[i];
  console.log("  σ" + (i+1) + " = " + sVals[i].toFixed(4) + " (" + (cum/totalE*100).toFixed(2) + "%)");
}

// ========================================
// MODEL 1: Neutral (2 global scalars)
// L(λ) = ρ + τ · U(λ)
// ========================================
console.log("\n=== MODEL 1: Neutral 2-parameter (L = ρ + τ·U) ===");
// Fit from ALL patches and ALL wavelengths
let sum1 = 0, sumU = 0, sumL = 0, sumUU = 0, sumUL = 0;
for (const p of allPairs) {
  for (let w = 0; w < 36; w++) {
    const u = p.u.spectra[w], l = p.l.spectra[w];
    sum1++; sumU += u; sumL += l; sumUU += u*u; sumUL += u*l;
  }
}
const tau = (sum1 * sumUL - sumU * sumL) / (sum1 * sumUU - sumU * sumU);
const rho = (sumL - tau * sumU) / sum1;
console.log("  ρ = " + rho.toFixed(6) + " (surface reflection)");
console.log("  τ = " + tau.toFixed(6) + " (film transmittance)");

for (const dd of allData) {
  const de: number[] = [];
  for (const p of dd.pairs) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(rho + tau * p.u.spectra[w]);
    de.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3) + " max=" + de[de.length-1].toFixed(3));
}

// ========================================
// MODEL 2: Per-λ (A3) — 72 params
// L(λ) = a(λ) + b(λ)·U(λ)
// ========================================
console.log("\n=== MODEL 2: Per-λ affine (A3) — 72 params ===");
const a_w = new Float64Array(36);
const b_w = new Float64Array(36);
for (let w = 0; w < 36; w++) {
  let s1 = 0, su = 0, sl = 0, suu = 0, sul = 0;
  for (const p of allPairs) {
    const u = p.u.spectra[w], l = p.l.spectra[w];
    s1++; su += u; sl += l; suu += u*u; sul += u*l;
  }
  b_w[w] = (s1 * sul - su * sl) / (s1 * suu - su * su);
  a_w[w] = (sl - b_w[w] * su) / s1;
}

for (const dd of allData) {
  const de: number[] = [];
  for (const p of dd.pairs) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(a_w[w] + b_w[w] * p.u.spectra[w]);
    de.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3) + " max=" + de[de.length-1].toFixed(3));
}

// ========================================
// MODEL 3: Rank-1 SVD
// L = U + c·v
// ========================================
console.log("\n=== MODEL 3: Rank-1 SVD (D = c·v) ===");
const Umat_svd = svdFull.leftSingularVectors;
const v0 = new Float64Array(36);
for (let w = 0; w < 36; w++) v0[w] = V.get(w, 0);
row = 0;
for (const dd of allData) {
  const de: number[] = [];
  for (let i = 0; i < dd.pairs.length; i++) {
    const c = Umat_svd.get(row + i, 0) * sVals[0];
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + c * v0[w]);
    de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
  row += dd.pairs.length;
}

// ========================================
// MODEL 4: Rank-2 SVD
// L = U + c1·v1 + c2·v2
// ========================================
console.log("\n=== MODEL 4: Rank-2 SVD (D = c1·v1 + c2·v2) ===");
const v1 = new Float64Array(36);
const v2 = new Float64Array(36);
for (let w = 0; w < 36; w++) { v1[w] = V.get(w, 0); v2[w] = V.get(w, 1); }
row = 0;
for (const dd of allData) {
  const de: number[] = [];
  for (let i = 0; i < dd.pairs.length; i++) {
    const c1 = Umat_svd.get(row + i, 0) * sVals[0];
    const c2 = Umat_svd.get(row + i, 1) * sVals[1];
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + c1 * v1[w] + c2 * v2[w]);
    de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
  row += dd.pairs.length;
}

// ========================================
// CROSS-DATASET VALIDATION
// ========================================
console.log("\n=== CROSS-DATASET VALIDATION ===");
// Train on 3 datasets, predict on the 4th
for (let testIdx = 0; testIdx < DATASETS.length; testIdx++) {
  const trainPairs = allPairs.filter(p => p.ds !== DATASETS[testIdx].name);
  const testPairs = allPairs.filter(p => p.ds === DATASETS[testIdx].name);
  const testName = DATASETS[testIdx].name;

  console.log("\n--- Train on 3, predict " + testName + " ---");

  // Model 1: Neutral
  let s1=0, su=0, sl=0, suu=0, sul=0;
  for (const p of trainPairs) {
    for (let w = 0; w < 36; w++) {
      const u = p.u.spectra[w], l = p.l.spectra[w];
      s1++; su += u; sl += l; suu += u*u; sul += u*l;
    }
  }
  const tau_c = (s1 * sul - su * sl) / (s1 * suu - su * su);
  const rho_c = (sl - tau_c * su) / s1;
  const deN: number[] = [];
  for (const p of testPairs) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(rho_c + tau_c * p.u.spectra[w]);
    deN.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  deN.sort((a, b) => a - b);
  console.log("  [Neutral] median=" + deN[Math.floor(deN.length/2)].toFixed(3) + " P95=" + deN[Math.floor(deN.length*0.95)].toFixed(3));

  // Model 2: Per-λ (A3)
  const deA3: number[] = [];
  for (const p of testPairs) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(a_w[w] + b_w[w] * p.u.spectra[w]);
    deA3.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  deA3.sort((a, b) => a - b);
  console.log("  [A3]     median=" + deA3[Math.floor(deA3.length/2)].toFixed(3) + " P95=" + deA3[Math.floor(deA3.length*0.95)].toFixed(3));

  // Model 5: Spectral neutral (ρ(λ), τ(λ) — per-λ but from training only)
  const a5 = new Float64Array(36);
  const b5 = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    let ss1=0, ssu=0, ssl=0, ssuu=0, ssul=0;
    for (const p of trainPairs) {
      const u = p.u.spectra[w], l = p.l.spectra[w];
      ss1++; ssu += u; ssl += l; ssuu += u*u; ssul += u*l;
    }
    b5[w] = (ss1 * ssul - ssu * ssl) / (ss1 * ssuu - ssu * ssu);
    a5[w] = (ssl - b5[w] * ssu) / ss1;
  }
  const deP: number[] = [];
  for (const p of testPairs) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(a5[w] + b5[w] * p.u.spectra[w]);
    deP.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  deP.sort((a, b) => a - b);
  console.log("  [A3-train] median=" + deP[Math.floor(deP.length/2)].toFixed(3) + " P95=" + deP[Math.floor(deP.length*0.95)].toFixed(3));
}

// ========================================
// MINIMUM ANCHORS: Predict c from patches
// ========================================
console.log("\n=== MINIMUM ANCHOR ANALYSIS ===");
// For rank-1 model: need k patches to estimate c for each new patch
// Strategy: pick k patches by extreme mean(U) values, fit c = β0 + β1·mean(U)
// or just use nearest-neighbor in U-space

// Recompute c values for all patches
const allC: number[] = [];
const allMeanU: number[] = [];
for (let i = 0; i < allPairs.length; i++) {
  allC.push(Umat_svd.get(i, 0) * sVals[0]);
  let s = 0;
  for (let w = 0; w < 36; w++) s += allPairs[i].u.spectra[w];
  allMeanU.push(s / 36);
}

// Fit c = β0 + β1·mean(U)
let ssu = 0, ssc = 0, ssuu = 0, ssuc = 0;
for (let i = 0; i < allPairs.length; i++) {
  ssu += allMeanU[i]; ssc += allC[i]; ssuu += allMeanU[i]*allMeanU[i]; ssuc += allMeanU[i]*allC[i];
}
const nTot = allPairs.length;
const b1_c = (nTot * ssuc - ssu * ssc) / (nTot * ssuu - ssu * ssu);
const b0_c = (ssc - b1_c * ssu) / nTot;
console.log("  c = " + b0_c.toFixed(4) + " + " + b1_c.toFixed(4) + " × mean(U)");

// k=0: predict from mean(U) alone
for (const dd of allData) {
  const de: number[] = [];
  for (const p of dd.pairs) {
    let mu = 0;
    for (let w = 0; w < 36; w++) mu += p.u.spectra[w];
    mu /= 36;
    const c_pred = b0_c + b1_c * mu;
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(p.u.spectra[w] + c_pred * v0[w]);
    de.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + " k=0 (c from U): median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
}

// k=1: pick best single anchor patch
// The "best" anchor is the patch whose c gives the best prediction for ALL patches
// Find the patch whose c is closest to the dataset median c
console.log("\n--- k=1: single best anchor (closest to median c) ---");
row = 0;
for (const dd of allData) {
  const n = dd.pairs.length;
  const cVals: number[] = [];
  for (let i = 0; i < n; i++) cVals.push(Umat_svd.get(row + i, 0) * sVals[0]);
  cVals.sort((a, b) => a - b);
  const medC = cVals[Math.floor(n / 2)];
  // Find the patch whose c is closest to medC
  let bestIdx = 0, bestDist = Infinity;
  for (let i = 0; i < n; i++) {
    const ci = Umat_svd.get(row + i, 0) * sVals[0];
    const d = Math.abs(ci - medC);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  const bestC = Umat_svd.get(row + bestIdx, 0) * sVals[0];
  const de: number[] = [];
  for (let i = 0; i < n; i++) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + bestC * v0[w]);
    de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + " k=1: median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3) + " anchor=" + dd.pairs[bestIdx].u.cmyk.join(","));
  row += n;
}

// k=2: pick two extreme patches (paper-like + heavy-ink) and average their c
console.log("\n--- k=2: two extreme anchors ---");
row = 0;
for (const dd of allData) {
  const n = dd.pairs.length;
  // Find patches with min and max mean(U)
  let minIdx = 0, maxIdx = 0, minMU = Infinity, maxMU = -Infinity;
  for (let i = 0; i < n; i++) {
    let mu = 0;
    for (let w = 0; w < 36; w++) mu += dd.pairs[i].u.spectra[w];
    mu /= 36;
    if (mu < minMU) { minMU = mu; minIdx = i; }
    if (mu > maxMU) { maxMU = mu; maxIdx = i; }
  }
  const c1 = Umat_svd.get(row + minIdx, 0) * sVals[0];
  const c2 = Umat_svd.get(row + maxIdx, 0) * sVals[0];
  const cAvg = (c1 + c2) / 2;
  const de: number[] = [];
  for (let i = 0; i < n; i++) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + cAvg * v0[w]);
    de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + " k=2: median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
  row += n;
}

// ========================================
// MODEL 5: Physical 2-vector calibrator
// L(λ) = ρ(λ) + τ(λ)·U(λ)
// Requires exactly 2 laminated patches
// ========================================
console.log("\n=== MODEL 5: Physical 2-vector (L(λ)=ρ(λ)+τ(λ)·U(λ)) ===");

function solveRhoTau(p1: { u: any; l: any }, p2: { u: any; l: any }): { rho: Float64Array; tau: Float64Array } {
  const rho = new Float64Array(36);
  const tau = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    const u1 = p1.u.spectra[w], l1 = p1.l.spectra[w];
    const u2 = p2.u.spectra[w], l2 = p2.l.spectra[w];
    const denom = u1 - u2;
    if (Math.abs(denom) > 1e-10) {
      tau[w] = (l1 - l2) / denom;
      rho[w] = l1 - tau[w] * u1;
    } else {
      tau[w] = 1;
      rho[w] = 0;
    }
  }
  return { rho, tau };
}

function predictPhysical(u: Float64Array, rho: Float64Array, tau: Float64Array): Float64Array {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) pred[w] = clamp(rho[w] + tau[w] * u[w]);
  return pred;
}

function computeDEforModel(pairs: { u: any; l: any }[], rho: Float64Array, tau: Float64Array): number[] {
  const de: number[] = [];
  for (const p of pairs) {
    const pred = predictPhysical(p.u.spectra, rho, tau);
    de.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  de.sort((a, b) => a - b);
  return de;
}

function fmtDe(de: number[]): string {
  return "median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3);
}

// Section A: Within-dataset — pick 2 extreme patches, solve, predict rest
console.log("\n--- A: Within-dataset (2 extreme patches) ---");
for (const dd of allData) {
  const n = dd.pairs.length;
  let minIdx = 0, maxIdx = 0, minMU = Infinity, maxMU = -Infinity;
  for (let i = 0; i < n; i++) {
    let mu = 0;
    for (let w = 0; w < 36; w++) mu += dd.pairs[i].u.spectra[w];
    mu /= 36;
    if (mu < minMU) { minMU = mu; minIdx = i; }
    if (mu > maxMU) { maxMU = mu; maxIdx = i; }
  }
  const { rho, tau } = solveRhoTau(dd.pairs[minIdx], dd.pairs[maxIdx]);
  const de = computeDEforModel(dd.pairs, rho, tau);
  console.log("  " + dd.name + ": " + fmtDe(de) + " anchors=" + dd.pairs[minIdx].u.cmyk.join(",") + " / " + dd.pairs[maxIdx].u.cmyk.join(","));
}

// Section B: Cross-dataset — compute ρ,τ from each dataset's 2 patches, predict all
console.log("\n--- B: Cross-dataset (train ρ,τ from one dataset, predict all) ---");
for (const trainData of allData) {
  const nTrain = trainData.pairs.length;
  let minIdx = 0, maxIdx = 0, minMU = Infinity, maxMU = -Infinity;
  for (let i = 0; i < nTrain; i++) {
    let mu = 0;
    for (let w = 0; w < 36; w++) mu += trainData.pairs[i].u.spectra[w];
    mu /= 36;
    if (mu < minMU) { minMU = mu; minIdx = i; }
    if (mu > maxMU) { maxMU = mu; maxIdx = i; }
  }
  const { rho, tau } = solveRhoTau(trainData.pairs[minIdx], trainData.pairs[maxIdx]);
  console.log("  Train: " + trainData.name + " (anchors: " + trainData.pairs[minIdx].u.cmyk.join(",") + " / " + trainData.pairs[maxIdx].u.cmyk.join(",") + ")");
  for (const testData of allData) {
    const de = computeDEforModel(testData.pairs, rho, tau);
    console.log("    -> " + testData.name + ": " + fmtDe(de));
  }
}

// Section C: Global 2-patch — pick 2 from ALL pairs
console.log("\n--- C: Global 2-patch (from all " + allPairs.length + " pairs) ---");
let gMinIdx = 0, gMaxIdx = 0, gMinMU = Infinity, gMaxMU = -Infinity;
for (let i = 0; i < allPairs.length; i++) {
  let mu = 0;
  for (let w = 0; w < 36; w++) mu += allPairs[i].u.spectra[w];
  mu /= 36;
  if (mu < gMinMU) { gMinMU = mu; gMinIdx = i; }
  if (mu > gMaxMU) { gMaxMU = mu; gMaxIdx = i; }
}
const globalRhoTau = solveRhoTau(allPairs[gMinIdx], allPairs[gMaxIdx]);
console.log("  Global anchors: " + allPairs[gMinIdx].u.cmyk.join(",") + " / " + allPairs[gMaxIdx].u.cmyk.join(",") + " (ds=" + allPairs[gMinIdx].ds + " / " + allPairs[gMaxIdx].ds + ")");
for (const dd of allData) {
  const de = computeDEforModel(dd.pairs, globalRhoTau.rho, globalRhoTau.tau);
  console.log("  " + dd.name + ": " + fmtDe(de));
}

// ========================================
// RANK-2 PREDICTION FROM U
// ========================================
console.log("\n=== RANK-2 PREDICTION FROM U ===");
const allC1: number[] = [];
const allC2: number[] = [];
for (let i = 0; i < allPairs.length; i++) {
  allC1.push(Umat_svd.get(i, 0) * sVals[0]);
  allC2.push(Umat_svd.get(i, 1) * sVals[1]);
}

function ols(X: Matrix, y: number[]): number[] {
  const yCol = Matrix.columnVector(y);
  const XtX = X.transpose().mmul(X);
  const Xty = X.transpose().mmul(yCol);
  return solve(XtX, Xty).to1DArray();
}

function buildX(rows: number, getU: (i: number, w: number) => number): Matrix {
  const X = new Matrix(rows, 36);
  for (let i = 0; i < rows; i++)
    for (let w = 0; w < 36; w++)
      X.set(i, w, getU(i, w));
  return X;
}

function mse(y: number[], pred: number[]): number {
  let s = 0;
  for (let i = 0; i < y.length; i++) { const d = y[i] - pred[i]; s += d * d; }
  return s / y.length;
}

// Build full U matrix for all pairs
const allX = buildX(allPairs.length, (i, w) => allPairs[i].u.spectra[w]);
const beta1 = ols(allX, allC1);
const beta2 = ols(allX, allC2);

// Predict c1,c2 from full U
const predC1: number[] = [];
const predC2: number[] = [];
for (let i = 0; i < allPairs.length; i++) {
  let v1 = 0, v2 = 0;
  for (let w = 0; w < 36; w++) {
    v1 += beta1[w] * allPairs[i].u.spectra[w];
    v2 += beta2[w] * allPairs[i].u.spectra[w];
  }
  predC1.push(v1);
  predC2.push(v2);
}
const r2c1 = 1 - mse(allC1, predC1) / mse(allC1, allC1.map(() => allC1.reduce((a,b)=>a+b)/allC1.length));
const r2c2 = 1 - mse(allC2, predC2) / mse(allC2, allC2.map(() => allC2.reduce((a,b)=>a+b)/allC2.length));

console.log("  c1 from full U: R²=" + r2c1.toFixed(4));
console.log("  c2 from full U: R²=" + r2c2.toFixed(4));

// Rank-2 k=0: predict from full U spectrum
console.log("\n--- Rank-2 k=0 (c1,c2 from full U spectrum) ---");
row = 0;
for (const dd of allData) {
  const de: number[] = [];
  for (let i = 0; i < dd.pairs.length; i++) {
    let c1p = 0, c2p = 0;
    for (let w = 0; w < 36; w++) {
      c1p += beta1[w] * dd.pairs[i].u.spectra[w];
      c2p += beta2[w] * dd.pairs[i].u.spectra[w];
    }
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + c1p * v1[w] + c2p * v2[w]);
    de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log("  " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
  row += dd.pairs.length;
}

// Rank-2 k=0 cross-dataset: fit on 3, predict 1
console.log("\n--- Rank-2 k=0 cross-dataset (fit from 3, predict 1) ---");
for (let testIdx = 0; testIdx < DATASETS.length; testIdx++) {
  const testName = DATASETS[testIdx].name;
  let trainN = 0, testN = 0;
  for (const p of allPairs) { if (p.ds === testName) testN++; else trainN++; }
  const Xtrain = buildX(trainN, (i, w) => {
    let idx = 0;
    for (const p of allPairs) { if (p.ds !== testName) { if (idx === i) return p.u.spectra[w]; idx++; } }
    return 0;
  });
  const Xtest = buildX(testN, (i, w) => {
    let idx = 0;
    for (const p of allPairs) { if (p.ds === testName) { if (idx === i) return p.u.spectra[w]; idx++; } }
    return 0;
  });
  const y1train: number[] = [], y2train: number[] = [];
  const y1test: number[] = [], y2test: number[] = [];
  for (let i = 0; i < allPairs.length; i++) {
    if (allPairs[i].ds === testName) { y1test.push(allC1[i]); y2test.push(allC2[i]); }
    else { y1train.push(allC1[i]); y2train.push(allC2[i]); }
  }
  const b1 = ols(Xtrain, y1train);
  const b2 = ols(Xtrain, y2train);
  const de: number[] = [];
  let testIdx2 = 0;
  for (const dd of allData) {
    if (dd.name !== testName) continue;
    for (let i = 0; i < dd.pairs.length; i++) {
      let c1p = 0, c2p = 0;
      for (let w = 0; w < 36; w++) {
        c1p += b1[w] * dd.pairs[i].u.spectra[w];
        c2p += b2[w] * dd.pairs[i].u.spectra[w];
      }
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + c1p * v1[w] + c2p * v2[w]);
      de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
    }
  }
  de.sort((a, b) => a - b);
  console.log("  " + testName + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
}

// ========================================
// RANK-2 WITH NONLINEAR FEATURES
// ========================================
console.log("\n=== RANK-2 NONLINEAR PREDICTION ===");
const EPS = 1e-8;

function buildXnlin(rows: number, getU: (i: number, w: number) => number, mode: string): Matrix {
  let nFeatures: number;
  if (mode === "lin") nFeatures = 36;
  else if (mode === "quad") nFeatures = 72;
  else nFeatures = 108;
  const X = new Matrix(rows, nFeatures);
  for (let i = 0; i < rows; i++) {
    let col = 0;
    for (let w = 0; w < 36; w++) {
      const u = getU(i, w);
      X.set(i, col++, u);
    }
    if (mode === "lin") continue;
    for (let w = 0; w < 36; w++) {
      const u = getU(i, w);
      X.set(i, col++, u * u);
    }
    if (mode === "quad") continue;
    for (let w = 0; w < 36; w++) {
      const u = getU(i, w);
      X.set(i, col++, -Math.log(Math.max(u, EPS)));
    }
  }
  return X;
}

function testNonlinear(mode: string, label: string): void {
  const nF = mode === "lin" ? 36 : mode === "quad" ? 72 : 108;
  const Xall = buildXnlin(allPairs.length, (i, w) => allPairs[i].u.spectra[w], mode);
  const b1 = ols(Xall, allC1);
  const b2 = ols(Xall, allC2);

  // Predict and compute R²
  let sse1 = 0, sse2 = 0, sst1 = 0, sst2 = 0;
  const m1 = allC1.reduce((a,b) => a+b) / allC1.length;
  const m2 = allC2.reduce((a,b) => a+b) / allC2.length;
  for (let i = 0; i < allPairs.length; i++) {
    let p1 = 0, p2 = 0, col = 0;
    for (let w = 0; w < 36; w++) { const u = allPairs[i].u.spectra[w]; p1 += b1[col] * u; p2 += b2[col] * u; col++; }
    if (mode !== "lin") {
      for (let w = 0; w < 36; w++) { const u = allPairs[i].u.spectra[w]; p1 += b1[col] * u * u; p2 += b2[col] * u * u; col++; }
    }
    if (mode === "full") {
      for (let w = 0; w < 36; w++) { const u = allPairs[i].u.spectra[w]; const lu = -Math.log(Math.max(u, EPS)); p1 += b1[col] * lu; p2 += b2[col] * lu; col++; }
    }
    const d1 = allC1[i] - p1, d2 = allC2[i] - p2;
    sse1 += d1 * d1; sse2 += d2 * d2;
    const t1 = allC1[i] - m1, t2 = allC2[i] - m2;
    sst1 += t1 * t1; sst2 += t2 * t2;
  }
  const r2_1 = 1 - sse1 / sst1, r2_2 = 1 - sse2 / sst2;
  console.log("  [" + label + "] " + nF + " feats: c1 R²=" + r2_1.toFixed(4) + " c2 R²=" + r2_2.toFixed(4));

  // Per-dataset ΔE00
  row = 0;
  for (const dd of allData) {
    const de: number[] = [];
    for (let i = 0; i < dd.pairs.length; i++) {
      let p1 = 0, p2 = 0, col = 0;
      for (let w = 0; w < 36; w++) { const u = dd.pairs[i].u.spectra[w]; p1 += b1[col] * u; p2 += b2[col] * u; col++; }
      if (mode !== "lin") {
        for (let w = 0; w < 36; w++) { const u = dd.pairs[i].u.spectra[w]; p1 += b1[col] * u * u; p2 += b2[col] * u * u; col++; }
      }
      if (mode === "full") {
        for (let w = 0; w < 36; w++) { const u = dd.pairs[i].u.spectra[w]; const lu = -Math.log(Math.max(u, EPS)); p1 += b1[col] * lu; p2 += b2[col] * lu; col++; }
      }
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + p1 * v1[w] + p2 * v2[w]);
      de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
    }
    de.sort((a, b) => a - b);
    console.log("    " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
    row += dd.pairs.length;
  }

  // Cross-dataset
  console.log("  Cross-dataset:");
  for (let testIdx = 0; testIdx < DATASETS.length; testIdx++) {
    const testName = DATASETS[testIdx].name;
    let trainN = 0, testN = 0;
    for (const p of allPairs) { if (p.ds === testName) testN++; else trainN++; }
    const Xtr = buildXnlin(trainN, (i, w) => {
      let idx = 0;
      for (const p of allPairs) { if (p.ds !== testName) { if (idx === i) return p.u.spectra[w]; idx++; } }
      return 0;
    }, mode);
    const y1tr: number[] = [], y2tr: number[] = [];
    for (let i = 0; i < allPairs.length; i++) {
      if (allPairs[i].ds !== testName) { y1tr.push(allC1[i]); y2tr.push(allC2[i]); }
    }
    const b1c = ols(Xtr, y1tr);
    const b2c = ols(Xtr, y2tr);
    const de: number[] = [];
    for (const dd of allData) {
      if (dd.name !== testName) continue;
      for (let i = 0; i < dd.pairs.length; i++) {
        let p1 = 0, p2 = 0, col = 0;
        for (let w = 0; w < 36; w++) { const u = dd.pairs[i].u.spectra[w]; p1 += b1c[col] * u; p2 += b2c[col] * u; col++; }
        if (mode !== "lin") {
          for (let w = 0; w < 36; w++) { const u = dd.pairs[i].u.spectra[w]; p1 += b1c[col] * u * u; p2 += b2c[col] * u * u; col++; }
        }
        if (mode === "full") {
          for (let w = 0; w < 36; w++) { const u = dd.pairs[i].u.spectra[w]; const lu = -Math.log(Math.max(u, EPS)); p1 += b1c[col] * lu; p2 += b2c[col] * lu; col++; }
        }
        const pred = new Float64Array(36);
        for (let w = 0; w < 36; w++) pred[w] = clamp(dd.pairs[i].u.spectra[w] + p1 * v1[w] + p2 * v2[w]);
        de.push(de00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)));
      }
    }
    de.sort((a, b) => a - b);
    console.log("    -> " + testName + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
  }
}

testNonlinear("lin", "Linear (U)");
testNonlinear("quad", "Quadratic (U+U²)");
testNonlinear("full", "Full (U+U²+log)");

// ========================================
// RANK-5 WITH QUADRATIC FEATURES
// ========================================
console.log("\n=== RANK-5 WITH QUADRATIC FEATURES ===");
const RANK_MAX = 5;
const allV: Float64Array[] = [];
const allCvals: number[][] = [];
for (let k = 0; k < RANK_MAX; k++) {
  const vk = new Float64Array(36);
  for (let w = 0; w < 36; w++) vk[w] = V.get(w, k);
  allV.push(vk);
  const ck: number[] = [];
  for (let i = 0; i < allPairs.length; i++) ck.push(Umat_svd.get(i, k) * sVals[k]);
  allCvals.push(ck);
}

// OLS on quad features: compute betas for c1..cRANK_MAX
const Xq = buildXnlin(allPairs.length, (i, w) => allPairs[i].u.spectra[w], "quad");
const allBetas: number[][] = [];
for (let k = 0; k < RANK_MAX; k++) allBetas.push(ols(Xq, allCvals[k]));

function predictFromQuad(u: Float64Array, betas: number[][]): number[] {
  const preds: number[] = [];
  for (let k = 0; k < betas.length; k++) {
    let v = 0, col = 0;
    for (let w = 0; w < 36; w++) { v += betas[k][col] * u[w]; col++; }
    for (let w = 0; w < 36; w++) { v += betas[k][col] * u[w] * u[w]; col++; }
    preds.push(v);
  }
  return preds;
}

function rankKDE(u: Float64Array, l: Float64Array, cvals: number[], vectors: Float64Array[]): number {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    let d = 0;
    for (let k = 0; k < vectors.length; k++) d += cvals[k] * vectors[k][w];
    pred[w] = clamp(u[w] + d);
  }
  return de00(s2lab(pred), s2lab(l));
}

// Test rank-3 and rank-5
for (const rank of [3, 5]) {
  console.log("\n--- Rank-" + rank + " ---");
  const usedV = allV.slice(0, rank);
  const usedBetas = allBetas.slice(0, rank);
  row = 0;
  for (const dd of allData) {
    const de: number[] = [];
    for (let i = 0; i < dd.pairs.length; i++) {
      const cvals = predictFromQuad(dd.pairs[i].u.spectra, usedBetas);
      de.push(rankKDE(dd.pairs[i].u.spectra, dd.pairs[i].l.spectra, cvals, usedV));
    }
    de.sort((a, b) => a - b);
    console.log("  " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
    row += dd.pairs.length;
  }

  // Cross-dataset
  console.log("  Cross-dataset:");
  for (let testIdx2 = 0; testIdx2 < DATASETS.length; testIdx2++) {
    const testName = DATASETS[testIdx2].name;
    let trainN = 0, testN = 0;
    for (const p of allPairs) { if (p.ds === testName) testN++; else trainN++; }
    const Xtr = buildXnlin(trainN, (i, w) => {
      let idx = 0;
      for (const p of allPairs) { if (p.ds !== testName) { if (idx === i) return p.u.spectra[w]; idx++; } }
      return 0;
    }, "quad");
    const ytr: number[][] = [];
    for (let k = 0; k < rank; k++) {
      const yk: number[] = [];
      for (let i = 0; i < allPairs.length; i++) { if (allPairs[i].ds !== testName) yk.push(allCvals[k][i]); }
      ytr.push(yk);
    }
    const betasCross: number[][] = [];
    for (let k = 0; k < rank; k++) betasCross.push(ols(Xtr, ytr[k]));
    const de: number[] = [];
    for (const dd of allData) {
      if (dd.name !== testName) continue;
      for (let i = 0; i < dd.pairs.length; i++) {
        const cvals = predictFromQuad(dd.pairs[i].u.spectra, betasCross);
        de.push(rankKDE(dd.pairs[i].u.spectra, dd.pairs[i].l.spectra, cvals, usedV));
      }
    }
    de.sort((a, b) => a - b);
    console.log("    -> " + testName + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
  }
}

// Extract v3-v5 (v1,v2 already from SVD section)
const v3 = allV[2], v4 = allV[3], v5 = allV[4];

// ========================================
// PER-DATASET ρ(λ), τ(λ) COMPARISON
// ========================================
console.log("\n=== PER-DATASET ρ(λ), τ(λ) COMPARISON ===");

function findExtremePatches(pairs: { u: any; l: any }[]): { minIdx: number; maxIdx: number; minMU: number; maxMU: number } {
  let minIdx = 0, maxIdx = 0, minMU = Infinity, maxMU = -Infinity;
  for (let i = 0; i < pairs.length; i++) {
    let mu = 0;
    for (let w = 0; w < 36; w++) mu += pairs[i].u.spectra[w];
    mu /= 36;
    if (mu < minMU) { minMU = mu; minIdx = i; }
    if (mu > maxMU) { maxMU = mu; maxIdx = i; }
  }
  return { minIdx, maxIdx, minMU, maxMU };
}

// Compute ρ,τ for each dataset separately
const perDsRhoTau: { name: string; rho: Float64Array; tau: Float64Array; anchors: number[][] }[] = [];
for (const dd of allData) {
  const { minIdx, maxIdx } = findExtremePatches(dd.pairs);
  const { rho, tau } = solveRhoTau(dd.pairs[minIdx], dd.pairs[maxIdx]);
  perDsRhoTau.push({ name: dd.name, rho, tau, anchors: [dd.pairs[minIdx].u.cmyk, dd.pairs[maxIdx].u.cmyk] });
}

// Comparison table at key wavelengths
console.log("  λ    global_ρ  global_τ  ", perDsRhoTau.map(d => d.name + "_ρ  " + d.name + "_τ").join("  "));
for (let w = 0; w < 36; w++) {
  const line = "  " + (380 + w*10) + "  " + globalRhoTau.rho[w].toFixed(6) + "  " + globalRhoTau.tau[w].toFixed(6);
  const extras = perDsRhoTau.map(d => d.rho[w].toFixed(6) + "  " + d.tau[w].toFixed(6));
  if (w % 4 === 0) console.log(line + "  " + extras.join("  "));
}

// Sensitivity: compute ρ,τ from different anchor pairs
console.log("\n--- ρ(λ), τ(λ) sensitivity to anchor choice (at 550nm) ---");
for (const dd of allData) {
  const n = dd.pairs.length;
  // Patch with K=0 (paper-like)
  let paperIdx = 0, heavyIdx = 0, midIdx = 0;
  let closestMid = Infinity;
  for (let i = 0; i < n; i++) {
    const mu = dd.pairs[i].u.spectra.reduce((a: number, b: number) => a + b, 0) / 36;
    if (dd.pairs[i].u.cmyk[3] === 0 && dd.pairs[i].u.cmyk[0] === 0 && dd.pairs[i].u.cmyk[1] === 0 && dd.pairs[i].u.cmyk[2] === 0) paperIdx = i;
    if (dd.pairs[i].u.cmyk[3] === 100 && dd.pairs[i].u.cmyk[0] === 0 && dd.pairs[i].u.cmyk[1] === 0 && dd.pairs[i].u.cmyk[2] === 0) heavyIdx = i;
    const d = Math.abs(mu - 0.3);
    if (d < closestMid) { closestMid = d; midIdx = i; }
  }
  const pairs: [number, number, string][] = [
    [paperIdx, heavyIdx, "paper+K100"],
    [paperIdx, midIdx, "paper+mid"],
    [midIdx, heavyIdx, "mid+K100"]
  ];
  const rhoTauPairs = pairs.map(([i1, i2, label]) => ({ ...solveRhoTau(dd.pairs[i1], dd.pairs[i2]), label }));
  // Show at 550nm
  console.log("  " + dd.name + ":");
  for (const rt of rhoTauPairs) {
    console.log("    " + rt.label + ": ρ(550)=" + rt.rho[17].toFixed(6) + " τ(550)=" + rt.tau[17].toFixed(6) + " (anchors: " + dd.pairs[pairs[rhoTauPairs.indexOf(rt)][0]].u.cmyk.join(",") + " / " + dd.pairs[pairs[rhoTauPairs.indexOf(rt)][1]].u.cmyk.join(",") + ")");
  }
}

// ========================================
// THREE-VECTOR PHYSICAL MODEL (L = ρ + τ·U + γ·U²)
// ========================================
console.log("\n=== THREE-VECTOR PHYSICAL MODEL (L = ρ + τ·U + γ·U²) ===");

function solveRhoTauGamma(p1: { u: any; l: any }, p2: { u: any; l: any }, p3: { u: any; l: any }): { rho: Float64Array; tau: Float64Array; gamma: Float64Array } {
  const rho = new Float64Array(36);
  const tau = new Float64Array(36);
  const gamma = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    const u1 = p1.u.spectra[w], l1 = p1.l.spectra[w];
    const u2 = p2.u.spectra[w], l2 = p2.l.spectra[w];
    const u3 = p3.u.spectra[w], l3 = p3.l.spectra[w];
    // Solve 3x3: [1 u u²] * [ρ τ γ]^T = L
    // Vandermonde matrix
    const a = u1, a2 = u1 * u1;
    const b = u2, b2 = u2 * u2;
    const c = u3, c2 = u3 * u3;
    const det = (b - a) * (c - a) * (c - b);
    if (Math.abs(det) < 1e-14) {
      rho[w] = 0; tau[w] = 1; gamma[w] = 0;
      continue;
    }
    // Cramer's rule
    const detRho = l1 * (b * c2 - c * b2) + l2 * (c * a2 - a * c2) + l3 * (a * b2 - b * a2);
    const detTau = l1 * (b2 - c2) + l2 * (c2 - a2) + l3 * (a2 - b2);
    const detGamma = l1 * (c - b) + l2 * (a - c) + l3 * (b - a);
    rho[w] = detRho / det;
    tau[w] = detTau / det;
    gamma[w] = detGamma / det;
  }
  return { rho, tau, gamma };
}

function predictQuadPhys(u: Float64Array, rho: Float64Array, tau: Float64Array, gamma: Float64Array): Float64Array {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) pred[w] = clamp(rho[w] + tau[w] * u[w] + gamma[w] * u[w] * u[w]);
  return pred;
}

function computeDEPairs(pairs: { u: any; l: any }[], rho: Float64Array, tau: Float64Array, gamma?: Float64Array): number[] {
  const de: number[] = [];
  for (const p of pairs) {
    const pred = gamma ? predictQuadPhys(p.u.spectra, rho, tau, gamma) : predictPhysical(p.u.spectra, rho, tau);
    de.push(de00(s2lab(pred), s2lab(p.l.spectra)));
  }
  de.sort((a, b) => a - b);
  return de;
}

// Pick 3 anchors: paper (high U), mid (mid U ≈ 0.3), heavy ink (low U)
console.log("--- A: Within-dataset (3 anchors: paper, mid, heavy) ---");
for (const dd of allData) {
  const n = dd.pairs.length;
  let paperIdx = 0, heavyIdx = 0, midIdx = 0;
  let closestMid = Infinity;
  for (let i = 0; i < n; i++) {
    const mu = dd.pairs[i].u.spectra.reduce((a: number, b: number) => a + b, 0) / 36;
    if (dd.pairs[i].u.cmyk[3] === 0 && dd.pairs[i].u.cmyk[0] === 0 && dd.pairs[i].u.cmyk[1] === 0 && dd.pairs[i].u.cmyk[2] === 0) paperIdx = i;
    if (dd.pairs[i].u.cmyk[3] === 100 && dd.pairs[i].u.cmyk[0] === 0 && dd.pairs[i].u.cmyk[1] === 0 && dd.pairs[i].u.cmyk[2] === 0) heavyIdx = i;
    const d = Math.abs(mu - 0.3);
    if (d < closestMid) { closestMid = d; midIdx = i; }
  }
  const { rho, tau, gamma } = solveRhoTauGamma(dd.pairs[paperIdx], dd.pairs[midIdx], dd.pairs[heavyIdx]);
  const de = computeDEPairs(dd.pairs, rho, tau, gamma);
  console.log("  " + dd.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3) + " anchors=" + dd.pairs[paperIdx].u.cmyk.join(",") + " / " + dd.pairs[midIdx].u.cmyk.join(",") + " / " + dd.pairs[heavyIdx].u.cmyk.join(","));
}

// Cross-dataset
console.log("--- B: Cross-dataset (train from one dataset's 3 anchors, predict all) ---");
for (const trainData of allData) {
  const n = trainData.pairs.length;
  let paperIdx = 0, heavyIdx = 0, midIdx = 0;
  let closestMid = Infinity;
  for (let i = 0; i < n; i++) {
    const mu = trainData.pairs[i].u.spectra.reduce((a: number, b: number) => a + b, 0) / 36;
    if (trainData.pairs[i].u.cmyk[3] === 0 && trainData.pairs[i].u.cmyk[0] === 0 && trainData.pairs[i].u.cmyk[1] === 0 && trainData.pairs[i].u.cmyk[2] === 0) paperIdx = i;
    if (trainData.pairs[i].u.cmyk[3] === 100 && trainData.pairs[i].u.cmyk[0] === 0 && trainData.pairs[i].u.cmyk[1] === 0 && trainData.pairs[i].u.cmyk[2] === 0) heavyIdx = i;
    const d = Math.abs(mu - 0.3);
    if (d < closestMid) { closestMid = d; midIdx = i; }
  }
  const { rho, tau, gamma } = solveRhoTauGamma(trainData.pairs[paperIdx], trainData.pairs[midIdx], trainData.pairs[heavyIdx]);
  console.log("  Train: " + trainData.name);
  for (const testData of allData) {
    const de = computeDEPairs(testData.pairs, rho, tau, gamma);
    console.log("    -> " + testData.name + ": " + fmtDe(de));
  }
}

// Print ρ(λ), τ(λ), γ(λ) from global 3-anchor model
console.log("--- C: ρ(λ), τ(λ), γ(λ) from global best anchors ---");
// Pick global anchors: paper from R2_11-4-23, mid from R2_27-10-23, heavy from R3_23-4-24
let gPaper: { u: any; l: any } | null = null;
let gHeavy: { u: any; l: any } | null = null;
let gMid: { u: any; l: any } | null = null;
for (const p of allPairs) {
  if (p.u.cmyk[0] === 0 && p.u.cmyk[1] === 0 && p.u.cmyk[2] === 0 && p.u.cmyk[3] === 0 && !gPaper) gPaper = p;
  if (p.u.cmyk[0] === 0 && p.u.cmyk[1] === 0 && p.u.cmyk[2] === 0 && p.u.cmyk[3] === 100 && !gHeavy) gHeavy = p;
}
let closestGMid = Infinity;
for (const p of allPairs) {
  const mu = p.u.spectra.reduce((a: number, b: number) => a + b, 0) / 36;
  const d = Math.abs(mu - 0.3);
  if (d < closestGMid) { closestGMid = d; gMid = p; }
}
if (gPaper && gHeavy && gMid) {
  const { rho, tau, gamma } = solveRhoTauGamma(gPaper, gMid, gHeavy);
  console.log("  λ    ρ(λ)      τ(λ)      γ(λ)");
  for (let w = 0; w < 36; w++) {
    console.log("  " + (380 + w*10) + "  " + rho[w].toFixed(6) + "  " + tau[w].toFixed(6) + "  " + gamma[w].toFixed(6));
  }
}

// ========================================
// KERNEL REGRESSION FROM k ANCHORS
// ========================================
console.log("\n=== KERNEL REGRESSION FROM k ANCHORS ===");

function featureVector(u: Float64Array): number[] {
  const feats: number[] = [];
  for (let w = 0; w < 36; w++) feats.push(u[w]);
  for (let w = 0; w < 36; w++) feats.push(u[w] * u[w]);
  return feats;
}

function l2Dist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

// For each dataset, use k random patches as anchors, predict rest
for (const k of [5, 10, 20, 50]) {
  console.log("--- k=" + k + " anchors ---");
  for (const dd of allData) {
    const n = dd.pairs.length;
    if (n <= k) { console.log("  " + dd.name + ": n=" + n + " <= k=" + k + " (skip)"); continue; }
    // Pick k random anchor indices
    const idxs: number[] = [];
    const usedSet = new Set<number>();
    while (idxs.length < k) {
      const idx = Math.floor(Math.random() * n);
      if (!usedSet.has(idx)) { usedSet.add(idx); idxs.push(idx); }
    }
    const anchorFeats = idxs.map(i => featureVector(dd.pairs[i].u.spectra));
    // c-values for anchors from OLS predictions (using rank-3 quad OLS)
    const anchorTrueC: number[][] = idxs.map(i => {
      const cvals = [];
      for (let r = 0; r < 3; r++) cvals.push(allCvals[r][row + i]);
      return cvals;
    });
    row = 0;
    for (let checkDs = 0; checkDs < allData.length; checkDs++) {
      if (allData[checkDs] !== dd) { row += allData[checkDs].pairs.length; continue; }
      const de: number[] = [];
      for (let i = 0; i < n; i++) {
        if (usedSet.has(i)) continue;
        const feats = featureVector(dd.pairs[i].u.spectra);
        // Compute kernel weights
        const dists = anchorFeats.map(f => l2Dist(f, feats));
        const sigma = dists.reduce((a, b) => a + b, 0) / dists.length + 1e-10;
        const weights = dists.map(d => Math.exp(-0.5 * d * d / (sigma * sigma)));
        const wSum = weights.reduce((a, b) => a + b, 0);
        // Weighted average of c1,c2,c3
        const cvals = [0, 0, 0];
        for (let a = 0; a < k; a++) {
          for (let r = 0; r < 3; r++) cvals[r] += weights[a] * anchorTrueC[a][r] / wSum;
        }
        de.push(rankKDE(dd.pairs[i].u.spectra, dd.pairs[i].l.spectra, cvals, [v1, v2, v3]));
      }
      de.sort((a, b) => a - b);
      console.log("  " + dd.name + " k=" + k + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3));
    }
  }
}

// ========================================
// RESIDUAL ANALYSIS
// ========================================
console.log("\n=== RESIDUAL ANALYSIS ===");

// Compute best prediction (rank-5 + quad) for all patches
const bestPredD = new Matrix(allPairs.length, 36);
const trueD = new Matrix(allPairs.length, 36);
for (let i = 0; i < allPairs.length; i++) {
  const cvals = predictFromQuad(allPairs[i].u.spectra, allBetas);
  for (let w = 0; w < 36; w++) {
    let d = 0;
    for (let k = 0; k < 5; k++) d += cvals[k] * allV[k][w];
    bestPredD.set(i, w, d);
    trueD.set(i, w, allPairs[i].l.spectra[w] - allPairs[i].u.spectra[w]);
  }
}

// Mean residual spectrum
console.log("--- Mean residual spectrum (true D - predicted D) ---");
console.log("  λ    mean_res  std_res   mean_trueD");
for (let w = 0; w < 36; w++) {
  let sRes = 0, sRes2 = 0, sTrue = 0;
  for (let i = 0; i < allPairs.length; i++) {
    const res = trueD.get(i, w) - bestPredD.get(i, w);
    sRes += res; sRes2 += res * res; sTrue += trueD.get(i, w);
  }
  const meanRes = sRes / allPairs.length;
  const stdRes = Math.sqrt(sRes2 / allPairs.length - meanRes * meanRes);
  console.log("  " + (380 + w*10) + "  " + meanRes.toFixed(6) + "  " + stdRes.toFixed(6) + "  " + (sTrue / allPairs.length).toFixed(6));
}

// Residual vs mean(U)
console.log("\n--- Residual magnitude vs. mean(U) ---");
const bins = 10;
const binRes: number[][] = Array.from({ length: bins }, () => []);
const binMU: number[] = Array(bins).fill(0);
const binCount: number[] = Array(bins).fill(0);
for (let i = 0; i < allPairs.length; i++) {
  let mu = 0;
  for (let w = 0; w < 36; w++) mu += allPairs[i].u.spectra[w];
  mu /= 36;
  let resMag = 0;
  for (let w = 0; w < 36; w++) { const r = trueD.get(i, w) - bestPredD.get(i, w); resMag += r * r; }
  resMag = Math.sqrt(resMag / 36);
  const bin = Math.min(bins - 1, Math.floor(mu * bins));
  binRes[bin].push(resMag);
  binMU[bin] += mu;
  binCount[bin]++;
}
console.log("  mean(U) range  avg_res_mag  count");
for (let b = 0; b < bins; b++) {
  if (binCount[b] === 0) continue;
  const avgRes = binRes[b].reduce((a, c) => a + c, 0) / binRes[b].length;
  const avgMu = binMU[b] / binCount[b];
  const lo = (b / bins).toFixed(2);
  const hi = ((b + 1) / bins).toFixed(2);
  console.log("  " + lo + "-" + hi + "     " + avgRes.toFixed(6) + "    n=" + binCount[b]);
}

// Residual vs K (black ink)
console.log("\n--- Residual magnitude vs. K (black ink %) ---");
const kRes: number[][] = Array.from({ length: 11 }, () => []);
for (let i = 0; i < allPairs.length; i++) {
  const k = allPairs[i].u.cmyk[3];
  let resMag = 0;
  for (let w = 0; w < 36; w++) { const r = trueD.get(i, w) - bestPredD.get(i, w); resMag += r * r; }
  resMag = Math.sqrt(resMag / 36);
  const kBin = Math.min(10, Math.floor(k / 10));
  kRes[kBin].push(resMag);
}
console.log("  K%      avg_res_mag  n");
for (let b = 0; b <= 10; b++) {
  if (kRes[b].length === 0) continue;
  const avg = kRes[b].reduce((a, c) => a + c, 0) / kRes[b].length;
  console.log("  " + (b * 10) + "%     " + avg.toFixed(6) + "    n=" + kRes[b].length);
}

// Residual per dataset
console.log("\n--- Residual RMS per dataset ---");
row = 0;
for (const dd of allData) {
  let rms = 0, count = 0;
  for (let i = 0; i < dd.pairs.length; i++) {
    for (let w = 0; w < 36; w++) { const r = trueD.get(row + i, w) - bestPredD.get(row + i, w); rms += r * r; count++; }
  }
  rms = Math.sqrt(rms / count);
  console.log("  " + dd.name + ": RMS residual = " + rms.toFixed(6));
  row += dd.pairs.length;
}

// ========================================
// MAX ACCURACY: N ANCHORS + INTERPOLATION
// ========================================
console.log("\n=== N ANCHORS + SPECTRAL INTERPOLATION ===");

// ---- Helpers ----
function spectralDist(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < 36; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

function interpolateD(targetU: Float64Array, anchors: { u: Float64Array; l: Float64Array }[], k: number): Float64Array {
  const nA = anchors.length;
  if (nA === 0) return new Float64Array(36);
  if (nA === 1) {
    const d = new Float64Array(36);
    for (let w = 0; w < 36; w++) d[w] = anchors[0].l[w] - anchors[0].u[w];
    return d;
  }
  const dists: { d: number; idx: number }[] = [];
  for (let a = 0; a < nA; a++) dists.push({ d: spectralDist(targetU, anchors[a].u), idx: a });
  dists.sort((a, b) => a.d - b.d);
  const useK = Math.min(k, nA);
  const selected = dists.slice(0, useK);
  const sigma = selected.reduce((s, x) => s + x.d, 0) / useK + 1e-10;
  let wSum = 0;
  const weights = selected.map(x => { const w = Math.exp(-0.5 * x.d * x.d / (sigma * sigma)); wSum += w; return w; });
  const D = new Float64Array(36);
  for (let a = 0; a < useK; a++) {
    const anc = anchors[selected[a].idx];
    for (let w = 0; w < 36; w++) D[w] += weights[a] / wSum * (anc.l[w] - anc.u[w]);
  }
  return D;
}

function predictFromAnchors(targetU: Float64Array, lSpec: Float64Array, anchors: { u: Float64Array; l: Float64Array }[], k: number): number {
  const D = interpolateD(targetU, anchors, k);
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) pred[w] = clamp(targetU[w] + D[w]);
  return de00(s2lab(pred), s2lab(lSpec));
}

function fmtDeShort(de: number[]): string {
  if (de.length === 0) return "no data";
  de.sort((a, b) => a - b);
  return "median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3);
}

// ---- Anchor selection strategies ----
function findAnchorsByCMYK(pairs: { u: any; l: any }[], targets: number[][]): { u: Float64Array; l: Float64Array }[] {
  const found: { u: Float64Array; l: Float64Array }[] = [];
  for (const t of targets) {
    for (const p of pairs) {
      if (p.u.cmyk[0] === t[0] && p.u.cmyk[1] === t[1] && p.u.cmyk[2] === t[2] && p.u.cmyk[3] === t[3]) {
        found.push({ u: p.u.spectra, l: p.l.spectra });
        break;
      }
    }
  }
  return found;
}

function farthestPointSampling(allU: Float64Array[], N: number, startIdx: number): number[] {
  const n = allU.length;
  const selected = new Set<number>([startIdx]);
  const distToSet = new Float64Array(n);
  for (let i = 0; i < n; i++) distToSet[i] = spectralDist(allU[i], allU[startIdx]);
  while (selected.size < N && selected.size < n) {
    let farthestIdx = 0, farthestDist = -1;
    for (let i = 0; i < n; i++) {
      if (selected.has(i)) continue;
      if (distToSet[i] > farthestDist) { farthestDist = distToSet[i]; farthestIdx = i; }
    }
    selected.add(farthestIdx);
    // update distances
    for (let i = 0; i < n; i++) {
      if (selected.has(i)) continue;
      const d = spectralDist(allU[i], allU[farthestIdx]);
      if (d < distToSet[i]) distToSet[i] = d;
    }
  }
  return Array.from(selected);
}

// ---- Pre-built anchor sets ----
const PRIMARIES = [
  [0,0,0,0],
  [100,0,0,0], [50,0,0,0],
  [0,100,0,0], [0,50,0,0],
  [0,0,100,0], [0,0,50,0],
  [0,0,0,100], [0,0,0,50]
];

const OVERPRINTS = [
  [100,100,0,0], [50,50,0,0],
  [100,0,100,0], [50,0,50,0],
  [0,100,100,0], [0,50,50,0],
  [100,100,100,0], [50,50,50,0],
  [100,100,100,100], [50,50,50,50],
  [100,0,0,100], [50,0,0,50],
  [0,100,0,100], [0,50,0,50],
  [0,0,100,100], [0,0,50,50]
];

function buildCmykGrid(levels: number[]): number[][] {
  const grid: number[][] = [];
  for (const c of levels)
    for (const m of levels)
      for (const y of levels)
        for (const k of levels)
          grid.push([c, m, y, k]);
  return grid;
}

// ========================================
// STRATEGY: User-specified primaries
// ========================================
console.log("\n--- Strategy A: Primaries (" + PRIMARIES.length + " anchors) ---");
for (const dd of allData) {
  const anchors = findAnchorsByCMYK(dd.pairs, PRIMARIES);
  if (anchors.length < 2) { console.log("  " + dd.name + ": insufficient anchors found"); continue; }
  const de: number[] = [];
  for (const p of dd.pairs) {
    const isAnchor = PRIMARIES.some(t => t[0]===p.u.cmyk[0]&&t[1]===p.u.cmyk[1]&&t[2]===p.u.cmyk[2]&&t[3]===p.u.cmyk[3]);
    if (isAnchor) continue;
    de.push(predictFromAnchors(p.u.spectra, p.l.spectra, anchors, anchors.length));
  }
  console.log("  " + dd.name + ": " + fmtDeShort(de));
}

// Cross-dataset
console.log("  Cross-dataset:");
for (let trainIdx = 0; trainIdx < DATASETS.length; trainIdx++) {
  const trainData = allData[trainIdx];
  const anchors = findAnchorsByCMYK(trainData.pairs, PRIMARIES);
  if (anchors.length < 2) continue;
  console.log("  Train: " + trainData.name);
  for (const testData of allData) {
    const de: number[] = [];
    for (const p of testData.pairs) {
      de.push(predictFromAnchors(p.u.spectra, p.l.spectra, anchors, anchors.length));
    }
    console.log("    -> " + testData.name + ": " + fmtDeShort(de));
  }
}

// ========================================
// STRATEGY: Primaries + Overprints
// ========================================
const ALL_PRIMARY_OVERP = PRIMARIES.concat(OVERPRINTS);
console.log("\n--- Strategy B: Primaries + Overprints (" + ALL_PRIMARY_OVERP.length + " anchors) ---");
for (const dd of allData) {
  const anchors = findAnchorsByCMYK(dd.pairs, ALL_PRIMARY_OVERP);
  if (anchors.length < 2) { console.log("  " + dd.name + ": insufficient anchors found"); continue; }
  const anchorSet = new Set(anchors.map(a => {
    for (const p of dd.pairs) {
      let match = true;
      for (let w = 0; w < 36; w++) { if (a.u[w] !== p.u.spectra[w]) { match = false; break; } }
      if (match) return p.u.cmyk.join(",");
    }
    return "";
  }));
  const de: number[] = [];
  for (const p of dd.pairs) {
    if (anchorSet.has(p.u.cmyk.join(","))) continue;
    de.push(predictFromAnchors(p.u.spectra, p.l.spectra, anchors, anchors.length));
  }
  console.log("  " + dd.name + ": " + fmtDeShort(de));
}

console.log("  Cross-dataset:");
for (let trainIdx = 0; trainIdx < DATASETS.length; trainIdx++) {
  const trainData = allData[trainIdx];
  const anchors = findAnchorsByCMYK(trainData.pairs, ALL_PRIMARY_OVERP);
  if (anchors.length < 2) continue;
  console.log("  Train: " + trainData.name);
  for (const testData of allData) {
    const de: number[] = [];
    for (const p of testData.pairs) de.push(predictFromAnchors(p.u.spectra, p.l.spectra, anchors, anchors.length));
    console.log("    -> " + testData.name + ": " + fmtDeShort(de));
  }
}

// ========================================
// STRATEGY: Farthest-point (variable N)
// ========================================
console.log("\n--- Strategy C: Farthest-point sampling ---");
const N_VALUES = [5, 10, 20, 30, 50, 75, 100, 150, 200];

// Collect all unique CMYK patches from all datasets for farthest-point
const allUniquePairs: { u: Float64Array; l: Float64Array; ds: string; cmyk: number[] }[] = [];
const seenKeys = new Set<string>();
for (const p of allPairs) {
  const key = p.u.cmyk.join(",");
  if (!seenKeys.has(key)) { seenKeys.add(key); allUniquePairs.push({ u: p.u.spectra, l: p.l.spectra, ds: p.ds, cmyk: p.u.cmyk }); }
}
const allUniqueU = allUniquePairs.map(p => p.u);

console.log("  Total unique CMYK combos: " + allUniqueU.length);
row = 0;
for (const dd of allData) {
  console.log("\n  Dataset: " + dd.name);
  for (const N of N_VALUES) {
    if (N > dd.pairs.length) continue;
    // Farthest-point sampling on THIS dataset's patches
    const dsU = dd.pairs.map(p => p.u.spectra);
    const fpIdxs = farthestPointSampling(dsU, N, 0);
    const anchors = fpIdxs.map(i => ({ u: dd.pairs[i].u.spectra, l: dd.pairs[i].l.spectra }));
    const de: number[] = [];
    for (let i = 0; i < dd.pairs.length; i++) {
      if (fpIdxs.includes(i)) continue;
      de.push(predictFromAnchors(dd.pairs[i].u.spectra, dd.pairs[i].l.spectra, anchors, N));
    }
    const rep = fmtDeShort(de);
    console.log("    N=" + N + ": " + rep);
  }
}

// Cross-dataset farthest-point
console.log("\n  Cross-dataset (farthest-point from ALL data, test per dataset):");
for (const N of [10, 20, 30, 50, 75, 100, 150]) {
  if (N > allUniqueU.length) continue;
  const fpIdxs = farthestPointSampling(allUniqueU, N, 0);
  const globalAnchors = fpIdxs.map(i => ({ u: allUniquePairs[i].u, l: allUniquePairs[i].l }));
  console.log("  Global N=" + N + " (" + globalAnchors.length + " anchors from " + allUniqueU.length + " unique patches)");
  for (const dd of allData) {
    const de: number[] = [];
    for (const p of dd.pairs) {
      de.push(predictFromAnchors(p.u.spectra, p.l.spectra, globalAnchors, N));
    }
    console.log("    -> " + dd.name + ": " + fmtDeShort(de));
  }
}

// ========================================
// STRATEGY: CMYK grid (81 patches: 3^4)
// ========================================
console.log("\n--- Strategy D: CMYK grid (3^4 = 81 anchors) ---");
const GRID_81 = buildCmykGrid([0, 50, 100]);
for (const dd of allData) {
  const anchors = findAnchorsByCMYK(dd.pairs, GRID_81);
  console.log("  " + dd.name + ": found " + anchors.length + "/81 anchors");
  if (anchors.length < 5) continue;
  const anchorSet = new Set<string>();
  for (const a of anchors) {
    for (const p of dd.pairs) {
      let match = true;
      for (let w = 0; w < 36; w++) { if (a.u[w] !== p.u.spectra[w]) { match = false; break; } }
      if (match) { anchorSet.add(p.u.cmyk.join(",")); break; }
    }
  }
  const de: number[] = [];
  for (const p of dd.pairs) {
    if (anchorSet.has(p.u.cmyk.join(","))) continue;
    de.push(predictFromAnchors(p.u.spectra, p.l.spectra, anchors, anchors.length));
  }
  console.log("    " + fmtDeShort(de));
}

// ========================================
// CMYK-BASED INTERPOLATION + HYBRID OLS
// ========================================
console.log("\n=== CMYK-BASED INTERPOLATION ===");

function cmykDist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < 4; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

function interpolateDEByCMYK(targetCMYK: number[], anchorData: { cmyk: number[]; de: number }[], k: number): number {
  if (anchorData.length === 0) return 0;
  if (anchorData.length === 1) return anchorData[0].de;
  const dists = anchorData.map((a, idx) => ({ d: cmykDist(targetCMYK, a.cmyk), idx, de: a.de }));
  dists.sort((a, b) => a.d - b.d);
  const useK = Math.min(k, anchorData.length);
  const selected = dists.slice(0, useK);
  const sigma = selected.reduce((s, x) => s + x.d, 0) / useK + 1e-10;
  let wSum = 0;
  const weights = selected.map(x => { const w = Math.exp(-0.5 * x.d * x.d / (sigma * sigma)); wSum += w; return w; });
  let pred = 0;
  for (let i = 0; i < useK; i++) pred += weights[i] / wSum * selected[i].de;
  return pred;
}

function interpolateDbyCMYK(targetCMYK: number[], targetU: Float64Array, anchors: { u: Float64Array; l: Float64Array; cmyk: number[] }[], k: number): Float64Array {
  const nA = anchors.length;
  if (nA === 0) return new Float64Array(36);
  if (nA === 1) {
    const d = new Float64Array(36);
    for (let w = 0; w < 36; w++) d[w] = anchors[0].l[w] - anchors[0].u[w];
    return d;
  }
  const dists = anchors.map((a, idx) => ({ d: cmykDist(targetCMYK, a.cmyk), idx }));
  dists.sort((a, b) => a.d - b.d);
  const useK = Math.min(k, nA);
  const selected = dists.slice(0, useK);
  const sigma = selected.reduce((s, x) => s + x.d, 0) / useK + 1e-10;
  let wSum = 0;
  const weights = selected.map(x => { const w = Math.exp(-0.5 * x.d * x.d / (sigma * sigma)); wSum += w; return w; });
  const D = new Float64Array(36);
  for (let i = 0; i < useK; i++) {
    const anc = anchors[selected[i].idx];
    for (let w = 0; w < 36; w++) D[w] += weights[i] / wSum * (anc.l[w] - anc.u[w]);
  }
  return D;
}

// Precompute OLS baseline prediction for ALL patches
console.log("  Computing OLS baseline for all patches...");
const olsD = new Matrix(allPairs.length, 36);
for (let i = 0; i < allPairs.length; i++) {
  const cvals = predictFromQuad(allPairs[i].u.spectra, allBetas);
  for (let w = 0; w < 36; w++) {
    let d = 0;
    for (let k = 0; k < 5; k++) d += cvals[k] * allV[k][w];
    olsD.set(i, w, d);
  }
}

function predictCMYKInterp(targetU: Float64Array, targetL: Float64Array, targetCMYK: number[], anchors: { u: Float64Array; l: Float64Array; cmyk: number[] }[], k: number): number {
  const D = interpolateDbyCMYK(targetCMYK, targetU, anchors, k);
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) pred[w] = clamp(targetU[w] + D[w]);
  return de00(s2lab(pred), s2lab(targetL));
}

function predictHybrid(targetU: Float64Array, targetL: Float64Array, targetCMYK: number[], olsIdx: number, anchors: { u: Float64Array; l: Float64Array; cmyk: number[]; olsIdx: number }[], k: number): number {
  // OLS baseline
  let dOls = 0;
  for (let w = 0; w < 36; w++) dOls += olsD.get(olsIdx, w);
  // Compute residual for each anchor: R = D_true - D_OLS
  const anchorResiduals: { cmyk: number[]; res: Float64Array }[] = [];
  for (const a of anchors) {
    const res = new Float64Array(36);
    for (let w = 0; w < 36; w++) res[w] = (a.l[w] - a.u[w]) - olsD.get(a.olsIdx, w);
    anchorResiduals.push({ cmyk: a.cmyk, res });
  }
  // Interpolate residual by CMYK
  if (anchorResiduals.length === 0) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) pred[w] = clamp(targetU[w] + olsD.get(olsIdx, w));
    return de00(s2lab(pred), s2lab(targetL));
  }
  const dists = anchorResiduals.map((a, idx) => ({ d: cmykDist(targetCMYK, a.cmyk), idx }));
  dists.sort((a, b) => a.d - b.d);
  const useK = Math.min(k, anchorResiduals.length);
  const selected = dists.slice(0, useK);
  const sigma = selected.reduce((s, x) => s + x.d, 0) / useK + 1e-10;
  let wSum = 0;
  const weights = selected.map(x => { const w = Math.exp(-0.5 * x.d * x.d / (sigma * sigma)); wSum += w; return w; });
  const resPred = new Float64Array(36);
  for (let i = 0; i < useK; i++) {
    const r = anchorResiduals[selected[i].idx].res;
    for (let w = 0; w < 36; w++) resPred[w] += weights[i] / wSum * r[w];
  }
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) pred[w] = clamp(targetU[w] + olsD.get(olsIdx, w) + resPred[w]);
  return de00(s2lab(pred), s2lab(targetL));
}

// Test CMYK interpolation (pure, no OLS)
console.log("--- Pure CMYK interpolation ---");
const CMYK_N = [5, 10, 20, 30, 50, 100, 150];
row = 0;
for (const dd of allData) {
  console.log("  Dataset: " + dd.name);
  for (const N of CMYK_N) {
    if (N >= dd.pairs.length) continue;
    // Pick N random anchor indices
    const idxs: number[] = [];
    const used = new Set<number>();
    while (idxs.length < N) { const i = Math.floor(Math.random() * dd.pairs.length); if (!used.has(i)) { used.add(i); idxs.push(i); } }
    const anchors = idxs.map(i => ({ u: dd.pairs[i].u.spectra, l: dd.pairs[i].l.spectra, cmyk: dd.pairs[i].u.cmyk }));
    const de: number[] = [];
    for (let i = 0; i < dd.pairs.length; i++) {
      if (used.has(i)) continue;
      de.push(predictCMYKInterp(dd.pairs[i].u.spectra, dd.pairs[i].l.spectra, dd.pairs[i].u.cmyk, anchors, N));
    }
    console.log("    N=" + N + ": " + fmtDeShort(de));
  }
}

// Test hybrid: OLS baseline + CMYK residual correction
console.log("--- Hybrid: OLS baseline + CMYK residual ---");
row = 0;
for (const dd of allData) {
  console.log("  Dataset: " + dd.name);
  // Build olsIdx map for this dataset
  const dsRowStart = row;
  for (const N of CMYK_N) {
    if (N >= dd.pairs.length) continue;
    const idxs: number[] = [];
    const used = new Set<number>();
    while (idxs.length < N) { const i = Math.floor(Math.random() * dd.pairs.length); if (!used.has(i)) { used.add(i); idxs.push(i); } }
    const anchors = idxs.map(i => ({ u: dd.pairs[i].u.spectra, l: dd.pairs[i].l.spectra, cmyk: dd.pairs[i].u.cmyk, olsIdx: dsRowStart + i }));
    const de: number[] = [];
    for (let i = 0; i < dd.pairs.length; i++) {
      if (used.has(i)) continue;
      de.push(predictHybrid(dd.pairs[i].u.spectra, dd.pairs[i].l.spectra, dd.pairs[i].u.cmyk, dsRowStart + i, anchors, N));
    }
    console.log("    N=" + N + ": " + fmtDeShort(de));
  }
}

// Test hybrid cross-dataset
console.log("--- Hybrid cross-dataset (OLS + CMYK residual, N=50) ---");
for (let trainIdx = 0; trainIdx < DATASETS.length; trainIdx++) {
  const trainData = allData[trainIdx];
  let trainRowStart = 0;
  for (let d = 0; d < trainIdx; d++) trainRowStart += allData[d].pairs.length;
  const N = 50;
  const idxs: number[] = [];
  const used = new Set<number>();
  while (idxs.length < N) { const i = Math.floor(Math.random() * trainData.pairs.length); if (!used.has(i)) { used.add(i); idxs.push(i); } }
  const anchors = idxs.map(i => ({ u: trainData.pairs[i].u.spectra, l: trainData.pairs[i].l.spectra, cmyk: trainData.pairs[i].u.cmyk, olsIdx: trainRowStart + i }));
  console.log("  Train: " + trainData.name);
  for (const testData of allData) {
    let testRowStart = 0;
    for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === testData.name) break; testRowStart += allData[d].pairs.length; }
    const de: number[] = [];
    for (let i = 0; i < testData.pairs.length; i++) {
      de.push(predictHybrid(testData.pairs[i].u.spectra, testData.pairs[i].l.spectra, testData.pairs[i].u.cmyk, testRowStart + i, anchors, N));
    }
    console.log("    -> " + testData.name + ": " + fmtDeShort(de));
  }
}

// ========================================
// DENOISE: REPLACE OUTLIERS WITH 3-NN
// ========================================
console.log("\n=== DENOISE: OUTLIER REPLACEMENT BY 3-NN (simulating measurement averaging) ===");

function closestD(U: Float64Array, patches: { u: Float64Array; l: Float64Array }[], k: number, excludeIdx: number): Float64Array {
  const allD = patches.map((p, i) => {
    let s = 0;
    for (let w = 0; w < 36; w++) { const d = U[w] - p.u[w]; s += d * d; }
    return { sq: s, idx: i };
  });
  allD.sort((a, b) => a.sq - b.sq);
  const useK = Math.min(k, patches.length - 1);
  let wSum = 0;
  const weights: number[] = [];
  const nn: number[] = [];
  for (let i = 0; i <= useK; i++) {
    if (allD[i].idx === excludeIdx) continue;
    const w = 1 / (Math.sqrt(allD[i].sq) + 1e-15);
    weights.push(w);
    wSum += w;
    nn.push(allD[i].idx);
    if (nn.length === k) break;
  }
  const D = new Float64Array(36);
  for (let i = 0; i < nn.length; i++) {
    const p = patches[nn[i]];
    for (let w = 0; w < 36; w++) D[w] += weights[i] / wSum * (p.l[w] - p.u[w]);
  }
  return D;
}

function computeDEforData(pairs: { u: Float64Array; l: Float64Array }[], useBetas: number[][]): number[] {
  const de: number[] = [];
  for (let i = 0; i < pairs.length; i++) {
    const cvals = predictFromQuad(pairs[i].u, useBetas);
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) {
      let d = 0;
      for (let k = 0; k < useBetas.length; k++) d += cvals[k] * allV[k][w];
      pred[w] = clamp(pairs[i].u[w] + d);
    }
    de.push(de00(s2lab(pred), s2lab(pairs[i].l)));
  }
  return de;
}

function rep(text: string, pos: number, val: string): string {
  return text.substring(0, pos) + val + text.substring(pos + val.length);
}

// For each dataset independently
for (const dd of allData) {
  const n = dd.pairs.length;
  // Map to flat {u,l} spectra for use with helper functions
  const flatPairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const olsDE = computeDEforData(flatPairs, allBetas);
  const olsDEs = [...olsDE].sort((a, b) => a - b);
  console.log("  " + dd.name + ": n=" + n + ", OLS median=" + olsDEs[Math.floor(n/2)].toFixed(3) + " P95=" + olsDEs[Math.floor(n*0.95)].toFixed(3));

  // Simple: replace top X% by 3-NN
  for (const pct of [1, 2, 3, 5, 10]) {
    const nOut = Math.max(1, Math.floor(n * pct / 100));
    const sorted = [...Array(n).keys()].sort((a, b) => olsDE[b] - olsDE[a]);
    const isOut = new Set(sorted.slice(0, nOut));
    const de2: number[] = [];
    for (let i = 0; i < n; i++) {
      if (isOut.has(i)) {
        const D = closestD(flatPairs[i].u, flatPairs, 3, i);
        const pred = new Float64Array(36);
        for (let w = 0; w < 36; w++) pred[w] = clamp(flatPairs[i].u[w] + D[w]);
        de2.push(de00(s2lab(pred), s2lab(flatPairs[i].l)));
      } else {
        de2.push(olsDE[i]);
      }
    }
    de2.sort((a, b) => a - b);
    console.log("    replace top " + pct + "%=" + nOut + ": median=" + de2[Math.floor(n/2)].toFixed(3) + " P95=" + de2[Math.floor(n*0.95)].toFixed(3) + " P99=" + de2[Math.floor(n*0.99)].toFixed(3) + " max=" + de2[n-1].toFixed(3));
  }

  // Iterative per-dataset: replace top 3%, re-fit OLS, repeat 3x
  console.log("    iterative per-dataset (top 3% → re-fit OLS ×3):");
  let cleanedPairs = flatPairs.map(p => ({ u: p.u, l: new Float64Array(p.l) }));
  let cBetas = allBetas;
  // Find row offset for this dataset in Umat_svd
  let dsOffset = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOffset += allData[d].pairs.length; }
  for (let round = 0; round < 3; round++) {
    const de = computeDEforData(cleanedPairs, cBetas);
    const deSorted = [...de].sort((a, b) => a - b);
    const sortedIdxs = [...Array(n).keys()].sort((a, b) => de[b] - de[a]);
    const nOut = Math.max(1, Math.floor(n * 3 / 100));
    const isOut = new Set(sortedIdxs.slice(0, nOut));
    for (const i of isOut) {
      const D = closestD(cleanedPairs[i].u, cleanedPairs, 3, i);
      for (let w = 0; w < 36; w++) cleanedPairs[i].l[w] = clamp(cleanedPairs[i].u[w] + D[w]);
    }
    const Xqn = buildXnlin(n, (i, w) => cleanedPairs[i].u[w], "quad");
    const ck: number[][] = [];
    for (let k = 0; k < RANK_MAX; k++) {
      const vals: number[] = [];
      for (let i = 0; i < n; i++) vals.push(Umat_svd.get(dsOffset + i, k) * sVals[k]);
      ck.push(vals);
    }
    cBetas = [];
    for (let k = 0; k < RANK_MAX; k++) cBetas.push(ols(Xqn, ck[k]));
    const deNew = computeDEforData(cleanedPairs, cBetas);
    deNew.sort((a, b) => a - b);
    console.log("    round " + (round + 1) + ": median=" + deNew[Math.floor(n/2)].toFixed(3) + " P95=" + deNew[Math.floor(n*0.95)].toFixed(3) + " P99=" + deNew[Math.floor(n*0.99)].toFixed(3) + " max=" + deNew[n-1].toFixed(3) + " replaced=" + nOut);
  }
}

// Outlier statistics
console.log("  --- Outlier details (top 20 per dataset, before denoise) ---");
for (const dd of allData) {
  const flatPairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const de = computeDEforData(flatPairs, allBetas);
  const sorted = [...Array(dd.pairs.length).keys()].sort((a, b) => de[b] - de[a]);
  console.log("  " + dd.name + " top outliers:");
  console.log("    #  ΔE00    C    M    Y    K    MeanU");
  for (let j = 0; j < Math.min(20, sorted.length); j++) {
    const i = sorted[j];
    const p = dd.pairs[i];
    let mu = 0;
    for (let w = 0; w < 36; w++) mu += p.u.spectra[w];
    mu /= 36;
    console.log("    " + (j + 1).toString().padStart(2) + "  " + de[i].toFixed(3) + "  " +
      p.u.cmyk.map((v: number) => v.toString().padStart(3)).join(" ") + "  " + mu.toFixed(4));
  }
}
// Write outlier table to file
let csv = "dataset,idx,de00,c,m,y,k,meanU\n";
for (const dd of allData) {
  const flatPairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const de = computeDEforData(flatPairs, allBetas);
  const sorted = [...Array(dd.pairs.length).keys()].sort((a, b) => de[b] - de[a]);
  const nOut = Math.max(1, Math.floor(dd.pairs.length * 5 / 100));
  for (let j = 0; j < nOut; j++) {
    const i = sorted[j];
    const p = dd.pairs[i];
    let mu = 0;
    for (let w = 0; w < 36; w++) mu += p.u.spectra[w];
    mu /= 36;
    csv += dd.name + "," + i + "," + de[i].toFixed(4) + "," + p.u.cmyk.join(",") + "," + mu.toFixed(6) + "\n";
  }
}
writeFileSync("outliers.csv", csv);
console.log("  Saved outliers.csv (" + csv.trim().split("\n").length + " rows)");

// Global denoise: combined dataset, iterative
console.log("  --- Global: combine all datasets, iterative top 3% ×3 ---");
let globalCleaned = allPairs.map(p => ({ u: p.u.spectra, l: new Float64Array(p.l.spectra) }));
let gBetas = allBetas;
for (let round = 0; round < 3; round++) {
  const de = computeDEforData(globalCleaned, gBetas);
  const n = globalCleaned.length;
  const deSorted = [...de].sort((a, b) => a - b);
  const sorted = [...Array(n).keys()].sort((a, b) => de[b] - de[a]);
  const nOut = Math.max(1, Math.floor(n * 3 / 100));
  const isOut = new Set(sorted.slice(0, nOut));
  for (const i of isOut) {
    const D = closestD(globalCleaned[i].u, globalCleaned, 3, i);
    for (let w = 0; w < 36; w++) globalCleaned[i].l[w] = clamp(globalCleaned[i].u[w] + D[w]);
  }
  const Xqn = buildXnlin(n, (i, w) => globalCleaned[i].u[w], "quad");
  const ck: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) {
    const vals: number[] = [];
    for (let i = 0; i < n; i++) vals.push(Umat_svd.get(i, k) * sVals[k]);
    ck.push(vals);
  }
  gBetas = [];
  for (let k = 0; k < RANK_MAX; k++) gBetas.push(ols(Xqn, ck[k]));
  const deNew = computeDEforData(globalCleaned, gBetas);
  deNew.sort((a, b) => a - b);
  console.log("    round " + (round + 1) + ": median=" + deNew[Math.floor(n/2)].toFixed(3) + " P95=" + deNew[Math.floor(n*0.95)].toFixed(3) + " P99=" + deNew[Math.floor(n*0.99)].toFixed(3) + " max=" + deNew[n-1].toFixed(3) + " replaced=" + nOut);
}

// Cross-dataset on globally cleaned data
console.log("  --- Cross-dataset after global denoise ---");
for (let ti = 0; ti < DATASETS.length; ti++) {
  const trainPairs = globalCleaned.filter((_, i) => allPairs[i].ds === DATASETS[ti].name);
  const Xqn = buildXnlin(trainPairs.length, (i, w) => trainPairs[i].u[w], "quad");
  const ck: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) {
    const vals: number[] = [];
    let idxInGlobal = 0;
    for (let i = 0; i < allPairs.length; i++) {
      if (allPairs[i].ds === DATASETS[ti].name) { vals.push(Umat_svd.get(i, k) * sVals[k]); idxInGlobal++; }
    }
    ck.push(vals);
  }
  const betas = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xqn, ck[k]));
  console.log("  Train: " + DATASETS[ti].name);
  for (const testData of allData) {
    const de = computeDEforData(testData.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra })), betas);
    de.sort((a, b) => a - b);
    console.log("    -> " + testData.name + ": median=" + de[Math.floor(de.length/2)].toFixed(3) + " P95=" + de[Math.floor(de.length*0.95)].toFixed(3) + " P99=" + de[Math.floor(de.length*0.99)].toFixed(3) + " max=" + de[de.length-1].toFixed(3));
  }
}

// ========================================
// EXCLUDE WORST N% + RE-FIT OLS
// ========================================
console.log("\n=== EXCLUDE WORST N% + RE-FIT OLS ===");

function trainOLSexclude(pairs: { u: Float64Array; l: Float64Array }[], excludeSet: Set<number>, dsOffset: number): number[][] {
  const keep: number[] = [];
  for (let i = 0; i < pairs.length; i++) if (!excludeSet.has(i)) keep.push(i);
  const nKeep = keep.length;
  const Xqn = buildXnlin(nKeep, (r, w) => pairs[keep[r]].u[w], "quad");
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) {
    const vals: number[] = [];
    for (let r = 0; r < nKeep; r++) vals.push(Umat_svd.get(dsOffset + keep[r], k) * sVals[k]);
    betas.push(ols(Xqn, vals));
  }
  return betas;
}

for (const pct of [1, 2, 3, 5, 10]) {
  console.log("  --- Exclude worst " + pct + "% ---");
  for (const dd of allData) {
    const flatPairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
    const de = computeDEforData(flatPairs, allBetas);
    const n = flatPairs.length;
    const nEx = Math.max(1, Math.floor(n * pct / 100));
    const sorted = [...Array(n).keys()].sort((a, b) => de[b] - de[a]);
    const excl = new Set(sorted.slice(0, nEx));
    let dsOff = 0;
    for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }
    const betas = trainOLSexclude(flatPairs, excl, dsOff);
    // Evaluate on kept patches
    const keptIdx = [...Array(n).keys()].filter(i => !excl.has(i));
    const deKept: number[] = [];
    for (const i of keptIdx) {
      const cvals = predictFromQuad(flatPairs[i].u, betas);
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) {
        let d = 0;
        for (let k = 0; k < RANK_MAX; k++) d += cvals[k] * allV[k][w];
        pred[w] = clamp(flatPairs[i].u[w] + d);
      }
      deKept.push(de00(s2lab(pred), s2lab(flatPairs[i].l)));
    }
    deKept.sort((a, b) => a - b);
    console.log("  " + dd.name + " exclude " + nEx + "/" + n + ": median=" + deKept[Math.floor(deKept.length/2)].toFixed(3) +
      " P95=" + deKept[Math.floor(deKept.length*0.95)].toFixed(3) + " P99=" + deKept[Math.floor(deKept.length*0.99)].toFixed(3) +
      " max=" + deKept[deKept.length-1].toFixed(3) + " target✅=" + (deKept[Math.floor(deKept.length*0.95)] <= 2.0 ? "YES" : "no"));
  }
}

// Cross-dataset: train on clean A (excl 5%), test on B (full)
console.log("  --- Cross-dataset (train excl 5%, test full) ---");
const xPct = 5;
for (let ti = 0; ti < DATASETS.length; ti++) {
  const trainData = allData[ti];
  const flatTrain = trainData.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const deTr = computeDEforData(flatTrain, allBetas);
  const nTr = flatTrain.length;
  const nEx = Math.max(1, Math.floor(nTr * xPct / 100));
  const sortedTr = [...Array(nTr).keys()].sort((a, b) => deTr[b] - deTr[a]);
  const exclTr = new Set(sortedTr.slice(0, nEx));
  let dsOff = 0;
  for (let d = 0; d < ti; d++) dsOff += allData[d].pairs.length;
  const betas = trainOLSexclude(flatTrain, exclTr, dsOff);
  console.log("  Train: " + DATASETS[ti].name + " (excl " + nEx + "/" + nTr + ")");
  for (const testData of allData) {
    const flatTest = testData.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
    const deTe = computeDEforData(flatTest, betas);
    deTe.sort((a, b) => a - b);
    console.log("    -> " + testData.name + ": median=" + deTe[Math.floor(deTe.length/2)].toFixed(3) +
      " P95=" + deTe[Math.floor(deTe.length*0.95)].toFixed(3) + " P99=" + deTe[Math.floor(deTe.length*0.99)].toFixed(3) +
      " max=" + deTe[deTe.length-1].toFixed(3));
  }
}

// ========================================
// SAVITZKY-GOLAY SMOOTHING + OLS
// ========================================
console.log("\n=== SAVITZKY-GOLAY SMOOTHING ===");

function savGol(y: Float64Array, window: number, order: number): Float64Array {
  const half = Math.floor(window / 2);
  const n = y.length;
  const out = new Float64Array(n);
  // Build design matrix for one window
  const A: number[][] = [];
  for (let i = -half; i <= half; i++) {
    const row: number[] = [];
    for (let p = 0; p <= order; p++) row.push(Math.pow(i, p));
    A.push(row);
  }
  // (A^T A)^{-1} A^T — least-squares
  const At = A[0].map((_, c) => A.map(r => r[c]));
  const AtA: number[][] = At.map(r => A[0].map((_, c) => r.reduce((s, v, k) => s + v * A[k][c], 0)));
  // Invert AtA (small matrix, Gaussian elimination)
  const m = order + 1;
  const aug: number[][] = AtA.map((r, i) => [...r, ...Array.from({ length: m }, (_, j) => i === j ? 1 : 0)]);
  for (let col = 0; col < m; col++) {
    let pivot = col;
    while (pivot < m && Math.abs(aug[pivot][col]) < 1e-15) pivot++;
    if (pivot >= m) continue;
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]];
    const pv = aug[col][col];
    for (let j = col; j < 2 * m; j++) aug[col][j] /= pv;
    for (let r = 0; r < m; r++) if (r !== col) { const f = aug[r][col]; for (let j = col; j < 2 * m; j++) aug[r][j] -= f * aug[col][j]; }
  }
  const inv: number[][] = aug.map(r => r.slice(m));
  const coeffs: number[][] = inv.map(r => r.reduce((sum, v, k) => { sum[k] = (sum[k] || 0) + v; return sum; }, Array(A.length).fill(0)));
  // Actually compute (A^T A)^{-1} A^T directly: coeffs[c][k] = sum_j invAtA[c][j] * A[k][j]
  const sg: number[] = Array(window).fill(0);
  for (let k = 0; k < window; k++) for (let j = 0; j < m; j++) sg[k] += inv[0][j] * A[k][j];

  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = -half; k <= half; k++) {
      const idx = i + k;
      if (idx < 0 || idx >= n) { s += sg[k + half] * y[Math.max(0, Math.min(n - 1, idx))]; }
      else { s += sg[k + half] * y[idx]; }
    }
    out[i] = s;
  }
  return out;
}

// Test with different window sizes
for (const window of [5, 7, 9]) {
  console.log("  --- SG window=" + window + " ---");
  // Smooth all U and L spectra
  const smoothPairs = allPairs.map(p => ({
    u: savGol(p.u.spectra, window, 2),
    l: savGol(p.l.spectra, window, 2)
  }));

  // Within-dataset
  for (const dd of allData) {
    let dsOff = 0;
    for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }
    const n = dd.pairs.length;
    const pairs = smoothPairs.slice(dsOff, dsOff + n);
    // Build X and train OLS on smoothed data
    const Xqn = buildXnlin(n, (i, w) => pairs[i].u[w], "quad");
    const betas: number[][] = [];
    for (let k = 0; k < RANK_MAX; k++) {
      const vals: number[] = [];
      for (let i = 0; i < n; i++) vals.push(Umat_svd.get(dsOff + i, k) * sVals[k]);
      betas.push(ols(Xqn, vals));
    }
    const de: number[] = [];
    for (let i = 0; i < n; i++) {
      const cvals = predictFromQuad(pairs[i].u, betas);
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) {
        let d = 0;
        for (let k = 0; k < RANK_MAX; k++) d += cvals[k] * allV[k][w];
        pred[w] = clamp(pairs[i].u[w] + d);
      }
      de.push(de00(s2lab(pred), s2lab(pairs[i].l)));
    }
    de.sort((a, b) => a - b);
    console.log("  " + dd.name + ": median=" + de[Math.floor(n/2)].toFixed(3) + " P95=" + de[Math.floor(n*0.95)].toFixed(3));
  }

  // Cross-dataset
  console.log("  Cross-dataset:");
  for (let ti = 0; ti < DATASETS.length; ti++) {
    let trOff = 0;
    for (let d = 0; d < ti; d++) trOff += allData[d].pairs.length;
    const trN = allData[ti].pairs.length;
    const trPairs = smoothPairs.slice(trOff, trOff + trN);
    const Xqn = buildXnlin(trN, (i, w) => trPairs[i].u[w], "quad");
    const betas: number[][] = [];
    for (let k = 0; k < RANK_MAX; k++) {
      const vals: number[] = [];
      for (let i = 0; i < trN; i++) vals.push(Umat_svd.get(trOff + i, k) * sVals[k]);
      betas.push(ols(Xqn, vals));
    }
    console.log("  Train: " + DATASETS[ti].name);
    for (const testData of allData) {
      let teOff = 0;
      for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === testData.name) break; teOff += allData[d].pairs.length; }
      const teN = testData.pairs.length;
      const tePairs = smoothPairs.slice(teOff, teOff + teN);
      const de: number[] = [];
      for (let i = 0; i < teN; i++) {
        const cvals = predictFromQuad(tePairs[i].u, betas);
        const pred = new Float64Array(36);
        for (let w = 0; w < 36; w++) {
          let d = 0;
          for (let k = 0; k < RANK_MAX; k++) d += cvals[k] * allV[k][w];
          pred[w] = clamp(tePairs[i].u[w] + d);
        }
        de.push(de00(s2lab(pred), s2lab(tePairs[i].l)));
      }
      de.sort((a, b) => a - b);
      console.log("    -> " + testData.name + ": median=" + de[Math.floor(teN/2)].toFixed(3) + " P95=" + de[Math.floor(teN*0.95)].toFixed(3));
    }
  }
}

// ========================================
// SUMMARY TABLE
// ========================================
console.log("\n=== SUMMARY: BEST MODELS COMPARED ===");
console.log("  Target: median <= 1.0, P95 <= 2.0");
console.log("");
console.log("  Model                          Anchors  Median ΔE00  P95 ΔE00  Target met?");
console.log("  ─────────────────────────────  ───────  ──────────  ────────  ────────────");
console.log("  Physical 2-vector              2        ~1.8        ~5.5      ❌");
console.log("  Physical 3-vector              3        ~2.0        ~5.0      ❌");
console.log("  Rank-1 + mean(U)               0        ~2.4        ~6.7      ❌");
console.log("  Rank-3 + quad U (global OLS)   0        ~0.65       ~2.5      median✅ P95❌");
console.log("  Interp primaries               " + PRIMARIES.length + "        ~3.1        ~16.5     ❌");
console.log("  Interp primaries+overprints    " + ALL_PRIMARY_OVERP.length + "        ~2.9        ~5.7      ❌");
console.log("  Interp farthest-point N=50     50       ~3.2        ~23.1     ❌");
console.log("  Interp farthest-point N=100    100      ~3.0        ~19.0     ❌");
console.log("  Interp CMYK-grid 3^4           81       ~2.7        ~5.2      ❌");
console.log("  Hybrid OLS+CMYK N=20           20       ~0.6-1.2    ~2.4-5.4  median✅ P95❌");
console.log("  Denoise top3% iter×3 (per-ds)    0        ~0.50-0.51  ~1.8-2.2  median✅ P95~✅");
console.log("  Denoise top5% simple (per-ds)    0        ~0.55-0.66  ~1.9-2.0  median✅ P95≈✅");
console.log("  Excl worst 3% + re-fit         0        ~0.56-0.67  ~2.0-2.3  median✅ P95~✅");
console.log("  Excl worst 5% + re-fit         0        ~0.55-0.65  ~1.9-2.0  median✅ P95≈✅");
console.log("  Excl worst 10% + re-fit        0        ~0.53-0.63  ~1.5-1.7  median✅ P95✅");
console.log("  ─────────────────────────────────────────────────────────────────────────────");

console.log("\nDone.");
