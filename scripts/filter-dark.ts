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
function buildX(rows: number, getU: (i: number, w: number) => number): Matrix {
  const X = new Matrix(rows, 72);
  for (let i = 0; i < rows; i++) {
    let col = 0;
    for (let w = 0; w < 36; w++) X.set(i, col++, getU(i, w));
    for (let w = 0; w < 36; w++) { const u = getU(i, w); X.set(i, col++, u * u); }
  }
  return X;
}
function ols(X: Matrix, y: number[]) { return solve(X.transpose().mmul(X), X.transpose().mmul(Matrix.columnVector(y))).to1DArray(); }
function predict(u: Float64Array, betas: number[][]): number[] {
  return betas.map(b => { let v = 0, c = 0; for (let w = 0; w < 36; w++) v += b[c++] * u[w]; for (let w = 0; w < 36; w++) v += b[c++] * u[w] * u[w]; return v; });
}
function computeDE(pairs: { u: Float64Array; l: Float64Array }[], betas: number[][], V: Float64Array[]): { de: number[]; lab: [number,number,number][] } {
  const de: number[] = [];
  const labs: [number,number,number][] = [];
  for (const p of pairs) {
    const c = predict(p.u, betas);
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < betas.length; k++) d += c[k] * V[k][w]; pred[w] = clamp(p.u[w] + d); }
    const ll = s2lab(p.l);
    de.push(deltaE00(s2lab(pred), ll));
    labs.push(ll);
  }
  return { de, lab: labs };
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
  for (let i = 0; i < dd.pairs.length; i++)
    for (let w = 0; w < 36; w++) fullD.set(gRow, w, dd.pairs[i].l.spectra[w] - dd.pairs[i].u.spectra[w]);
  gRow++;
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
// MAIN ANALYSIS
// ========================================
console.log("=== ПРАКТИЧЕСКАЯ ОЦЕНКА: КАКОЙ ТОЧНОСТИ ОЖИДАТЬ ===\n");

// 1. Count dark patches
console.log("1. СКОЛЬКО ТЁМНЫХ ПАТЧЕЙ (L* < 10) В КАЖДОМ ДАТАСЕТЕ:\n");
for (const dd of allData) {
  const nDark = dd.pairs.filter(p => s2lab(p.l.spectra)[0] < 10).length;
  const nDarkU = dd.pairs.filter(p => s2lab(p.u.spectra)[0] < 10).length;
  console.log(`  ${dd.name}: ${nDark} ламинированных, ${nDarkU} неламинированных (из ${dd.pairs.length})`);
}

// 2. Per-dataset: train 80%, evaluate on L* ≥ 10 only
console.log("\n2. PER-DATASET ПРЕДИКТОР (OLS rank-5, U+U²):\n");
console.log("  Обучаем на 80% случайных патчей, тестируем на 20%");
console.log("  ΔE00 считаем ТОЛЬКО для патчей с L* ≥ 10 ламинированного\n");
console.log("  Dataset        median     P95       P99       max       n_test   ✅ P95≤2.0\n" + "  " + "─".repeat(75));

const NREPS = 20;
for (const dd of allData) {
  let dsOff = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }

  const pairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const allMeds: number[] = [], allP95: number[] = [], allP99: number[] = [], allMax: number[] = [], allN: number[] = [];

  for (let rep = 0; rep < NREPS; rep++) {
    const idx = [...Array(pairs.length).keys()];
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const nTrain = Math.floor(pairs.length * 80 / 100);
    const trIdx = idx.slice(0, nTrain), teIdx = idx.slice(nTrain);

    const Xtr = buildX(trIdx.length, (i, w) => pairs[trIdx[i]].u[w]);
    const betas: number[][] = [];
    for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xtr, trIdx.map(i => allCvals[k][dsOff + i])));

    const { de, lab } = computeDE(teIdx.map(i => pairs[i]), betas, allV);
    const kept = de.filter((_, i) => lab[i][0] >= 10);
    if (kept.length < 3) continue;
    kept.sort((a, b) => a - b);
    allMeds.push(kept[Math.floor(kept.length / 2)]);
    allP95.push(kept[Math.floor(kept.length * 0.95)]);
    allP99.push(kept[Math.floor(kept.length * 0.99)]);
    allMax.push(kept[kept.length - 1]);
    allN.push(kept.length);
  }

  const avg = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const ok = avg(allP95) <= 2.0 ? "✅" : "❌";
  console.log(`  ${dd.name.padEnd(14)} ${avg(allMeds).toFixed(3).padStart(8)} ${avg(allP95).toFixed(3).padStart(8)} ${avg(allP99).toFixed(3).padStart(8)} ${avg(allMax).toFixed(3).padStart(8)} n=${Math.round(avg(allN)).toString().padStart(4)} ${ok}`);
}

// 3. Cross-dataset with L* filter
console.log("\n3. УНИВЕРСАЛЬНЫЙ ПРЕДИКТОР (один для всех):\n");
console.log("  Обучаем на одном датасете, применяем ко всем остальным\n");
console.log("  ΔE00 ТОЛЬКО для L* ≥ 10\n");

console.log("  Train → Test          median     P95       P99       max       ✅ P95≤2.0\n" + "  " + "─".repeat(65));

for (let ti = 0; ti < DATASETS.length; ti++) {
  const trainDs = allData[ti];
  const trainPairs = trainDs.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  let trOff = 0;
  for (let d = 0; d < ti; d++) trOff += allData[d].pairs.length;

  // train + excl 5%
  const Xtr = buildX(trainPairs.length, (i, w) => trainPairs[i].u[w]);
  const bAll: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) bAll.push(ols(Xtr, allCvals[k].slice(trOff, trOff + trainPairs.length)));
  const { de: deTr } = computeDE(trainPairs, bAll, allV);
  const sTr = [...Array(trainPairs.length).keys()].sort((a, b) => deTr[b] - deTr[a]);
  const excl = new Set(sTr.slice(0, Math.max(1, Math.floor(trainPairs.length * 5 / 100))));
  const keep = [...Array(trainPairs.length).keys()].filter(i => !excl.has(i));
  const Xk = buildX(keep.length, (r, w) => trainPairs[keep[r]].u[w]);
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xk, keep.map(i => allCvals[k][trOff + i])));

  for (let testIdx = 0; testIdx < DATASETS.length; testIdx++) {
    if (testIdx === ti) continue;
    const testDs = allData[testIdx];
    const testPairs = testDs.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
    const { de, lab } = computeDE(testPairs, betas, allV);
    const kept: number[] = [];
    for (let i = 0; i < de.length; i++) if (lab[i][0] >= 10) kept.push(de[i]);
    kept.sort((a, b) => a - b);
    if (kept.length < 2) continue;
    const med = kept[Math.floor(kept.length / 2)];
    const p95 = kept[Math.floor(kept.length * 0.95)];
    const p99 = kept[Math.floor(kept.length * 0.99)];
    const mx = kept[kept.length - 1];
    const ok = p95 <= 2.0 ? "✅" : "❌";
    console.log(`  ${DATASETS[ti].name.padEnd(12)} → ${DATASETS[testIdx].name.padEnd(12)} ${med.toFixed(3).padStart(8)} ${p95.toFixed(3).padStart(8)} ${p99.toFixed(3).padStart(8)} ${mx.toFixed(3).padStart(8)} ${ok}`);
  }
}

// 4. R2→R2 (same substrate)
console.log("\n4. R2→R2 (одна и та же подложка, разные прогоны):\n");
console.log("  Train → Test          median     P95       ✅ P95≤2.0\n" + "  " + "─".repeat(45));

const r2Data = DATASETS.filter(d => d.name.startsWith("R2"));
for (let ti = 0; ti < r2Data.length; ti++) {
  const trainDs = allData.find(d => d.name === r2Data[ti].name)!;
  const trainPairs = trainDs.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  let trOff = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === r2Data[ti].name) break; trOff += allData[d].pairs.length; }

  const Xtr = buildX(trainPairs.length, (i, w) => trainPairs[i].u[w]);
  const bAll: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) bAll.push(ols(Xtr, allCvals[k].slice(trOff, trOff + trainPairs.length)));
  const { de: deTr } = computeDE(trainPairs, bAll, allV);
  const sTr = [...Array(trainPairs.length).keys()].sort((a, b) => deTr[b] - deTr[a]);
  const excl = new Set(sTr.slice(0, Math.max(1, Math.floor(trainPairs.length * 5 / 100))));
  const keep = [...Array(trainPairs.length).keys()].filter(i => !excl.has(i));
  const Xk = buildX(keep.length, (r, w) => trainPairs[keep[r]].u[w]);
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xk, keep.map(i => allCvals[k][trOff + i])));

  for (let testIdx = 0; testIdx < r2Data.length; testIdx++) {
    if (testIdx === ti) continue;
    const testDs = allData.find(d => d.name === r2Data[testIdx].name)!;
    const testPairs = testDs.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
    const { de, lab } = computeDE(testPairs, betas, allV);
    const kept: number[] = [];
    for (let i = 0; i < de.length; i++) if (lab[i][0] >= 10) kept.push(de[i]);
    kept.sort((a, b) => a - b);
    if (kept.length < 2) continue;
    const med = kept[Math.floor(kept.length / 2)];
    const p95 = kept[Math.floor(kept.length * 0.95)];
    console.log(`  ${r2Data[ti].name.padEnd(12)} → ${r2Data[testIdx].name.padEnd(12)} ${med.toFixed(3).padStart(8)} ${p95.toFixed(3).padStart(8)} ${p95 <= 2.0 ? "✅" : "❌"}`);
  }
}

// 5. Distribution analysis: what's in the worst 5%
console.log("\n5. АНАЛИЗ ХВОСТА (худшие 5% патчей, per-dataset OLS):\n");
console.log("  Какие патчи дают ΔE00 > 2.0 и попадают в P95 хвост?\n");

for (const dd of allData) {
  let dsOff = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }
  const pairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const n = pairs.length;
  const Xtr = buildX(n, (i, w) => pairs[i].u[w]);
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xtr, allCvals[k].slice(dsOff, dsOff + n)));
  const { de, lab } = computeDE(pairs, betas, allV);
  const sorted = [...Array(n).keys()].sort((a, b) => de[b] - de[a]);
  const nWorst = Math.max(1, Math.floor(n * 5 / 100));

  console.log(`  ${dd.name} (top ${nWorst} из ${n}):`);
  console.log("    #  ΔE00     L*       C    M    Y    K");
  for (let j = 0; j < Math.min(nWorst, 10); j++) {
    const i = sorted[j];
    const p = dd.pairs[i];
    console.log(`    ${(j+1).toString().padStart(2)}  ${de[i].toFixed(3).padStart(7)}  ${lab[i][0].toFixed(2).padStart(6)}  ${p.u.cmyk.map((v:number) => v.toString().padStart(3)).join(" ")}`);
  }

  // How many of worst 5% have L* < 10?
  const darkInTail = sorted.slice(0, nWorst).filter(i => lab[i][0] < 10).length;
  const darkInTail15 = sorted.slice(0, nWorst).filter(i => lab[i][0] < 15).length;
  console.log(`    Из них L*<10: ${darkInTail}/${nWorst}, L*<15: ${darkInTail15}/${nWorst}\n`);
}
