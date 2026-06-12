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
// KEY QUESTION: What causes the P95 tail?
// ========================================
console.log("=== АНАЛИЗ ХВОСТА: какие патчи дают ΔE00 > 2.0? ===\n");

for (const dd of allData) {
  let dsOff = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }
  const pairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const n = pairs.length;

  // Train on ALL data to compute per-patch ΔE00
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

  const Xtr = buildX(n, (i, w) => pairs[i].u[w]);
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) betas.push(ols(Xtr, allCvals[k].slice(dsOff, dsOff + n)));

  // Compute ΔE00 for ALL patches
  const allDE: number[] = [];
  const allLab: [number,number,number][] = [];
  for (const p of pairs) {
    const c = predict(p.u, betas);
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < RANK_MAX; k++) d += c[k] * allV[k][w]; pred[w] = clamp(p.u[w] + d); }
    const ll = s2lab(p.l);
    allDE.push(deltaE00(s2lab(pred), ll));
    allLab.push(ll);
  }

  // Sort by ΔE00 descending
  const sortedIdx = [...Array(n).keys()].sort((a, b) => allDE[b] - allDE[a]);

  // Stats by L* bins
  console.log(`  ${dd.name}: распределение ΔE00 по яркости L*\n`);
  console.log("    L* bin     count    >2.0     >3.0     >5.0     median   P95");
  const bins = [
    [0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 100]
  ];
  for (const [lo, hi] of bins) {
    const inBin = sortedIdx.filter(i => allLab[i][0] >= lo && allLab[i][0] < hi);
    if (inBin.length === 0) continue;
    const deVals = inBin.map(i => allDE[i]).sort((a, b) => a - b);
    const ct = inBin.length;
    const gt2 = inBin.filter(i => allDE[i] > 2.0).length;
    const gt3 = inBin.filter(i => allDE[i] > 3.0).length;
    const gt5 = inBin.filter(i => allDE[i] > 5.0).length;
    const med = deVals[Math.floor(deVals.length / 2)];
    const p95 = deVals[Math.floor(deVals.length * 0.95)];
    console.log(`    ${lo.toString().padStart(2)}-${hi.toString().padStart(3)}   ${ct.toString().padStart(6)} ${gt2.toString().padStart(7)} ${gt3.toString().padStart(7)} ${gt5.toString().padStart(7)} ${med.toFixed(3).padStart(8)} ${p95.toFixed(3).padStart(7)}`);
  }

  // What are the worst patches?
  const nWorst = Math.max(1, Math.floor(n * 5 / 100));
  console.log(`\n    Худшие ${nWorst} (5%):`);
  console.log("    #  ΔE00     L*      C   M   Y   K");
  for (let j = 0; j < Math.min(nWorst, 15); j++) {
    const i = sortedIdx[j];
    const p = dd.pairs[i];
    console.log(`    ${(j+1).toString().padStart(2)}  ${allDE[i].toFixed(3).padStart(7)}  ${allLab[i][0].toFixed(2).padStart(6)}  ${p.u.cmyk.map((v:number) => v.toString().padStart(3)).join(" ")}`);
  }
  console.log("");
}
