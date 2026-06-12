import { readFileSync, writeFileSync } from "fs";
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

function matchByCMYK(unlam: any[], lam: any[]): { u: any; l: any }[] {
  const lamMap = new Map<string, any[]>();
  for (const p of lam) {
    const key = p.cmyk.join(",");
    if (!lamMap.has(key)) lamMap.set(key, []);
    lamMap.get(key)!.push(p);
  }
  const pairs: { u: any; l: any }[] = [];
  for (const pu of unlam) {
    const key = pu.cmyk.join(",");
    const matches = lamMap.get(key);
    if (matches && matches.length > 0) {
      pairs.push({ u: pu, l: matches[0] });
      matches.shift();
    }
  }
  return pairs;
}

function s2lab(s: Float64Array): [number, number, number] {
  return xyzToLab(spectraToXyz(s), D50_WP);
}

function clamp(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function ols(X: Matrix, y: number[]): number[] {
  const yCol = Matrix.columnVector(y);
  const XtX = X.transpose().mmul(X);
  const Xty = X.transpose().mmul(yCol);
  return solve(XtX, Xty).to1DArray();
}

function savGol(y: Float64Array, window: number, order: number): Float64Array {
  const half = Math.floor(window / 2);
  const n = y.length;
  const out = new Float64Array(n);
  const A: number[][] = [];
  for (let i = -half; i <= half; i++) {
    const row: number[] = [];
    for (let p = 0; p <= order; p++) row.push(Math.pow(i, p));
    A.push(row);
  }
  const m = order + 1;
  const At = A[0].map((_, c) => A.map(r => r[c]));
  const AtA: number[][] = At.map(r => A[0].map((_, c) => r.reduce((s, v, k) => s + v * A[k][c], 0)));
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
  const sg: number[] = Array(window).fill(0);
  for (let k = 0; k < window; k++) for (let j = 0; j < m; j++) sg[k] += inv[0][j] * A[k][j];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = -half; k <= half; k++) {
      const idx = i + k;
      if (idx < 0 || idx >= n) s += sg[k + half] * y[Math.max(0, Math.min(n - 1, idx))];
      else s += sg[k + half] * y[idx];
    }
    out[i] = s;
  }
  return out;
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

console.log("Loaded " + allPairs.length + " total pairs");

// Global SVD
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
const RANK_MAX = 5;
const allV: Float64Array[] = [];
for (let k = 0; k < RANK_MAX; k++) {
  const vk = new Float64Array(36);
  for (let w = 0; w < 36; w++) vk[w] = V.get(w, k);
  allV.push(vk);
}

// c-values for all patches
const Umat_svd = svdFull.leftSingularVectors;
const allCvals: number[][] = [];
for (let k = 0; k < RANK_MAX; k++) {
  const ck: number[] = [];
  for (let i = 0; i < allPairs.length; i++) ck.push(Umat_svd.get(i, k) * sVals[k]);
  allCvals.push(ck);
}

function buildXnlin(rows: number, getU: (i: number, w: number) => number, mode: string): Matrix {
  let nFeatures: number;
  if (mode === "lin") nFeatures = 36;
  else nFeatures = 72;
  const X = new Matrix(rows, nFeatures);
  for (let i = 0; i < rows; i++) {
    let col = 0;
    for (let w = 0; w < 36; w++) X.set(i, col++, getU(i, w));
    for (let w = 0; w < 36; w++) X.set(i, col++, getU(i, w) * getU(i, w));
  }
  return X;
}

function predictFromQuad(u: Float64Array, betas: number[][]): number[] {
  const preds: number[] = [];
  for (let k = 0; k < betas.length; k++) {
    let v = 0, col = 0;
    for (let w = 0; w < 36; w++) v += betas[k][col++] * u[w];
    for (let w = 0; w < 36; w++) v += betas[k][col++] * u[w] * u[w];
    preds.push(v);
  }
  return preds;
}

function computeDEforData(pairs: { u: Float64Array; l: Float64Array }[], betas: number[][]): number[] {
  const de: number[] = [];
  for (let i = 0; i < pairs.length; i++) {
    const cvals = predictFromQuad(pairs[i].u, betas);
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) {
      let d = 0;
      for (let k = 0; k < betas.length; k++) d += cvals[k] * allV[k][w];
      pred[w] = clamp(pairs[i].u[w] + d);
    }
    de.push(deltaE00(s2lab(pred), s2lab(pairs[i].l)));
  }
  return de;
}

function trainOLSexclude(pairs: { u: Float64Array; l: Float64Array }[], excludeSet: Set<number>, dsOffset: number): number[][] {
  const keep: number[] = [];
  for (let i = 0; i < pairs.length; i++) if (!excludeSet.has(i)) keep.push(i);
  const nKeep = keep.length;
  const Xqn = buildXnlin(nKeep, (r, w) => pairs[keep[r]].u[w], "quad");
  const betas: number[][] = [];
  for (let k = 0; k < RANK_MAX; k++) {
    const vals: number[] = [];
    for (let r = 0; r < nKeep; r++) vals.push(allCvals[k][dsOffset + keep[r]]);
    betas.push(ols(Xqn, vals));
  }
  return betas;
}

// ========================================
// COMBINED: SG + EXCLUDE + RE-FIT
// ========================================
console.log("\n=== SG SMOOTHING + EXCLUDE WORST N% + RE-FIT OLS ===");

for (const window of [5, 7]) {
  console.log("\n--- SG window=" + window + " ---");
  const smoothPairs = allPairs.map(p => ({
    u: savGol(p.u.spectra, window, 2),
    l: savGol(p.l.spectra, window, 2)
  }));

  for (const pct of [1, 2, 3, 5, 10]) {
    console.log("  Exclude worst " + pct + "%:");

    for (const dd of allData) {
      let dsOff = 0;
      for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }
      const n = dd.pairs.length;
      const pairs = smoothPairs.slice(dsOff, dsOff + n);

      // Train OLS on ALL smoothed data to identify outliers
      const XqnAll = buildXnlin(n, (i, w) => pairs[i].u[w], "quad");
      const betasAll: number[][] = [];
      for (let k = 0; k < RANK_MAX; k++) {
        const vals: number[] = [];
        for (let i = 0; i < n; i++) vals.push(allCvals[k][dsOff + i]);
        betasAll.push(ols(XqnAll, vals));
      }
      const de = computeDEforData(pairs, betasAll);

      // Exclude worst N%
      const nEx = Math.max(1, Math.floor(n * pct / 100));
      const sorted = [...Array(n).keys()].sort((a, b) => de[b] - de[a]);
      const excl = new Set(sorted.slice(0, nEx));

      // Re-fit on remaining
      const betas = trainOLSexclude(pairs, excl, dsOff);

      // Evaluate on kept (smoothed) patches
      const keep = [...Array(n).keys()].filter(i => !excl.has(i));
      const deKept: number[] = [];
      for (const i of keep) {
        const cvals = predictFromQuad(pairs[i].u, betas);
        const pred = new Float64Array(36);
        for (let w = 0; w < 36; w++) {
          let d = 0;
          for (let k = 0; k < RANK_MAX; k++) d += cvals[k] * allV[k][w];
          pred[w] = clamp(pairs[i].u[w] + d);
        }
        deKept.push(deltaE00(s2lab(pred), s2lab(pairs[i].l)));
      }
      deKept.sort((a, b) => a - b);
      const p95 = deKept[Math.floor(deKept.length * 0.95)];
      console.log("  " + dd.name + " SG" + window + " excl" + pct + " n=" + deKept.length + ": med=" + deKept[Math.floor(deKept.length / 2)].toFixed(3) +
        " P95=" + p95.toFixed(3) + " P99=" + deKept[Math.floor(deKept.length * 0.99)].toFixed(3) +
        " max=" + deKept[deKept.length - 1].toFixed(3) + " target✅=" + (p95 <= 2.0 ? "YES" : "no"));
    }
  }
}

// Also do baseline (no SG) for comparison with same exclude logic
console.log("\n=== BASELINE (NO SG) + EXCLUDE + RE-FIT ===");
for (const pct of [1, 2, 3, 5, 10]) {
  console.log("  Exclude worst " + pct + "%:");
  for (const dd of allData) {
    const flatPairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
    const de = computeDEforData(flatPairs, allCvals.map(cv => {
      const Xq = buildXnlin(flatPairs.length, (i, w) => flatPairs[i].u[w], "quad");
      return ols(Xq, cv);
    }));
    const n = flatPairs.length;
    const nEx = Math.max(1, Math.floor(n * pct / 100));
    const sorted = [...Array(n).keys()].sort((a, b) => de[b] - de[a]);
    const excl = new Set(sorted.slice(0, nEx));
    let dsOff = 0;
    for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; dsOff += allData[d].pairs.length; }
    const betas = trainOLSexclude(flatPairs, excl, dsOff);
    const keep = [...Array(n).keys()].filter(i => !excl.has(i));
    const deKept: number[] = [];
    for (const i of keep) {
      const cvals = predictFromQuad(flatPairs[i].u, betas);
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) {
        let d = 0;
        for (let k = 0; k < RANK_MAX; k++) d += cvals[k] * allV[k][w];
        pred[w] = clamp(flatPairs[i].u[w] + d);
      }
      deKept.push(deltaE00(s2lab(pred), s2lab(flatPairs[i].l)));
    }
    deKept.sort((a, b) => a - b);
    const p95 = deKept[Math.floor(deKept.length * 0.95)];
    console.log("  " + dd.name + " excl" + pct + " n=" + deKept.length + ": med=" + deKept[Math.floor(deKept.length / 2)].toFixed(3) +
      " P95=" + p95.toFixed(3) + " P99=" + deKept[Math.floor(deKept.length * 0.99)].toFixed(3) +
      " max=" + deKept[deKept.length - 1].toFixed(3) + " target✅=" + (p95 <= 2.0 ? "YES" : "no"));
  }
}
