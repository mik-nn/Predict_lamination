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

function s2lab(s: Float64Array) { return xyzToLab(spectraToXyz(s), D50_WP); }

function matchByCMYK(unlam: any[], lam: any[]) {
  const lamMap = new Map<string, any[]>();
  for (const p of lam) { const k = p.cmyk.join(","); if (!lamMap.has(k)) lamMap.set(k, []); lamMap.get(k)!.push(p); }
  const pairs: { u: any; l: any }[] = [];
  for (const pu of unlam) { const k = pu.cmyk.join(","); const m = lamMap.get(k); if (m && m.length > 0) { pairs.push({ u: pu, l: m[0] }); m.shift(); } }
  return pairs;
}

// ---- Neugebauer model ----
const ALL_PRIMARIES = [
  "0000","1000","0100","0010","0001",
  "1100","1010","1001","0110","0101","0011",
  "1110","1101","1011","0111","1111"
];

function demichel(cmyk: number[]): number[] {
  const [c, m, y, k] = cmyk.map(v => v / 100);
  const w=(1-c)*(1-m)*(1-y)*(1-k);const C=c*(1-m)*(1-y)*(1-k);const M=(1-c)*m*(1-y)*(1-k);
  const Y=(1-c)*(1-m)*y*(1-k);const K=(1-c)*(1-m)*(1-y)*k;const CM=c*m*(1-y)*(1-k);
  const CY=c*(1-m)*y*(1-k);const CK=c*(1-m)*(1-y)*k;const MY=(1-c)*m*y*(1-k);
  const MK=(1-c)*m*(1-y)*k;const YK=(1-c)*(1-m)*y*k;const CMY=c*m*y*(1-k);
  const CMK=c*m*(1-y)*k;const CYK=c*(1-m)*y*k;const MYK=(1-c)*m*y*k;const CMYK=c*m*y*k;
  return [w,C,M,Y,K,CM,CY,CK,MY,MK,YK,CMY,CMK,CYK,MYK,CMYK];
}

// Neugebauer with PER-INK n factors
function predictNeugePerInk(cmyk: number[], primaries: Map<string, Float64Array>, nC: number, nM: number, nY: number, nK: number): Float64Array {
  const [c, m, y, k] = cmyk.map(v => v / 100);
  // Effective n for this patch = weighted by ink coverage
  // Actually, per-ink n means: each ink has its own n for halftone blending
  // The model: R^1/n = sum a_i * R_i^1/n — but which n?
  // Standard approach: use the n of the dominant ink
  // Better: interpolate n by ink coverage (total ink n = sum(ink_i * n_i) / sum(ink_i))
  const totalInk = c + m + y + k;
  const nEff = totalInk > 0 ? (c * nC + m * nM + y * nY + k * nK) / totalInk : 1.0;

  const areas = demichel(cmyk);
  const out = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    let sum = 0;
    for (let p = 0; p < 16; p++) {
      const spec = primaries.get(ALL_PRIMARIES[p]);
      if (spec) sum += areas[p] * Math.pow(spec[w], 1 / nEff);
    }
    out[w] = Math.pow(Math.max(sum, 0), nEff);
  }
  return out;
}

// Standard Neugebauer (single n)
function predictNeuge(cmyk: number[], primaries: Map<string, Float64Array>, n: number): Float64Array {
  const areas = demichel(cmyk);
  const out = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    let sum = 0;
    for (let p = 0; p < 16; p++) {
      const spec = primaries.get(ALL_PRIMARIES[p]);
      if (spec) sum += areas[p] * Math.pow(spec[w], 1/n);
    }
    out[w] = Math.pow(Math.max(sum, 0), n);
  }
  return out;
}

function rmsError(pred: Float64Array, meas: Float64Array): number {
  let s = 0;
  for (let w = 0; w < 36; w++) { const d = pred[w] - meas[w]; s += d * d; }
  return Math.sqrt(s / 36);
}

function fitPerInkN(primaries: Map<string, Float64Array>, pairs: { cmyk: number[]; spectra: Float64Array }[], inkIdx: number, nRange: number[]): number {
  // Fit n for a specific ink using only single-ink patches
  const singleInk = pairs.filter(p => p.cmyk.every((v, i) => i === inkIdx || v === 0));
  if (singleInk.length < 2) return 1.0;

  // Use the OTHER inks' n as the best single-n fit
  // Actually: find n that minimizes RMSE on patches where only this ink varies
  let bestN = 1, bestRMSE = Infinity;
  for (const n of nRange) {
    // For single-ink test, use the global n as the "other ink" n is irrelevant
    // since there's only one ink active
    const otherN = n; // doesn't matter, only one ink
    let totalErr = 0;
    for (const p of singleInk) {
      const pred = predictNeuge(p.cmyk, primaries, n);
      totalErr += rmsError(pred, p.spectra);
    }
    const avg = totalErr / singleInk.length;
    if (avg < bestRMSE) { bestRMSE = avg; bestN = n; }
  }
  return bestN;
}

// ---- OLS model ----
function buildX(rows: number, getU: (i: number, w: number) => number): Matrix {
  const X = new Matrix(rows, 72);
  for (let i = 0; i < rows; i++) {
    let col = 0;
    for (let w = 0; w < 36; w++) X.set(i, col++, getU(i, w));
    for (let w = 0; w < 36; w++) { const u = getU(i, w); X.set(i, col++, u * u); }
  }
  return X;
}
function olsFn(X: Matrix, y: number[]) { return solve(X.transpose().mmul(X), X.transpose().mmul(Matrix.columnVector(y))).to1DArray(); }

console.log("=== NEUGEBAUER (per-ink n) vs OLS — предсказание БЕЗ ламинации ===\n");

const nGrid = Array.from({ length: 49 }, (_, i) => 1.0 + i * 0.1); // 1.0 to 5.8

for (const ds of DATASETS) {
  const pairs = matchByCMYK(parseCgatsFile(ds.unlam), parseCgatsFile(ds.lam));
  console.log(`========== ${ds.name} ==========\n`);

  // Extract U primaries
  const uPrimaries = new Map<string, Float64Array>();
  for (const p of pairs) {
    if (p.u.cmyk.every(v => v === 0 || v === 100)) {
      const key = p.u.cmyk.map(v => v >= 50 ? 1 : 0).join("");
      if (!uPrimaries.has(key)) uPrimaries.set(key, p.u.spectra);
    }
  }
  console.log(`  Примари: ${uPrimaries.size}/16`);

  // ---- 1. Fit global n on ALL single-ink patches ----
  const allSingles = pairs.filter(p => p.u.cmyk.filter(v => v > 0).length <= 1).map(p => ({ cmyk: p.u.cmyk, spectra: p.u.spectra }));

  let bestGlobalN = 1, bestGlobalRMSE = Infinity;
  for (const n of nGrid) {
    let err = 0;
    for (const p of allSingles) {
      const pred = predictNeuge(p.cmyk, uPrimaries, n);
      err += rmsError(pred, p.spectra);
    }
    const avg = err / allSingles.length;
    if (avg < bestGlobalRMSE) { bestGlobalRMSE = avg; bestGlobalN = n; }
  }

  // ---- 2. Fit per-ink n on single-ink patches ----
  const nC = fitPerInkN(uPrimaries, allSingles, 0, nGrid);
  const nM = fitPerInkN(uPrimaries, allSingles, 1, nGrid);
  const nY = fitPerInkN(uPrimaries, allSingles, 2, nGrid);
  const nK = fitPerInkN(uPrimaries, allSingles, 3, nGrid);

  console.log(`  n(C)=${nC.toFixed(1)}  n(M)=${nM.toFixed(1)}  n(Y)=${nY.toFixed(1)}  n(K)=${nK.toFixed(1)}  (global n=${bestGlobalN.toFixed(1)})`);

  // ---- 3. Neugebauer linear (n=1) — just Demichel average ----
  // ---- 4. OLS rank-5 on full spectra ----
  // Train OLS on unlaminated data only
  const uPairs = pairs.map(p => ({ u: p.u.spectra, l: p.u.spectra })); // predict U from U (identity, but using OLS structure)

  // For OLS, we need the SVD basis from U data alone for comparison
  const uSp = pairs.map(p => p.u.spectra);
  const nAll = uSp.length;
  const Umat = new Matrix(nAll, 36);
  for (let i = 0; i < nAll; i++) for (let w = 0; w < 36; w++) Umat.set(i, w, uSp[i][w]);
  const svdU = new SVD(Umat, { autoTranspose: true });
  const sVals = svdU.diagonal;
  const Vmat = svdU.rightSingularVectors;
  const RANK_MAX = 5;
  const basis: Float64Array[] = [];
  for (let k = 0; k < RANK_MAX; k++) { const vk = new Float64Array(36); for (let w = 0; w < 36; w++) vk[w] = Vmat.get(w, k); basis.push(vk); }

  // Actually OLS for U→U is trivial (identity). Let's build a proper comparison:
  // Train OLS on unlaminated to predict laminated. But the user wants to see if Neugebauer matches OLS for UNLAMINATED prediction.
  // So let's just test: given unlaminated primaries + n, how well does Neugebauer predict all unlaminated patches?

  // ---- Test all models on UNLAMINATED data ----
  function testModel(label: string, predictFn: (cmyk: number[]) => Float64Array): number[] {
    const de: number[] = [];
    for (const p of pairs) {
      const pred = predictFn(p.u.cmyk);
      de.push(deltaE00(s2lab(pred), s2lab(p.u.spectra)));
    }
    de.sort((a, b) => a - b);
    return de;
  }

  const deLinear = testModel("n=1", cmyk => predictNeuge(cmyk, uPrimaries, 1));
  const deGlobalN = testModel(`n=${bestGlobalN.toFixed(1)}`, cmyk => predictNeuge(cmyk, uPrimaries, bestGlobalN));
  const dePerInk = testModel("per-ink n", cmyk => predictNeugePerInk(cmyk, uPrimaries, nC, nM, nY, nK));

  const fmt = (de: number[]) =>
    `med=${de[Math.floor(de.length/2)].toFixed(3)}  P95=${de[Math.floor(de.length*0.95)].toFixed(3)}  P99=${de[Math.floor(de.length*0.99)].toFixed(3)}  max=${de[de.length-1].toFixed(3)}`;

  console.log(`\n  --- Предсказание НЕламинированных патчей (ΔE00) ---`);
  console.log(`  Модель                            median   P95      P99      max`);
  console.log(`  ${"─".repeat(55)}`);
  console.log(`  Neugebauer n=1 (Demichel)         ${fmt(deLinear)}`);
  console.log(`  Neugebauer n=${bestGlobalN.toFixed(1)} (global)       ${fmt(deGlobalN)}`);
  console.log(`  Neugebauer per-ink n              ${fmt(dePerInk)}`);

  // Also show RMSE distributions
  console.log(`\n  --- Средний RMSE по длинам волн ---`);
  for (const [label, de, predFn] of [
    ["n=1", deLinear, (cmyk: number[]) => predictNeuge(cmyk, uPrimaries, 1)],
    ["global n="+bestGlobalN.toFixed(1), deGlobalN, (cmyk: number[]) => predictNeuge(cmyk, uPrimaries, bestGlobalN)],
    ["per-ink n", dePerInk, (cmyk: number[]) => predictNeugePerInk(cmyk, uPrimaries, nC, nM, nY, nK)]
  ] as [string, number[], Function][]) {
    let rmsTotal = 0;
    for (const p of pairs) {
      const pred = predFn(p.u.cmyk);
      rmsTotal += rmsError(pred, p.u.spectra);
    }
    console.log(`  ${label.padEnd(15)}  avg RMSE=${(rmsTotal/pairs.length*1000).toFixed(3)}×10⁻³`);
  }

  // ---- Show per-L* bin error ----
  console.log(`\n  --- Per-L* bin (Neugebauer per-ink n) ---`);
  const lut = pairs.map((p, i) => ({ de: dePerInk[i], l: s2lab(p.u.spectra)[0], cmyk: p.u.cmyk }));
  const bins = [[0,20],[20,40],[40,60],[60,80],[80,100]];
  for (const [lo, hi] of bins) {
    const inBin = lut.filter(x => x.l >= lo && x.l < hi).map(x => x.de).sort((a, b) => a - b);
    if (inBin.length < 2) continue;
    console.log(`  L* ${lo}-${hi} (n=${inBin.length}): med=${inBin[Math.floor(inBin.length/2)].toFixed(3)} P95=${inBin[Math.floor(inBin.length*0.95)].toFixed(3)}`);
  }

  // ---- Distribution of errors for worst patches ----
  const worst10 = [...Array(pairs.length).keys()].sort((a, b) => dePerInk[b] - dePerInk[a]).slice(0, 10);
  console.log(`\n  Худшие 10 (Neugebauer per-ink n, U):`);
  console.log("  #  ΔE00     L*      C   M   Y   K");
  for (let j = 0; j < 10; j++) {
    const i = worst10[j];
    const p = pairs[i];
    const lab = s2lab(p.u.spectra);
    console.log(`  ${(j+1).toString().padStart(2)}  ${dePerInk[i].toFixed(3).padStart(7)}  ${lab[0].toFixed(2).padStart(6)}  ${p.u.cmyk.map((v:number) => v.toString().padStart(3)).join(" ")}`);
  }
  console.log("");
}

// ---- CROSS-DATASET Neugebauer ----
console.log("\n=== CROSS-DATASET: Neugebauer c одними примари, тест на других ===\n");
console.log("Обучаем примари и n на одном датасете, предсказываем все патчи другого\n");

for (let ti = 0; ti < DATASETS.length; ti++) {
  const trainPairs = matchByCMYK(parseCgatsFile(DATASETS[ti].unlam), parseCgatsFile(DATASETS[ti].lam));

  // Build primaries + fit n from training dataset (unlaminated only)
  const uPrim = new Map<string, Float64Array>();
  for (const p of trainPairs) {
    if (p.u.cmyk.every(v => v === 0 || v === 100)) {
      const key = p.u.cmyk.map(v => v >= 50 ? 1 : 0).join("");
      if (!uPrim.has(key)) uPrim.set(key, p.u.spectra);
    }
  }
  const allSingles = trainPairs.filter(p => p.u.cmyk.filter(v => v > 0).length <= 1).map(p => ({ cmyk: p.u.cmyk, spectra: p.u.spectra }));
  const nC = fitPerInkN(uPrim, allSingles, 0, nGrid);
  const nM = fitPerInkN(uPrim, allSingles, 1, nGrid);
  const nY = fitPerInkN(uPrim, allSingles, 2, nGrid);
  const nK = fitPerInkN(uPrim, allSingles, 3, nGrid);

  console.log(`  Примари+ n из ${DATASETS[ti].name}: n(C)=${nC.toFixed(1)} n(M)=${nM.toFixed(1)} n(Y)=${nY.toFixed(1)} n(K)=${nK.toFixed(1)}`);

  for (let testIdx = 0; testIdx < DATASETS.length; testIdx++) {
    if (testIdx === ti) continue;
    const testPairs = matchByCMYK(parseCgatsFile(DATASETS[testIdx].unlam), parseCgatsFile(DATASETS[testIdx].lam));

    const de: number[] = [];
    for (const p of testPairs) {
      const pred = predictNeugePerInk(p.u.cmyk, uPrim, nC, nM, nY, nK);
      de.push(deltaE00(s2lab(pred), s2lab(p.u.spectra)));
    }
    de.sort((a, b) => a - b);
    console.log(`    → ${DATASETS[testIdx].name} (U): med=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)}`);
  }
}
