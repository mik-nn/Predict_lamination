import * as tf from '@tensorflow/tfjs';
import { Matrix, SVD, solve } from "ml-matrix";
import { parseCgatsFile } from "../src/cgats-parser.js";
import { spectraToXyz, xyzToLab, deltaE00 } from "../src/color-math.js";

const D50_WP: [number, number, number] = [96.42, 100, 82.49];
const DATASETS = [
  { name: "R2_11-4-23", u: "Data/CGATS/R2_11-4-23.txt", l: "Data/CGATS/R2_11-4-23_lam.txt" },
  { name: "R2_27-10-23", u: "Data/CGATS/R2_27-10-23.txt", l: "Data/CGATS/R2_27-10-23_lam.txt" },
  { name: "R2_13-02-24", u: "Data/CGATS/R2_13-02-24.txt", l: "Data/CGATS/R2_13-02-24_lam.txt" },
  { name: "R3_23-4-24",  u: "Data/CGATS/R3_23-4-24.txt", l: "Data/CGATS/R3_23-4-24_lam.txt" },
];
const REFERENCE_DS = 0;
const WINDOW_SIZES = [25, 30, 40, 50, 64];
const LAMBDAS = [0.01, 0.1, 1.0, 10.0];

function matchByCMYK(unlam: any[], lam: any[]) {
  const lm = new Map<string, any[]>();
  for (const p of lam) { const k = p.cmyk.join(","); if (!lm.has(k)) lm.set(k, []); lm.get(k)!.push(p); }
  const out: { u: any; l: any }[] = [];
  for (const pu of unlam) { const k = pu.cmyk.join(","); const m = lm.get(k); if (m && m.length) { out.push({ u: pu, l: m[0] }); m.shift(); } }
  return out;
}
function s2lab(s: Float64Array) { return xyzToLab(spectraToXyz(s), D50_WP); }
function clamp(v: number) { return Math.max(0, Math.min(1, v)); }
function shuffle(arr: number[]) { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; }

// ---- Load & build global SVD ----
const allData: { name: string; pairs: { u: any; l: any }[] }[] = [];
const allPairs: { u: any; l: any; ds: string }[] = [];
for (const ds of DATASETS) {
  const pairs = matchByCMYK(parseCgatsFile(ds.u), parseCgatsFile(ds.l));
  allData.push({ name: ds.name, pairs });
  for (const p of pairs) allPairs.push({ ...p, ds: ds.name });
}

const fullD = new Matrix(allPairs.length, 36);
let gRow = 0;
for (const dd of allData)
  for (let i = 0; i < dd.pairs.length; i++) {
    for (let w = 0; w < 36; w++) fullD.set(gRow, w, dd.pairs[i].l.spectra[w] - dd.pairs[i].u.spectra[w]);
    gRow++;
  }
const svdFull = new SVD(fullD, { autoTranspose: true });
const V = svdFull.rightSingularVectors;
const RANK = 5;
const allV: Float64Array[] = [];
for (let k = 0; k < RANK; k++) {
  const vk = new Float64Array(36);
  for (let w = 0; w < 36; w++) vk[w] = V.get(w, k);
  allV.push(vk);
}
const allCvals: number[][] = Array.from({ length: RANK }, () => []);
for (let i = 0; i < allPairs.length; i++)
  for (let k = 0; k < RANK; k++)
    allCvals[k].push(svdFull.leftSingularVectors.get(i, k) * svdFull.diagonal[k]);

function getFeat(p: { u: any }): number[] {
  const f: number[] = [];
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w]);
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w] * p.u.spectra[w]);
  return f;
}
function getCvals(gi: number): number[] {
  const c: number[] = [];
  for (let k = 0; k < RANK; k++) c.push(allCvals[k][gi]);
  return c;
}
function reconstructL(u: Float64Array, c: number[]): Float64Array {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < RANK; k++) d += c[k] * allV[k][w]; pred[w] = clamp(u[w] + d); }
  return pred;
}

// ---- ResNet ----
function buildResNet(): tf.Sequential {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, inputShape: [72] }));
  model.add(tf.layers.batchNormalization({ momentum: 0.9 }));
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.1 }));
  for (let i = 0; i < 3; i++) {
    model.add(tf.layers.dense({ units: 64 }));
    model.add(tf.layers.batchNormalization({ momentum: 0.9 }));
    model.add(tf.layers.activation({ activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
  }
  model.add(tf.layers.dense({ units: 32 }));
  model.add(tf.layers.batchNormalization({ momentum: 0.9 }));
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.1 }));
  model.add(tf.layers.dense({ units: RANK }));
  return model;
}

async function trainResNet(X: number[][], Y: number[][]): Promise<tf.Sequential> {
  const model = buildResNet();
  model.compile({ optimizer: tf.train.adam(1e-3), loss: 'meanSquaredError' });
  const xt = tf.tensor2d(X);
  const yt = tf.tensor2d(Y);
  await model.fit(xt, yt, { epochs: 200, batchSize: 64, validationSplit: 0.1, verbose: 0 });
  xt.dispose(); yt.dispose();
  return model;
}

// ---- Ridge adapter (closed-form) ----
function ridgeFit(X: number[][], Y: number[][], lambda: number): number[][] {
  const n = X.length, p = X[0].length, q = Y[0].length;
  const Xm = new Matrix(X);
  const Ym = new Matrix(Y);
  const XtX = Xm.transpose().mmul(Xm);
  for (let j = 0; j < p; j++) XtX.set(j, j, XtX.get(j, j) + lambda);
  const XtY = Xm.transpose().mmul(Ym);
  const betas = solve(XtX, XtY);
  const result: number[][] = [];
  for (let j = 0; j < p; j++) { const row: number[] = []; for (let k = 0; k < q; k++) row.push(betas.get(j, k)); result.push(row); }
  return result;
}
function ridgePredict(X: number[][], betas: number[][]): number[][] {
  const p = betas.length, q = betas[0].length;
  return X.map(row => { const out = new Array(q).fill(0); for (let j = 0; j < p; j++) for (let k = 0; k < q; k++) out[k] += row[j] * betas[j][k]; return out; });
}

// ---- Compute ΔE00 for predicted c-values vs actual ----
function evaluateDE(pairs: { u: any; l: any }[], predC: number[][], indices: number[]): number[] {
  const de: number[] = [];
  for (let i = 0; i < indices.length; i++) {
    const pi = indices[i];
    de.push(deltaE00(s2lab(reconstructL(pairs[pi].u.spectra, predC[i])), s2lab(pairs[pi].l.spectra)));
  }
  return de;
}

function printDE(de: number[], label: string) {
  de.sort((a, b) => a - b);
  console.log(`  ${label.padEnd(16)} median=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)} P99=${de[Math.floor(de.length*0.99)].toFixed(3)} max=${de[de.length-1].toFixed(3)} n=${de.length}`);
}

function cmykFingerprint(pairs: { u: any }[], indices: number[]): string {
  let cM = 0, mM = 0, yM = 0, kM = 0, cMin = 100, cMax = 0, mMin = 100, mMax = 0, yMin = 100, yMax = 0, kMin = 100, kMax = 0;
  for (const i of indices) {
    const cmyk = pairs[i].u.cmyk;
    cM += cmyk[0]; mM += cmyk[1]; yM += cmyk[2]; kM += cmyk[3];
    if (cmyk[0] < cMin) cMin = cmyk[0]; if (cmyk[0] > cMax) cMax = cmyk[0];
    if (cmyk[1] < mMin) mMin = cmyk[1]; if (cmyk[1] > mMax) mMax = cmyk[1];
    if (cmyk[2] < yMin) yMin = cmyk[2]; if (cmyk[2] > yMax) yMax = cmyk[2];
    if (cmyk[3] < kMin) kMin = cmyk[3]; if (cmyk[3] > kMax) kMax = cmyk[3];
  }
  const n = indices.length;
  return `C${cMin}-${cMax} M${mMin}-${mMax} Y${yMin}-${yMax} K${kMin}-${kMax} (avg ${(cM/n).toFixed(0)}/${(mM/n).toFixed(0)}/${(yM/n).toFixed(0)}/${(kM/n).toFixed(0)})`;
}

// ======================================================================
async function main() {
console.log("=== RESNET TRANSFER: SLIDING WINDOW ANALYSIS ===\n");

const refDs = allData[REFERENCE_DS];
const refPairs = refDs.pairs;
const refOff = 0;

// ---- Step 1: Pre-train ResNet on reference dataset ----
console.log(`Step 1: Pre-training ResNet on ${refDs.name} (${refPairs.length} pairs)...`);
const refFeat = refPairs.map(p => getFeat(p));
const refCvals = [...Array(refPairs.length).keys()].map(i => getCvals(i));

// Normalize reference data
function normData(X: number[][], Y: number[][]) {
  const p = X[0].length, n = X.length;
  const xm = new Float64Array(p), xs = new Float64Array(p);
  for (let j = 0; j < p; j++) { let s = 0; for (const r of X) s += r[j]; xm[j] = s / n; }
  for (let j = 0; j < p; j++) { let ss = 0; for (const r of X) ss += (r[j] - xm[j]) ** 2; xs[j] = Math.sqrt(ss / n) + 1e-10; }
  const Xn = X.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));
  const q = Y[0].length;
  const ym = new Float64Array(q), ys = new Float64Array(q);
  for (let j = 0; j < q; j++) { let s = 0; for (const r of Y) s += r[j]; ym[j] = s / n; }
  for (let j = 0; j < q; j++) { let ss = 0; for (const r of Y) ss += (r[j] - ym[j]) ** 2; ys[j] = Math.sqrt(ss / n) + 1e-10; }
  const Yn = Y.map(r => r.map((v, j) => (v - ym[j]) / ys[j]));
  return { Xn, xm, xs, Yn, ym, ys };
}
const { Xn: refXn, xm, xs, Yn: refYn, ym, ys } = normData(refFeat, refCvals);

const refModel = await trainResNet(refXn, refYn);
console.log("  Done.\n");

// Build feature extractor (output of 32-dim ReLU layer, index 14)
const featureModel = tf.model({
  inputs: refModel.input,
  outputs: refModel.layers[14].output as tf.SymbolicTensor
});
featureModel.trainable = false;

// Extract 32-dim features for all patches
async function extractFeatures(pairs: { u: any }[]): Promise<number[][]> {
  const feats = pairs.map(p => getFeat(p));
  const Xn = feats.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));
  const t = tf.tensor2d(Xn);
  const f = featureModel.predict(t) as tf.Tensor;
  const arr = await f.array() as number[][];
  t.dispose(); f.dispose();
  return arr;
}

console.log("Extracting 32-dim frozen features for all datasets...");
const allFeatures = await Promise.all(allData.map(dd => extractFeatures(dd.pairs)));
console.log("  Done.\n");

// ---- Step 2: Pre-compute frozen outputs ----
console.log("Pre-computing frozen ResNet outputs for all datasets...");
const allFrozenOuts: number[][][] = []; // [dataset][patch][k] = denormalized c-value
for (let di = 0; di < DATASETS.length; di++) {
  const dd = allData[di];
  const n = dd.pairs.length;
  const rawFeat = [...Array(n).keys()].map(i => getFeat(dd.pairs[i]));
  const normFeat = rawFeat.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));
  const ft = tf.tensor2d(normFeat);
  const fp = refModel.predict(ft) as tf.Tensor;
  const fArr = await fp.array() as number[][];
  ft.dispose(); fp.dispose();
  allFrozenOuts.push(fArr.map(row => row.map((v, k) => v * ys[k] + ym[k])));
}
console.log("  Done.\n");

// ---- Step 3: Sliding window with 3 adapter types ----
console.log("Step 3: Sliding window evaluation");
console.log("Window sizes:", WINDOW_SIZES.join(", "));
console.log("Ridge λ values:", LAMBDAS.join(", "));
console.log("Adapter types: 32→5 (direct), 5→5 (residual), 37→5 (combined)\n");

type AdapterInfo = { label: string; key: string; dim: number; isResidual: boolean };

const ADAPTERS: AdapterInfo[] = [
  { label: "32→5 direct",   key: "32to5", dim: 32, isResidual: false },
  { label: "5→5 residual",  key: "5to5",  dim: 5,  isResidual: true  },
  { label: "37→5 combined", key: "37to5", dim: 37, isResidual: false },
];

function makeFeatures(features: number[][], frozenOut: number[][], adapter: AdapterInfo): number[][] {
  if (adapter.key === "32to5") return features;
  if (adapter.key === "5to5") return frozenOut;
  // 37to5: concatenate
  return features.map((f, i) => [...f, ...frozenOut[i]]);
}

function olsFit(X: number[][], Y: number[][]): number[][] {
  const n = X.length, p = X[0].length, q = Y[0].length;
  if (n < p + 5) return [];
  const Xm = new Matrix(X);
  const Ym = new Matrix(Y);
  try {
    const betas = solve(Xm.transpose().mmul(Xm), Xm.transpose().mmul(Ym));
    const result: number[][] = [];
    for (let j = 0; j < p; j++) { const row: number[] = []; for (let k = 0; k < q; k++) row.push(betas.get(j, k)); result.push(row); }
    return result;
  } catch { return []; }
}

for (let di = 0; di < DATASETS.length; di++) {
  const dd = allData[di];
  const n = dd.pairs.length;
  const off = allData.slice(0, di).reduce((s, d) => s + d.pairs.length, 0);
  const features = allFeatures[di];
  const frozenOut = allFrozenOuts[di];
  const allC = [...Array(n).keys()].map(i => getCvals(off + i));
  const isRef = di === REFERENCE_DS;

  console.log("=".repeat(80));
  console.log(`Target: ${dd.name} (n=${n})${isRef ? " [REFERENCE]" : ""}\n`);

  // Baselines
  console.log("  Baselines:");
  const fullFeat = [...Array(n).keys()].map(i => getFeat(dd.pairs[i]));
  const fullBeta = olsFit(fullFeat, allC);
  if (fullBeta.length) {
    const allDe = evaluateDE(dd.pairs, ridgePredict(fullFeat, fullBeta), [...Array(n).keys()]);
    printDE(allDe, "OLS all-data");
  }
  const frozenDe = evaluateDE(dd.pairs, frozenOut, [...Array(n).keys()]);
  printDE(frozenDe, "Frozen ResNet");
  console.log("");

  // For each window size
  for (const W of WINDOW_SIZES) {
    if (W >= n) continue;
    const nPos = n - W + 1;
    const step = Math.max(1, Math.floor(nPos / 60));

    // Per-adapter tracking
    const adapterBest: Record<string, { pos: number; p95: number; med: number; lambda: number }> = {};
    for (const a of ADAPTERS) adapterBest[a.key] = { pos: -1, p95: Infinity, med: 0, lambda: 0 };

    let sumP95_all: Record<string, number> = {};
    let countP95_all: Record<string, number> = {};
    for (const a of ADAPTERS) { sumP95_all[a.key] = 0; countP95_all[a.key] = 0; }

    // Track per-λ best for 32→5 to report later
    const best32 = { pos: -1, p95: Infinity, lambda: 0 };

    for (let pos = 0; pos < nPos; pos += step) {
      const anchorIdx = [...Array(W).keys()].map(i => pos + i);
      const testIdx = [...Array(n).keys()].filter(i => !anchorIdx.includes(i));

      const anchorC = anchorIdx.map(i => allC[i]);

      for (const a of ADAPTERS) {
        const anchorFeat = makeFeatures(features, frozenOut, a).filter((_, i) => anchorIdx.includes(i));
        const testFeat = makeFeatures(features, frozenOut, a).filter((_, i) => testIdx.includes(i));

        // Try each λ
        let bestP95local = Infinity, bestLambdaLocal = 0, bestMedLocal = 0;
        for (const lambda of LAMBDAS) {
          let targetC: number[][];
          if (a.isResidual) {
            // Residual: predict delta = actual_c - frozen_c
            const anchorDelta = anchorIdx.map(i => allC[i].map((v, k) => v - frozenOut[i][k]));
            const betas = ridgeFit(anchorFeat, anchorDelta, lambda);
            const deltaPred = ridgePredict(testFeat, betas);
            targetC = testIdx.map((ti, i) => frozenOut[ti].map((v, k) => v + deltaPred[i][k]));
          } else {
            const betas = ridgeFit(anchorFeat, anchorC, lambda);
            targetC = ridgePredict(testFeat, betas);
          }
          const de = evaluateDE(dd.pairs, targetC, testIdx);
          if (de.length === 0) continue;
          const p95 = de.sort((a, b) => a - b)[Math.floor(de.length * 0.95)];
          if (p95 < bestP95local) { bestP95local = p95; bestLambdaLocal = lambda; bestMedLocal = de[Math.floor(de.length / 2)]; }
        }

        if (bestP95local < adapterBest[a.key].p95) {
          adapterBest[a.key] = { pos, p95: bestP95local, med: bestMedLocal, lambda: bestLambdaLocal };
        }
        sumP95_all[a.key] += bestP95local;
        countP95_all[a.key]++;

        if (a.key === "32to5" && bestP95local < best32.p95) {
          best32.pos = pos; best32.p95 = bestP95local; best32.lambda = bestLambdaLocal;
        }
      }
    }

    // Print per-adapter results
    console.log(`  W=${W} (${nPos} positions, step=${step}):`);
    for (const a of ADAPTERS) {
      const b = adapterBest[a.key];
      const avg = countP95_all[a.key] > 0 ? (sumP95_all[a.key] / countP95_all[a.key]) : 0;
      const anchor = [...Array(W).keys()].map(i => b.pos + i);
      console.log(`    ${a.label.padEnd(16)} best P95=${b.p95.toFixed(3)} @ pos=${b.pos} λ=${b.lambda} avg=${avg.toFixed(3)}  [${cmykFingerprint(dd.pairs, anchor)}]`);
    }

    // OLS on best 32→5 block
    if (W >= 72) {
      const bestAnchor = [...Array(W).keys()].map(i => best32.pos + i);
      const olsBeta = olsFit(bestAnchor.map(i => getFeat(dd.pairs[i])), bestAnchor.map(i => allC[i]));
      if (olsBeta.length) {
        const testIdx = [...Array(n).keys()].filter(i => !bestAnchor.includes(i));
        const olsDe = evaluateDE(dd.pairs, ridgePredict(testIdx.map(i => getFeat(dd.pairs[i])), olsBeta), testIdx);
        printDE(olsDe, `  OLS @ best pos`);
      }
    }
    console.log("");
  }
}

// ---- Step 4: Best strips per adapter type ----
console.log("\n\nStep 4: Best anchor strips (all 3 adapters)");
console.log("=".repeat(80));

for (let di = 0; di < DATASETS.length; di++) {
  const dd = allData[di];
  const n = dd.pairs.length;
  const off = allData.slice(0, di).reduce((s, d) => s + d.pairs.length, 0);
  const features = allFeatures[di];
  const frozenOut = allFrozenOuts[di];
  const allC = [...Array(n).keys()].map(i => getCvals(off + i));

  console.log(`\n${dd.name}:`);

  for (const a of ADAPTERS) {
    console.log(`  ${a.label}:`);
    for (const W of WINDOW_SIZES) {
      if (W >= n) continue;
      const nPos = n - W + 1;
      let bestPos = -1, bestP95 = Infinity, bestLambda = 0;
      const step = Math.max(1, Math.floor(nPos / 100));

      for (let pos = 0; pos < nPos; pos += step) {
        const anchorIdx = [...Array(W).keys()].map(i => pos + i);
        const testIdx = [...Array(n).keys()].filter(i => !anchorIdx.includes(i));
        for (const lambda of LAMBDAS) {
          const anchorFeat = makeFeatures(features, frozenOut, a).filter((_, i) => anchorIdx.includes(i));
          const testFeat = makeFeatures(features, frozenOut, a).filter((_, i) => testIdx.includes(i));
          let targetC: number[][];
          if (a.isResidual) {
            const anchorDelta = anchorIdx.map(i => allC[i].map((v, k) => v - frozenOut[i][k]));
            const betas = ridgeFit(anchorFeat, anchorDelta, lambda);
            const deltaPred = ridgePredict(testFeat, betas);
            targetC = testIdx.map((ti, i) => frozenOut[ti].map((v, k) => v + deltaPred[i][k]));
          } else {
            const betas = ridgeFit(anchorFeat, anchorIdx.map(i => allC[i]), lambda);
            targetC = ridgePredict(testFeat, betas);
          }
          const de = evaluateDE(dd.pairs, targetC, testIdx);
          if (de.length === 0) continue;
          const p95 = de.sort((a, b) => a - b)[Math.floor(de.length * 0.95)];
          if (p95 < bestP95) { bestP95 = p95; bestPos = pos; bestLambda = lambda; }
        }
      }
      const anchor = [...Array(W).keys()].map(i => bestPos + i);
      console.log(`    W=${W.toString().padStart(2)}: P95=${bestP95.toFixed(3)} λ=${bestLambda.toFixed(1).padStart(4)} pos=${bestPos.toString().padStart(4)} [${cmykFingerprint(dd.pairs, anchor)}]`);
    }
  }
}

console.log("\nDone.");
refModel.dispose();
featureModel.dispose();
}

main().catch(e => { console.error(e); process.exit(1); });
