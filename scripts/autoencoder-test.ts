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

// ---- NN utilities ----
function tanh(x: number): number { return Math.tanh(x); }
function tanhDeriv(x: number): number { const t = Math.tanh(x); return 1 - t * t; }

class Layer {
  W: number[][]; b: number[]; activation: string;
  constructor(inDim: number, outDim: number, activation: string) {
    const scale = Math.sqrt(2 / inDim);
    this.W = Array.from({ length: outDim }, () => Array.from({ length: inDim }, () => (Math.random() - 0.5) * 2 * scale));
    this.b = new Array(outDim).fill(0);
    this.activation = activation;
  }
  forward(input: number[]) {
    const z = this.b.map((b, i) => { let s = b; for (let j = 0; j < input.length; j++) s += this.W[i][j] * input[j]; return s; });
    const output = this.activation === "tanh" ? z.map(tanh) : [...z];
    return { output, cache: { input, z } };
  }
  backward(gradOutput: number[], cache: { input: number[]; z: number[] }) {
    const { input, z } = cache;
    const gradZ = this.activation === "tanh" ? gradOutput.map((g, i) => g * tanhDeriv(z[i])) : [...gradOutput];
    const dW = gradZ.map(g => input.map(x => g * x));
    return { gradInput: this.W[0].map((_, j) => gradZ.reduce((s, g, i) => s + g * this.W[i][j], 0)), dW, db: [...gradZ] };
  }
}

class NN {
  layers: Layer[]; lr: number;
  constructor(layerSizes: number[], activations: string[], lr = 0.01) {
    this.layers = []; this.lr = lr;
    for (let i = 0; i < layerSizes.length - 1; i++) this.layers.push(new Layer(layerSizes[i], layerSizes[i + 1], activations[i]));
  }
  forward(input: number[]) { const caches: any[] = []; let curr = input; for (const l of this.layers) { const r = l.forward(curr); caches.push(r.cache); curr = r.output; } return { output: curr, caches }; }
  predict(input: number[]) { return this.forward(input).output; }
  encode(input: number[], layerIdx: number) { let curr = input; for (let i = 0; i <= layerIdx; i++) curr = this.layers[i].forward(curr).output; return curr; }
  trainOn(data: { input: number[]; target: number[] }[], epochs: number, batchSize = 128, logEvery = 0) {
    for (let ep = 0; ep < epochs; ep++) {
      for (let i = data.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [data[i], data[j]] = [data[j], data[i]]; }
      let totalLoss = 0, nB = 0;
      for (let b = 0; b < data.length; b += batchSize) {
        const batch = data.slice(b, b + batchSize);
        const dwA = this.layers.map(l => l.W.map(r => r.map(() => 0)));
        const dbA = this.layers.map(l => l.b.map(() => 0));
        for (const s of batch) {
          const { output, caches } = this.forward(s.input);
          const grad = output.map((o, i) => (o - s.target[i]) / s.target.length);
          totalLoss += output.reduce((s2, o, i) => s2 + (o - s.target[i]) * (o - s.target[i]), 0);
          let g = grad;
          for (let li = this.layers.length - 1; li >= 0; li--) {
            const r = this.layers[li].backward(g, caches[li]);
            for (let i = 0; i < r.dW.length; i++) for (let j = 0; j < r.dW[i].length; j++) dwA[li][i][j] += r.dW[i][j];
            for (let i = 0; i < r.db.length; i++) dbA[li][i] += r.db[i];
            g = r.gradInput;
          }
        }
        for (let li = 0; li < this.layers.length; li++) {
          for (let i = 0; i < this.layers[li].W.length; i++) {
            for (let j = 0; j < this.layers[li].W[i].length; j++) this.layers[li].W[i][j] -= this.lr * dwA[li][i][j] / batch.length;
            this.layers[li].b[i] -= this.lr * dbA[li][i] / batch.length;
          }
        }
        totalLoss += 0; nB++;
      }
      if (logEvery > 0 && ep % logEvery === 0) process.stdout.write(`\r    ep ${ep} loss=${(totalLoss / Math.max(1,data.length/batchSize)).toFixed(4)}`);
    }
    if (logEvery > 0) process.stdout.write("\n");
  }
}

// Load data
const allPairs: { u: any; l: any; ds: string }[] = [];
const allData: { name: string; pairs: { u: any; l: any }[] }[] = [];
for (const ds of DATASETS) {
  const pairs = matchByCMYK(parseCgatsFile(ds.unlam), parseCgatsFile(ds.lam));
  allData.push({ name: ds.name, pairs });
  for (const p of pairs) allPairs.push({ ...p, ds: ds.name });
}

// Global SVD of D = L-U
const fullD = new Matrix(allPairs.length, 36);
let gRow = 0;
for (const dd of allData)
  for (let i = 0; i < dd.pairs.length; i++) {
    for (let w = 0; w < 36; w++) fullD.set(gRow, w, dd.pairs[i].l.spectra[w] - dd.pairs[i].u.spectra[w]);
    gRow++;
  }
const svdFull = new SVD(fullD, { autoTranspose: true });
const sVals = svdFull.diagonal;
const V = svdFull.rightSingularVectors;
const RANK = 5;
const allV: Float64Array[] = [];
const allCvals: number[][] = Array.from({ length: RANK }, () => []);
for (let k = 0; k < RANK; k++) {
  const vk = new Float64Array(36);
  for (let w = 0; w < 36; w++) vk[w] = V.get(w, k);
  allV.push(vk);
}
const Umat = svdFull.leftSingularVectors;
for (let i = 0; i < allPairs.length; i++)
  for (let k = 0; k < RANK; k++)
    allCvals[k].push(Umat.get(i, k) * sVals[k]);

// OLS helpers
function buildQuadX(rows: number, getU: (i: number, w: number) => number) {
  const X = new Matrix(rows, 72);
  for (let i = 0; i < rows; i++) {
    let c = 0;
    for (let w = 0; w < 36; w++) X.set(i, c++, getU(i, w));
    for (let w = 0; w < 36; w++) { const u = getU(i, w); X.set(i, c++, u * u); }
  }
  return X;
}
function ols(X: Matrix, y: number[]) { return solve(X.transpose().mmul(X), X.transpose().mmul(Matrix.columnVector(y))).to1DArray(); }
function predictQuad(u: Float64Array, betas: number[][]) {
  return betas.map(b => { let v = 0, c = 0; for (let w = 0; w < 36; w++) v += b[c++] * u[w]; for (let w = 0; w < 36; w++) v += b[c++] * u[w] * u[w]; return v; });
}

// Evaluate OLS model on pairs, return sorted ΔE00 array
function evalOLS(pairs: { u: Float64Array; l: Float64Array }[], betas: number[][]): number[] {
  const de: number[] = [];
  for (const p of pairs) {
    const c = predictQuad(p.u, betas);
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < RANK; k++) d += c[k] * allV[k][w]; pred[w] = clamp(p.u[w] + d); }
    de.push(deltaE00(s2lab(pred), s2lab(p.l)));
  }
  return de.sort((a, b) => a - b);
}

// ---- Normalization ----
function computeMeanStd(data: number[][]) {
  const d = data[0].length;
  const mean = new Float64Array(d); const std = new Float64Array(d);
  for (let w = 0; w < d; w++) { let s = 0; for (const v of data) s += v[w]; mean[w] = s / data.length; }
  for (let w = 0; w < d; w++) { let s2 = 0; for (const v of data) s2 += (v[w] - mean[w]) ** 2; std[w] = Math.sqrt(s2 / data.length) + 1e-10; }
  return { mean, std };
}

const allUSpec = allPairs.map(p => { const a: number[] = []; for (let w = 0; w < 36; w++) a.push(p.u.spectra[w]); return a; });
const allLSpec = allPairs.map(p => { const a: number[] = []; for (let w = 0; w < 36; w++) a.push(p.l.spectra[w]); return a; });
const uStats = computeMeanStd(allUSpec);
const lStats = computeMeanStd(allLSpec);
const normU = (u: number[]) => u.map((v, i) => (v - uStats.mean[i]) / uStats.std[i]);
const denormL = (u: number[]) => u.map((v, i) => v * lStats.std[i] + lStats.mean[i]);

function printDE(de: number[], label: string) {
  const m = de[Math.floor(de.length/2)];
  const p95 = de[Math.floor(de.length*0.95)];
  const p99 = de[Math.floor(de.length*0.99)];
  const mx = de[de.length-1];
  console.log(`  ${label.padEnd(14)} median=${m.toFixed(3)} P95=${p95.toFixed(3)} P99=${p99.toFixed(3)} max=${mx.toFixed(3)}`);
}

// ======================================================================
console.log("=== АВТОЭНКОДЕР vs PCA vs OLS ===\n");

// ---- 0. Reference: OLS per-dataset (honest 80/20) ----
console.log("0. OLS rank-5 (U+U²) per-dataset, 80/20 split:");
for (const dd of allData) {
  let off = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; off += allData[d].pairs.length; }
  const n = dd.pairs.length;
  const pairs = dd.pairs.map(p => ({ u: p.u.spectra, l: p.l.spectra }));
  const idx = [...Array(n).keys()]; for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nTr = Math.floor(n * 0.8);
  const trIdx = idx.slice(0, nTr), teIdx = idx.slice(nTr);
  const Xtr = buildQuadX(nTr, (i, w) => pairs[trIdx[i]].u[w]);
  const b: number[][] = [];
  for (let k = 0; k < RANK; k++) b.push(ols(Xtr, trIdx.map(i => allCvals[k][off + i])));
  const de = evalOLS(teIdx.map(i => pairs[i]), b);
  printDE(de, dd.name);
}

// ---- 1. Linear autoencoder (PCA of U) ----
console.log("\n1. Linear autoencoder (PCA of U) — U→PCA→inverse→U':");
const UmatAll = new Matrix(allUSpec.length, 36);
for (let i = 0; i < allUSpec.length; i++) for (let w = 0; w < 36; w++) UmatAll.set(i, w, allUSpec[i][w]);
const svdU = new SVD(UmatAll, { autoTranspose: true });
const UPCA_v = svdU.rightSingularVectors; // 36×36

function pcaReconstruct(u: Float64Array, nPC: number): Float64Array {
  // Project to nPC and back
  const scores = new Float64Array(nPC);
  for (let k = 0; k < nPC; k++) { let s = 0; for (let w = 0; w < 36; w++) s += u[w] * UPCA_v.get(w, k); scores[k] = s; }
  const recon = new Float64Array(36);
  for (let w = 0; w < 36; w++) { let s = 0; for (let k = 0; k < nPC; k++) s += scores[k] * UPCA_v.get(w, k); recon[w] = s; }
  return recon;
}

for (const nPC of [5, 10, 15, 20]) {
  const de: number[] = [];
  for (const p of allPairs) {
    const recon = pcaReconstruct(p.u.spectra, nPC);
    for (let w = 0; w < 36; w++) recon[w] = Math.max(0, Math.min(1, recon[w]));
    de.push(deltaE00(s2lab(recon), s2lab(p.u.spectra)));
  }
  de.sort((a, b) => a - b);
  console.log(`  PC=${nPC}: median=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)} P99=${de[Math.floor(de.length*0.99)].toFixed(3)}`);
}

// PCA → predict L from scores (OLS on scores)
console.log("\n   PCA scores → OLS → L (per-dataset 80/20):");
for (const nPC of [5, 10, 15, 20]) {
  const scoreData = allUSpec.map(u => {
    const s = new Float64Array(nPC);
    for (let k = 0; k < nPC; k++) { let v = 0; for (let w = 0; w < 36; w++) v += u[w] * UPCA_v.get(w, k); s[k] = v; }
    return s;
  });

  console.log(`\n   PC=${nPC}:`);
  for (const dd of allData) {
    let off = 0;
    for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; off += allData[d].pairs.length; }
    const n = dd.pairs.length;
    const idx = [...Array(n).keys()]; for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const nTr = Math.floor(n * 0.8);
    const trIdx = idx.slice(0, nTr), teIdx = idx.slice(nTr);

    // Use scores + scores² as features
    const Xtr = new Matrix(nTr, nPC * 2);
    for (let i = 0; i < nTr; i++) {
      for (let k = 0; k < nPC; k++) { Xtr.set(i, k, scoreData[off + trIdx[i]][k]); Xtr.set(i, k + nPC, scoreData[off + trIdx[i]][k] ** 2); }
    }

    const betas: number[][] = [];
    for (let w = 0; w < 36; w++) betas.push(ols(Xtr, trIdx.map(i => dd.pairs[i].l.spectra[w])));

    const de: number[] = [];
    for (const ti of teIdx) {
      const feat: number[] = [];
      for (let k = 0; k < nPC; k++) { feat.push(scoreData[off + ti][k]); }
      for (let k = 0; k < nPC; k++) { feat.push(scoreData[off + ti][k] ** 2); }
      const pred = new Float64Array(36);
      for (let w = 0; w < 36; w++) { let v = 0; for (let j = 0; j < feat.length; j++) v += betas[w][j] * feat[j]; pred[w] = clamp(v); }
      de.push(deltaE00(s2lab(pred), s2lab(dd.pairs[ti].l.spectra)));
    }
    de.sort((a, b) => a - b);
    printDE(de, dd.name);
  }
}

// ---- 2. Nonlinear autoencoder (tanh) ----
console.log("\n2. Nonlinear autoencoder: 36→20(tanh)→10(linear)→20(tanh)→36");
const ae = new NN([36, 20, 10, 20, 36], ["tanh", "linear", "tanh", "linear"], 0.03);
const aeData = allUSpec.map(u => ({ input: normU(u), target: normU(u) }));
console.log("  Training (300 epochs)...");
ae.trainOn(aeData, 300, 128, 25);

// Evaluate U→AE→U'
const reconDE: number[] = [];
for (const p of allPairs) {
  const inp = normU(Array.from(p.u.spectra));
  const recon = denormL(ae.predict(inp));
  const s = new Float64Array(36); for (let w = 0; w < 36; w++) s[w] = Math.max(0, Math.min(1, recon[w]));
  reconDE.push(deltaE00(s2lab(s), s2lab(p.u.spectra)));
}
reconDE.sort((a, b) => a - b);
console.log(`  U→AE→U': median=${reconDE[Math.floor(reconDE.length/2)].toFixed(3)} P95=${reconDE[Math.floor(reconDE.length*0.95)].toFixed(3)}`);

// AE latent → predict L
console.log("\n  AE latent (10D tanh) → OLS → L (per-dataset 80/20):");
const latentData = allUSpec.map(u => ae.encode(normU(u), 1)); // after 36→20→10

const aeLatentMean = new Float64Array(10); const aeLatentStd = new Float64Array(10);
for (let k = 0; k < 10; k++) { let s = 0; for (const l of latentData) s += l[k]; aeLatentMean[k] = s / latentData.length; }
for (let k = 0; k < 10; k++) { let s2 = 0; for (const l of latentData) s2 += (l[k] - aeLatentMean[k]) ** 2; aeLatentStd[k] = Math.sqrt(s2 / latentData.length) + 1e-10; }
const normLat = (l: number[]) => l.map((v, k) => (v - aeLatentMean[k]) / aeLatentStd[k]);

for (const dd of allData) {
  let off = 0;
  for (let d = 0; d < DATASETS.length; d++) { if (DATASETS[d].name === dd.name) break; off += allData[d].pairs.length; }
  const n = dd.pairs.length;
  const idx = [...Array(n).keys()]; for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nTr = Math.floor(n * 0.8);
  const trIdx = idx.slice(0, nTr), teIdx = idx.slice(nTr);

  const nFeat = 10;
  const Xtr = new Matrix(nTr, nFeat * 2);
  for (let i = 0; i < nTr; i++) {
    const nl = normLat(latentData[off + trIdx[i]]);
    for (let k = 0; k < nFeat; k++) { Xtr.set(i, k, nl[k]); Xtr.set(i, k + nFeat, nl[k] * nl[k]); }
  }

  const betas: number[][] = [];
  for (let w = 0; w < 36; w++) betas.push(ols(Xtr, trIdx.map(i => dd.pairs[i].l.spectra[w])));

  const de: number[] = [];
  for (const ti of teIdx) {
    const nl = normLat(latentData[off + ti]);
    const feat: number[] = [];
    for (let k = 0; k < nFeat; k++) { feat.push(nl[k]); }
    for (let k = 0; k < nFeat; k++) { feat.push(nl[k] * nl[k]); }
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) { let v = 0; for (let j = 0; j < feat.length; j++) v += betas[w][j] * feat[j]; pred[w] = clamp(v); }
    de.push(deltaE00(s2lab(pred), s2lab(dd.pairs[ti].l.spectra)));
  }
  de.sort((a, b) => a - b);
  printDE(de, dd.name);
}

// ---- 3. Direct NN U→L ----
console.log("\n3. Direct NN U→L: 36→30(tanh)→30(tanh)→36(linear)");
const nn = new NN([36, 30, 30, 36], ["tanh", "tanh", "linear"], 0.01);
const nnData = allUSpec.map((u, i) => ({ input: normU(u), target: allLSpec[i].map((v, w) => (v - lStats.mean[w]) / lStats.std[w]) }));
console.log("  Training (500 epochs)...");
nn.trainOn(nnData, 500, 128, 50);

console.log("  Per-dataset (trained on ALL 4):");
for (const dd of allData) {
  const de: number[] = [];
  for (const p of dd.pairs) {
    const pred = denormL(nn.predict(normU(Array.from(p.u.spectra))));
    const s = new Float64Array(36); for (let w = 0; w < 36; w++) s[w] = clamp(pred[w]);
    de.push(deltaE00(s2lab(s), s2lab(p.l.spectra)));
  }
  de.sort((a, b) => a - b);
  printDE(de, dd.name);
}
