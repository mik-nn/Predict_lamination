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

// Load & build global SVD
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
for (let i = 0; i < allPairs.length; i++)
  for (let k = 0; k < RANK; k++)
    allCvals[k].push(svdFull.leftSingularVectors.get(i, k) * sVals[k]);

function getFeat(p: { u: any }): number[] {
  const f: number[] = [];
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w]);
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w] * p.u.spectra[w]);
  return f;
}
function getCvals(i: number, off: number): number[] {
  const c: number[] = [];
  for (let k = 0; k < RANK; k++) c.push(allCvals[k][off + i]);
  return c;
}
function reconstructL(u: Float64Array, c: number[]): Float64Array {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < RANK; k++) d += c[k] * allV[k][w]; pred[w] = clamp(u[w] + d); }
  return pred;
}

function normData(X: number[][], Y?: number[][]) {
  const p = X[0].length, n = X.length;
  const xm = new Float64Array(p), xs = new Float64Array(p);
  for (let j = 0; j < p; j++) { let s = 0; for (const r of X) s += r[j]; xm[j] = s / n; }
  for (let j = 0; j < p; j++) { let ss = 0; for (const r of X) ss += (r[j] - xm[j]) ** 2; xs[j] = Math.sqrt(ss / n) + 1e-10; }
  const Xn = X.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));

  if (!Y) return { Xn, xm, xs };
  const q = Y[0].length;
  const ym = new Float64Array(q), ys = new Float64Array(q);
  for (let j = 0; j < q; j++) { let s = 0; for (const r of Y) s += r[j]; ym[j] = s / n; }
  for (let j = 0; j < q; j++) { let ss = 0; for (const r of Y) ss += (r[j] - ym[j]) ** 2; ys[j] = Math.sqrt(ss / n) + 1e-10; }
  const Yn = Y.map(r => r.map((v, j) => (v - ym[j]) / ys[j]));
  return { Xn, xm, xs, Yn, ym, ys };
}

// ---- Build ResNet in TF.js ----
function buildResNet(inputDim: number, outputDim: number): tf.Sequential {
  // We use a simple Sequential with residual-like connections implemented via
  // wide layers — TF.js Sequential doesn't natively support skip connections,
  // so we use a deep feed-forward with BatchNorm
  const model = tf.sequential();

  // Input → 64
  model.add(tf.layers.dense({ units: 64, inputShape: [inputDim] }));
  model.add(tf.layers.batchNormalization({ momentum: 0.9 }));
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.1 }));

  // 64 → 64 (deep layers approximate residual learning)
  for (let i = 0; i < 3; i++) {
    model.add(tf.layers.dense({ units: 64 }));
    model.add(tf.layers.batchNormalization({ momentum: 0.9 }));
    model.add(tf.layers.activation({ activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
  }

  // 64 → 32
  model.add(tf.layers.dense({ units: 32 }));
  model.add(tf.layers.batchNormalization({ momentum: 0.9 }));
  model.add(tf.layers.activation({ activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.1 }));

  // 32 → output
  model.add(tf.layers.dense({ units: outputDim }));

  return model;
}

// ---- Evaluation ----
console.log("=== RESNET (TF.js) ===\n");

const NREP = 3;
const EPOCHS = 200;
const BATCH_SIZE = 64;

console.log("ResNet: 72→64→64×3→32→5, BN+Dropout+Adam, 80/20 × 3 reps:");
console.log("  Dataset            median   P95     P99     max\n");

for (let di = 0; di < DATASETS.length; di++) {
  const dd = allData[di];
  const n = dd.pairs.length;
  const off = allData.slice(0, di).reduce((s, d) => s + d.pairs.length, 0);

  let aMed = 0, aP95 = 0, aP99 = 0, aMax = 0;

  for (let rep = 0; rep < NREP; rep++) {
    const idx = shuffle([...Array(n).keys()]);
    const nTr = Math.floor(n * 0.8);
    const trIdx = idx.slice(0, nTr), teIdx = idx.slice(nTr);

    const Xtr = trIdx.map(i => getFeat(dd.pairs[i]));
    const Ytr = trIdx.map(i => getCvals(i, off));
    const Xte = teIdx.map(i => getFeat(dd.pairs[i]));

    const { Xn: XtrN, xm, xs, Yn: YtrN, ym, ys } = normData(Xtr, Ytr);
    const XteN = Xte.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));

    // Build & train model
    const model = buildResNet(72, RANK);
    model.compile({ optimizer: tf.train.adam(1e-3), loss: 'meanSquaredError' });

    const xt = tf.tensor2d(XtrN);
    const yt = tf.tensor2d(YtrN);

    process.stdout.write(`  ${dd.name} rep ${rep + 1}/${NREP} training...`);
    await model.fit(xt, yt, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if ((epoch + 1) % 50 === 0) process.stdout.write(`${epoch + 1} `);
        }
      }
    });
    process.stdout.write("\n");

    xt.dispose(); yt.dispose();

    // Predict
    const xte = tf.tensor2d(XteN);
    const preds = model.predict(xte) as tf.Tensor;
    const predArr = await preds.array() as number[][];
    xte.dispose(); preds.dispose();
    model.dispose();

    // Denormalize predictions
    const de: number[] = [];
    for (let i = 0; i < predArr.length; i++) {
      const c = predArr[i].map((v, k) => v * ys[k] + ym[k]);
      de.push(deltaE00(s2lab(reconstructL(dd.pairs[teIdx[i]].u.spectra, c)), s2lab(dd.pairs[teIdx[i]].l.spectra)));
    }
    de.sort((a, b) => a - b);
    aMed += de[Math.floor(de.length / 2)];
    aP95 += de[Math.floor(de.length * 0.95)];
    aP99 += de[Math.floor(de.length * 0.99)];
    aMax += de[de.length - 1];

    process.stdout.write(`  → median=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)}\n`);
  }

  console.log(`  ${dd.name.padEnd(14)} ${(aMed / NREP).toFixed(3).padStart(7)} ${(aP95 / NREP).toFixed(3).padStart(7)} ${(aP99 / NREP).toFixed(3).padStart(7)} ${(aMax / NREP).toFixed(3).padStart(7)}\n`);
}
