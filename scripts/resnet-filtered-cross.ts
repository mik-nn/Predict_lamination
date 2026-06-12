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

function olsPredict(pairs: { u: { spectra: Float64Array }; l: { spectra: Float64Array } }[], off: number): number[] {
  const n = pairs.length;
  const X = new Matrix(n, 72);
  for (let i = 0; i < n; i++) { let c = 0; for (let w = 0; w < 36; w++) X.set(i, c++, pairs[i].u.spectra[w]); for (let w = 0; w < 36; w++) { const u = pairs[i].u.spectra[w]; X.set(i, c++, u * u); } }
  const Y = new Matrix(n, RANK);
  for (let i = 0; i < n; i++) for (let k = 0; k < RANK; k++) Y.set(i, k, allCvals[k][off + i]);
  const betas = solve(X.transpose().mmul(X), X.transpose().mmul(Y));
  const de: number[] = [];
  for (let i = 0; i < n; i++) {
    const feat: number[] = []; for (let w = 0; w < 36; w++) feat.push(pairs[i].u.spectra[w]); for (let w = 0; w < 36; w++) feat.push(pairs[i].u.spectra[w] * pairs[i].u.spectra[w]);
    const c = new Float64Array(RANK); for (let k = 0; k < RANK; k++) { let v = 0; for (let j = 0; j < 72; j++) v += feat[j] * betas.get(j, k); c[k] = v; }
    de.push(deltaE00(s2lab(reconstructL(pairs[i].u.spectra, Array.from(c))), s2lab(pairs[i].l.spectra)));
  }
  return de;
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

async function trainResNet(XtrN: number[][], YtrN: number[][], epochs = 200): Promise<tf.Sequential> {
  const model = buildResNet();
  model.compile({ optimizer: tf.train.adam(1e-3), loss: 'meanSquaredError' });
  const xt = tf.tensor2d(XtrN);
  const yt = tf.tensor2d(YtrN);
  await model.fit(xt, yt, { epochs, batchSize: 64, validationSplit: 0.1, verbose: 0 });
  xt.dispose(); yt.dispose();
  return model;
}

async function resNetPredict(model: tf.Sequential, XteN: number[][], ym: Float64Array, ys: Float64Array): Promise<number[][]> {
  const xte = tf.tensor2d(XteN);
  const preds = model.predict(xte) as tf.Tensor;
  const arr = await preds.array() as number[][];
  xte.dispose(); preds.dispose();
  return arr.map(r => r.map((v, k) => v * ys[k] + ym[k]));
}

function printDE(de: number[], label: string) {
  de.sort((a, b) => a - b);
  console.log(`  ${label.padEnd(16)} median=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)} P99=${de[Math.floor(de.length*0.99)].toFixed(3)} max=${de[de.length-1].toFixed(3)} n=${de.length}`);
}

// ======================================================================
console.log("=== RESNET: OUTLIER FILTER + CROSS-DATASET ===\n");

// ---- Step 1: Outlier filtering per dataset using OLS ----
console.log("Step 1: Filter patches with ΔE00 > 3σ (from OLS fit)\n");

const filteredData: { name: string; entries: { u: any; l: any; gIdx: number }[]; off: number; removed: number }[] = [];

for (let di = 0; di < DATASETS.length; di++) {
  const dd = allData[di];
  const off = allData.slice(0, di).reduce((s, d) => s + d.pairs.length, 0);

  // OLS on all pairs to find outliers (by CMYK)
  const n = dd.pairs.length;
  const X = new Matrix(n, 72);
  for (let i = 0; i < n; i++) { let c = 0; for (let w = 0; w < 36; w++) X.set(i, c++, dd.pairs[i].u.spectra[w]); for (let w = 0; w < 36; w++) { const u = dd.pairs[i].u.spectra[w]; X.set(i, c++, u * u); } }
  const Y = new Matrix(n, RANK);
  for (let i = 0; i < n; i++) for (let k = 0; k < RANK; k++) Y.set(i, k, allCvals[k][off + i]);
  const betas = solve(X.transpose().mmul(X), X.transpose().mmul(Y));

  const de: { idx: number; de: number; key: string }[] = [];
  for (let i = 0; i < n; i++) {
    const feat: number[] = []; for (let w = 0; w < 36; w++) feat.push(dd.pairs[i].u.spectra[w]); for (let w = 0; w < 36; w++) feat.push(dd.pairs[i].u.spectra[w] * dd.pairs[i].u.spectra[w]);
    const c = new Float64Array(RANK); for (let k = 0; k < RANK; k++) { let v = 0; for (let j = 0; j < 72; j++) v += feat[j] * betas.get(j, k); c[k] = v; }
    const pred = reconstructL(dd.pairs[i].u.spectra, Array.from(c));
    de.push({ idx: i, de: deltaE00(s2lab(pred), s2lab(dd.pairs[i].l.spectra)), key: dd.pairs[i].u.cmyk.join(",") });
  }

  const mean = de.reduce((s, v) => s + v.de, 0) / de.length;
  const std = Math.sqrt(de.reduce((s, v) => s + (v.de - mean) ** 2, 0) / de.length);
  const threshold = mean + 3 * std;

  const kept = de.filter(v => v.de <= threshold);
  const removed = de.filter(v => v.de > threshold);
  const entries = kept.map(v => ({ u: dd.pairs[v.idx].u, l: dd.pairs[v.idx].l, gIdx: v.idx }));
  filteredData.push({ name: dd.name, entries, off, removed: removed.length });

  console.log(`  ${dd.name}: ${removed.length}/${n} removed (${(removed.length/n*100).toFixed(1)}%), threshold=mean+3σ=${threshold.toFixed(2)}`);
  if (removed.length > 0) {
    const rde = removed.map(v => v.de);
    console.log(`    ΔE removed: ${rde.reduce((a,b)=>Math.min(a,b),Infinity).toFixed(2)} - ${rde.reduce((a,b)=>Math.max(a,b),-Infinity).toFixed(2)}`);
    for (let r = 0; r < Math.min(3, removed.length); r++) console.log(`    [${removed[r].key}] ΔE=${removed[r].de.toFixed(2)}`);
  }
}

// Build CMYK-keyed cross-dataset lookup
const cmykIndex = filteredData.map(dd => {
  const map = new Map<string, number>();
  for (let i = 0; i < dd.entries.length; i++) map.set(dd.entries[i].u.cmyk.join(","), i);
  return map;
});

// ---- Step 2: ResNet per-dataset (after filtering) ----
console.log("\n\nStep 2: ResNet per-dataset (filtered data, 80/20 × 3 reps)");
console.log("=".repeat(80));

const NREP = 2;
const PER_DATASET_EPOCHS = 150;

for (let di = 0; di < DATASETS.length; di++) {
  const dd = filteredData[di];
  const n = dd.entries.length;
  if (n < 50) { console.log(`  ${dd.name}: too few patches (${n}), skipping`); continue; }

  let aMed = 0, aP95 = 0, aP99 = 0, aMax = 0;
  for (let rep = 0; rep < NREP; rep++) {
    const idx = shuffle([...Array(n).keys()]);
    const nTr = Math.floor(n * 0.8);
    const trIdx = idx.slice(0, nTr), teIdx = idx.slice(nTr);

    const Xtr = trIdx.map(i => getFeat(dd.entries[i]));
    const Ytr = trIdx.map(i => getCvals(dd.entries[i].gIdx, dd.off));
    const Xte = teIdx.map(i => getFeat(dd.entries[i]));

    const { Xn: XtrN, xm, xs, Yn: YtrN, ym, ys } = normData(Xtr, Ytr);
    const XteN = Xte.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));

    process.stdout.write(`  ${dd.name} rep ${rep + 1}/${NREP}...`);
    const model = await trainResNet(XtrN, YtrN, PER_DATASET_EPOCHS);
    const preds = await resNetPredict(model, XteN, ym, ys);

    const de: number[] = [];
    for (let i = 0; i < preds.length; i++)
      de.push(deltaE00(s2lab(reconstructL(dd.entries[teIdx[i]].u.spectra, preds[i])), s2lab(dd.entries[teIdx[i]].l.spectra)));
    de.sort((a, b) => a - b);
    aMed += de[Math.floor(de.length / 2)];
    aP95 += de[Math.floor(de.length * 0.95)];
    aP99 += de[Math.floor(de.length * 0.99)];
    aMax += de[de.length - 1];
    process.stdout.write(` median=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)}\n`);
    model.dispose();
  }
  console.log(`  → ${dd.name.padEnd(14)} median=${(aMed/NREP).toFixed(3)} P95=${(aP95/NREP).toFixed(3)} P99=${(aP99/NREP).toFixed(3)} max=${(aMax/NREP).toFixed(3)}`);
}

// ---- Step 3: Reference OLS per-dataset (filtered) ----
  console.log("\n\n  Reference OLS on filtered data (80/20 × 2 reps):");
for (let di = 0; di < DATASETS.length; di++) {
  const dd = filteredData[di];
  const n = dd.entries.length;
  if (n < 50) continue;
  const off = dd.off;
  let aMed = 0, aP95 = 0, aP99 = 0, aMax = 0;
  for (let rep = 0; rep < NREP; rep++) {
    const idx = shuffle([...Array(n).keys()]);
    const nTr = Math.floor(n * 0.8);
    const trIdx = idx.slice(0, nTr), teIdx = idx.slice(nTr);
    const trEntries = trIdx.map(i => dd.entries[i]);
    const teEntries = teIdx.map(i => dd.entries[i]);
    // Build OLS using original global indices for cvals
    const Xtr = new Matrix(trEntries.length, 72);
    for (let i = 0; i < trEntries.length; i++) { let c = 0; for (let w = 0; w < 36; w++) Xtr.set(i, c++, trEntries[i].u.spectra[w]); for (let w = 0; w < 36; w++) { const u = trEntries[i].u.spectra[w]; Xtr.set(i, c++, u * u); } }
    const Ytr = new Matrix(trEntries.length, RANK);
    for (let i = 0; i < trEntries.length; i++) for (let k = 0; k < RANK; k++) Ytr.set(i, k, allCvals[k][off + trEntries[i].gIdx]);
    const betas = solve(Xtr.transpose().mmul(Xtr), Xtr.transpose().mmul(Ytr));

    const de: number[] = [];
    for (let i = 0; i < teEntries.length; i++) {
      const feat: number[] = []; for (let w = 0; w < 36; w++) feat.push(teEntries[i].u.spectra[w]); for (let w = 0; w < 36; w++) feat.push(teEntries[i].u.spectra[w] * teEntries[i].u.spectra[w]);
      const c = new Float64Array(RANK); for (let k = 0; k < RANK; k++) { let v = 0; for (let j = 0; j < 72; j++) v += feat[j] * betas.get(j, k); c[k] = v; }
      de.push(deltaE00(s2lab(reconstructL(teEntries[i].u.spectra, Array.from(c))), s2lab(teEntries[i].l.spectra)));
    }
    de.sort((a, b) => a - b);
    aMed += de[Math.floor(de.length / 2)];
    aP95 += de[Math.floor(de.length * 0.95)];
    aP99 += de[Math.floor(de.length * 0.99)];
    aMax += de[de.length - 1];
  }
  console.log(`  ${dd.name.padEnd(14)} median=${(aMed/NREP).toFixed(3)} P95=${(aP95/NREP).toFixed(3)} P99=${(aP99/NREP).toFixed(3)} max=${(aMax/NREP).toFixed(3)}`);
}

// ---- Step 4: ResNet cross-dataset (filtered) ----
console.log("\n\nStep 3: ResNet cross-dataset (filtered, train/test, 2 reps)");
console.log("=".repeat(80));

for (let trainDi = 0; trainDi < DATASETS.length; trainDi++) {
  const trainDD = filteredData[trainDi];
  if (trainDD.entries.length < 50) continue;

  console.log(`\n  Train: ${trainDD.name} (n=${trainDD.entries.length})`);
  console.log("  Test dataset      median   P95     P99     max");

  for (let testDi = 0; testDi < DATASETS.length; testDi++) {
    const testDD = filteredData[testDi];
    if (testDD.entries.length < 10) continue;

    let aMed = 0, aP95 = 0, aP99 = 0, aMax = 0;
    const reps = testDi === trainDi ? NREP : 1;
    const actualReps = reps;

    for (let rep = 0; rep < reps; rep++) {
      // Train on full training dataset
      const Xtr = trainDD.entries.map(p => getFeat(p));
      const Ytr = trainDD.entries.map(e => { const c: number[] = []; for (let k = 0; k < RANK; k++) c.push(allCvals[k][trainDD.off + e.gIdx]); return c; });
      const { Xn: XtrN, xm, xs, Yn: YtrN, ym, ys } = normData(Xtr, Ytr);

      process.stdout.write(`    → ${testDD.name.padEnd(14)} rep ${rep + 1}/${reps}`);
      const model = await trainResNet(XtrN, YtrN, 200);

      // Test on test dataset
      const Xte = testDD.entries.map(p => getFeat(p));
      const XteN = Xte.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));
      const preds = await resNetPredict(model, XteN, ym, ys);

      const de: number[] = [];
      for (let i = 0; i < preds.length; i++)
        de.push(deltaE00(s2lab(reconstructL(testDD.entries[i].u.spectra, preds[i])), s2lab(testDD.entries[i].l.spectra)));
      de.sort((a, b) => a - b);
      aMed += de[Math.floor(de.length / 2)];
      aP95 += de[Math.floor(de.length * 0.95)];
      aP99 += de[Math.floor(de.length * 0.99)];
      aMax += de[de.length - 1];
      process.stdout.write(` med=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)}\n`);
      model.dispose();
    }
    console.log(`  → ${testDD.name.padEnd(14)} ${(aMed/actualReps).toFixed(3).padStart(7)} ${(aP95/actualReps).toFixed(3).padStart(7)} ${(aP99/actualReps).toFixed(3).padStart(7)} ${(aMax/actualReps).toFixed(3).padStart(7)}`);
  }
}
