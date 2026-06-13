// Export baked-in params for web app
// Reads reference dataset (R2_11-4-23), runs SVD, outputs JSON + TF.js model weights

import * as tf from '@tensorflow/tfjs';
import { Matrix, SVD } from "ml-matrix";
import { parseCgatsFile } from "../src/cgats-parser.node.js";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { spectralToXYZ, xyzToLab, deltaE00 } from '../src/color-math.ts';

function matchByCMYK(unlam: any[], lam: any[]) {
  const lm = new Map<string, any[]>();
  for (const p of lam) { const k = p.cmyk.join(","); if (!lm.has(k)) lm.set(k, []); lm.get(k)!.push(p); }
  const out: { u: any; l: any }[] = [];
  for (const pu of unlam) { const k = pu.cmyk.join(","); const m = lm.get(k); if (m && m.length) { out.push({ u: pu, l: m[0] }); m.shift(); } }
  return out;
}

function getFeat(p: { u: any }): number[] {
  const f: number[] = [];
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w]);
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w] * p.u.spectra[w]);
  return f;
}

function s2lab(s: Float64Array): [number, number, number] {
  const [X, Y, Z] = spectralToXYZ(Array.from(s));
  return xyzToLab(X, Y, Z);
}

async function main() {
  console.log("Exporting baked-in params from reference dataset...\n");

  // ---- Load reference data ----
  const unlam = parseCgatsFile("Data/CGATS/R2_11-4-23.txt");
  const lam = parseCgatsFile("Data/CGATS/R2_11-4-23_lam.txt");
  const pairs = matchByCMYK(unlam, lam);
  console.log(`Loaded ${pairs.length} pairs from R2_11-4-23`);

  // ---- SVD of Δ-spectra ----
  const D = new Matrix(pairs.length, 36);
  for (let i = 0; i < pairs.length; i++)
    for (let w = 0; w < 36; w++)
      D.set(i, w, pairs[i].l.spectra[w] - pairs[i].u.spectra[w]);

  const svd = new SVD(D, { autoTranspose: true });
  const RANK = 5;
  const V: number[][] = [];
  for (let k = 0; k < RANK; k++) {
    const vk: number[] = [];
    for (let w = 0; w < 36; w++) vk.push(svd.rightSingularVectors.get(w, k));
    V.push(vk);
  }
  const expVar = svd.diagonal.slice(0, RANK).reduce((a, b) => a + b, 0) / svd.diagonal.reduce((a, b) => a + b, 0) * 100;
  console.log(`SVD basis: 5×36 (explained var: ${expVar.toFixed(1)}%)`);

  // ---- Compute c-values and feature normalization ----
  const features = pairs.map(p => getFeat(p));
  const cvals = pairs.map((_, i) => {
    const c: number[] = [];
    for (let k = 0; k < RANK; k++) {
      let v = 0;
      for (let w = 0; w < 36; w++) v += D.get(i, w) * V[k][w];
      c.push(v);
    }
    return c;
  });

  // Normalization params
  const pDim = features[0].length, n = features.length;
  const xm = new Float64Array(pDim), xs = new Float64Array(pDim);
  for (let j = 0; j < pDim; j++) { let s = 0; for (const r of features) s += r[j]; xm[j] = s / n; }
  for (let j = 0; j < pDim; j++) { let ss = 0; for (const r of features) ss += (r[j] - xm[j]) ** 2; xs[j] = Math.sqrt(ss / n) + 1e-10; }

  const q = cvals[0].length;
  const ym = new Float64Array(q), ys = new Float64Array(q);
  for (let j = 0; j < q; j++) { let s = 0; for (const r of cvals) s += r[j]; ym[j] = s / n; }
  for (let j = 0; j < q; j++) { let ss = 0; for (const r of cvals) ss += (r[j] - ym[j]) ** 2; ys[j] = Math.sqrt(ss / n) + 1e-10; }

  console.log(`Feature dim: ${pDim}, c-value dim: ${q}`);

  // ---- Train ResNet ----
  const Xn = features.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));
  const Yn = cvals.map(r => r.map((v, j) => (v - ym[j]) / ys[j]));

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

  model.compile({ optimizer: tf.train.adam(1e-3), loss: 'meanSquaredError' });
  const xt = tf.tensor2d(Xn);
  const yt = tf.tensor2d(Yn);
  console.log("\nTraining ResNet (200 epochs)...");
  const history = await model.fit(xt, yt, { epochs: 200, batchSize: 64, validationSplit: 0.1, verbose: 1 });
  const finalLoss = history.history.loss[history.history.loss.length - 1];
  console.log(`Final loss: ${finalLoss.toFixed(6)}`);

  // ---- Evaluate frozen model ----
  const fp = model.predict(tf.tensor2d(Xn)) as tf.Tensor;
  const fArr = await fp.array() as number[][];
  fp.dispose();
  const denormC = fArr.map(row => row.map((v, k) => v * ys[k] + ym[k]));

  const de: number[] = [];
  for (let i = 0; i < pairs.length; i++) {
    const pred = new Float64Array(36);
    for (let w = 0; w < 36; w++) {
      let d = 0;
      for (let k = 0; k < RANK; k++) d += denormC[i][k] * V[k][w];
      pred[w] = Math.max(0, Math.min(1, pairs[i].u.spectra[w] + d));
    }
    const labPred = s2lab(pred);
    const labAct = s2lab(pairs[i].l.spectra);
    de.push(deltaE00(labPred[0], labPred[1], labPred[2], labAct[0], labAct[1], labAct[2]));
  }
  de.sort((a, b) => a - b);
  console.log(`\nFrozen ResNet: median=${de[Math.floor(de.length/2)].toFixed(3)} P95=${de[Math.floor(de.length*0.95)].toFixed(3)}`);

  // ---- Export baked-params.json ----
  const bakedParams = {
    V,
    xm: Array.from(xm),
    xs: Array.from(xs),
    ym: Array.from(ym),
    ys: Array.from(ys),
    modelUrl: '/model/model.json',
  };

  writeFileSync(join('public', 'baked-params.json'), JSON.stringify(bakedParams));
  console.log(`\n→ public/baked-params.json`);

  // ---- Export TF.js model using save handler ----
  const modelDir = join('public', 'model');
  if (!existsSync(modelDir)) mkdirSync(modelDir, { recursive: true });

  // Use withSaveHandler to get the artifacts directly
  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      weightsManifest: [{
        paths: ['weights.bin'],
        weights: artifacts.weightSpecs,
      }],
      format: 'tfjs-layers',
      generatedBy: 'lamination-export',
      convertedAt: new Date().toISOString(),
    };
    writeFileSync(join(modelDir, 'model.json'), JSON.stringify(modelJson));

    // Write weights.bin (already packed Float32 bytes)
    writeFileSync(join(modelDir, 'weights.bin'), Buffer.from(artifacts.weightData));
    console.log(`→ ${modelDir}/model.json + weights.bin (${artifacts.weightData.byteLength} bytes)`);
  }));

  console.log("\nDone!");
  model.dispose();
  xt.dispose(); yt.dispose();
}

main().catch(console.error);
