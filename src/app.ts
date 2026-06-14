// Lamination DeviceLink App — browser entry point
// Loads frozen ResNet model, fits Ridge adapter, builds ICC DeviceLink

// @tensorflow/globalThis.tfjs is loaded as a global via CDN (<script src="...">)
// We reference it via globalThis.globalThis.tf throughout this module.
import { buildDeviceLink, allocateCLUT, labToLab8 } from './icc-writer.ts';
import { parseCgatsText, parseCGATS, extractSpectralData } from './cgats-parser.ts';
import { spectralToXYZ, xyzToLab, deltaE00 } from './color-math.ts';
import { ridgeFit, ridgePredict } from './ridge.ts';
import type { Patch } from './types.ts';

import { computeRows, rowDiversityScores, generateSubsetCGATS, verifySubsetMatch } from './strip-matcher.ts';

// Re-export for browser bundle consumers
export { buildDeviceLink, allocateCLUT, labToLab8, downloadICC };
export { parseCgatsText, parseCGATS, extractSpectralData, computeRows, rowDiversityScores, generateSubsetCGATS, verifySubsetMatch };
export { spectralToXYZ, xyzToLab, deltaE00, srgbToLab, labToSrgb } from './color-math.ts';
export { ridgeFit, ridgePredict };
export { applyCLUT, lab8ToLab } from './icc-writer.ts';

// Module-level cache for analyzeRows → processDeviceLinkWithRows
let _cache: {
  uPatches: Patch[];
  allFrozenOut: number[][];
  allDeltaC: number[][];
  normAll: number[][];
  uLab: [number, number, number][];
  predictedLab: [number, number, number][];
  predictedLc: number[][];
} | null = null;

export function clearCache(): void {
  _cache = null;
}

// ---- Baked-in params (from reference dataset R2_11-4-23) ----
// These will be pre-computed and embedded at build time
// For now, they're populated from a JSON endpoint

let BAKED: {
  V: number[][];          // SVD basis: 5×36
  xm: number[];           // feature means (72)
  xs: number[];           // feature stds (72)
  ym: number[];           // c-value means (5)
  ys: number[];           // c-value stds (5)
  modelUrl: string;       // URL to TF.js model
} | null = null;

export async function loadBakedParams(url: string = '/baked-params.json'): Promise<void> {
  const resp = await fetch(url);
  BAKED = await resp.json();
}

// ---- Feature extraction ----
function getFeat(p: Patch): number[] {
  const f: number[] = [];
  for (let w = 0; w < 36; w++) f.push(p.spectra[w]);
  for (let w = 0; w < 36; w++) f.push(p.spectra[w] * p.spectra[w]);
  return f;
}

// ---- Match patches by CMYK ----
function matchByCMYK(unlam: Patch[], lam: Patch[]): { u: Patch; l: Patch }[] {
  const lm = new Map<string, Patch[]>();
  for (const p of lam) {
    const k = p.cmyk.join(",");
    if (!lm.has(k)) lm.set(k, []);
    lm.get(k)!.push(p);
  }
  const out: { u: Patch; l: Patch }[] = [];
  for (const pu of unlam) {
    const k = pu.cmyk.join(",");
    const m = lm.get(k);
    if (m && m.length > 0) {
      out.push({ u: pu, l: m[0] });
      m.shift();
    }
  }
  return out;
}

// ---- Reconstruct L spectra from c-values ----
function clamp(v: number): number {
  return Math.max(0, Math.min(1, v));
}
function reconstructL(u: Float64Array, c: number[], V: number[][]): Float64Array {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    let d = 0;
    for (let k = 0; k < 5; k++) d += c[k] * V[k][w];
    pred[w] = clamp(u[w] + d);
  }
  return pred;
}

// ---- ΔE00 computation with spectral ----
function s2lab(s: Float64Array): [number, number, number] {
  const [X, Y, Z] = spectralToXYZ(Array.from(s));
  return xyzToLab(X, Y, Z);
}

// ---- Full processing pipeline ----
export async function processDeviceLink(
  uText: string,
  lText: string,
  options: {
    clutPoints?: number;
    iccVersion?: 'v2' | 'v4';
    model?: globalThis.tf.GraphModel | globalThis.tf.LayersModel;
  } = {}
): Promise<{ buffer: ArrayBuffer; stats: any }> {
  if (!BAKED) throw new Error("Baked params not loaded. Call loadBakedParams() first.");
  const { V, xm, xs, ym, ys } = BAKED;
  const gp = options.clutPoints ?? 17;
  const RANK = 5;

  // Parse CGATS
  const uPatches = parseCgatsText(uText);
  const lPatches = parseCgatsText(lText);
  const pairs = matchByCMYK(uPatches, lPatches);

  if (pairs.length < 10) {
    throw new Error(`Only ${pairs.length} matched pairs. Need at least 10.`);
  }

  // Extract features for all unlaminated patches
  const allFeatures = uPatches.map(p => getFeat(p));
  const anchorFeatures = pairs.map(p => getFeat(p.u));

  // Normalize features
  const normAll = allFeatures.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));
  const normAnchor = anchorFeatures.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));

  // Run frozen ResNet model
  let model = options.model;
  if (!model) throw new Error("No model provided. Load TF.js model first.");

  const allT = globalThis.tf.tensor2d(normAll);
  const allPred = model.predict(allT) as globalThis.tf.Tensor;
  const allFrozenOut = (await allPred.array() as number[][])
    .map(row => row.map((v, k) => v * ys[k] + ym[k]));
  allT.dispose(); allPred.dispose();

  const anchorT = globalThis.tf.tensor2d(normAnchor);
  const anchorPred = model.predict(anchorT) as globalThis.tf.Tensor;
  const anchorFrozenOut = (await anchorPred.array() as number[][])
    .map(row => row.map((v, k) => v * ys[k] + ym[k]));
  anchorT.dispose(); anchorPred.dispose();

  // Get actual c-values for anchor patches via global SVD
  // We need to compute c-values from the U→L Δ
  // c_k = Δ · V_k  (dot product of Δ-spectra with SVD basis vectors)
  const allDeltaC: number[][] = pairs.map((p, i) => {
    const c = new Array(RANK).fill(0);
    for (let w = 0; w < 36; w++) {
      const d = p.l.spectra[w] - p.u.spectra[w];
      for (let k = 0; k < RANK; k++) c[k] += d * V[k][w];
    }
    return c;
  });

  // Fit Ridge(5→5) residual: predict Δc from frozen c-values
  const anchorDelta = allDeltaC.map((c, i) =>
    c.map((v, k) => v - anchorFrozenOut[i][k])
  );
  const ridgeModel = ridgeFit(anchorFrozenOut, anchorDelta, 0.01);

  // Predict L c-values for ALL unlaminated patches
  const predictedLc: number[][] = allFrozenOut.map((c, i) => {
    const predDelta = new Array(RANK).fill(0);
    for (let j = 0; j < RANK; j++) {
      const cj = c[j];
      if (cj === 0) continue;
      for (let k = 0; k < RANK; k++)
        predDelta[k] += cj * ridgeModel.weights[j][k];
    }
    return c.map((v, k) => v + predDelta[k]);
  });

  // Reconstruct L spectra and compute Lab
  const predictedLab: [number, number, number][] = predictedLc.map((c, i) => {
    const Lspec = reconstructL(uPatches[i].spectra, c, V);
    return s2lab(Lspec);
  });

  // Compute actual Lab from L CGATS for anchor patches
  const actualLab = pairs.map(p => s2lab(p.l.spectra));

  // Compute ΔE00 on anchor patches
  const anchorDe: number[] = [];
  const matchedUIndices = pairs.map(p => uPatches.indexOf(p.u));
  for (let i = 0; i < pairs.length; i++) {
    const pi = matchedUIndices[i];
    if (pi === -1) continue;
    const predLab = predictedLab[pi];
    const actLab = actualLab[i];
    anchorDe.push(deltaE00(predLab[0], predLab[1], predLab[2], actLab[0], actLab[1], actLab[2]));
  }
  anchorDe.sort((a, b) => a - b);

  // ---- Build DeviceLink ICC profile ----
  // For Lab→Lab: CLUT maps unlaminated Lab to laminated Lab
  // Grid: gp^3 entries, each output is Lab8

  // Compute Lab for ALL unlaminated patches
  const uLab = uPatches.map(p => s2lab(p.spectra));

  // Build interpolation look-up
  // We'll grid the Lab cube and interpolate predicted L values
  const Lmin = 0, Lmax = 100;
  const amin = -128, amax = 127;
  const bmin = -128, bmax = 127;

  const clut = allocateCLUT(3, 3, gp);
  for (let i = 0; i < gp; i++) {
    for (let j = 0; j < gp; j++) {
      for (let k = 0; k < gp; k++) {
        const Lt = Lmin + (Lmax - Lmin) * i / (gp - 1);
        const at = amin + (amax - amin) * j / (gp - 1);
        const bt = bmin + (bmax - bmin) * k / (gp - 1);

        // Simple NN interpolation: find nearest unlaminated patch
        let bestD = Infinity;
        let bestIdx = 0;
        for (let pi = 0; pi < uLab.length; pi++) {
          const d = Math.hypot(uLab[pi][0] - Lt, uLab[pi][1] - at, uLab[pi][2] - bt);
          if (d < bestD) { bestD = d; bestIdx = pi; }
        }

        const [L8, a8, b8] = labToLab8(
          predictedLab[bestIdx][0],
          predictedLab[bestIdx][1],
          predictedLab[bestIdx][2]
        );
        clut.setter([i, j, k], [L8, a8, b8]);
      }
    }
  }

  const stats = {
    totalPatches: uPatches.length,
    anchorPatches: pairs.length,
    medianDE: anchorDe[Math.floor(anchorDe.length / 2)] ?? 0,
    p95DE: anchorDe[Math.floor(anchorDe.length * 0.95)] ?? 0,
    maxDE: anchorDe[anchorDe.length - 1] ?? 0,
  };

  const buffer = buildDeviceLink({
    inputChannels: 3,
    outputChannels: 3,
    clutPoints: gp,
    clutData: clut.array,
    description: `Unlaminated→Laminated DeviceLink (${stats.anchorPatches} anchors, P95=${stats.p95DE.toFixed(2)})`,
  });

  return { buffer, stats };
}

// ---- Step-by-step workflow helpers ----

export interface AnalyzeRowsResult {
  totalPatches: number;
  patchesPerRow: number;
  rows: { index: number; patchCount: number; diversity: number; isRecommended: boolean }[];
}

export async function analyzeRows(
  uText: string,
  totalRows: number,
  model: globalThis.tf.LayersModel
): Promise<{
  result: AnalyzeRowsResult;
  uPatches: Patch[];
  frozenCvals: number[][];
}> {
  if (!BAKED) throw new Error("Baked params not loaded. Call loadBakedParams() first.");
  const { xm, xs, ym, ys, V } = BAKED;
  const RANK = 5;

  const uPatches = parseCgatsText(uText);
  const totalPatches = uPatches.length;

  const allFeatures = uPatches.map(p => getFeat(p));
  const normAll = allFeatures.map(r => r.map((v, j) => (v - xm[j]) / xs[j]));

  const t = globalThis.tf.tensor2d(normAll);
  const pred = model.predict(t) as globalThis.tf.Tensor;
  const arr = await pred.array() as number[][];
  t.dispose(); pred.dispose();

  const frozenCvals = arr.map(row => row.map((v, k) => v * ys[k] + ym[k]));

  const { rows, patchesPerRow } = computeRows(totalPatches, totalRows);
  const diversity = rowDiversityScores(frozenCvals, patchesPerRow, totalPatches, 1);

  const allPatchesPerRow = Math.ceil(totalPatches / totalRows);
  const result: AnalyzeRowsResult = {
    totalPatches,
    patchesPerRow: allPatchesPerRow,
    rows: diversity.map(r => ({
      index: r.rowIndex,
      patchCount: r.patchCount,
      diversity: r.diversity,
      isRecommended: r.isRecommended,
    })),
  };

  return { result, uPatches, frozenCvals };
}


