import { Matrix, SVD, solve, inverse } from "ml-matrix";
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

// ---- OLS helpers ----
function makeQuadX(rows: number, getU: (i: number, w: number) => number): Matrix {
  const X = new Matrix(rows, 72);
  for (let i = 0; i < rows; i++) { let c = 0; for (let w = 0; w < 36; w++) X.set(i, c++, getU(i, w)); for (let w = 0; w < 36; w++) { const u = getU(i, w); X.set(i, c++, u * u); } }
  return X;
}
function ols(X: Matrix, Y: Matrix): Matrix { return solve(X.transpose().mmul(X), X.transpose().mmul(Y)); }
function ridge(X: Matrix, Y: Matrix, lambda: number): Matrix {
  const p = X.columns;
  return solve(X.transpose().mmul(X).add(Matrix.eye(p).mul(lambda)), X.transpose().mmul(Y));
}
function predictCvals(feat: number[], betas: Matrix): Float64Array {
  const c = new Float64Array(RANK);
  for (let k = 0; k < RANK; k++) { let v = 0; for (let j = 0; j < feat.length; j++) v += feat[j] * betas.get(j, k); c[k] = v; }
  return c;
}
function reconstructL(u: Float64Array, c: Float64Array): Float64Array {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < RANK; k++) d += c[k] * allV[k][w]; pred[w] = clamp(u[w] + d); }
  return pred;
}
function getFeat(p: { u: any }): number[] {
  const f: number[] = [];
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w]);
  for (let w = 0; w < 36; w++) f.push(p.u.spectra[w] * p.u.spectra[w]);
  return f;
}

// ======================================================================
// PART 1: D-OPTIMAL ANCHORS
// ======================================================================
console.log("=== PART 1: D-OPTIMAL ANCHORS ===\n");

function dOptimal(allFeat: number[][], N: number, seeds: number[], lambda: number): number[] {
  const n = allFeat.length, p = allFeat[0].length;
  const sel = new Set(seeds);
  let X = new Matrix(seeds.length, p);
  for (let i = 0; i < seeds.length; i++)
    for (let j = 0; j < p; j++) X.set(i, j, allFeat[seeds[i]][j]);
  let XtX = X.transpose().mmul(X);
  for (let j = 0; j < p; j++) XtX.set(j, j, XtX.get(j, j) + lambda);
  let Xinv = inverse(XtX);

  while (sel.size < Math.min(N, n)) {
    let bestI = -1; let bestH = -1;
    for (let i = 0; i < n; i++) {
      if (sel.has(i)) continue;
      let h = 0;
      for (let a = 0; a < p; a++) {
        let sa = 0; for (let b = 0; b < p; b++) sa += Xinv.get(a, b) * allFeat[i][b];
        h += allFeat[i][a] * sa;
      }
      if (h > bestH) { bestH = h; bestI = i; }
    }
    if (bestI < 0) break;
    sel.add(bestI);
    const xi = allFeat[bestI];
    const v = new Array(p);
    for (let a = 0; a < p; a++) { let s = 0; for (let b = 0; b < p; b++) s += Xinv.get(a, b) * xi[b]; v[a] = s; }
    let denom = 1; for (let a = 0; a < p; a++) denom += xi[a] * v[a];
    for (let a = 0; a < p; a++)
      for (let b = 0; b < p; b++)
        Xinv.set(a, b, Xinv.get(a, b) - v[a] * v[b] / denom);
  }
  return [...sel].slice(0, Math.min(N, n));
}

function evalAnchors(di: number, anchors: number[], lambda: number) {
  const dd = allData[di];
  const n = dd.pairs.length;
  const off = allData.slice(0, di).reduce((s, d) => s + d.pairs.length, 0);
  const as = new Set(anchors);
  if (as.size === n) as.clear(); // if all anchors, evaluate all (no exclude)

  const Xa = makeQuadX(anchors.length, (i, w) => dd.pairs[anchors[i]].u.spectra[w]);
  const Ya = new Matrix(anchors.length, RANK);
  for (let i = 0; i < anchors.length; i++)
    for (let k = 0; k < RANK; k++) Ya.set(i, k, allCvals[k][off + anchors[i]]);
  const betas = ridge(Xa, Ya, lambda);

  const de: number[] = [];
  for (let i = 0; i < n; i++) {
    if (as.has(i) && as.size > 0) continue;
    const c = predictCvals(getFeat(dd.pairs[i]), betas);
    de.push(deltaE00(s2lab(reconstructL(dd.pairs[i].u.spectra, c)), s2lab(dd.pairs[i].l.spectra)));
  }
  de.sort((a, b) => a - b);
  return { med: de[Math.floor(de.length/2)], p95: de[Math.floor(de.length*0.95)], p99: de[Math.floor(de.length*0.99)], max: de[de.length-1], n: de.length };
}

// Seed: farthest-first (k-means++ style, up to 5)
function farthestSeeds(allFeat: number[][], nSeeds: number): number[] {
  const n = allFeat.length;
  const seeds: number[] = [];
  const pwIdx = allData[DATASETS.length - 1].pairs.findIndex(p => p.u.cmyk.every((v: number) => v === 0)); // fallback
  const pw = allFeat.findIndex(f => f.every(v => v === 0));
  seeds.push(pw >= 0 ? pw : 0);
  while (seeds.length < nSeeds && seeds.length < n) {
    let bestD = -1, bestI = -1;
    for (let i = 0; i < n; i++) {
      if (seeds.includes(i)) continue;
      let minD = Infinity;
      for (const s of seeds) { let d = 0; for (let j = 0; j < allFeat[i].length; j++) { const diff = allFeat[i][j] - allFeat[s][j]; d += diff * diff; } if (d < minD) minD = d; }
      if (minD > bestD) { bestD = minD; bestI = i; }
    }
    if (bestI >= 0) seeds.push(bestI); else break;
  }
  return seeds;
}

const LAMBDA = 1e-6;
const ANCHOR_COUNTS = [3, 5, 10, 15, 20, 30, 50, 100, 200, 500];

for (let di = 0; di < DATASETS.length; di++) {
  const dd = allData[di];
  const n = dd.pairs.length;
  const allFeat = [...Array(n).keys()].map(i => getFeat(dd.pairs[i]));
  const seeds = farthestSeeds(allFeat, 5);

  console.log(`\n${dd.name} (n=${n}):`);
  console.log("  N    median   P95     P99     max    (non-anchor)");
  for (const N of ANCHOR_COUNTS) {
    if (N > n) continue;
    const anchors = dOptimal(allFeat, N, seeds.slice(0, Math.min(N, 5)), LAMBDA);
    const r = evalAnchors(di, anchors, LAMBDA);
    console.log(`  ${String(N).padStart(4)} ${r.med.toFixed(3).padStart(7)} ${r.p95.toFixed(3).padStart(7)} ${r.p99.toFixed(3).padStart(7)} ${r.max.toFixed(3).padStart(7)}  (n_eval=${r.n})`);
  }
  const rAll = evalAnchors(di, [...Array(n).keys()], LAMBDA);
  console.log(`  all  ${rAll.med.toFixed(3).padStart(7)} ${rAll.p95.toFixed(3).padStart(7)} ${rAll.p99.toFixed(3).padStart(7)} ${rAll.max.toFixed(3).padStart(7)}`);
}

// D-optimal vs Random comparison
console.log("\n--- D-optimal vs Random (R2_11-4-23) ---");
const CMP_DS = 0;
const cmpFeat = [...Array(allData[CMP_DS].pairs.length).keys()].map(i => getFeat(allData[CMP_DS].pairs[i]));
const cmpSeeds = farthestSeeds(cmpFeat, 1);
console.log("  N    Dopt-med  Dopt-P95  Rand-med  Rand-P95  Ratio");
for (const N of ANCHOR_COUNTS) {
  if (N > allData[CMP_DS].pairs.length) continue;
  const dopt = dOptimal(cmpFeat, N, cmpSeeds.slice(0, Math.min(N, 1)), LAMBDA);
  const rD = evalAnchors(CMP_DS, dopt, LAMBDA);
  let sMed = 0, sP95 = 0;
  for (let rep = 0; rep < 5; rep++) {
    const rI = shuffle([...Array(allData[CMP_DS].pairs.length).keys()]).slice(0, N);
    const rR = evalAnchors(CMP_DS, rI, LAMBDA);
    sMed += rR.med; sP95 += rR.p95;
  }
  console.log(`  ${String(N).padStart(4)}  ${rD.med.toFixed(3).padStart(8)}  ${rD.p95.toFixed(3).padStart(8)}  ${(sMed/5).toFixed(3).padStart(8)}  ${(sP95/5).toFixed(3).padStart(8)}  ${(rD.p95/(sP95/5)).toFixed(2).padStart(6)}`);
}

// ======================================================================
// PART 2: RANDOM FOREST
// ======================================================================
console.log("\n\n=== PART 2: RANDOM FOREST ===\n");

class RFNode {
  featIdx = 0; threshold = 0; left: RFNode | null = null; right: RFNode | null = null;
  isLeaf = false; values = new Float64Array(RANK);
}

function buildTree(X: number[][], Y: number[][], idx: number[], depth: number, maxDepth: number, minSamp: number): RFNode {
  const n = idx.length;
  const node = new RFNode();
  if (n <= minSamp || depth >= maxDepth) {
    node.isLeaf = true;
    for (let k = 0; k < RANK; k++) { let s = 0; for (const i of idx) s += Y[i][k]; node.values[k] = s / n; }
    return node;
  }

  // Precompute total sums for faster split search
  const totalSum = new Float64Array(RANK);
  const totalSqSum = new Float64Array(RANK);
  for (const i of idx) for (let k = 0; k < RANK; k++) { const v = Y[i][k]; totalSum[k] += v; totalSqSum[k] += v * v; }
  let totalSS = 0;
  for (let k = 0; k < RANK; k++) totalSS += totalSqSum[k] - totalSum[k] * totalSum[k] / n;

  const p = X[0].length, nTry = Math.max(1, Math.floor(p / 3));
  const featChoices: number[] = [];
  const used = new Set<number>();
  while (featChoices.length < nTry) { const f = Math.floor(Math.random() * p); if (!used.has(f)) { used.add(f); featChoices.push(f); } }

  let bestFeat = -1, bestThresh = 0, bestImp = -1;
  for (const f of featChoices) {
    const pairs = idx.map(i => ({ x: X[i][f], i }));
    pairs.sort((a, b) => a.x - b.x);
    const sums = new Float64Array(RANK);
    const sqSums = new Float64Array(RANK);
    // Try thresholds at ~20 quantiles for speed
    const nCand = Math.min(20, n - 1);
    const step = Math.max(1, Math.floor((n - 1) / nCand));
    for (let sp = step - 1; sp < n - 1; sp += step) {
      if (pairs[sp].x === pairs[sp + 1].x) continue;
      const nL = sp + 1, nR = n - nL;
      if (nL < minSamp || nR < minSamp) continue;
      // Compute sums up to sp (incremental)
      for (let s = (sp - step + 1 < 0 ? 0 : sp - step + 1); s <= sp; s++) {
        const ii = pairs[s].i;
        for (let k = 0; k < RANK; k++) { const v = Y[ii][k]; sums[k] += v; sqSums[k] += v * v; }
      }
      const thresh = (pairs[sp].x + pairs[sp + 1].x) / 2;
      let imp = 0;
      for (let k = 0; k < RANK; k++) {
        const lVar = sqSums[k] / nL - (sums[k] / nL) ** 2;
        const rSum = totalSum[k] - sums[k];
        const rSq = totalSqSum[k] - sqSums[k];
        const rVar = rSq / nR - (rSum / nR) ** 2;
        imp += totalSS - lVar * nL - rVar * nR;
      }
      if (imp > bestImp) { bestImp = imp; bestFeat = f; bestThresh = thresh; }
    }
  }

  if (bestFeat < 0 || bestImp <= 1e-12) {
    node.isLeaf = true;
    for (let k = 0; k < RANK; k++) { let s = 0; for (const i of idx) s += Y[i][k]; node.values[k] = s / n; }
    return node;
  }
  node.featIdx = bestFeat;
  node.threshold = bestThresh;
  const li = idx.filter(i => X[i][bestFeat] <= bestThresh);
  const ri = idx.filter(i => X[i][bestFeat] > bestThresh);
  node.left = buildTree(X, Y, li, depth + 1, maxDepth, minSamp);
  node.right = buildTree(X, Y, ri, depth + 1, maxDepth, minSamp);
  return node;
}

function predictTree(node: RFNode, x: number[]): Float64Array {
  let n = node;
  while (!n.isLeaf) n = x[n.featIdx] <= n.threshold ? n.left! : n.right!;
  return n.values;
}

function trainForest(X: number[][], Y: number[][], nTrees: number, maxDepth: number, minSamp: number): RFNode[] {
  const n = X.length;
  const forest: RFNode[] = [];
  for (let t = 0; t < nTrees; t++) {
    const bx: number[][] = []; const by: number[][] = [];
    for (let i = 0; i < n; i++) { const ri = Math.floor(Math.random() * n); bx.push(X[ri]); by.push(Y[ri]); }
    forest.push(buildTree(bx, by, bx.map((_, i) => i), 0, maxDepth, minSamp));
    if ((t + 1) % 25 === 0) process.stdout.write(` ${t + 1}`);
  }
  process.stdout.write("\n");
  return forest;
}

function predictForest(forest: RFNode[], x: number[]): Float64Array {
  const sum = new Float64Array(RANK);
  for (const tree of forest) { const p = predictTree(tree, x); for (let k = 0; k < RANK; k++) sum[k] += p[k]; }
  for (let k = 0; k < RANK; k++) sum[k] /= forest.length;
  return sum;
}

function normFeatures(X: number[][]) {
  const p = X[0].length, n = X.length;
  const m = new Float64Array(p), s = new Float64Array(p);
  for (let j = 0; j < p; j++) { let sum = 0; for (const r of X) sum += r[j]; m[j] = sum / n; }
  for (let j = 0; j < p; j++) { let ss = 0; for (const r of X) ss += (r[j] - m[j]) ** 2; s[j] = Math.sqrt(ss / n) + 1e-10; }
  const nX = X.map(r => r.map((v, j) => (v - m[j]) / s[j]));
  return { normed: nX, mean: m, std: s };
}

process.stdout.write("Training RF per dataset (50 trees, maxDepth=5)...\n");
const NREP = 3;
console.log("\n  Dataset (80/20 × 3 reps)  median   P95     P99     max");
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
    const Ytr = trIdx.map(i => { const y: number[] = []; for (let k = 0; k < RANK; k++) y.push(allCvals[k][off + i]); return y; });
    const Xte = teIdx.map(i => getFeat(dd.pairs[i]));

    const { normed: XtrN, mean: fm, std: fs } = normFeatures(Xtr);
    const XteN = Xte.map(r => r.map((v, j) => (v - fm[j]) / fs[j]));

    process.stdout.write(`  ${dd.name} rep ${rep + 1}/${NREP} trees:`);
    const forest = trainForest(XtrN, Ytr, 50, 5, 5);

    const de: number[] = [];
    for (let i = 0; i < XteN.length; i++) {
      const c = predictForest(forest, XteN[i]);
      de.push(deltaE00(s2lab(reconstructL(dd.pairs[teIdx[i]].u.spectra, c)), s2lab(dd.pairs[teIdx[i]].l.spectra)));
    }
    de.sort((a, b) => a - b);
    aMed += de[Math.floor(de.length / 2)];
    aP95 += de[Math.floor(de.length * 0.95)];
    aP99 += de[Math.floor(de.length * 0.99)];
    aMax += de[de.length - 1];
  }
  console.log(`  ${dd.name.padEnd(14)} ${(aMed / NREP).toFixed(3).padStart(7)} ${(aP95 / NREP).toFixed(3).padStart(7)} ${(aP99 / NREP).toFixed(3).padStart(7)} ${(aMax / NREP).toFixed(3).padStart(7)}`);
}

// ======================================================================
// PART 3: REFERENCE OLS
// ======================================================================
console.log("\n=== PART 3: REFERENCE OLS (U+U² rank-5, 80/20 × 3 reps) ===");
console.log("  Dataset            median   P95     P99     max");
for (let di = 0; di < DATASETS.length; di++) {
  const dd = allData[di];
  const n = dd.pairs.length;
  const off = allData.slice(0, di).reduce((s, d) => s + d.pairs.length, 0);

  let aMed = 0, aP95 = 0, aP99 = 0, aMax = 0;
  for (let rep = 0; rep < NREP; rep++) {
    const idx = shuffle([...Array(n).keys()]);
    const nTr = Math.floor(n * 0.8);
    const trIdx = idx.slice(0, nTr), teIdx = idx.slice(nTr);

    const Xtr = makeQuadX(nTr, (i, w) => dd.pairs[trIdx[i]].u.spectra[w]);
    const Ytr = new Matrix(nTr, RANK);
    for (let i = 0; i < nTr; i++)
      for (let k = 0; k < RANK; k++) Ytr.set(i, k, allCvals[k][off + trIdx[i]]);
    const betas = ols(Xtr, Ytr);

    const de: number[] = [];
    for (const ti of teIdx) {
      const c = predictCvals(getFeat(dd.pairs[ti]), betas);
      de.push(deltaE00(s2lab(reconstructL(dd.pairs[ti].u.spectra, c)), s2lab(dd.pairs[ti].l.spectra)));
    }
    de.sort((a, b) => a - b);
    aMed += de[Math.floor(de.length / 2)];
    aP95 += de[Math.floor(de.length * 0.95)];
    aP99 += de[Math.floor(de.length * 0.99)];
    aMax += de[de.length - 1];
  }
  console.log(`  ${dd.name.padEnd(14)} ${(aMed / NREP).toFixed(3).padStart(7)} ${(aP95 / NREP).toFixed(3).padStart(7)} ${(aP99 / NREP).toFixed(3).padStart(7)} ${(aMax / NREP).toFixed(3).padStart(7)}`);
}
