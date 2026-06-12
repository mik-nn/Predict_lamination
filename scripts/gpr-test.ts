import { Matrix, SVD, solve, inverse, CholeskyDecomposition } from "ml-matrix";
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

// Load & build global SVD (same as before)
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
function reconstructL(u: Float64Array, c: Float64Array): Float64Array {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) { let d = 0; for (let k = 0; k < RANK; k++) d += c[k] * allV[k][w]; pred[w] = clamp(u[w] + d); }
  return pred;
}

// ---- Gaussian Process Regression ----
class GPR {
  X: number[][]; // training features (normalized)
  y: Float64Array; // training targets (normalized)
  alpha: Float64Array; // weights (N)
  l: number; // lengthscale
  sf2: number; // signal variance
  sn2: number; // noise variance
  n: number;
  dim: number;
  Kchol: Matrix | null; // Cholesky of K + sn2*I

  constructor(l: number, sf2: number, sn2: number) {
    this.l = l; this.sf2 = sf2; this.sn2 = sn2;
    this.X = []; this.y = new Float64Array(0);
    this.alpha = new Float64Array(0);
    this.n = 0; this.dim = 0;
    this.Kchol = null;
  }

  // RBF kernel: k(x,z) = sf² * exp(-||x-z||² / 2ℓ²)
  private kernel(x: number[], z: number[]): number {
    let d2 = 0;
    for (let i = 0; i < x.length; i++) { const d = x[i] - z[i]; d2 += d * d; }
    return this.sf2 * Math.exp(-d2 / (2 * this.l * this.l));
  }

  fit(X: number[][], y: number[]) {
    this.X = X; this.y = new Float64Array(y); this.n = X.length; this.dim = X[0].length;

    // Build kernel matrix K (N×N)
    const K = new Matrix(this.n, this.n);

    // Use X*Xᵀ trick for efficiency
    const Xmat = new Matrix(this.n, this.dim);
    for (let i = 0; i < this.n; i++)
      for (let j = 0; j < this.dim; j++) Xmat.set(i, j, X[i][j]);

    const Xt = Xmat.transpose();
    const XXt = Xmat.mmul(Xt); // N×N dot products

    const norms = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) norms[i] = XXt.get(i, i);

    // K_ij = sf² * exp(-(||xi||² + ||xj||² - 2*<xi,xj>) / 2ℓ²)
    const inv2l2 = 1 / (2 * this.l * this.l);
    for (let i = 0; i < this.n; i++) {
      K.set(i, i, this.sf2 + this.sn2);
      for (let j = i + 1; j < this.n; j++) {
        const d2 = norms[i] + norms[j] - 2 * XXt.get(i, j);
        const v = this.sf2 * Math.exp(-d2 * inv2l2);
        K.set(i, j, v);
        K.set(j, i, v);
      }
    }

    // Cholesky: LLᵀ = K
    const chol = new CholeskyDecomposition(K);
    this.Kchol = chol;

    // α = Lᵀ \ (L \ y)
    // CholeskyDecomposition in ml-matrix gives L such that L*Lᵀ = K
    // solve(L * z = y) → z = L \ y
    // then solve(Lᵀ * α = z) → α = Lᵀ \ z
    const L = chol.lowerTriangularMatrix; // L is lower triangular
    const Lt = L.transpose();

    const yVec = Matrix.columnVector(y);
    const z = solve(L, yVec);
    this.alpha = solve(Lt, z).to1DArray();
  }

  predict(x: number[]): { mean: number; var: number } {
    // k* = kernel between test point and all training points
    const ks = new Float64Array(this.n);
    let kss = this.sf2; // k(x*, x*) = sf² + sn² (without noise for test)
    for (let i = 0; i < this.n; i++) {
      ks[i] = this.kernel(x, this.X[i]);
    }

    // μ = k*ᵀ α
    let mean = 0;
    for (let i = 0; i < this.n; i++) mean += ks[i] * this.alpha[i];

    // v = L \ k*
    const kVec = Matrix.columnVector(Array.from(ks));
    const L = this.Kchol!.lowerTriangularMatrix;
    const v = solve(L, kVec);

    // var = kss - vᵀ v
    let var_ = kss;
    for (let i = 0; i < this.n; i++) var_ -= v.get(i, 0) ** 2;

    return { mean, var: var_ + this.sn2 }; // add noise variance for predictive variance
  }
}

function normFeat(X: number[][]) {
  const p = X[0].length, n = X.length;
  const m = new Float64Array(p), s = new Float64Array(p);
  for (let j = 0; j < p; j++) { let sum = 0; for (const r of X) sum += r[j]; m[j] = sum / n; }
  for (let j = 0; j < p; j++) { let ss = 0; for (const r of X) ss += (r[j] - m[j]) ** 2; s[j] = Math.sqrt(ss / n) + 1e-10; }
  return { normed: X.map(r => r.map((v, j) => (v - m[j]) / s[j])), mean: m, std: s };
}

function trainGPR(XtrN: number[][], y: number[]): { gpr: GPR; yMean: number; yStd: number } {
  const yMean = y.reduce((s, v) => s + v, 0) / y.length;
  const yStd = Math.sqrt(y.reduce((s, v) => s + (v - yMean) ** 2, 0) / y.length) + 1e-10;
  const ytrN = y.map(v => (v - yMean) / yStd);

  const gpr = new GPR(1.0, 1.0, 0.01);
  gpr.fit(XtrN, ytrN);
  return { gpr, yMean, yStd };
}

// ---- Main evaluation ----
console.log("=== GAUSSIAN PROCESS REGRESSION ===\n");

const NREP = 3;

// Evaluate per dataset
console.log("GPR (RBF kernel, grid-search ℓ, per-dataset 80/20 × 3 reps):");
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

    const Xtr = trIdx.map(i => getFeat(dd.pairs[i]));
    const Xte = teIdx.map(i => getFeat(dd.pairs[i]));

    // Normalize
    const { normed: XtrN, mean: fm, std: fs } = normFeat(Xtr);
    const XteN = Xte.map(r => r.map((v, j) => (v - fm[j]) / fs[j]));

    // Use random subset of 300 training points for GPR (full N is too slow)
    const subsetN = Math.min(300, nTr);
    const subIdx = shuffle([...Array(nTr).keys()]).slice(0, subsetN);
    const Xsub = subIdx.map(i => XtrN[i]);
    const subOff = subIdx.map(i => trIdx[i]);

    const gprModels: { gpr: GPR; yMean: number; yStd: number }[] = [];
    for (let k = 0; k < RANK; k++) {
      const yk = subOff.map(i => allCvals[k][off + i]);
      process.stdout.write(`    ${dd.name} rep ${rep + 1}/${NREP} c${k} (n=${subsetN})...`);
      const m = trainGPR(Xsub, yk);
      gprModels.push(m);
      process.stdout.write("ok\n");
    }

    // Evaluate
    const de: number[] = [];
    for (let i = 0; i < XteN.length; i++) {
      const c = new Float64Array(RANK);
      for (let k = 0; k < RANK; k++) {
        const p = gprModels[k].gpr.predict(XteN[i]);
        c[k] = p.mean * gprModels[k].yStd + gprModels[k].yMean;
      }
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

// Reference OLS
console.log("\nReference OLS (U+U² rank-5, 80/20 × 3 reps):");
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

    const Xtr = new Matrix(nTr, 72);
    for (let i = 0; i < nTr; i++) { let c = 0; for (let w = 0; w < 36; w++) Xtr.set(i, c++, dd.pairs[trIdx[i]].u.spectra[w]); for (let w = 0; w < 36; w++) { const u = dd.pairs[trIdx[i]].u.spectra[w]; Xtr.set(i, c++, u * u); } }
    const Ytr = new Matrix(nTr, RANK);
    for (let i = 0; i < nTr; i++) for (let k = 0; k < RANK; k++) Ytr.set(i, k, allCvals[k][off + trIdx[i]]);
    const betas = solve(Xtr.transpose().mmul(Xtr), Xtr.transpose().mmul(Ytr));

    const de: number[] = [];
    for (const ti of teIdx) {
      const feat: number[] = []; for (let w = 0; w < 36; w++) feat.push(dd.pairs[ti].u.spectra[w]); for (let w = 0; w < 36; w++) feat.push(dd.pairs[ti].u.spectra[w] * dd.pairs[ti].u.spectra[w]);
      const c = new Float64Array(RANK); for (let k = 0; k < RANK; k++) { let v = 0; for (let j = 0; j < 72; j++) v += feat[j] * betas.get(j, k); c[k] = v; }
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
