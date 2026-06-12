import { readFileSync } from "fs";
import { Matrix, SVD } from "ml-matrix";

function loadPatches(filePath: string): any[] {
  const text = readFileSync(filePath, "utf-8");
  const lines = text.split(/\r?\n/);
  let dataStart = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === "BEGIN_DATA") { dataStart = i; break; }
  }
  const patches: any[] = [];
  for (let i = dataStart + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line === "END_DATA") break;
    const cols = line.split(/\t/);
    const s = new Float64Array(36);
    for (let w = 0; w < 36; w++) s[w] = parseFloat(cols[5 + w]);
    patches.push({
      id: cols[0],
      c: parseFloat(cols[1]), m: parseFloat(cols[2]), y: parseFloat(cols[3]), k: parseFloat(cols[4]),
      spectra: s
    });
  }
  return patches;
}

function matchByCMYK(unlam: any[], lam: any[]): any[] {
  const lamByCMYK = new Map<string, any[]>();
  for (const p of lam) {
    const key = p.c+","+p.m+","+p.y+","+p.k;
    if (!lamByCMYK.has(key)) lamByCMYK.set(key, []);
    lamByCMYK.get(key)!.push(p);
  }
  const pairs: any[] = [];
  for (const pu of unlam) {
    const key = pu.c+","+pu.m+","+pu.y+","+pu.k;
    const matches = lamByCMYK.get(key);
    if (matches && matches.length > 0) {
      pairs.push({ unlam: pu, lam: matches[0], cmyk: [pu.c, pu.m, pu.y, pu.k] });
    }
  }
  return pairs;
}

function analyze() {
  const datasets = [
    { name: "R2_11-4-23", unlam: "Data/CGATS/R2_11-4-23.txt", lam: "Data/CGATS/R2_11-4-23_lam.txt" },
    { name: "R2_27-10-23", unlam: "Data/CGATS/R2_27-10-23.txt", lam: "Data/CGATS/R2_27-10-23_lam.txt" },
    { name: "R2_13-02-24", unlam: "Data/CGATS/R2_13-02-24.txt", lam: "Data/CGATS/R2_13-02-24_lam.txt" },
    { name: "R3_23-4-24", unlam: "Data/CGATS/R3_23-4-24.txt", lam: "Data/CGATS/R3_23-4-24_lam.txt" },
  ];

  // Load all data
  const allPairsByDS: { name: string; pairs: any[] }[] = [];
  for (const ds of datasets) {
    const unlam = loadPatches(ds.unlam);
    const lam = loadPatches(ds.lam);
    const pairs = matchByCMYK(unlam, lam);
    allPairsByDS.push({ name: ds.name, pairs });
  }

  // Build D = L - U matrix
  const allPairs: any[] = allPairsByDS.flatMap(ds => ds.pairs);
  console.log("Total pairs: " + allPairs.length);
  
  const n = allPairs.length;
  const D = new Matrix(n, 36);
  for (let i = 0; i < n; i++) {
    for (let w = 0; w < 36; w++) {
      D.set(i, w, allPairs[i].lam.spectra[w] - allPairs[i].unlam.spectra[w]);
    }
  }

  // SVD
  console.log("Computing SVD of D (" + n + " x 36)...");
  const svd = new SVD(D, { autoTranspose: false });
  
  const s = svd.diagonal;
  const totalE = s.reduce((sum, v) => sum + v * v, 0);
  
  console.log("\n=== Singular Values ===");
  let cumE = 0;
  for (let i = 0; i < Math.min(10, s.length); i++) {
    cumE += s[i] * s[i];
    console.log("  σ" + (i+1) + " = " + s[i].toFixed(6) + " (" + (s[i]/s[0]*100).toFixed(2) + "% of σ1, cumE=" + (cumE/totalE*100).toFixed(4) + "%)");
  }
  console.log("  σ1/σ2 = " + (s[0]/s[1]).toFixed(1));

  // v(λ) = right singular vector for σ1
  const V = svd.V;
  console.log("\n=== v(λ) — First Right Singular Vector (spectral shape of lamination) ===");
  console.log("Wavelength | v(λ) normalized");
  for (let w = 0; w < 36; w++) {
    const wl = 380 + w * 10;
    console.log("  " + wl + "nm | " + V.get(w, 0).toFixed(6));
  }

  // c(i) = left singular vector * σ1
  const U = svd.U;
  const coeffs: { ds: string; c: number; m: number; y: number; k: number; ci: number }[] = [];
  for (let i = 0; i < n; i++) {
    coeffs.push({
      ds: allPairs[i].cmyk ? "all" : "all",
      c: allPairs[i].cmyk[0],
      m: allPairs[i].cmyk[1],
      y: allPairs[i].cmyk[2],
      k: allPairs[i].cmyk[3],
      ci: U.get(i, 0) * s[0]
    });
  }

  // Show c(i) by dataset
  let idx = 0;
  for (const ds of allPairsByDS) {
    const dsCoeffs = coeffs.slice(idx, idx + ds.pairs.length);
    idx += ds.pairs.length;
    const vals = dsCoeffs.map(c => c.ci);
    vals.sort((a, b) => a - b);
    const median = vals[Math.floor(vals.length/2)];
    const min = vals[0];
    const max = vals[vals.length - 1];
    const q1 = vals[Math.floor(vals.length*0.25)];
    const q3 = vals[Math.floor(vals.length*0.75)];
    console.log("\n=== c(i) for " + ds.name + " (" + vals.length + " patches) ===");
    console.log("  min=" + min.toFixed(4) + " Q1=" + q1.toFixed(4) + " median=" + median.toFixed(4) + " Q3=" + q3.toFixed(4) + " max=" + max.toFixed(4));
    
    // Show c(i) by K-channel
    const k0 = dsCoeffs.filter(p => p.k === 0).map(p => p.ci);
    const k20 = dsCoeffs.filter(p => p.k === 20).map(p => p.ci);
    if (k0.length > 0) {
      k0.sort((a, b) => a - b);
      const m0 = k0[Math.floor(k0.length/2)];
      const min0 = k0[0], max0 = k0[k0.length-1];
      console.log("  K=0 (n=" + k0.length + "): median=" + m0.toFixed(4) + " range=[" + min0.toFixed(4) + "," + max0.toFixed(4) + "]");
    }
    if (k20.length > 0) {
      k20.sort((a, b) => a - b);
      const m20 = k20[Math.floor(k20.length/2)];
      const min20 = k20[0], max20 = k20[k20.length-1];
      console.log("  K=20 (n=" + k20.length + "): median=" + m20.toFixed(4) + " range=[" + min20.toFixed(4) + "," + max20.toFixed(4) + "]");
    }
  }

  // Now test: predict L = U + c*v for each dataset with c estimated per-dataset
  console.log("\n=== Per-Dataset 1-Parameter Model: L = U + c*σ1*v(λ) ===");
  idx = 0;
  for (const ds of allPairsByDS) {
    const dsPairs = ds.pairs;
    const dsCoeffs: number[] = [];
    for (let i = 0; i < dsPairs.length; i++) {
      dsCoeffs.push(U.get(idx + i, 0) * s[0]);
    }
    idx += dsPairs.length;
    
    const medianC = dsCoeffs.sort((a, b) => a - b)[Math.floor(dsCoeffs.length/2)];
    
    let mae = 0, maxErr = 0, count = 0;
    for (let i = 0; i < dsPairs.length; i++) {
      const c = dsCoeffs[i]; // using per-patch exact c
      for (let w = 0; w < 36; w++) {
        const pred = dsPairs[i].unlam.spectra[w] + c * V.get(w, 0);
        const err = Math.abs(dsPairs[i].lam.spectra[w] - pred);
        mae += err;
        maxErr = Math.max(maxErr, err);
        count++;
      }
    }
    console.log("  " + ds.name + " (per-patch c): MAE=" + (mae/count).toFixed(6) + " maxErr=" + maxErr.toFixed(6));

    // Fixed c for the entire dataset
    mae = 0; maxErr = 0; count = 0;
    for (let i = 0; i < dsPairs.length; i++) {
      for (let w = 0; w < 36; w++) {
        const pred = dsPairs[i].unlam.spectra[w] + medianC * V.get(w, 0);
        const err = Math.abs(dsPairs[i].lam.spectra[w] - pred);
        mae += err;
        maxErr = Math.max(maxErr, err);
        count++;
      }
    }
    console.log("  " + ds.name + " (fixed c=" + medianC.toFixed(4) + "): MAE=" + (mae/count).toFixed(6) + " maxErr=" + maxErr.toFixed(6));
  }

  // Now test: can c(i) be predicted from U(i)?
  console.log("\n=== Predicting c(i) from unlaminated spectrum ===");
  const cVals: number[] = [];
  const meanUVals: number[] = [];
  for (let i = 0; i < n; i++) {
    let sumU = 0;
    for (let w = 0; w < 36; w++) sumU += allPairs[i].unlam.spectra[w];
    meanUVals.push(sumU / 36);
    cVals.push(U.get(i, 0) * s[0]);
  }
  
  // Linear regression: c = β0 + β1 * meanU
  let su = 0, sc = 0, suu = 0, suc = 0;
  for (let i = 0; i < n; i++) {
    su += meanUVals[i];
    sc += cVals[i];
    suu += meanUVals[i] * meanUVals[i];
    suc += meanUVals[i] * cVals[i];
  }
  const β1 = (n * suc - su * sc) / (n * suu - su * su);
  const β0 = (sc - β1 * su) / n;
  console.log("  c = " + β0.toFixed(6) + " + " + β1.toFixed(6) + " × mean(U)");
  
  let maeC = 0, maxErrC = 0;
  for (let i = 0; i < n; i++) {
    const predC = β0 + β1 * meanUVals[i];
    const err = Math.abs(cVals[i] - predC);
    maeC += err;
    maxErrC = Math.max(maxErrC, err);
  }
  console.log("  c prediction: MAE=" + (maeC/n).toFixed(6) + " maxErr=" + maxErrC.toFixed(6));
  
  // Full model: U + (β0 + β1*meanU)*v(λ)
  mae = 0; maxErr = 0; count = 0;
  for (let i = 0; i < n; i++) {
    const predC = β0 + β1 * meanUVals[i];
    for (let w = 0; w < 36; w++) {
      const pred = allPairs[i].unlam.spectra[w] + predC * V.get(w, 0);
      const err = Math.abs(allPairs[i].lam.spectra[w] - pred);
      mae += err;
      maxErr = Math.max(maxErr, err);
      count++;
    }
  }
  console.log("  Full rank-1 model (c from U): MAE=" + (mae/count).toFixed(6) + " maxErr=" + maxErr.toFixed(6) + " (across ALL datasets)");
}

analyze();
