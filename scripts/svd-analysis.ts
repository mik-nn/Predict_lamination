import { readFileSync } from "fs";

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
      pairs.push({ unlam: pu, lam: matches[0] });
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

  const allPairs: any[] = [];

  for (const ds of datasets) {
    let unlam = loadPatches(ds.unlam);
    let lam = loadPatches(ds.lam);
    let pairs = matchByCMYK(unlam, lam);
    console.log(ds.name + ": unlam=" + unlam.length + " lam=" + lam.length + " matched=" + pairs.length);
    for (const p of pairs) allPairs.push(p);
  }

  console.log("Total pairs across all datasets: " + allPairs.length);

  // Build matrix: each column = one wavelength, rows interleaved
  // For SVD of (L - U), we want difference per patch
  const n = allPairs.length;
  const diffMatrix: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let w = 0; w < 36; w++) {
      row.push(allPairs[i].lam.spectra[w] - allPairs[i].unlam.spectra[w]);
    }
    diffMatrix.push(row);
  }

  // Compute SVD via covariance (Gram) matrix: D^T D (36x36)
  const gram = new Array(36);
  for (let i = 0; i < 36; i++) {
    gram[i] = new Float64Array(36);
    for (let j = 0; j < 36; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += diffMatrix[k][i] * diffMatrix[k][j];
      gram[i][j] = sum;
    }
  }

  // Jacobi eigenvalue decomposition of Gram matrix
  // The eigenvalues of gram = σ² of original matrix
  let eigenvectors = gram.map((_, i) => { const v = new Float64Array(36); v[i] = 1; return v; });
  let eigenvalues = gram.map((_, i) => gram[i][i]);
  let converged = false;
  for (let iter = 0; iter < 500 && !converged; iter++) {
    converged = true;
    for (let p = 0; p < 35; p++) {
      for (let q = p + 1; q < 36; q++) {
        const a_pp = eigenvalues[p];
        const a_qq = eigenvalues[q];
        const a_pq = gram[p][q];
        const theta = 0.5 * Math.atan2(2 * a_pq, a_qq - a_pp);
        const c = Math.cos(theta);
        const s = Math.sin(theta);

        // Update eigenvalues
        const new_pp = c*c*a_pp + s*s*a_qq - 2*s*c*a_pq;
        const new_qq = s*s*a_pp + c*c*a_qq + 2*s*c*a_pq;
        if (Math.abs(new_pp - a_pp) > 1e-12 || Math.abs(new_qq - a_qq) > 1e-12) converged = false;
        eigenvalues[p] = new_pp;
        eigenvalues[q] = new_qq;

        // Update off-diagonal
        gram[p][q] = c*s*(a_pp - a_qq) + (c*c - s*s)*a_pq;
        gram[q][p] = gram[p][q];

        // Update eigenvectors
        for (let r = 0; r < 36; r++) {
          const old_rp = eigenvectors[r][p];
          const old_rq = eigenvectors[r][q];
          eigenvectors[r][p] = c * old_rp - s * old_rq;
          eigenvectors[r][q] = s * old_rp + c * old_rq;
        }

        // Update remaining off-diagonals
        for (let r = 0; r < 36; r++) {
          if (r !== p && r !== q) {
            const a_rp = gram[Math.min(r,p)][Math.max(r,p)];
            const a_rq = gram[Math.min(r,q)][Math.max(r,q)];
            const new_a_rp = c * a_rp - s * a_rq;
            const new_a_rq = s * a_rp + c * a_rq;
            gram[Math.min(r,p)][Math.max(r,p)] = new_a_rp >= 0 ? new_a_rp : 0;
            gram[Math.min(r,q)][Math.max(r,q)] = new_a_rq >= 0 ? new_a_rq : 0;
          }
        }
      }
    }
  }

  // Sort descending
  const sv = eigenvalues.map((v, i) => ({ val: Math.sqrt(Math.max(0, v)), idx: i }));
  sv.sort((a, b) => b.val - a.val);

  console.log("\n=== Singular Values of (Laminated - Unlaminated) ===");
  console.log("(Pooled across all " + n + " patch pairs)");
  const totalEnergy = sv.reduce((s, v) => s + v.val * v.val, 0);
  let cumEnergy = 0;
  for (let i = 0; i < 10; i++) {
    cumEnergy += sv[i].val * sv[i].val;
    console.log("  σ" + (i+1) + " = " + sv[i].val.toFixed(6) + " (cumulative " + (cumEnergy/totalEnergy*100).toFixed(2) + "%)");
  }
  if (sv.length > 10) {
    const remEnergy = sv.slice(10).reduce((s, v) => s + v.val * v.val, 0);
    console.log("  ... remaining σ11-σ36: " + Math.sqrt(remEnergy/(sv.length-10)).toFixed(6) + " avg");
    console.log("  Total energy (Σσ²) = " + totalEnergy.toFixed(6));
  }

  // Now test the 2-parameter model: L = a + b*U
  // Fit a, b by pooling all patches and all wavelengths
  let sumU = 0, sumL = 0, sumUU = 0, sumUL = 0, count = 0;
  for (const p of allPairs) {
    for (let w = 0; w < 36; w++) {
      const u = p.unlam.spectra[w];
      const l = p.lam.spectra[w];
      sumU += u; sumL += l; sumUU += u*u; sumUL += u*l; count++;
    }
  }
  const b = (count * sumUL - sumU * sumL) / (count * sumUU - sumU * sumU);
  const a = (sumL - b * sumU) / count;
  console.log("\n=== 2-Parameter Model: L = a + b*U ===");
  console.log("  a = " + a.toFixed(6) + " (surface reflection)");
  console.log("  b = " + b.toFixed(6) + " (film transmittance)");

  // Per-dataset residuals
  for (const ds of datasets) {
    let unlam = loadPatches(ds.unlam);
    let lam = loadPatches(ds.lam);
    let pairs = matchByCMYK(unlam, lam);
    let residSum = 0, residCount = 0;
    let maxResid = 0;
    for (const p of pairs) {
      for (let w = 0; w < 36; w++) {
        const pred = a + b * p.unlam.spectra[w];
        const resid = Math.abs(p.lam.spectra[w] - pred);
        residSum += resid;
        residCount++;
        if (resid > maxResid) maxResid = resid;
      }
    }
    console.log("  " + ds.name + ": MAE=" + (residSum/residCount).toFixed(6) + " maxErr=" + maxResid.toFixed(6));
  }
}
analyze();
