import { readFileSync } from "node:fs";
import { parseCgatsFile } from "../src/cgats-parser.js";
import { spectraToXyz, xyzToLab, patchToLab } from "../src/color-math.js";

const D50_WP: [number, number, number] = [96.42, 100, 82.49];

const DATASETS = [
  { name: "R2_11-4-23", path: "Data/CGATS/R2_11-4-23.txt" },
  { name: "R2_27-10-23", path: "Data/CGATS/R2_27-10-23.txt" },
  { name: "R2_13-02-24", path: "Data/CGATS/R2_13-02-24.txt" },
  { name: "R3_23-4-24", path: "Data/CGATS/R3_23-4-24.txt" },
];

const dsData = DATASETS.map(ds => ({
  name: ds.name,
  patches: parseCgatsFile(ds.path),
}));

// --- Paper white (CMYK=0) ---
console.log("=== PAPER WHITE (CMYK=0) COMPARISON ===\n");
console.log("Dataset        L*        a*        b*        R380     R550     R730     meanU");
for (const dd of dsData) {
  const pw = dd.patches.find(p => p.cmyk[0] === 0 && p.cmyk[1] === 0 && p.cmyk[2] === 0 && p.cmyk[3] === 0);
  if (!pw) { console.log(dd.name + ": paper white NOT FOUND"); continue; }
  const lab = patchToLab(pw.spectra, D50_WP);
  let mu = 0;
  for (let w = 0; w < 36; w++) mu += pw.spectra[w];
  mu /= 36;
  console.log(dd.name.padEnd(15) + " " +
    lab[0].toFixed(3).padStart(7) + " " + lab[1].toFixed(3).padStart(7) + " " + lab[2].toFixed(3).padStart(7) + " " +
    pw.spectra[0].toFixed(6).padStart(9) + " " + pw.spectra[17].toFixed(6).padStart(9) + " " + pw.spectra[35].toFixed(6).padStart(9) + " " + mu.toFixed(6));
}

// --- Paper white spectral difference ---
console.log("\n=== PAPER WHITE SPECTRAL Δ (vs R2_11-4-23 baseline) ===\n");
const refDs = dsData[0];
const refPW = refDs.patches.find(p => p.cmyk[0] === 0 && p.cmyk[1] === 0 && p.cmyk[2] === 0 && p.cmyk[3] === 0)!;
console.log("λ     Δ(" + dsData[1].name + ")  Δ(" + dsData[2].name + ")  Δ(" + dsData[3].name + ")");
for (let w = 0; w < 36; w++) {
  const vals = dsData.slice(1).map(dd => {
    const pw = dd.patches.find(p => p.cmyk[0] === 0 && p.cmyk[1] === 0 && p.cmyk[2] === 0 && p.cmyk[3] === 0);
    if (!pw) return NaN;
    return pw.spectra[w] - refPW.spectra[w];
  });
  console.log((380 + w * 10).toString().padStart(4) + "  " +
    vals.map(v => isNaN(v) ? "  N/A   " : (v * 1000).toFixed(3).padStart(8)).join(""));
}

// --- Dark patches ---
console.log("\n=== DARK PATCHES (highest K, highest total ink) COMPARISON ===\n");
console.log("Dataset        CMYK           L*        a*        b*        meanU");
for (const dd of dsData) {
  const sorted = [...dd.patches].sort((a, b) => b.cmyk[3] - a.cmyk[3] || (b.cmyk[0]+b.cmyk[1]+b.cmyk[2]) - (a.cmyk[0]+a.cmyk[1]+a.cmyk[2]));
  const dark = sorted[0];
  const lab = patchToLab(dark.spectra, D50_WP);
  let mu = 0;
  for (let w = 0; w < 36; w++) mu += dark.spectra[w];
  mu /= 36;
  console.log(dd.name.padEnd(15) + " " +
    dark.cmyk.join(",").padEnd(14) + " " +
    lab[0].toFixed(3).padStart(7) + " " + lab[1].toFixed(3).padStart(7) + " " + lab[2].toFixed(3).padStart(7) + " " + mu.toFixed(6));
}

// --- Mean U spectra ---
console.log("\n=== MEAN U SPECTRUM COMPARISON (% reflectance) ===\n");
const meanSpecs: Float64Array[] = dsData.map(dd => {
  const mu = new Float64Array(36);
  for (const p of dd.patches) for (let w = 0; w < 36; w++) mu[w] += p.spectra[w];
  for (let w = 0; w < 36; w++) mu[w] /= dd.patches.length;
  return mu;
});
console.log("λ     " + dsData.map(d => d.name.padStart(12)).join(" "));
for (let w = 0; w < 36; w++) {
  console.log((380 + w * 10).toString().padStart(4) + "  " +
    dsData.map((_, i) => (meanSpecs[i][w] * 100).toFixed(3).padStart(12)).join(""));
}

// --- RMS difference between dataset mean spectra ---
console.log("\n=== RMS DIFFERENCE BETWEEN MEAN U SPECTRA (×10⁻³) ===\n");
for (let i = 0; i < dsData.length; i++) {
  for (let j = i + 1; j < dsData.length; j++) {
    let rms = 0;
    for (let w = 0; w < 36; w++) { const d = meanSpecs[i][w] - meanSpecs[j][w]; rms += d * d; }
    rms = Math.sqrt(rms / 36);
    console.log("  " + dsData[i].name + " vs " + dsData[j].name + ": RMS = " + (rms * 1000).toFixed(4));
  }
}

// --- Paper white Lab ΔE00 ---
console.log("\n=== PAPER WHITE ΔE00 BETWEEN DATASETS ===\n");
const pwLabs = dsData.map(dd => {
  const pw = dd.patches.find(p => p.cmyk[0] === 0 && p.cmyk[1] === 0 && p.cmyk[2] === 0 && p.cmyk[3] === 0);
  return pw ? patchToLab(pw.spectra, D50_WP) : null;
});
for (let i = 0; i < dsData.length; i++) {
  for (let j = i + 1; j < dsData.length; j++) {
    if (!pwLabs[i] || !pwLabs[j]) continue;
    const de = deltaE00(pwLabs[i]!, pwLabs[j]!);
    console.log("  " + dsData[i].name + " vs " + dsData[j].name + ": ΔE00 = " + de.toFixed(4));
  }
}

function deltaE00(lab1: [number, number, number], lab2: [number, number, number]): number {
  const [L1, a1, b1] = lab1;
  const [L2, a2, b2] = lab2;
  const avgL = (L1 + L2) / 2;
  const C1 = Math.sqrt(a1 * a1 + b1 * b1);
  const C2 = Math.sqrt(a2 * a2 + b2 * b2);
  const avgC = (C1 + C2) / 2;
  const avgC7 = Math.pow(avgC, 7);
  const G = 0.5 * (1 - Math.sqrt(avgC7 / (avgC7 + Math.pow(25, 7))));
  const a1p = a1 * (1 + G);
  const a2p = a2 * (1 + G);
  const C1p = Math.sqrt(a1p * a1p + b1 * b1);
  const C2p = Math.sqrt(a2p * a2p + b2 * b2);
  const avgCp = (C1p + C2p) / 2;
  const h1p = Math.atan2(b1, a1p) * 180 / Math.PI;
  const h2p = Math.atan2(b2, a2p) * 180 / Math.PI;
  const h1pDeg = h1p < 0 ? h1p + 360 : h1p;
  const h2pDeg = h2p < 0 ? h2p + 360 : h2p;
  const deltaLp = L2 - L1;
  const deltaCp = C2p - C1p;
  let deltahp: number;
  const diffH = h2pDeg - h1pDeg;
  if (C1p * C2p === 0) deltahp = 0;
  else if (Math.abs(diffH) <= 180) deltahp = diffH;
  else if (diffH > 180) deltahp = diffH - 360;
  else deltahp = diffH + 360;
  const deltaHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(deltahp * Math.PI / 360);
  const avgHp = (C1p * C2p === 0) ? h1pDeg + h2pDeg :
    Math.abs(h1pDeg - h2pDeg) > 180 ? (h1pDeg + h2pDeg + 360) / 2 : (h1pDeg + h2pDeg) / 2;
  const T = 1 - 0.17 * Math.cos((avgHp - 30) * Math.PI / 180)
    + 0.24 * Math.cos(2 * avgHp * Math.PI / 180)
    + 0.32 * Math.cos((3 * avgHp + 6) * Math.PI / 180)
    - 0.2 * Math.cos((4 * avgHp - 63) * Math.PI / 180);
  const SL = 1 + 0.015 * Math.pow(avgL - 50, 2) / Math.sqrt(20 + Math.pow(avgL - 50, 2));
  const SC = 1 + 0.045 * avgCp;
  const SH = 1 + 0.015 * avgCp * T;
  const deltaTheta = 30 * Math.exp(-Math.pow((avgHp - 275) / 25, 2));
  const avgCp7 = Math.pow(avgCp, 7);
  const RC = 2 * Math.sqrt(avgCp7 / (avgCp7 + Math.pow(25, 7)));
  const RT = -RC * Math.sin(2 * deltaTheta * Math.PI / 180);
  return Math.sqrt(
    Math.pow(deltaLp / SL, 2) +
    Math.pow(deltaCp / SC, 2) +
    Math.pow(deltaHp / SH, 2) +
    RT * (deltaCp / SC) * (deltaHp / SH)
  );
}
