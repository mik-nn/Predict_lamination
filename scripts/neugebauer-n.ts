import { readFileSync } from "fs";
import { parseCgatsFile } from "../src/cgats-parser.js";
import { spectraToXyz, xyzToLab, deltaE00 } from "../src/color-math.js";

const D50_WP: [number, number, number] = [96.42, 100, 82.49];
const DATASETS = [
  { name: "R2_11-4-23", unlam: "Data/CGATS/R2_11-4-23.txt", lam: "Data/CGATS/R2_11-4-23_lam.txt" },
  { name: "R2_27-10-23", unlam: "Data/CGATS/R2_27-10-23.txt", lam: "Data/CGATS/R2_27-10-23_lam.txt" },
  { name: "R2_13-02-24", unlam: "Data/CGATS/R2_13-02-24.txt", lam: "Data/CGATS/R2_13-02-24_lam.txt" },
  { name: "R3_23-4-24", unlam: "Data/CGATS/R3_23-4-24.txt", lam: "Data/CGATS/R3_23-4-24_lam.txt" },
];

function s2lab(s: Float64Array) { return xyzToLab(spectraToXyz(s), D50_WP); }

function matchByCMYK(unlam: any[], lam: any[]) {
  const lamMap = new Map<string, any[]>();
  for (const p of lam) { const k = p.cmyk.join(",");   if (!lamMap.has(k)) lamMap.set(k, []); lamMap.get(k)!.push(p); }
  const pairs: { u: any; l: any }[] = [];
  for (const pu of unlam) { const k = pu.cmyk.join(","); const m = lamMap.get(k); if (m && m.length > 0) { pairs.push({ u: pu, l: m[0] }); m.shift(); } }
  return pairs;
}

// Build Neugebauer primary key from CMYK (0/100 only)
function neugeKey(cmyk: number[]): string {
  return cmyk.map(v => v >= 50 ? 1 : 0).join("");
}

const ALL_PRIMARIES = [
  "0000", "1000", "0100", "0010", "0001",
  "1100", "1010", "1001", "0110", "0101", "0011",
  "1110", "1101", "1011", "0111",
  "1111"
];

// Demichel area coverage
function demichel(cmyk: number[]): number[] {
  const [c, m, y, k] = cmyk.map(v => v / 100);
  const w = (1-c)*(1-m)*(1-y)*(1-k);
  const C = c*(1-m)*(1-y)*(1-k);
  const M = (1-c)*m*(1-y)*(1-k);
  const Y = (1-c)*(1-m)*y*(1-k);
  const K = (1-c)*(1-m)*(1-y)*k;
  const CM = c*m*(1-y)*(1-k);
  const CY = c*(1-m)*y*(1-k);
  const CK = c*(1-m)*(1-y)*k;
  const MY = (1-c)*m*y*(1-k);
  const MK = (1-c)*m*(1-y)*k;
  const YK = (1-c)*(1-m)*y*k;
  const CMY = c*m*y*(1-k);
  const CMK = c*m*(1-y)*k;
  const CYK = c*(1-m)*y*k;
  const MYK = (1-c)*m*y*k;
  const CMYK = c*m*y*k;
  return [w, C, M, Y, K, CM, CY, CK, MY, MK, YK, CMY, CMK, CYK, MYK, CMYK];
}

// Yule-Nielsen Neugebauer prediction
function predictNeuge(cmyk: number[], primaries: Map<string, Float64Array>, n: number): Float64Array {
  const areas = demichel(cmyk);
  const out = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    let sum = 0;
    for (let p = 0; p < 16; p++) {
      const key = ALL_PRIMARIES[p];
      const spec = primaries.get(key);
      if (spec) sum += areas[p] * Math.pow(spec[w], 1/n);
    }
    out[w] = Math.pow(Math.max(sum, 0), n);
  }
  return out;
}

// RMS error between prediction and measurement (across all wavelengths)
function rmsError(pred: Float64Array, meas: Float64Array): number {
  let s = 0;
  for (let w = 0; w < 36; w++) { const d = pred[w] - meas[w]; s += d * d; }
  return Math.sqrt(s / 36);
}

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

console.log("=== АНАЛИЗ YULE-NIELSEN NEUGEBAUER: U vs L ===\n");

for (const ds of DATASETS) {
  const pairs = matchByCMYK(parseCgatsFile(ds.unlam), parseCgatsFile(ds.lam));
  console.log(`========== ${ds.name} ==========\n`);

  // Extract Neugebauer primaries (0/100% only)
  const uPrimaries = new Map<string, Float64Array>();
  const lPrimaries = new Map<string, Float64Array>();

  for (const p of pairs) {
    const key = neugeKey(p.u.cmyk);
    // Only store 0/100 patches (not 50% etc.)
    if (p.u.cmyk.every(v => v === 0 || v === 100)) {
      if (!uPrimaries.has(key)) uPrimaries.set(key, p.u.spectra);
      if (!lPrimaries.has(key)) lPrimaries.set(key, p.l.spectra);
    }
  }

  console.log(`  Найдено примари: ${uPrimaries.size}/16 (U), ${lPrimaries.size}/16 (L)`);

  // Find best n for unlaminated by grid search
  function findBestN(primaries: Map<string, Float64Array>, testPairs: { u?: any; l?: any; cmyk: number[]; spectra: Float64Array }[], label: string): { n: number; rmse: number; deList: number[] } {
    let bestN = 1;
    let bestRMSE = Infinity;
    let bestDE: number[] = [];

    // Grid search n from 1 to 5 with step 0.1
    for (let n = 1.0; n <= 5.01; n += 0.1) {
      let totalRMSE = 0;
      const de: number[] = [];
      let count = 0;
      for (const p of testPairs) {
        const pred = predictNeuge(p.cmyk, primaries, n);
        totalRMSE += rmsError(pred, p.spectra);
        const labP = s2lab(p.spectra);
        de.push(deltaE00(s2lab(pred), labP));
        count++;
      }
      const avgRMSE = totalRMSE / count;
      if (avgRMSE < bestRMSE) {
        bestRMSE = avgRMSE;
        bestN = n;
        bestDE = de;
      }
    }
    return { n: bestN, rmse: bestRMSE, deList: bestDE };
  }

  // Test on ALL pairs (not just primaries - to see generalization)
  const uTest = pairs.map(p => ({ cmyk: p.u.cmyk, spectra: p.u.spectra }));
  const lTest = pairs.map(p => ({ cmyk: p.l.cmyk, spectra: p.l.spectra }));

  // Find n for ALL primaries (0/100 only)
  const uNeugePairs = pairs.filter(p => p.u.cmyk.every(v => v === 0 || v === 100))
    .map(p => ({ cmyk: p.u.cmyk, spectra: p.u.spectra }));
  const lNeugePairs = pairs.filter(p => p.l.cmyk.every(v => v === 0 || v === 100))
    .map(p => ({ cmyk: p.l.cmyk, spectra: p.l.spectra }));

  // Fit n on primaries
  console.log(`\n  --- Подбор n на примари (0/100) ---`);
  const uResult = findBestN(uPrimaries, uNeugePairs, "U");
  const lResult = findBestN(lPrimaries, lNeugePairs, "L");

  console.log(`  U (без ламинации): n=${uResult.n.toFixed(2)}, RMSE=${(uResult.rmse*1000).toFixed(3)}×10⁻³`);
  uResult.deList.sort((a, b) => a - b);
  console.log(`    ΔE00 на примари: median=${median(uResult.deList).toFixed(3)}, P95=${uResult.deList[Math.floor(uResult.deList.length*0.95)].toFixed(3)}, max=${uResult.deList[uResult.deList.length-1].toFixed(3)}`);

  console.log(`\n  L (с ламинацией): n=${lResult.n.toFixed(2)}, RMSE=${(lResult.rmse*1000).toFixed(3)}×10⁻³`);
  lResult.deList.sort((a, b) => a - b);
  console.log(`    ΔE00 на примари: median=${median(lResult.deList).toFixed(3)}, P95=${lResult.deList[Math.floor(lResult.deList.length*0.95)].toFixed(3)}, max=${lResult.deList[lResult.deList.length-1].toFixed(3)}`);

  // Test on ALL patches (generalization)
  console.log(`\n  --- Обобщение на ВСЕ патчи с n(U) и n(L) ---`);

  // With U's n, predict U for all patches
  const uAllDE: number[] = [];
  const lAllDE: number[] = [];
  const lCrossDE: number[] = []; // use U's n to predict L

  for (const p of pairs) {
    const cmyk = p.u.cmyk;

    // U prediction (n from U)
    const predU = predictNeuge(cmyk, uPrimaries, uResult.n);
    uAllDE.push(deltaE00(s2lab(predU), s2lab(p.u.spectra)));

    // L prediction (n from L)
    const predL = predictNeuge(cmyk, lPrimaries, lResult.n);
    lAllDE.push(deltaE00(s2lab(predL), s2lab(p.l.spectra)));

    // Cross: use U primaries + U's n to predict L (this won't work, just for comparison)
    // Actually: use L primaries but U's n
    const predCross = predictNeuge(cmyk, lPrimaries, uResult.n);
    lCrossDE.push(deltaE00(s2lab(predCross), s2lab(p.l.spectra)));
  }

  const fmt = (de: number[]) => {
    de.sort((a, b) => a - b);
    return `med=${de[Math.floor(de.length/2)].toFixed(3)}  P95=${de[Math.floor(de.length*0.95)].toFixed(3)}  P99=${de[Math.floor(de.length*0.99)].toFixed(3)}`;
  };
  console.log(`  U (n=${uResult.n.toFixed(2)}):       ${fmt(uAllDE)}`);
  console.log(`  L (n=${lResult.n.toFixed(2)}):       ${fmt(lAllDE)}`);

  // Try using L primaries with U's n
  const lCrossDE2: number[] = [];
  for (const p of pairs) {
    const pred = predictNeuge(p.u.cmyk, lPrimaries, uResult.n);
    lCrossDE2.push(deltaE00(s2lab(pred), s2lab(p.l.spectra)));
  }
  console.log(`  L с n(U)=${uResult.n.toFixed(2)}:        ${fmt(lCrossDE2)}`);

  // Try using U primaries with L's n to predict L
  const lCrossDE3: number[] = [];
  for (const p of pairs) {
    const pred = predictNeuge(p.u.cmyk, uPrimaries, lResult.n);
    lCrossDE3.push(deltaE00(s2lab(pred), s2lab(p.l.spectra)));
  }
  console.log(`  U->L с n(L)=${lResult.n.toFixed(2)}:     ${fmt(lCrossDE3)}`);

  // Compare n spectra: show how n varies by wavelength
  console.log(`\n  --- Поиск n для каждого λ отдельно ---`);
  function perWavelengthN(primaries: Map<string, Float64Array>, testPairs: { cmyk: number[]; spectra: Float64Array }[]): number[] {
    const nBest = new Float64Array(36);
    for (let w = 0; w < 36; w++) {
      let bestN = 1, bestErr = Infinity;
      for (let n = 1.0; n <= 5.01; n += 0.1) {
        let err = 0;
        for (const p of testPairs) {
          const areas = demichel(p.cmyk);
          let sum = 0;
          for (let pi = 0; pi < 16; pi++) {
            const spec = primaries.get(ALL_PRIMARIES[pi]);
            if (spec) sum += areas[pi] * Math.pow(spec[w], 1/n);
          }
          const pred = Math.pow(Math.max(sum, 0), n);
          const d = pred - p.spectra[w];
          err += d * d;
        }
        err = Math.sqrt(err / testPairs.length);
        if (err < bestErr) { bestErr = err; bestN = n; }
      }
      nBest[w] = bestN;
    }
    return Array.from(nBest);
  }

  const uPerWL = perWavelengthN(uPrimaries, uNeugePairs);
  const lPerWL = perWavelengthN(lPrimaries, lNeugePairs);

  console.log("  λ    n(U)    n(L)    Δn");
  for (let w = 0; w < 36; w += 2) {
    console.log(`  ${380+w*10}  ${uPerWL[w].toFixed(2)}   ${lPerWL[w].toFixed(2)}   ${(lPerWL[w] - uPerWL[w]).toFixed(2)}`);
  }

  // Does n change with ink coverage? Try fitting n on halftones vs solids
  console.log(`\n  --- n на разных процентах краски ---`);
  for (const ink of ["C", "M", "Y", "K"]) {
    const idx = ["C","M","Y","K"].indexOf(ink);
    // Collect patches with only this ink at various levels
    const inkPatches = pairs.filter(p => {
      const cmk = p.u.cmyk;
      return cmk.every((v, i) => i === idx || v === 0);
    });

    if (inkPatches.length < 3) continue;

    // For U: fit n on all these single-ink patches
    function fitNOnSubset(testPairs: { cmyk: number[]; spectra: Float64Array }[], primaries: Map<string, Float64Array>): { n: number; rmse: number } {
      let bestN = 1, bestRMSE = Infinity;
      for (let n = 1.0; n <= 5.01; n += 0.1) {
        let totalErr = 0;
        for (const p of testPairs) {
          const pred = predictNeuge(p.cmyk, primaries, n);
          totalErr += rmsError(pred, p.spectra);
        }
        const avg = totalErr / testPairs.length;
        if (avg < bestRMSE) { bestRMSE = avg; bestN = n; }
      }
      return { n: bestN, rmse: bestRMSE };
    }

    const uInk = fitNOnSubset(inkPatches.map(p => ({ cmyk: p.u.cmyk, spectra: p.u.spectra })), uPrimaries);
    const lInk = fitNOnSubset(inkPatches.map(p => ({ cmyk: p.l.cmyk, spectra: p.l.spectra })), lPrimaries);
    console.log(`  ${ink}: n(U)=${uInk.n.toFixed(2)} (RMSE=${(uInk.rmse*1000).toFixed(2)}×10⁻³) → n(L)=${lInk.n.toFixed(2)} (RMSE=${(lInk.rmse*1000).toFixed(2)}×10⁻³)`);
  }
  console.log("");
}
