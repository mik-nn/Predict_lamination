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

function matchByCMYK(unlam: any[], lam: any[]) {
  const lamMap = new Map<string, any[]>();
  for (const p of lam) { const k = p.cmyk.join(","); if (!lamMap.has(k)) lamMap.set(k, []); lamMap.get(k)!.push(p); }
  const pairs: { u: any; l: any }[] = [];
  for (const pu of unlam) { const k = pu.cmyk.join(","); const m = lamMap.get(k); if (m && m.length > 0) { pairs.push({ u: pu, l: m[0] }); m.shift(); } }
  return pairs;
}

function s2lab(s: Float64Array) { return xyzToLab(spectraToXyz(s), D50_WP); }

type Delta = { cmyk: number[]; D: Float64Array; meanD: number; uSpec: Float64Array; lSpec: Float64Array };

// Load
console.log("=== АНАЛИЗ ВЛИЯНИЯ ЛАМИНАЦИИ ПО ТИПАМ КРАСОК ===\n");

for (const ds of DATASETS) {
  const pairs = matchByCMYK(parseCgatsFile(ds.unlam), parseCgatsFile(ds.lam));
  console.log(`\n========== ${ds.name} ==========\n`);

  // Classify patches
  const paper: Delta[] = [];
  const primaries: { [key: string]: Delta[] } = { C: [], M: [], Y: [], K: [] };
  const binaries: { [key: string]: Delta[] } = {};
  const ternaries: Delta[] = [];
  const quads: Delta[] = [];

  for (const p of pairs) {
    const [c, m, y, k] = p.u.cmyk;
    const D = new Float64Array(36);
    let meanD = 0;
    for (let w = 0; w < 36; w++) { D[w] = p.l.spectra[w] - p.u.spectra[w]; meanD += D[w]; }
    meanD /= 36;
    const delta: Delta = { cmyk: p.u.cmyk, D, meanD, uSpec: p.u.spectra, lSpec: p.l.spectra };

    if (c === 0 && m === 0 && y === 0 && k === 0) paper.push(delta);
    else {
      const inks = [c > 0 ? 1 : 0, m > 0 ? 1 : 0, y > 0 ? 1 : 0, k > 0 ? 1 : 0];
      const nInks = inks.reduce((s: number, v: number) => s + v, 0);
      const key = ["C","M","Y","K"].filter((_, i) => inks[i]).join("");
      const isPure = inks.every(v => v === 0) || inks.filter(v => v === 1).length === nInks;

      if (nInks === 1 && isPure) {
        primaries[key].push(delta);
      } else if (nInks === 2 && isPure) {
        if (!binaries[key]) binaries[key] = [];
        binaries[key].push(delta);
      } else if (nInks === 3 && isPure) {
        ternaries.push(delta);
      } else if (nInks === 4 && isPure) {
        quads.push(delta);
      }
    }
  }

  // 1. Spectral delta for paper
  console.log("--- 1. БУМАГА (0% красок) ---");
  for (const p of paper) {
    const lab = s2lab(p.lSpec);
    console.log(`  D(λ): ΔL=${(p.meanD*1000).toFixed(2)}×10⁻³  L*=${lab[0].toFixed(2)}`);
    if (p === paper[0]) {
      console.log("  λ    U(λ)       L(λ)       D(λ)×10³");
      for (let w = 0; w < 36; w += 3)
        console.log(`  ${380+w*10}  ${(p.uSpec[w]*100).toFixed(2)}%   ${(p.lSpec[w]*100).toFixed(2)}%   ${(p.D[w]*1000).toFixed(2)}`);
    }
  }

  // 2. Primaries at 100%
  console.log("\n--- 2. ПРАЙМАРИС 100% ---");
  for (const [name, list] of Object.entries(primaries)) {
    const full = list.find(p => p.cmyk.every(v => v === 0) || (!p.cmyk.every(v => v === 0) && p.cmyk.some(v => v === 100)));
    const half = list.find(p => p.cmyk.some(v => v === 50));
    if (!full) continue;

    // Find matching full coverage: all non-zero inks = 100
    const fullPatch = list.filter(p => p.cmyk.every((v, i) => v === 0 || v === 100))[0];
    const halfPatch = list.filter(p => p.cmyk.every((v, i) => v === 0 || v === 50))[0];

    if (fullPatch) {
      const lab = s2lab(fullPatch.lSpec);
      console.log(`  ${name}100: meanD=${(fullPatch.meanD*1000).toFixed(2)}×10⁻³  L*=${lab[0].toFixed(2)}`);
      console.log("  λ    U(λ)       L(λ)       D(λ)×10³");
      for (let w = 0; w < 36; w += 3)
        console.log(`  ${380+w*10}  ${(fullPatch.uSpec[w]*100).toFixed(2)}%   ${(fullPatch.lSpec[w]*100).toFixed(2)}%   ${(fullPatch.D[w]*1000).toFixed(2)}`);
    }
    if (halfPatch) {
      const lab = s2lab(halfPatch.lSpec);
      console.log(`\n  ${name}50: meanD=${(halfPatch.meanD*1000).toFixed(2)}×10⁻³  L*=${lab[0].toFixed(2)}`);
      console.log("  λ    U(λ)       L(λ)       D(λ)×10³");
      for (let w = 0; w < 36; w += 3)
        console.log(`  ${380+w*10}  ${(halfPatch.uSpec[w]*100).toFixed(2)}%   ${(halfPatch.lSpec[w]*100).toFixed(2)}%   ${(halfPatch.D[w]*1000).toFixed(2)}`);
    }
  }

  // 3. Is D additive? D(CM) ≈ D(C) + D(M)?
  console.log("\n--- 3. АДДИТИВНОСТЬ D(λ) ДЛЯ БИНАРОВ ---");
  const inkPairs = ["CM", "CY", "CK", "MY", "MK", "YK"];
  for (const pair of inkPairs) {
    const in1 = pair[0], in2 = pair[1];
    const p1 = primaries[in1]?.find(p => p.cmyk.every((v, i) => v === 0 || (["C","M","Y","K"].indexOf(in1) === i ? v === 100 : v === 0)));
    const p2 = primaries[in2]?.find(p => p.cmyk.every((v, i) => v === 0 || (["C","M","Y","K"].indexOf(in2) === i ? v === 100 : v === 0)));
    const bin = binaries[pair]?.find(p => p.cmyk.every(v => v === 0 || v === 100));

    if (!p1 || !p2 || !bin) continue;

    // Compute D_pred = D1 + D2 (per wavelength)
    const Dpred = new Float64Array(36);
    let rms = 0, maxDev = 0;
    for (let w = 0; w < 36; w++) {
      Dpred[w] = p1.D[w] + p2.D[w];
      const dev = Dpred[w] - bin.D[w];
      rms += dev * dev;
      if (Math.abs(dev) > maxDev) maxDev = Math.abs(dev);
    }
    rms = Math.sqrt(rms / 36);
    const r2 = 1 - rms * rms / (bin.D.reduce((s, v) => s + v*v, 0) / 36);

    // Predict L_pred = U_binary + Dpred, compute ΔE00
    const Lpred = new Float64Array(36);
    for (let w = 0; w < 36; w++) Lpred[w] = Math.max(0, Math.min(1, bin.uSpec[w] + Dpred[w]));
    const de = deltaE00(s2lab(Lpred), s2lab(bin.lSpec));

    console.log(`  ${pair} (${p1.cmyk.join(",")} + ${p2.cmyk.join(",")} → ${bin.cmyk.join(",")}):`);
    console.log(`    D(D_pred=D1+D2): RMS mismatch=${(rms*1000).toFixed(3)}×10⁻³, max dev=${(maxDev*1000).toFixed(2)}×10⁻³`);
    console.log(`    ΔE00(pred D1+D2 vs actual): ${de.toFixed(3)}`);
    console.log(`    R²(actual-binary vs D1+D2): ${r2.toFixed(4)}`);
  }

  // 4. Compare D for 50% vs 100% of same primary
  console.log("\n--- 4. ЛИНЕЙНОСТЬ D(λ) ПО ПРОЦЕНТУ КРАСКИ ---");
  for (const [name, list] of Object.entries(primaries)) {
    const full = list.find(p => p.cmyk.every((v, i) => v === 0 || v === 100));
    const half = list.find(p => p.cmyk.every((v, i) => v === 0 || v === 50));
    if (!full || !half) continue;

    // Addivity: D(50%) ≈ 0.5 * D(100%) ?
    let rms = 0, maxDev = 0;
    for (let w = 0; w < 36; w++) {
      const dev = half.D[w] - 0.5 * full.D[w];
      rms += dev * dev;
      if (Math.abs(dev) > maxDev) maxDev = Math.abs(dev);
    }
    rms = Math.sqrt(rms / 36);
    const de = deltaE00(
      s2lab((()=>{const p=new Float64Array(36);for(let w=0;w<36;w++)p[w]=Math.max(0,Math.min(1,half.uSpec[w]+0.5*full.D[w]));return p;})()),
      s2lab(half.lSpec)
    );
    console.log(`  ${name}: D50 vs 0.5×D100: RMS=${(rms*1000).toFixed(3)}×10⁻³, maxDev=${(maxDev*1000).toFixed(2)}×10⁻³, ΔE00=${de.toFixed(3)}`);
  }

  // 5. Simple model: predict laminated from unlaminated using just primaries data
  console.log("\n--- 5. ПРОСТАЯ МОДЕЛЬ: ИНТЕРПОЛЯЦИЯ ПО ПРАЙМАРИС ---");
  // For any patch with CMYK, decompose into primaries and predict D
  // D_pred(c,m,y,k) = Σ f_i(c) + Σ f_ij(c,m) + ...
  // Simplest: only use 100% primaries D values, scale by ink percentage
  // D_pred = c/100 * D_C100 + m/100 * D_M100 + y/100 * D_Y100 + k/100 * D_K100

  const c100 = primaries["C"]?.find(p => p.cmyk[0]===100);
  const m100 = primaries["M"]?.find(p => p.cmyk[1]===100);
  const y100 = primaries["Y"]?.find(p => p.cmyk[2]===100);
  const k100 = primaries["K"]?.find(p => p.cmyk[3]===100);

  if (c100 && m100 && y100 && k100) {
    const de: number[] = [];
    for (const p of pairs) {
      const [c, m, y, k] = p.u.cmyk;
      const Dpred = new Float64Array(36);
      for (let w = 0; w < 36; w++)
        Dpred[w] = (c/100)*c100.D[w] + (m/100)*m100.D[w] + (y/100)*y100.D[w] + (k/100)*k100.D[w];
      const Lpred = new Float64Array(36);
      for (let w = 0; w < 36; w++) Lpred[w] = Math.max(0, Math.min(1, p.u.spectra[w] + Dpred[w]));
      de.push(deltaE00(s2lab(Lpred), s2lab(p.l.spectra)));
    }
    de.sort((a, b) => a - b);
    console.log(`  D_pred = c*D(C100) + m*D(M100) + y*D(Y100) + k*D(K100) / 100`);
    console.log(`  median=${de[Math.floor(de.length/2)].toFixed(3)}  P95=${de[Math.floor(de.length*0.95)].toFixed(3)}  P99=${de[Math.floor(de.length*0.99)].toFixed(3)}`);

    // With binaries correction
    console.log("\n  С коррекцией бинаров:");
    const cm100 = binaries["CM"]?.find(p => p.cmyk[0]===100&&p.cmyk[1]===100);
    const cy100 = binaries["CY"]?.find(p => p.cmyk[0]===100&&p.cmyk[2]===100);
    const ck100 = binaries["CK"]?.find(p => p.cmyk[0]===100&&p.cmyk[3]===100);
    const my100 = binaries["MY"]?.find(p => p.cmyk[1]===100&&p.cmyk[2]===100);
    const mk100 = binaries["MK"]?.find(p => p.cmyk[1]===100&&p.cmyk[3]===100);
    const yk100 = binaries["YK"]?.find(p => p.cmyk[2]===100&&p.cmyk[3]===100);

    if (cm100 && cy100 && ck100 && my100 && mk100 && yk100) {
      // Neugebauer-like:
      // D_pred = Σ a_i * D_i + Σ a_ij * (D_ij - D_i - D_j)
      // where a_i = ink_i/100, a_ij = min(ink_i, ink_j)/100 for Demichel
      const de2: number[] = [];
      for (const p of pairs) {
        const [c, m, y, k] = p.u.cmyk.map(v => v/100);
        const Dpred = new Float64Array(36);
        for (let w = 0; w < 36; w++) {
          let dp = c*c100.D[w] + m*m100.D[w] + y*y100.D[w] + k*k100.D[w];
          // Subtract overlap (Demichel)
          const cm = Math.min(c, m); const cy = Math.min(c, y); const ck = Math.min(c, k);
          const my = Math.min(m, y); const mk = Math.min(m, k); const yk = Math.min(y, k);
          // Binary correction: use excess D - D_i - D_j
          if (cm100) dp += cm * (cm100.D[w] - c100.D[w] - m100.D[w]);
          if (cy100) dp += cy * (cy100.D[w] - c100.D[w] - y100.D[w]);
          if (ck100) dp += ck * (ck100.D[w] - c100.D[w] - k100.D[w]);
          if (my100) dp += my * (my100.D[w] - m100.D[w] - y100.D[w]);
          if (mk100) dp += mk * (mk100.D[w] - m100.D[w] - k100.D[w]);
          if (yk100) dp += yk * (yk100.D[w] - y100.D[w] - k100.D[w]);
          Dpred[w] = dp;
        }
        const Lpred = new Float64Array(36);
        for (let w = 0; w < 36; w++) Lpred[w] = Math.max(0, Math.min(1, p.u.spectra[w] + Dpred[w]));
        de2.push(deltaE00(s2lab(Lpred), s2lab(p.l.spectra)));
      }
      de2.sort((a, b) => a - b);
      console.log(`  median=${de2[Math.floor(de2.length/2)].toFixed(3)}  P95=${de2[Math.floor(de2.length*0.95)].toFixed(3)}  P99=${de2[Math.floor(de2.length*0.99)].toFixed(3)}`);
    }
  }
}
