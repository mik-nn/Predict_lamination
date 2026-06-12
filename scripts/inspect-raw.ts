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

for (const ds of [
  { name: "R2_13-02-24", unlam: "Data/CGATS/R2_13-02-24.txt", lam: "Data/CGATS/R2_13-02-24_lam.txt" },
  { name: "R3_23-4-24", unlam: "Data/CGATS/R3_23-4-24.txt", lam: "Data/CGATS/R3_23-4-24_lam.txt" },
]) {
  console.log("\n=== " + ds.name + " ===");
  const unlam = loadPatches(ds.unlam);
  const lam = loadPatches(ds.lam);
  
  // Check for patches with spectra outside [0,1] or near-zero
  let badCount = 0;
  for (const p of unlam) {
    for (let w = 0; w < 36; w++) {
      if (p.spectra[w] < -0.001 || p.spectra[w] > 1.001) {
        badCount++;
        if (badCount <= 5) console.log("UNLAM BAD: id=" + p.id + " CMYK=" + p.c+","+p.m+","+p.y+","+p.k + " w=" + (380+w*10) + " val=" + p.spectra[w]);
        break;
      }
    }
  }
  for (const p of lam) {
    for (let w = 0; w < 36; w++) {
      if (p.spectra[w] < -0.001 || p.spectra[w] > 1.001) {
        badCount++;
        if (badCount <= 5) console.log("LAM BAD: id=" + p.id + " CMYK=" + p.c+","+p.m+","+p.y+","+p.k + " w=" + (380+w*10) + " val=" + p.spectra[w]);
        break;
      }
    }
  }
  console.log("Bad spectra count: " + badCount);
  
  // Show sample patches: first 5
  console.log("Sample patches:");
  for (let i = 0; i < Math.min(5, unlam.length); i++) {
    const pu = unlam[i], pl = lam[i];
    console.log("  id=" + pu.id + " CMYK=" + pu.c+","+pu.m+","+pu.y+","+pu.k);
    console.log("    U: " + Array.from(pu.spectra.slice(0, 5)).map(v => v.toFixed(4)).join(" ") + "...");
    console.log("    L: " + Array.from(pl.spectra.slice(0, 5)).map(v => v.toFixed(4)).join(" ") + "...");
  }
  
  // Find near-white patches (low ink coverage)
  const sorted = unlam.map((p, i) => ({ idx: i, sum: Array.from(p.spectra).reduce((s, v) => s + v, 0) }));
  sorted.sort((a, b) => b.sum - a.sum);
  console.log("Top 5 patches by reflectance (near-white):");
  for (let i = 0; i < 5; i++) {
    const idx = sorted[i].idx;
    const pu = unlam[idx], pl = lam[idx];
    console.log("  id=" + pu.id + " CMYK=" + pu.c+","+pu.m+","+pu.y+","+pu.k + " sumU=" + sorted[i].sum.toFixed(4));
    console.log("    U380=" + pu.spectra[0].toFixed(4) + " L380=" + pl.spectra[0].toFixed(4) + " diff=" + (pl.spectra[0]-pu.spectra[0]).toFixed(4));
  }
}
