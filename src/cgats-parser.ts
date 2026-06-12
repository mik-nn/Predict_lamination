import { readFileSync } from "node:fs";
import { Patch } from "../src/types.js";

const WAVELENGTH_COLS = 36;
const SPECTRAL_START = 380;
const SPECTRAL_STEP = 10;

export function parseCgatsFile(filePath: string): Patch[] {
  const text = readFileSync(filePath, "utf-8");
  const lines = text.split(/\r?\n/);

  let dataFormatLine = -1;
  let dataStartLine = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === "BEGIN_DATA_FORMAT") dataFormatLine = i;
    if (t === "BEGIN_DATA") dataStartLine = i;
  }

  if (dataFormatLine === -1 || dataStartLine === -1) {
    throw new Error("Failed to find BEGIN_DATA_FORMAT or BEGIN_DATA in " + filePath);
  }

  const headerLine = lines[dataFormatLine + 1].trim();
  const fieldNames = headerLine.split(/\t/);

  const sampleIdIdx = fieldNames.indexOf("SAMPLE_ID");
  const cIdx = fieldNames.indexOf("CMYK_C");
  const mIdx = fieldNames.indexOf("CMYK_M");
  const yIdx = fieldNames.indexOf("CMYK_Y");
  const kIdx = fieldNames.indexOf("CMYK_K");

  const spectralIndices: number[] = [];
  for (let i = 0; i < WAVELENGTH_COLS; i++) {
    const wl = SPECTRAL_START + i * SPECTRAL_STEP;
    const name = "SPECTRAL_NM_" + wl;
    const idx = fieldNames.indexOf(name);
    if (idx === -1) throw new Error("Column " + name + " not found in " + filePath);
    spectralIndices.push(idx);
  }

  const patches: Patch[] = [];
  for (let i = dataStartLine + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line === "END_DATA") break;
    const cols = line.split(/\t/);
    if (cols.length < spectralIndices[spectralIndices.length - 1] + 1) continue;

    const spectra = new Float64Array(WAVELENGTH_COLS);
    for (let j = 0; j < WAVELENGTH_COLS; j++) {
      spectra[j] = parseFloat(cols[spectralIndices[j]]);
    }

    patches.push({
      sampleId: cols[sampleIdIdx],
      cmyk: [
        parseFloat(cols[cIdx]),
        parseFloat(cols[mIdx]),
        parseFloat(cols[yIdx]),
        parseFloat(cols[kIdx]),
      ],
      spectra,
    });
  }

  return patches;
}

export function matchPatches(unlam: Patch[], lam: Patch[]): { unlam: Patch; lam: Patch }[] {
  const lamMap = new Map<string, Patch>();
  for (const p of lam) lamMap.set(p.sampleId, p);

  const pairs: { unlam: Patch; lam: Patch }[] = [];
  for (const p of unlam) {
    const match = lamMap.get(p.sampleId);
    if (match) pairs.push({ unlam: p, lam: match });
  }
  return pairs;
}

export function findPaperWhite(patches: Patch[]): Patch | undefined {
  return patches.find(
    p => p.cmyk[0] === 0 && p.cmyk[1] === 0 && p.cmyk[2] === 0 && p.cmyk[3] === 0
  );
}
