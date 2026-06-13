// CGATS parser — browser-compatible (text input → Patch[])
import { Patch } from "./types.js";

const SPECTRAL_START = 380;
const SPECTRAL_STEP = 10;

export function parseCgatsText(text: string): Patch[] {
  const lines = text.split(/\r?\n/);

  let dataFormatLine = -1;
  let dataStartLine = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === "BEGIN_DATA_FORMAT") dataFormatLine = i;
    if (t === "BEGIN_DATA") dataStartLine = i;
  }
  if (dataFormatLine === -1 || dataStartLine === -1) {
    throw new Error("Failed to find BEGIN_DATA_FORMAT or BEGIN_DATA");
  }

  const headerLine = lines[dataFormatLine + 1].trim();
  const fieldNames = headerLine.split(/\t/);

  const sampleIdIdx = fieldNames.indexOf("SAMPLE_ID");
  const cIdx = fieldNames.indexOf("CMYK_C");
  const mIdx = fieldNames.indexOf("CMYK_M");
  const yIdx = fieldNames.indexOf("CMYK_Y");
  const kIdx = fieldNames.indexOf("CMYK_K");

  const spectralIdxs: number[] = [];
  for (let w = 0; w < 36; w++) {
    const nm = SPECTRAL_START + w * SPECTRAL_STEP;
    const col = `SPECTRAL_NM_${nm}`;
    const idx = fieldNames.indexOf(col);
    if (idx !== -1) spectralIdxs.push(idx);
  }

  const patches: Patch[] = [];
  for (let i = dataStartLine + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "END_DATA" || line.length === 0) break;
    const parts = line.split(/\t/);
    if (parts.length < fieldNames.length) continue;

    const sampleId = sampleIdIdx !== -1 ? parts[sampleIdIdx] : `${i}`;
    const cmyk: [number, number, number, number] = [
      cIdx !== -1 ? parseFloat(parts[cIdx]) : 0,
      mIdx !== -1 ? parseFloat(parts[mIdx]) : 0,
      yIdx !== -1 ? parseFloat(parts[yIdx]) : 0,
      kIdx !== -1 ? parseFloat(parts[kIdx]) : 0,
    ];
    const spectra = new Float64Array(36);
    for (let w = 0; w < spectralIdxs.length; w++) {
      spectra[w] = parseFloat(parts[spectralIdxs[w]]);
    }
    patches.push({ sampleId, cmyk, spectra });
  }
  return patches;
}

// Browser-compatible re-exports from strip-matcher
export { parseCGATS, extractSpectralData } from './strip-matcher.ts';
